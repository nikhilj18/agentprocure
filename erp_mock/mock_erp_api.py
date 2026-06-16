# erp_mock/mock_erp_api.py
# PURPOSE: Simulates a SAP/Oracle ERP REST API for AgentProcure.
#
# In production, AgentProcure would connect to a real SAP S/4HANA
# or Oracle Fusion API. This mock server replicates those endpoints
# so the integration can be demonstrated without enterprise licensing.
#
# Endpoints:
#   GET  /api/bom/{bom_id}          → fetch BOM from ERP
#   POST /api/purchase_requisition  → create PR in ERP
#   GET  /api/suppliers             → fetch supplier master from ERP
#   GET  /api/health                → server health check
#
# Run server: python3 erp_mock/mock_erp_api.py
# Then test:  curl http://localhost:8000/api/health

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, date
import uvicorn
import random

app = FastAPI(
    title="AgentProcure Mock ERP API",
    description="Simulates SAP S/4HANA / Oracle Fusion procurement endpoints",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MOCK ERP DATA
# ─────────────────────────────────────────────
MOCK_BOMS = {
    "BOM-2024-001": {
        "bom_id":      "BOM-2024-001",
        "description": "LVCT Current Transformer Assembly",
        "created_by":  "nikhil@company.com",
        "plant":       "PLANT-IN-01",
        "items": [
            {"part_no": "MAG-CORE-ETD39", "quantity": 500, "unit": "EA", "target_price": 1.30},
            {"part_no": "MAG-WIRE-1.0MM", "quantity": 50,  "unit": "KG", "target_price": 10.50},
            {"part_no": "ASM-XFMR-LVCT", "quantity": 100, "unit": "EA", "target_price": 19.00},
            {"part_no": "CON-TB-2P",      "quantity": 200, "unit": "EA", "target_price": 0.26},
        ]
    },
    "BOM-2024-002": {
        "bom_id":      "BOM-2024-002",
        "description": "1kW Inverter Core Assembly",
        "created_by":  "nikhil@company.com",
        "plant":       "PLANT-IN-01",
        "items": [
            {"part_no": "IGBT-G4PC50W",   "quantity": 8,   "unit": "EA", "target_price": 4.50},
            {"part_no": "MOSFET-IRF540",  "quantity": 12,  "unit": "EA", "target_price": 0.90},
            {"part_no": "CAP-1210-470U",  "quantity": 16,  "unit": "EA", "target_price": 0.25},
            {"part_no": "ASM-PCB-PWR",    "quantity": 4,   "unit": "EA", "target_price": 36.00},
            {"part_no": "ASM-INV-1KW",    "quantity": 2,   "unit": "EA", "target_price": 87.00},
        ]
    },
    "BOM-2024-003": {
        "bom_id":      "BOM-2024-003",
        "description": "Digital Panel Meter Assembly",
        "created_by":  "nikhil@company.com",
        "plant":       "PLANT-IN-02",
        "items": [
            {"part_no": "IC-MCU-STM32",   "quantity": 10,  "unit": "EA", "target_price": 4.00},
            {"part_no": "IC-MCU-ESP32",   "quantity": 10,  "unit": "EA", "target_price": 3.10},
            {"part_no": "RES-0402-10K",   "quantity": 2000,"unit": "EA", "target_price": 0.003},
            {"part_no": "CAP-0402-100N",  "quantity": 3000,"unit": "EA", "target_price": 0.009},
            {"part_no": "ASM-PANEL-DPM",  "quantity": 20,  "unit": "EA", "target_price": 42.00},
        ]
    }
}

# Purchase Requisition store (in-memory)
PR_STORE = {}
PR_COUNTER = 1000


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────
class PRLineItem(BaseModel):
    part_no:              str
    quantity:             int
    recommended_supplier: str
    unit_price:           float
    currency:             str = "USD"
    delivery_date:        Optional[str] = None

class PurchaseRequisitionRequest(BaseModel):
    bom_id:       str
    requester:    str
    plant:        str
    priority:     str = "Normal"
    scenario:     str = "balanced"
    line_items:   List[PRLineItem]
    total_value:  float
    notes:        Optional[str] = None

class PRResponse(BaseModel):
    pr_number:    str
    status:       str
    message:      str
    created_at:   str
    total_value:  float
    line_count:   int


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    """Server health check — confirms ERP API is running."""
    return {
        "status":    "healthy",
        "service":   "AgentProcure Mock ERP API",
        "version":   "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "message":   "Simulating SAP S/4HANA procurement endpoints"
    }


@app.get("/api/bom/{bom_id}")
def get_bom(bom_id: str):
    """
    Fetch a Bill of Materials from the ERP system.
    Simulates SAP MM module BOM retrieval.
    """
    if bom_id not in MOCK_BOMS:
        raise HTTPException(
            status_code=404,
            detail=f"BOM {bom_id} not found in ERP system"
        )
    bom = MOCK_BOMS[bom_id].copy()
    bom['retrieved_at'] = datetime.now().isoformat()
    bom['erp_system']   = "SAP S/4HANA (Mock)"
    return bom


@app.get("/api/bom")
def list_boms():
    """List all available BOMs in the ERP system."""
    return {
        "boms": [
            {
                "bom_id":      k,
                "description": v["description"],
                "item_count":  len(v["items"]),
                "plant":       v["plant"]
            }
            for k, v in MOCK_BOMS.items()
        ],
        "total": len(MOCK_BOMS)
    }


@app.post("/api/purchase_requisition", response_model=PRResponse)
def create_purchase_requisition(pr: PurchaseRequisitionRequest):
    """
    Creates a Purchase Requisition in the ERP system.
    Simulates SAP MM Purchase Requisition (ME51N) creation.

    In real SAP: this would create a PR document that goes
    through approval workflow before becoming a PO.
    """
    global PR_COUNTER
    PR_COUNTER += 1
    pr_number = f"PR-{datetime.now().year}-{PR_COUNTER:05d}"

    # Store the PR
    PR_STORE[pr_number] = {
        "pr_number":   pr_number,
        "bom_id":      pr.bom_id,
        "requester":   pr.requester,
        "plant":       pr.plant,
        "priority":    pr.priority,
        "scenario":    pr.scenario,
        "line_items":  [item.dict() for item in pr.line_items],
        "total_value": pr.total_value,
        "line_count":  len(pr.line_items),
        "status":      "Pending Approval",
        "created_at":  datetime.now().isoformat(),
        "notes":       pr.notes
    }

    return PRResponse(
        pr_number   = pr_number,
        status      = "Pending Approval",
        message     = f"PR {pr_number} created successfully. "
                      f"Pending approval from procurement manager.",
        created_at  = datetime.now().isoformat(),
        total_value = pr.total_value,
        line_count  = len(pr.line_items)
    )


@app.get("/api/purchase_requisition/{pr_number}")
def get_pr(pr_number: str):
    """Retrieve a specific Purchase Requisition by number."""
    if pr_number not in PR_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"PR {pr_number} not found"
        )
    return PR_STORE[pr_number]


@app.get("/api/purchase_requisitions")
def list_prs():
    """List all Purchase Requisitions created this session."""
    return {
        "purchase_requisitions": [
            {
                "pr_number":   v["pr_number"],
                "bom_id":      v["bom_id"],
                "total_value": v["total_value"],
                "line_count":  v["line_count"],
                "status":      v["status"],
                "created_at":  v["created_at"]
            }
            for v in PR_STORE.values()
        ],
        "total": len(PR_STORE)
    }


@app.get("/api/suppliers")
def get_erp_suppliers():
    """
    Fetch supplier master data from ERP.
    Simulates SAP Supplier Master (BP transaction).
    """
    suppliers = [
        {"vendor_id":"V001","name":"Shenzhen Components Co",  "country":"CN","payment_terms":"Net60","status":"Active"},
        {"vendor_id":"V002","name":"Shanghai Electronics Ltd","country":"CN","payment_terms":"Net45","status":"Active"},
        {"vendor_id":"V003","name":"Taiwan Semiconductor Corp","country":"TW","payment_terms":"Net45","status":"Active"},
        {"vendor_id":"V004","name":"Chennai Precision Parts",  "country":"IN","payment_terms":"Net45","status":"Active"},
        {"vendor_id":"V005","name":"Murata Europe GmbH",       "country":"DE","payment_terms":"Net30","status":"Active"},
        {"vendor_id":"V006","name":"Korea Advanced Components","country":"KR","payment_terms":"Net45","status":"Active"},
    ]
    return {
        "suppliers":    suppliers,
        "total":        len(suppliers),
        "erp_system":   "SAP S/4HANA (Mock)",
        "retrieved_at": datetime.now().isoformat()
    }


@app.get("/api/price_history/{part_no}")
def get_price_history(part_no: str, months: int = 12):
    """
    Fetch historical PO prices for a part from ERP.
    Simulates SAP MM purchasing info records.
    """
    # Generate realistic mock price history
    import numpy as np
    base_prices = {
        "IGBT-G4PC50W": 4.25, "IC-MCU-STM32": 3.85,
        "ASM-XFMR-LVCT": 18.50, "RES-0402-10K": 0.0025,
        "MAG-CORE-ETD39": 1.25
    }
    base = base_prices.get(part_no, 1.0)
    np.random.seed(hash(part_no) % 1000)

    history = []
    for i in range(months):
        month_date = date(2024, max(1, 13-months+i), 1)
        price = base * (1 + np.random.normal(0, 0.05))
        history.append({
            "month":      str(month_date),
            "avg_price":  round(float(price), 4),
            "order_count": int(np.random.randint(2, 8))
        })

    return {
        "part_no":      part_no,
        "price_history":history,
        "months":       months,
        "currency":     "USD"
    }


# ─────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AgentProcure Mock ERP API")
    print("  Simulating SAP S/4HANA procurement endpoints")
    print("="*55)
    print("\n  Endpoints available:")
    print("  GET  http://localhost:8000/api/health")
    print("  GET  http://localhost:8000/api/bom")
    print("  GET  http://localhost:8000/api/bom/{bom_id}")
    print("  POST http://localhost:8000/api/purchase_requisition")
    print("  GET  http://localhost:8000/api/suppliers")
    print("  GET  http://localhost:8000/api/price_history/{part_no}")
    print("\n  Docs: http://localhost:8000/docs")
    print("  Stop: Ctrl+C\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")