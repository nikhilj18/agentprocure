import requests, pandas as pd
from datetime import date, timedelta

ERP_BASE_URL = "http://localhost:8000/api"

class ERPConnector:
    def __init__(self, base_url=ERP_BASE_URL):
        self.base_url = base_url

    def check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200, r.json()
        except Exception as e:
            return False, {"error": str(e)}

    def fetch_bom(self, bom_id):
        r = requests.get(f"{self.base_url}/bom/{bom_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = [{"part_no":i["part_no"],"quantity_required":i["quantity"],
                 "target_price":i.get("target_price"),"preferred_supplier":None}
                for i in data["items"]]
        return pd.DataFrame(rows)

    def list_boms(self):
        r = requests.get(f"{self.base_url}/bom", timeout=10)
        r.raise_for_status()
        return r.json()["boms"]

    def create_pr(self, recommendations, bom_id, requester="AgentProcure", scenario="balanced"):
        delivery = str(date.today() + timedelta(days=45))
        items = [{"part_no":rec["part_no"],"quantity":int(rec.get("quantity_required",1)),
                  "recommended_supplier":rec["recommended_supplier"],
                  "unit_price":float(rec["landed_cost"]),"currency":"USD",
                  "delivery_date":delivery} for rec in recommendations]
        total = sum(float(r.get("line_value",0)) for r in recommendations)
        payload = {"bom_id":bom_id,"requester":requester,"plant":"PLANT-IN-01",
                   "priority":"Normal","scenario":scenario,"line_items":items,
                   "total_value":round(total,2),"notes":f"Auto-generated | {scenario}"}
        r = requests.post(f"{self.base_url}/purchase_requisition", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()

    def list_prs(self):
        r = requests.get(f"{self.base_url}/purchase_requisitions", timeout=10)
        r.raise_for_status()
        return r.json()["purchase_requisitions"]

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  ERP Connector Test")
    print("="*50)
    erp = ERPConnector()
    ok, info = erp.check_connection()
    if not ok:
        print(f"  ERP not reachable: {info}")
        exit(1)
    print(f"  Connected: {info['service']}")
    print("\nBOMs in ERP:")
    for b in erp.list_boms():
        print(f"  {b['bom_id']}: {b['description']} ({b['item_count']} items)")
    print("\nFetching BOM-2024-001:")
    df = erp.fetch_bom("BOM-2024-001")
    print(df.to_string(index=False))
    print("\nCreating Purchase Requisition...")
    recs = [{"part_no":"MAG-CORE-ETD39","recommended_supplier":"Shanghai Electronics Ltd",
             "landed_cost":1.35,"line_value":675.0,"quantity_required":500},
            {"part_no":"ASM-XFMR-LVCT","recommended_supplier":"Chennai Precision Parts",
             "landed_cost":20.20,"line_value":2020.0,"quantity_required":100}]
    pr = erp.create_pr(recs, "BOM-2024-001", "nikhil@company.com")
    print(f"  PR: {pr['pr_number']} | Status: {pr['status']} | Value: ${pr['total_value']:,.2f}")
    print("\nERP test complete!")