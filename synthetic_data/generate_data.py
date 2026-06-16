# synthetic_data/generate_data.py
# Run with: python3 synthetic_data/generate_data.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from datetime import date, timedelta
from database.db_connect import run_insert_many, run_query

np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────
# SECTION 1: 45 COMPONENTS
# ──────────────────────────────────────────────
def generate_components():
    print("Generating components...")
    components = [
        ("RES-0402-10K",  "10K Ohm Resistor 0402",              "Passive","Resistor",   "Commodity",0.0025,"C","Active"),
        ("RES-0402-100R", "100 Ohm Resistor 0402",              "Passive","Resistor",   "Commodity",0.0025,"C","Active"),
        ("RES-0603-1K",   "1K Ohm Resistor 0603",               "Passive","Resistor",   "Commodity",0.0030,"C","Active"),
        ("RES-0603-4K7",  "4.7K Ohm Resistor 0603",             "Passive","Resistor",   "Commodity",0.0030,"C","Active"),
        ("CAP-0402-100N", "100nF Capacitor 0402 50V",           "Passive","Capacitor",  "Commodity",0.0080,"C","Active"),
        ("CAP-0603-10U",  "10uF Capacitor 0603 25V",            "Passive","Capacitor",  "Commodity",0.0450,"C","Active"),
        ("CAP-0805-100U", "100uF Electrolytic 0805",            "Passive","Capacitor",  "Commodity",0.1200,"B","Active"),
        ("CAP-1210-470U", "470uF Electrolytic 1210",            "Passive","Capacitor",  "Commodity",0.2200,"B","Active"),
        ("IND-0603-10U",  "10uH Inductor 0603 500mA",           "Passive","Inductor",   "Commodity",0.0650,"C","Active"),
        ("IND-1210-100U", "100uH Inductor 1210 1A",             "Passive","Inductor",   "Commodity",0.1800,"C","Active"),
        ("CON-PCB-2P",    "2-Pin PCB Header 2.54mm",            "Connector","PCB Header","Commodity",0.0850,"C","Active"),
        ("CON-PCB-4P",    "4-Pin PCB Header 2.54mm",            "Connector","PCB Header","Commodity",0.1200,"C","Active"),
        ("CON-TB-2P",     "2-Pin Terminal Block 5.08mm",        "Connector","Terminal",  "Commodity",0.2500,"C","Active"),
        ("CON-TB-4P",     "4-Pin Terminal Block 5.08mm",        "Connector","Terminal",  "Commodity",0.4200,"C","Active"),
        ("CON-RJ45-STD",  "RJ45 Ethernet Connector",            "Connector","RJ45",      "Commodity",0.8500,"B","Active"),
        ("IC-REG-7805",   "LM7805 5V Linear Regulator TO220",  "Semiconductor","Regulator","Commodity",0.3500,"B","Active"),
        ("IC-REG-LM317",  "LM317 Adjustable Regulator",        "Semiconductor","Regulator","Commodity",0.4200,"B","Active"),
        ("IC-OP-LM358",   "LM358 Dual Op-Amp SOIC8",           "Semiconductor","Op-Amp", "Commodity",0.2800,"B","Active"),
        ("IC-OP-TL072",   "TL072 Low-Noise Op-Amp SOIC8",      "Semiconductor","Op-Amp", "Commodity",0.3500,"B","Active"),
        ("IC-MCU-STM32",  "STM32F103 ARM Cortex-M3 MCU",       "Semiconductor","MCU",    "Critical", 3.8500,"A","Active"),
        ("IC-MCU-ESP32",  "ESP32-WROOM-32 WiFi BT Module",     "Semiconductor","MCU",    "Critical", 2.9500,"A","Active"),
        ("IC-GATE-74HC",  "74HC00 Quad NAND Gate SOIC14",      "Semiconductor","Logic",  "Commodity",0.1800,"C","Active"),
        ("MOSFET-IRF540", "IRF540N N-Ch Power MOSFET TO220",   "Semiconductor","MOSFET", "Critical", 0.8500,"B","Active"),
        ("MOSFET-FQP30N", "FQP30N06L 60V MOSFET TO220",       "Semiconductor","MOSFET", "Critical", 0.7200,"B","Active"),
        ("IGBT-G4PC50W",  "IRG4PC50W 600V 55A IGBT TO247",    "Semiconductor","IGBT",   "Critical", 4.2500,"A","Active"),
        ("IGBT-FGA25N120","FGA25N120 1200V IGBT TO3P",         "Semiconductor","IGBT",   "Critical", 5.8500,"A","Active"),
        ("DIODE-1N4007",  "1N4007 1A 1000V Rectifier Diode",  "Semiconductor","Diode",  "Commodity",0.0350,"C","Active"),
        ("DIODE-SB540",   "SB540 5A 40V Schottky Diode",      "Semiconductor","Diode",  "Commodity",0.1200,"C","Active"),
        ("MAG-CORE-ETD39","ETD39 Ferrite Core N87",            "Magnetic","Ferrite Core","Critical", 1.2500,"A","Active"),
        ("MAG-CORE-EE25", "EE25 Ferrite Core PC40",            "Magnetic","Ferrite Core","Critical", 0.8500,"B","Active"),
        ("MAG-CORE-PQ3230","PQ32/30 Ferrite Core N97",         "Magnetic","Ferrite Core","Critical", 1.6500,"A","Active"),
        ("MAG-WIRE-0.5MM","Enameled Copper Wire 0.5mm 1kg",   "Magnetic","Copper Wire", "Commodity",8.5000,"B","Active"),
        ("MAG-WIRE-1.0MM","Enameled Copper Wire 1.0mm 1kg",   "Magnetic","Copper Wire", "Commodity",10.200,"B","Active"),
        ("MAG-WIRE-1.5MM","Enameled Copper Wire 1.5mm 1kg",   "Magnetic","Copper Wire", "Commodity",12.800,"B","Active"),
        ("ASM-XFMR-LVCT", "LVCT 5A/5mA Current Transformer",  "Assembly","Transformer", "Custom",   18.500,"A","Active"),
        ("ASM-XFMR-100VA","100VA EI Core Transformer",        "Assembly","Transformer", "Custom",   22.000,"A","Active"),
        ("ASM-XFMR-500VA","500VA Toroidal Transformer",       "Assembly","Transformer", "Custom",   45.000,"A","Active"),
        ("ASM-HARNESS-5W","5-Wire Control Panel Harness",     "Assembly","Wire Harness","Custom",   3.8500,"B","Active"),
        ("ASM-HARNESS-10W","10-Wire Power Harness 1000mm",    "Assembly","Wire Harness","Custom",   6.2000,"B","Active"),
        ("ASM-PCB-CTRL",  "Control PCB Sub-Assembly Rev3",    "Assembly","PCB Sub-Assy","Custom",   28.500,"A","Active"),
        ("ASM-PCB-PWR",   "Power PCB Sub-Assembly Rev2",      "Assembly","PCB Sub-Assy","Custom",   35.000,"A","Active"),
        ("ASM-PANEL-DPM", "Digital Panel Meter Assembly",     "Assembly","Panel Meter", "Custom",   42.000,"A","Active"),
        ("ASM-INV-1KW",   "1kW Inverter Core Assembly",       "Assembly","Inverter",    "Custom",   85.000,"A","Active"),
        ("ASM-CT-PANEL",  "CT Panel Complete Assembly",       "Assembly","Panel",       "Custom",   120.00,"A","Active"),
        ("ASM-CTRL-BOX",  "Control Box Assembly 24VDC",       "Assembly","Control Box", "Custom",   95.000,"A","NRND"),
    ]
    sql = """INSERT INTO component_master
             (part_no,description,category,sub_category,component_class,
              unit_cost_avg,weight_class,lifecycle_status)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
             ON CONFLICT (part_no) DO NOTHING"""
    run_insert_many(sql, components)
    print(f"  ✅ {len(components)} components inserted")
    return [c[0] for c in components]


# ──────────────────────────────────────────────
# SECTION 2: 18 SUPPLIERS
# ──────────────────────────────────────────────
def generate_suppliers():
    print("Generating suppliers...")
    suppliers = [
        ("Shenzhen Components Co",   "China",      "Asia-Pacific","Shenzhen", 1,"Net 60","Active",True),
        ("Shanghai Electronics Ltd", "China",      "Asia-Pacific","Shanghai", 1,"Net 45","Active",True),
        ("Guangzhou Parts Supply",   "China",      "Asia-Pacific","Guangzhou",2,"Net 60","Active",False),
        ("Dongguan FastParts",       "China",      "Asia-Pacific","Dongguan", 3,"Net 30","Active",False),
        ("Chennai Precision Parts",  "India",      "Asia-Pacific","Chennai",  2,"Net 45","Active",True),
        ("Pune Electronics Mfg",     "India",      "Asia-Pacific","Pune",     2,"Net 30","Active",True),
        ("Mumbai Magnetic Works",    "India",      "Asia-Pacific","Mumbai",   2,"Net 45","Active",False),
        ("Bangalore Tech Supplies",  "India",      "Asia-Pacific","Bangalore",3,"Net 30","Active",False),
        ("Taiwan Semiconductor Corp","Taiwan",     "Asia-Pacific","Taipei",   1,"Net 45","Active",True),
        ("Korea Advanced Components","South Korea","Asia-Pacific","Seoul",    1,"Net 45","Active",True),
        ("Murata Europe GmbH",       "Germany",    "Europe",      "Munich",   1,"Net 30","Active",True),
        ("Schaffner AG",             "Germany",    "Europe",      "Frankfurt",1,"Net 30","Active",True),
        ("Vishay Europe Ltd",        "Germany",    "Europe",      "Hamburg",  2,"Net 45","Active",True),
        ("Mouser Electronics USA",   "USA",        "North America","Texas",   1,"Net 30","Active",True),
        ("Digi-Key Corporation",     "USA",        "North America","Minnesota",1,"Net 30","Active",True),
        ("Arrow Electronics",        "USA",        "North America","Colorado", 2,"Net 45","Active",True),
        ("Acme Magnetics Pvt Ltd",   "India",      "Asia-Pacific","Hyderabad",3,"Net 60","Active",False),
        ("Vietnam Circuit Co",       "Vietnam",    "Asia-Pacific","Hanoi",    3,"Net 60","Active",False),
    ]
    sql = """INSERT INTO supplier_master
             (supplier_name,country,region,city,supplier_tier,
              payment_terms,status,iso_certified)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
    run_insert_many(sql, suppliers)
    df = run_query("SELECT supplier_id, supplier_name FROM supplier_master ORDER BY supplier_id")
    print(f"  ✅ {len(df)} suppliers inserted")
    return df


# ──────────────────────────────────────────────
# SECTION 3: AVL
# ──────────────────────────────────────────────
def generate_avl(part_nos, supplier_df):
    print("Generating AVL...")
    sid = supplier_df.set_index('supplier_name')['supplier_id'].to_dict()
    avl_map = {
        "RES-":       ["Shenzhen Components Co","Shanghai Electronics Ltd","Murata Europe GmbH"],
        "CAP-":       ["Shenzhen Components Co","Taiwan Semiconductor Corp","Murata Europe GmbH"],
        "IND-":       ["Shanghai Electronics Ltd","Shenzhen Components Co","Vishay Europe Ltd"],
        "CON-":       ["Shenzhen Components Co","Guangzhou Parts Supply","Dongguan FastParts"],
        "IC-":        ["Taiwan Semiconductor Corp","Korea Advanced Components","Mouser Electronics USA","Digi-Key Corporation"],
        "MOSFET-":    ["Taiwan Semiconductor Corp","Shenzhen Components Co","Vishay Europe Ltd"],
        "IGBT-":      ["Taiwan Semiconductor Corp","Korea Advanced Components","Schaffner AG"],
        "DIODE-":     ["Shenzhen Components Co","Shanghai Electronics Ltd","Vishay Europe Ltd"],
        "MAG-CORE":   ["Shanghai Electronics Ltd","Pune Electronics Mfg","Murata Europe GmbH"],
        "MAG-WIRE":   ["Mumbai Magnetic Works","Acme Magnetics Pvt Ltd","Vietnam Circuit Co"],
        "ASM-XFMR":   ["Chennai Precision Parts","Pune Electronics Mfg","Acme Magnetics Pvt Ltd"],
        "ASM-HARNESS":["Chennai Precision Parts","Bangalore Tech Supplies","Vietnam Circuit Co"],
        "ASM-PCB":    ["Shenzhen Components Co","Taiwan Semiconductor Corp","Chennai Precision Parts"],
        "ASM-PANEL":  ["Chennai Precision Parts","Pune Electronics Mfg"],
        "ASM-INV":    ["Chennai Precision Parts"],
        "ASM-CT":     ["Pune Electronics Mfg"],
        "ASM-CTRL":   ["Chennai Precision Parts","Acme Magnetics Pvt Ltd"],
    }
    rows = []
    base = date(2022, 1, 1)
    for pno in part_nos:
        sups = next((v for k, v in avl_map.items() if pno.startswith(k)),
                    ["Shenzhen Components Co","Shanghai Electronics Ltd"])
        for sname in sups:
            if sname in sid:
                rows.append((pno, sid[sname], "Qualified",
                             base + timedelta(days=random.randint(0, 365))))
    sql = """INSERT INTO approved_vendor_list
             (part_no,supplier_id,qualification_status,qualified_date)
             VALUES (%s,%s,%s,%s)
             ON CONFLICT (part_no,supplier_id) DO NOTHING"""
    run_insert_many(sql, rows)
    print(f"  ✅ {len(rows)} AVL records inserted")


# ──────────────────────────────────────────────
# SECTION 4: 700 PO HISTORY RECORDS
# ──────────────────────────────────────────────
def generate_po_history(part_nos, supplier_df):
    print("Generating PO history (700 records)...")
    profiles = {
        1: (0.96,0.99,0.03,"stable"),
        2: (0.94,0.98,0.04,"stable"),
        3: (0.82,0.91,0.07,"improving"),
        4: (0.72,0.85,0.10,"degrading"),
        5: (0.93,0.97,0.04,"stable"),
        6: (0.91,0.96,0.05,"stable"),
        7: (0.80,0.88,0.08,"improving"),
        8: (0.75,0.83,0.09,"degrading"),
        9: (0.97,0.99,0.02,"stable"),
        10:(0.96,0.99,0.03,"stable"),
        11:(0.98,0.99,0.02,"stable"),
        12:(0.97,0.99,0.02,"stable"),
        13:(0.95,0.98,0.03,"stable"),
        14:(0.99,1.00,0.01,"stable"),
        15:(0.99,1.00,0.01,"stable"),
        16:(0.94,0.97,0.04,"stable"),
        17:(0.70,0.80,0.12,"degrading"),
        18:(0.78,0.87,0.08,"stable"),
    }
    avl_df  = run_query("SELECT part_no, supplier_id FROM approved_vendor_list")
    avl_lkp = avl_df.groupby('part_no')['supplier_id'].apply(list).to_dict()
    comp_df = run_query("SELECT part_no, unit_cost_avg FROM component_master")
    price_m = comp_df.set_index('part_no')['unit_cost_avg'].to_dict()

    rows = []
    start = date(2024, 1, 1)
    ctr   = 1000

    for _ in range(700):
        pno    = random.choice([p for p in part_nos if p in avl_lkp])
        sup_id = random.choice(avl_lkp[pno])
        prof   = profiles.get(sup_id, (0.85,0.92,0.06,"stable"))
        otif, qual_rate, price_vol, trend = prof

        days_off  = random.randint(0, 364)
        po_date   = start + timedelta(days=days_off)
        month     = po_date.month
        tf        = days_off / 364.0

        if trend == "improving":
            otif      = min(0.98, otif + 0.10*tf)
            qual_rate = min(0.99, qual_rate + 0.05*tf)
        elif trend == "degrading":
            otif      = max(0.55, otif - 0.15*tf)
            qual_rate = max(0.65, qual_rate - 0.10*tf)

        base_price  = float(price_m.get(pno, 1.0))
        noise       = np.random.normal(0, price_vol * base_price)
        if pno in ("IGBT-G4PC50W","IC-MCU-STM32") and month in (6,7):
            noise += base_price * 0.20
        unit_price  = max(base_price*0.7, base_price+noise)

        if "ASM-" in pno:
            qty = random.randint(10,100)
        elif any(pno.startswith(p) for p in ["IGBT-","IC-MCU","MOSFET-","MAG-CORE"]):
            qty = random.randint(50,500)
        else:
            qty = random.randint(500,5000)
        if month >= 10:
            qty = int(qty * 1.4)

        lt_days       = random.randint(14,112)
        promised_date = po_date + timedelta(days=lt_days)
        on_time       = random.random() < otif
        actual_days   = lt_days + (random.randint(-3,2) if on_time else random.randint(5,30))
        actual_date   = po_date + timedelta(days=max(1,actual_days))

        if random.random() < qual_rate:
            qs, rq = "Pass", 0
        else:
            qs = random.choice(["Fail","Partial"])
            rq = random.randint(1, max(1, int(qty*0.15)))

        rows.append((f"PO-2024-{ctr:04d}", pno, sup_id, qty,
                     round(unit_price,4), po_date,
                     promised_date, actual_date,
                     on_time, qs, rq))
        ctr += 1

    sql = """INSERT INTO po_history
             (po_number,part_no,supplier_id,quantity,unit_price,
              po_date,promised_date,actual_delivery_date,
              on_time_flag,quality_status,reject_quantity)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    run_insert_many(sql, rows)
    print(f"  ✅ {len(rows)} PO history records inserted")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AgentProcure — Synthetic Data Generator")
    print("="*50)

    part_nos    = generate_components()
    supplier_df = generate_suppliers()
    generate_avl(part_nos, supplier_df)
    generate_po_history(part_nos, supplier_df)

    print("\n" + "="*50)
    print("  Verifying record counts...")
    print("="*50)
    for t in ["component_master","supplier_master","approved_vendor_list","po_history"]:
        df = run_query(f"SELECT COUNT(*) as n FROM {t}")
        print(f"  {t:30s} → {int(df['n'][0]):>4} records")

    print("\n🎉 All synthetic data generated! Ready for Phase 3.\n")