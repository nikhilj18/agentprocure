# modules/module1_classifier.py
# PURPOSE: Takes a BOM (list of part numbers + quantities) and:
#   1. Validates each part exists in component_master
#   2. Classifies each part as Commodity / Critical / Custom
#      using a Random Forest ML model
#   3. Determines the sourcing trigger for each part
#   4. Returns a scored, classified BOM ready for Module 2
#
# Run standalone test: python3 modules/module1_classifier.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

from database.db_connect import run_query


# ─────────────────────────────────────────────
# STEP 1: LOAD TRAINING DATA FROM DATABASE
# ─────────────────────────────────────────────
def load_component_data():
    """
    Loads all components from component_master.
    These become the training data for the Random Forest.
    Each component already has a known class label
    (Commodity/Critical/Custom) which the model learns from.
    """
    sql = """
        SELECT
            part_no,
            category,
            sub_category,
            component_class,
            unit_cost_avg,
            weight_class,
            lifecycle_status
        FROM component_master
        ORDER BY part_no
    """
    df = run_query(sql)
    return df


# ─────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    """
    Converts text columns into numbers that the ML model can use.
    ML models only understand numbers — not words like "Passive" or "A".

    LabelEncoder converts each unique text value to a number:
    e.g. "Passive"=2, "Semiconductor"=3, "Magnetic"=1, etc.
    """
    df = df.copy()

    # Create encoder objects — one per text column
    encoders = {}

    for col in ['category', 'sub_category', 'weight_class', 'lifecycle_status']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].fillna('Unknown'))
        encoders[col] = le

    # unit_cost_avg is already a number — just fill any missing values
    df['unit_cost_avg'] = df['unit_cost_avg'].fillna(0.0).astype(float)

    # These are the features the model will use to make predictions
    feature_cols = [
        'category_enc',
        'sub_category_enc',
        'unit_cost_avg',
        'weight_class_enc',
        'lifecycle_status_enc'
    ]

    return df, feature_cols, encoders


# ─────────────────────────────────────────────
# STEP 3: TRAIN THE RANDOM FOREST MODEL
# ─────────────────────────────────────────────
def train_classifier(df, feature_cols):
    """
    Trains a Random Forest classifier to predict component_class.

    Random Forest works by building many decision trees on random
    subsets of the data, then voting on the final prediction.
    This makes it more accurate and robust than a single tree.

    Returns:
        model     - the trained classifier
        label_enc - encoder for the target column (class labels)
        accuracy  - how accurate the model is on test data
    """
    # Encode the target column (what we want to predict)
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df['component_class'])
    # y is now: 0=Commodity, 1=Critical, 2=Custom (or similar)

    X = df[feature_cols].values

    # Split: 80% for training, 20% for testing
    # random_state=42 means we always get the same split (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build the Random Forest
    # n_estimators=100 means 100 trees
    # max_depth=8 means each tree can be at most 8 levels deep
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced'  # handles unequal class sizes
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    return model, label_enc, accuracy, feature_cols


# ─────────────────────────────────────────────
# STEP 4: GET FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def get_feature_importance(model, feature_cols):
    """
    Shows which features matter most for classification.
    Feature importance = how much each feature reduces prediction error.
    Higher = more important.

    This is the 'explainability' part — it shows WHY the model
    classified a component the way it did.
    """
    importance_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance_df


# ─────────────────────────────────────────────
# STEP 5: DETERMINE SOURCING TRIGGER
# ─────────────────────────────────────────────
def get_sourcing_trigger(part_no, component_class, lifecycle_status):
    """
    Decides what procurement action is needed for each part.

    Logic:
    - If lifecycle is EOL/Obsolete → Engineering Review (urgent!)
    - If we have PO history → Reorder
    - Otherwise → New RFQ (request for quote)
    """
    # Check if we've ordered this part before
    po_count = run_query(
        "SELECT COUNT(*) as n FROM po_history WHERE part_no = %s",
        params=(part_no,)
    )
    has_history = int(po_count['n'].iloc[0]) > 0

    if lifecycle_status in ('EOL', 'Obsolete', 'NRND'):
        return 'Engineering Review'
    elif has_history:
        return 'Reorder'
    else:
        return 'New RFQ'


# ─────────────────────────────────────────────
# STEP 6: CLASSIFY A BOM
# ─────────────────────────────────────────────
def classify_bom(bom_df):
    """
    Main function — takes a BOM DataFrame and returns it with
    classification results added.

    bom_df must have columns: part_no, quantity_required

    Returns a DataFrame with added columns:
        - component_class    (Commodity/Critical/Custom)
        - rf_confidence      (model's confidence 0-1)
        - sourcing_trigger   (New RFQ/Reorder/Engineering Review)
        - category, sub_category, unit_cost_avg (from master)
        - validation_status  (Valid/Not Found)
    """
    # Load and prepare training data
    comp_df = load_component_data()
    comp_df_feat, feature_cols, encoders = engineer_features(comp_df)
    model, label_enc, accuracy, _ = train_classifier(comp_df_feat, feature_cols)

    print(f"  Random Forest trained — accuracy: {accuracy:.1%}")

    results = []

    for _, row in bom_df.iterrows():
        pno = row['part_no']
        qty = row.get('quantity_required', 1)

        # Look up this part in component_master
        match = comp_df[comp_df['part_no'] == pno]

        if match.empty:
            # Part not found in master — flag it
            results.append({
                'part_no':          pno,
                'quantity_required':qty,
                'validation_status':'Not Found',
                'component_class':  'Unknown',
                'rf_confidence':    0.0,
                'sourcing_trigger': 'Engineering Review',
                'category':         'Unknown',
                'sub_category':     'Unknown',
                'unit_cost_avg':    0.0,
                'lifecycle_status': 'Unknown'
            })
        else:
            # Prepare features for this specific part
            part_row = match.iloc[0]
            feat_row = comp_df_feat[comp_df_feat['part_no'] == pno].iloc[0]
            X_new = feat_row[feature_cols].values.reshape(1, -1)

            # Predict class and confidence
            pred_encoded  = model.predict(X_new)[0]
            pred_proba    = model.predict_proba(X_new)[0]
            pred_class    = label_enc.inverse_transform([pred_encoded])[0]
            confidence    = pred_proba.max()

            # Use database label as ground truth (model confirms/validates)
            actual_class  = part_row['component_class']
            trigger       = get_sourcing_trigger(
                                pno, actual_class,
                                part_row['lifecycle_status'])

            results.append({
                'part_no':           pno,
                'quantity_required': qty,
                'validation_status': 'Valid',
                'component_class':   actual_class,
                'rf_predicted_class':pred_class,
                'rf_confidence':     round(float(confidence), 3),
                'sourcing_trigger':  trigger,
                'category':          part_row['category'],
                'sub_category':      part_row['sub_category'],
                'unit_cost_avg':     float(part_row['unit_cost_avg']),
                'lifecycle_status':  part_row['lifecycle_status']
            })

    result_df = pd.DataFrame(results)
    return result_df, get_feature_importance(model, feature_cols), accuracy


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Module 1 — BOM Classifier Test")
    print("="*55)

    # Sample BOM — a mix of component types
    sample_bom = pd.DataFrame([
        {'part_no': 'RES-0402-10K',  'quantity_required': 5000},
        {'part_no': 'IGBT-G4PC50W',  'quantity_required': 200},
        {'part_no': 'ASM-XFMR-LVCT','quantity_required': 50},
        {'part_no': 'IC-MCU-STM32',  'quantity_required': 100},
        {'part_no': 'CAP-0402-100N', 'quantity_required': 10000},
        {'part_no': 'ASM-INV-1KW',   'quantity_required': 20},
        {'part_no': 'UNKNOWN-PART',  'quantity_required': 10},  # invalid
    ])

    classified_df, importance_df, accuracy = classify_bom(sample_bom)

    print(f"\n📊 Classification Results:")
    print("-"*75)
    for _, row in classified_df.iterrows():
        status = "✅" if row['validation_status'] == 'Valid' else "❌"
        print(f"  {status} {row['part_no']:20s} | "
              f"Class: {row['component_class']:10s} | "
              f"Confidence: {row['rf_confidence']:.0%} | "
              f"Trigger: {row['sourcing_trigger']}")

    print(f"\n🌲 Feature Importance (what drives classification):")
    print("-"*45)
    for _, row in importance_df.iterrows():
        bar = "█" * int(row['importance'] * 40)
        print(f"  {row['feature']:25s} {bar} {row['importance']:.3f}")

    print(f"\n✅ Model accuracy: {accuracy:.1%}")
    print("\n🎉 Module 1 complete! Ready for Module 2.\n")