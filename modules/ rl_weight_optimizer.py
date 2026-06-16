# modules/rl_weight_optimizer.py
# PURPOSE: Reinforcement Learning weight optimiser for TOPSIS.
#
# HOW IT WORKS:
#   A multi-armed bandit (epsilon-greedy) learns which TOPSIS
#   weight configurations produce recommendations that buyers
#   accept vs override.
#
#   Each "arm" is a weight configuration for the 6 TOPSIS criteria.
#   When a buyer accepts a recommendation → reward = 1.0
#   When a buyer overrides a recommendation → reward = 0.0
#   Over time, the bandit learns which arms produce accepted recs.
#
#   The learned weights then replace the fixed scenario weights
#   in Module 2, making recommendations smarter over time.
#
# Run: python3 modules/rl_weight_optimizer.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from database.db_connect import run_query, run_insert


# ─────────────────────────────────────────────
# WEIGHT ARMS — each arm = one weight config
# ─────────────────────────────────────────────
# 6 criteria: [otif, quality, cost, lead_time, iso, esg]
# Each row sums to 1.0

WEIGHT_ARMS = {
    # Arm 0: Balanced (default)
    0: {'name': 'Balanced',       'weights': [0.22, 0.22, 0.22, 0.15, 0.09, 0.10]},
    # Arm 1: Cost-focused
    1: {'name': 'Cost-Focused',   'weights': [0.15, 0.15, 0.42, 0.12, 0.08, 0.08]},
    # Arm 2: Quality-focused
    2: {'name': 'Quality-Focused','weights': [0.18, 0.38, 0.18, 0.12, 0.07, 0.07]},
    # Arm 3: Delivery-focused
    3: {'name': 'Delivery-Focused','weights': [0.38, 0.20, 0.15, 0.15, 0.06, 0.06]},
    # Arm 4: ESG-focused
    4: {'name': 'ESG-Focused',    'weights': [0.18, 0.20, 0.18, 0.12, 0.12, 0.20]},
    # Arm 5: Risk-averse
    5: {'name': 'Risk-Averse',    'weights': [0.30, 0.25, 0.12, 0.18, 0.08, 0.07]},
}

# Component class modifiers — scale base arm weights
CLASS_MODIFIERS = {
    'Commodity': [0.85, 0.85, 1.40, 0.90, 0.90, 0.90],  # boost cost
    'Critical':  [1.30, 1.10, 0.70, 1.20, 0.90, 0.90],  # boost delivery
    'Custom':    [0.90, 1.30, 0.75, 0.90, 1.10, 1.15],  # boost quality+esg
}

# Epsilon for exploration (10% random, 90% exploit best arm)
EPSILON = 0.10

# Storage file for bandit state
STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'rl_bandit_state.json'
)


# ─────────────────────────────────────────────
# BANDIT STATE MANAGEMENT
# ─────────────────────────────────────────────
def load_bandit_state():
    """
    Loads the bandit's memory from disk.
    State tracks: how many times each arm was pulled,
    and the cumulative reward for each arm.
    """
    default = {
        str(arm_id): {
            'pulls':   0,
            'rewards': 0.0,
            'avg_reward': 0.0
        }
        for arm_id in WEIGHT_ARMS.keys()
    }

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_bandit_state(state):
    """Persists the bandit state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────
# EPSILON-GREEDY ARM SELECTION
# ─────────────────────────────────────────────
def select_arm(state, component_class=None):
    """
    Selects which weight arm to use for this recommendation.

    Epsilon-greedy strategy:
    - With probability EPSILON: explore (random arm)
    - With probability 1-EPSILON: exploit (best arm so far)

    New arms (0 pulls) are always tried first to ensure
    every arm gets at least one chance.

    Returns:
        arm_id:  int — which arm was selected
        weights: list — the 6 TOPSIS weights to use
    """
    # Always try unpulled arms first
    unpulled = [int(k) for k, v in state.items() if v['pulls'] == 0]
    if unpulled:
        arm_id = unpulled[0]
    elif np.random.random() < EPSILON:
        # Explore: random arm
        arm_id = np.random.randint(0, len(WEIGHT_ARMS))
    else:
        # Exploit: arm with highest average reward
        arm_id = max(
            [int(k) for k in state.keys()],
            key=lambda k: state[str(k)]['avg_reward']
        )

    base_weights = np.array(WEIGHT_ARMS[arm_id]['weights'])

    # Apply class modifier if provided
    if component_class and component_class in CLASS_MODIFIERS:
        modifier = np.array(CLASS_MODIFIERS[component_class])
        adjusted = base_weights * modifier
        # Renormalise to sum to 1.0
        weights = (adjusted / adjusted.sum()).tolist()
    else:
        weights = base_weights.tolist()

    return arm_id, weights


# ─────────────────────────────────────────────
# RECORD FEEDBACK
# ─────────────────────────────────────────────
def record_feedback(arm_id, accepted, component_class=None):
    """
    Records buyer feedback for a recommendation.

    accepted = True  → reward 1.0 (buyer accepted recommendation)
    accepted = False → reward 0.0 (buyer overrode recommendation)

    Updates the bandit state and saves to disk.

    Args:
        arm_id:          int   — which arm produced this recommendation
        accepted:        bool  — did the buyer accept it?
        component_class: str   — optional, for logging
    """
    state   = load_bandit_state()
    reward  = 1.0 if accepted else 0.0
    key     = str(arm_id)

    state[key]['pulls']   += 1
    state[key]['rewards'] += reward
    state[key]['avg_reward'] = (
        state[key]['rewards'] / state[key]['pulls']
    )

    save_bandit_state(state)

    # Also log to database for audit trail
    try:
        run_insert(
            """INSERT INTO buyer_feedback
               (arm_id, arm_name, component_class, accepted,
                reward, recorded_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (arm_id,
             WEIGHT_ARMS[arm_id]['name'],
             component_class or 'Unknown',
             accepted,
             reward,
             datetime.now())
        )
    except Exception:
        pass  # Table may not exist yet — file state is enough

    return state


# ─────────────────────────────────────────────
# GET BEST WEIGHTS FOR PRODUCTION
# ─────────────────────────────────────────────
def get_best_weights(component_class=None):
    """
    Returns the current best-performing weight configuration
    based on buyer feedback history.

    Falls back to balanced weights if no feedback yet.

    Args:
        component_class: str — 'Commodity', 'Critical', or 'Custom'

    Returns:
        weights:   list of 6 floats summing to 1.0
        arm_id:    int — which arm is being used
        arm_name:  str — human-readable name
        confidence: float — how confident we are (based on pulls)
    """
    state = load_bandit_state()

    # Find arm with most pulls that has decent reward
    best_arm = max(
        [int(k) for k in state.keys()],
        key=lambda k: (
            state[str(k)]['avg_reward']
            if state[str(k)]['pulls'] >= 3
            else -1  # penalise arms with < 3 pulls
        )
    )

    # Confidence: scales from 0 to 1 as pulls increase (saturates at 20)
    total_pulls = state[str(best_arm)]['pulls']
    confidence  = min(1.0, total_pulls / 20.0)

    base_weights = np.array(WEIGHT_ARMS[best_arm]['weights'])

    if component_class and component_class in CLASS_MODIFIERS:
        modifier = np.array(CLASS_MODIFIERS[component_class])
        adjusted = base_weights * modifier
        weights  = (adjusted / adjusted.sum()).tolist()
    else:
        weights = base_weights.tolist()

    return (
        weights,
        best_arm,
        WEIGHT_ARMS[best_arm]['name'],
        round(confidence, 3)
    )


# ─────────────────────────────────────────────
# SETUP: CREATE FEEDBACK TABLE
# ─────────────────────────────────────────────
def setup_feedback_table():
    """Creates the buyer_feedback table if it doesn't exist."""
    from database.db_connect import get_connection
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS buyer_feedback (
                feedback_id    SERIAL PRIMARY KEY,
                arm_id         INTEGER,
                arm_name       VARCHAR(50),
                component_class VARCHAR(20),
                accepted       BOOLEAN,
                reward         FLOAT,
                recorded_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  Note: {e}")
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────
# SIMULATE TRAINING (for demo purposes)
# ─────────────────────────────────────────────
def simulate_training(n_rounds=50):
    """
    Simulates buyer feedback to demonstrate the bandit learning.
    In production this would be replaced by real buyer clicks.

    Simulates realistic buyer behaviour:
    - Cost-focused arm performs well for Commodity
    - Delivery-focused arm performs well for Critical
    - Quality-focused arm performs well for Custom
    """
    np.random.seed(42)
    print(f"\n  Simulating {n_rounds} rounds of buyer feedback...")

    # True acceptance rates per arm per class (what a real buyer would prefer)
    true_rates = {
        'Commodity': {0:0.70, 1:0.88, 2:0.60, 3:0.65, 4:0.55, 5:0.68},
        'Critical':  {0:0.72, 1:0.55, 2:0.70, 3:0.85, 4:0.60, 5:0.80},
        'Custom':    {0:0.68, 1:0.52, 2:0.85, 3:0.60, 4:0.75, 5:0.65},
    }

    classes = ['Commodity', 'Critical', 'Custom']
    state   = load_bandit_state()

    for round_i in range(n_rounds):
        cls     = np.random.choice(classes)
        arm_id, weights = select_arm(state, cls)

        # Simulate buyer decision based on true acceptance rate
        rate     = true_rates[cls][arm_id]
        accepted = np.random.random() < rate

        state = record_feedback(arm_id, accepted, cls)

    return state


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RL Weight Optimiser — Multi-Armed Bandit Test")
    print("="*60)

    # Setup database table
    print("\n📋 Setting up feedback table...")
    setup_feedback_table()
    print("  ✅ buyer_feedback table ready")

    # Show initial state
    print("\n🎰 Initial bandit state (no feedback yet):")
    state = load_bandit_state()
    print(f"  {'Arm':<5} {'Name':<18} {'Pulls':>6} {'Avg Reward':>11}")
    print("  " + "-"*45)
    for arm_id, arm_data in WEIGHT_ARMS.items():
        s = state[str(arm_id)]
        print(f"  #{arm_id:<4} {arm_data['name']:<18} "
              f"{s['pulls']:>6} {s['avg_reward']:>11.3f}")

    # Simulate training
    print("\n🏋️  Training bandit with simulated buyer feedback...")
    state = simulate_training(n_rounds=60)

    # Show learned state
    print("\n📊 Bandit state after 60 rounds of feedback:")
    print(f"  {'Arm':<5} {'Name':<18} {'Pulls':>6} {'Avg Reward':>11} {'Status'}")
    print("  " + "-"*55)
    best_arm = max([int(k) for k in state.keys()],
                   key=lambda k: state[str(k)]['avg_reward'])
    for arm_id, arm_data in WEIGHT_ARMS.items():
        s    = state[str(arm_id)]
        star = " ← BEST" if arm_id == best_arm else ""
        print(f"  #{arm_id:<4} {arm_data['name']:<18} "
              f"{s['pulls']:>6} {s['avg_reward']:>11.3f}{star}")

    # Show best weights per component class
    print(f"\n⚖️  Learned optimal weights per component class:")
    print(f"  {'Class':<12} {'Best Arm':<18} {'OTIF':>6} {'Qual':>6} "
          f"{'Cost':>6} {'LT':>6} {'ISO':>6} {'ESG':>6} {'Conf':>6}")
    print("  " + "-"*72)
    for cls in ['Commodity', 'Critical', 'Custom']:
        w, aid, aname, conf = get_best_weights(cls)
        print(f"  {cls:<12} {aname:<18} "
              f"{w[0]:>6.2f} {w[1]:>6.2f} {w[2]:>6.2f} "
              f"{w[3]:>6.2f} {w[4]:>6.2f} {w[5]:>6.2f} "
              f"{conf:>6.0%}")

    print(f"\n✅ RL weight optimiser complete!")
    print(f"   In production: buyer Accept/Override clicks train this bandit.")
    print(f"   Weights improve automatically with each procurement cycle.\n")