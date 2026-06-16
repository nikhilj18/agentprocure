import sys, os, numpy as np, json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHT_ARMS = {
    0: {'name': 'Balanced',        'weights': [0.22,0.22,0.22,0.15,0.09,0.10]},
    1: {'name': 'Cost-Focused',    'weights': [0.15,0.15,0.42,0.12,0.08,0.08]},
    2: {'name': 'Quality-Focused', 'weights': [0.18,0.38,0.18,0.12,0.07,0.07]},
    3: {'name': 'Delivery-Focused','weights': [0.38,0.20,0.15,0.15,0.06,0.06]},
    4: {'name': 'ESG-Focused',     'weights': [0.18,0.20,0.18,0.12,0.12,0.20]},
    5: {'name': 'Risk-Averse',     'weights': [0.30,0.25,0.12,0.18,0.08,0.07]},
}
CLASS_MODIFIERS = {
    'Commodity':[0.85,0.85,1.40,0.90,0.90,0.90],
    'Critical': [1.30,1.10,0.70,1.20,0.90,0.90],
    'Custom':   [0.90,1.30,0.75,0.90,1.10,1.15],
}
EPSILON = 0.10
STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data','rl_bandit_state.json')

def load_bandit_state():
    default = {str(i):{'pulls':0,'rewards':0.0,'avg_reward':0.0} for i in WEIGHT_ARMS}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: return json.load(f)
        except: pass
    return default

def save_bandit_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE,'w') as f: json.dump(state,f,indent=2)

def select_arm(state, component_class=None):
    unpulled = [int(k) for k,v in state.items() if v['pulls']==0]
    if unpulled: arm_id = unpulled[0]
    elif np.random.random() < EPSILON: arm_id = np.random.randint(0,len(WEIGHT_ARMS))
    else: arm_id = max([int(k) for k in state],key=lambda k:state[str(k)]['avg_reward'])
    w = np.array(WEIGHT_ARMS[arm_id]['weights'])
    if component_class and component_class in CLASS_MODIFIERS:
        m = np.array(CLASS_MODIFIERS[component_class]); w = w*m; w = w/w.sum()
    return arm_id, w.tolist()

def record_feedback(arm_id, accepted, component_class=None):
    state = load_bandit_state()
    reward = 1.0 if accepted else 0.0
    k = str(arm_id)
    state[k]['pulls'] += 1
    state[k]['rewards'] += reward
    state[k]['avg_reward'] = state[k]['rewards']/state[k]['pulls']
    save_bandit_state(state)
    return state

def get_best_weights(component_class=None):
    state = load_bandit_state()
    best = max([int(k) for k in state],key=lambda k:state[str(k)]['avg_reward'] if state[str(k)]['pulls']>=3 else -1)
    conf = min(1.0, state[str(best)]['pulls']/20.0)
    w = np.array(WEIGHT_ARMS[best]['weights'])
    if component_class and component_class in CLASS_MODIFIERS:
        m = np.array(CLASS_MODIFIERS[component_class]); w = w*m; w = w/w.sum()
    return w.tolist(), best, WEIGHT_ARMS[best]['name'], round(conf,3)

def setup_feedback_table():
    from database.db_connect import get_connection
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS buyer_feedback (
            feedback_id SERIAL PRIMARY KEY, arm_id INTEGER,
            arm_name VARCHAR(50), component_class VARCHAR(20),
            accepted BOOLEAN, reward FLOAT,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()
    except Exception as e: conn.rollback()
    finally: cur.close(); conn.close()

def simulate_training(n_rounds=50):
    np.random.seed(42)
    true_rates = {
        'Commodity':{0:0.70,1:0.88,2:0.60,3:0.65,4:0.55,5:0.68},
        'Critical': {0:0.72,1:0.55,2:0.70,3:0.85,4:0.60,5:0.80},
        'Custom':   {0:0.68,1:0.52,2:0.85,3:0.60,4:0.75,5:0.65},
    }
    state = load_bandit_state()
    for _ in range(n_rounds):
        cls = np.random.choice(['Commodity','Critical','Custom'])
        arm_id, _ = select_arm(state, cls)
        accepted = np.random.random() < true_rates[cls][arm_id]
        state = record_feedback(arm_id, accepted, cls)
    return state

if __name__ == '__main__':
    print('\n' + '='*55)
    print('  RL Weight Optimiser — Multi-Armed Bandit Test')
    print('='*55)
    setup_feedback_table()
    print('\n Training bandit with 60 rounds...')
    state = simulate_training(60)
    print('\n Results after training:')
    print(f"  {'Arm':<5} {'Name':<18} {'Pulls':>6} {'Avg Reward':>11}")
    best = max([int(k) for k in state],key=lambda k:state[str(k)]['avg_reward'])
    for i,a in WEIGHT_ARMS.items():
        s=state[str(i)]; star=' <- BEST' if i==best else ''
        print(f"  #{i:<4} {a['name']:<18} {s['pulls']:>6} {s['avg_reward']:>11.3f}{star}")
    print('\n Best weights per class:')
    for cls in ['Commodity','Critical','Custom']:
        w,aid,aname,conf = get_best_weights(cls)
        print(f"  {cls:<12} -> {aname} (conf: {conf:.0%})")
    print('\nRL optimiser complete!')