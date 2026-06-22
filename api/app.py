"""
Flask API server for SQRS
Integrates with existing routing system without modifying core logic
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
from typing import Dict, Optional, Any
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from data.synthetic_data import SyntheticDataGenerator, data_gen
from models.uplift_model import UpliftModel, CapacityModel
from routing.scoring import RoutingScorer
from routing.assignment import AssignmentSolver
from simulation.simulator import RoutingSimulator, cucb_ota_policy
from evaluation.metrics import MetricsTracker

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
simulator_state = {
    'simulator': None,
    'data_generator': None,
    'uplift_model': None,
    'capacity_model': None,
    'scorer': None,
    'is_running': False,
    'current_policy': 'CUCB-OTA'
}


def initialize_models():
    """Initialize models (called once at startup)"""
    if simulator_state['uplift_model'] is None:
        print("Initializing models...")
        data_generator = data_gen
        uplift_model = UpliftModel()
        capacity_model = CapacityModel()
        
        # Train models
        uplift_model.train(data_generator.historical_data, data_generator.agents)
        capacity_model.train(data_generator.historical_data, data_generator.agents)
        
        simulator_state['data_generator'] = data_generator
        simulator_state['uplift_model'] = uplift_model
        simulator_state['capacity_model'] = capacity_model
        simulator_state['scorer'] = RoutingScorer(uplift_model, capacity_model)
        
        print("✓ Models initialized")
    
    return simulator_state


# Initialize on startup
initialize_models()


def convert_to_native_types(obj: Any) -> Any:
    """
    Convert numpy/pandas types to native Python types for JSON serialization.
    
    Recursively converts:
    - numpy.int64, numpy.float64 -> int, float
    - pandas Series/DataFrame -> dict/list
    - numpy arrays -> lists
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return {str(k): convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'SQRS API is running'})


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'num_agents': config.NUM_AGENTS,
        'num_customers_per_batch': config.NUM_CUSTOMERS_PER_BATCH,
        'channels': config.CHANNELS,
        'skills': config.SKILLS,
        'capacity_rules': config.CAPACITY_RULES,
        'max_aht_minutes': config.MAX_AHT_MINUTES,
        'max_sla_violation_rate': config.MAX_SLA_VIOLATION_RATE,
        'fairness_gini_threshold': config.FAIRNESS_GINI_THRESHOLD,
        'sla_thresholds': config.SLA_THRESHOLDS
    })


@app.route('/api/metrics/current', methods=['GET'])
def get_current_metrics():
    """Get current KPIs and metrics"""
    state = simulator_state
    
    if state['simulator'] is None:
        # Return default/empty metrics
        return jsonify({
            'csat': 0.0,
            'aht': 0.0,
            'sla_met_rate': 0.0,
            'gini': 0.0,
            'throughput': 0,
            'total_assignments': 0
        })
    
    metrics = state['simulator'].metrics.get_summary_stats()
    
    # Convert numpy/pandas types to native Python types for JSON serialization
    response_data = {
        'csat': convert_to_native_types(metrics.get('overall_avg_csat', 0.0)),
        'aht': convert_to_native_types(metrics.get('overall_avg_aht', 0.0)),
        'sla_met_rate': convert_to_native_types(metrics.get('overall_sla_met_rate', 0.0)),
        'gini': convert_to_native_types(metrics.get('avg_gini', 0.0)),
        'throughput': convert_to_native_types(metrics.get('total_assignments', 0)),
        'total_assignments': convert_to_native_types(metrics.get('total_assignments', 0))
    }
    
    return jsonify(response_data)


@app.route('/api/metrics/historical', methods=['GET'])
def get_historical_metrics():
    """Get historical metrics"""
    state = simulator_state
    
    if state['simulator'] is None:
        return jsonify({'data': []})
    
    df = state['simulator'].metrics.get_dataframe()
    # Convert DataFrame to dict, ensuring all types are JSON serializable
    data = df.to_dict('records')
    return jsonify({'data': convert_to_native_types(data)})


@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get all agents with their current status"""
    state = simulator_state
    
    if state['simulator'] is None:
        # Return agents from data generator
        if state['data_generator'] is not None:
            agents_df = state['data_generator'].agents
        else:
            return jsonify({'agents': []})
    else:
        agents_df = state['simulator'].agents
    
    agents = []
    for idx, agent in agents_df.iterrows():
        # Calculate current load per channel
        channel_loads = {}
        total_load = 0
        for channel in config.CHANNELS:
            load = agent.get(f'load_{channel}', 0)
            channel_loads[channel] = convert_to_native_types(load)
            total_load += convert_to_native_types(load)
        
        agent_data = {
            'agent_id': agent['agent_id'],
            'name': f"Agent {agent['agent_id']}",
            'status': 'busy' if total_load > 0 else 'available',
            'current_load': convert_to_native_types(total_load),
            'channel_loads': channel_loads,
            'capacity': config.CAPACITY_RULES,
            'skills': {skill: convert_to_native_types(agent.get(f'skill_{skill}', 0)) for skill in config.SKILLS},
            'avg_csat': convert_to_native_types(agent.get('avg_csat', 0.0)),
            'avg_aht': convert_to_native_types(agent.get('avg_aht', 0.0)),
            'experience_years': convert_to_native_types(agent.get('experience_years', 0))
        }
        agents.append(agent_data)
    
    return jsonify({'agents': convert_to_native_types(agents)})


@app.route('/api/agents/<agent_id>', methods=['GET'])
def get_agent(agent_id):
    """Get specific agent details"""
    state = simulator_state
    agents_df = state['simulator'].agents if state['simulator'] else state['data_generator'].agents
    
    agent = agents_df[agents_df['agent_id'] == agent_id]
    if len(agent) == 0:
        return jsonify({'error': 'Agent not found'}), 404
    
    agent = agent.iloc[0]
    agent_data = {
        'agent_id': agent['agent_id'],
        'skills': {skill: agent.get(f'skill_{skill}', 0) for skill in config.SKILLS},
        'avg_csat': agent.get('avg_csat', 0.0),
        'avg_aht': agent.get('avg_aht', 0.0),
        'experience_years': agent.get('experience_years', 0)
    }
    return jsonify(convert_to_native_types(agent_data))


@app.route('/api/assignments/active', methods=['GET'])
def get_active_assignments():
    """Get recent active assignments"""
    state = simulator_state
    
    if state['simulator'] is None:
        return jsonify({'assignments': []})
    
    # Get last N assignments from history
    history = state['simulator'].metrics.assignment_history
    recent = history[-50:] if len(history) > 50 else history
    
    assignments = []
    for assignment in recent:
        assignment_data = {
            'customer_id': assignment.get('customer_id', ''),
            'agent_id': assignment.get('agent_id', ''),
            'channel': assignment.get('channel', ''),
            'batch_id': convert_to_native_types(assignment.get('batch_id', 0)),
            'csat': convert_to_native_types(assignment.get('csat', 0.0)),
            'aht': convert_to_native_types(assignment.get('aht', 0.0)),
            'sla_met': bool(assignment.get('sla_met', False))
        }
        assignments.append(assignment_data)
    
    return jsonify({'assignments': convert_to_native_types(assignments)})


@app.route('/api/routing/matrix', methods=['POST'])
def get_routing_matrix():
    """Compute and return routing score matrix for given customers"""
    state = simulator_state
    
    if state['scorer'] is None:
        return jsonify({'error': 'Scorer not initialized'}), 500
    
    data = request.json
    batch_size = data.get('batch_size', 10)
    
    # Generate sample customers
    customers = state['data_generator'].generate_customer_batch(batch_size)
    agents = state['simulator'].agents if state['simulator'] else state['data_generator'].agents
    
    # Compute routing matrix
    RS_matrix, metadata = state['scorer'].compute_routing_matrix(
        customers, agents, exploration=False
    )
    
    # Convert to JSON-serializable format
    return jsonify({
        'routing_matrix': RS_matrix.tolist(),
        'metadata': {
            'uplift': metadata['uplift'].tolist(),
            'aht_penalty': metadata['aht_penalty'].tolist(),
            'capacity_mask': metadata['capacity_mask'].tolist()
        },
        'customers': customers[['customer_id', 'channel', 'skill_needed', 'priority']].to_dict('records'),
        'agents': [{'agent_id': a['agent_id']} for _, a in agents.iterrows()]
    })


@app.route('/api/constraints/dual', methods=['GET'])
def get_dual_variables():
    """Get current Lagrangian dual variables"""
    state = simulator_state
    
    if state['scorer'] is None:
        return jsonify({
            'lambda_aht': float(config.LAMBDA_INIT),
            'lambda_sla': float(config.LAMBDA_INIT),
            'lambda_fairness': float(config.LAMBDA_INIT)
        })
    
    dual_state = state['scorer'].get_dual_state()
    return jsonify(convert_to_native_types(dual_state))


@app.route('/api/policies/compare', methods=['GET'])
def get_policy_comparison():
    """Get policy comparison data"""
    # Try to load from CSV files if they exist
    import pandas as pd
    import os
    
    results = {}
    policies = ['CUCB-OTA', 'FCFS', 'Skill-Greedy']
    
    for policy in policies:
        csv_path = f'data/logs/{policy}_metrics.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                # Convert all values to native Python types for JSON serialization
                results[policy] = {
                    'avg_csat': convert_to_native_types(df['avg_csat'].mean()),
                    'avg_aht': convert_to_native_types(df['avg_aht'].mean()),
                    'sla_met_rate': convert_to_native_types(df['sla_met_rate'].mean()),
                    'gini': convert_to_native_types(df['gini_coefficient'].mean()),
                    'total_assignments': convert_to_native_types(df['n_assignments'].sum())
                }
    
    return jsonify({'policies': convert_to_native_types(results)})


@app.route('/api/report/latest', methods=['GET'])
def get_latest_report():
    """Return the most recent saved report, or build one from current data"""
    import os, glob, pandas as pd

    # 1. Try to find the latest saved report file
    report_files = sorted(
        glob.glob('data/logs/final_report_*.txt'),
        reverse=True
    )
    if report_files:
        try:
            with open(report_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({'report': content, 'source': report_files[0]})
        except Exception:
            pass

    # 2. Fallback: build a report from live data + CSV files
    lines = []
    lines.append('=' * 80)
    lines.append(' ' * 15 + 'SMART QUEUE ROUTING SYSTEM - REPORT')
    lines.append('=' * 80)
    lines.append('')

    # Live metrics
    state = simulator_state
    if state['simulator'] is not None:
        try:
            m = state['simulator'].metrics.get_summary_stats()
            lines.append('LIVE METRICS:')
            lines.append('-' * 40)
            lines.append(f"Total Assignments : {m.get('total_assignments', 0)}")
            lines.append(f"Avg CSAT          : {m.get('overall_avg_csat', 0):.4f}")
            lines.append(f"Avg AHT           : {m.get('overall_avg_aht', 0):.2f} min")
            lines.append(f"SLA Met Rate      : {m.get('overall_sla_met_rate', 0)*100:.1f}%")
            lines.append(f"Fairness (Gini)   : {m.get('avg_gini', 0):.4f}")
            lines.append('')
        except Exception:
            pass

    # Policy comparison from CSVs
    policies = ['CUCB-OTA', 'FCFS', 'Skill-Greedy']
    pol_data = {}
    for policy in policies:
        csv_path = f'data/logs/{policy}_metrics.csv'
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    pol_data[policy] = {
                        'assignments': int(df['n_assignments'].sum()),
                        'avg_csat': float(df['avg_csat'].mean()),
                        'avg_aht': float(df['avg_aht'].mean()),
                        'sla_met_rate': float(df['sla_met_rate'].mean()),
                        'gini': float(df['gini_coefficient'].mean()),
                    }
            except Exception:
                pass

    if pol_data:
        lines.append('POLICY COMPARISON:')
        lines.append('-' * 40)
        for name, p in pol_data.items():
            lines.append(f"{name}:")
            lines.append(f"  Total Assignments : {p['assignments']:,}")
            lines.append(f"  Avg CSAT          : {p['avg_csat']:.4f}")
            lines.append(f"  Avg AHT           : {p['avg_aht']:.2f} min")
            lines.append(f"  SLA Met Rate      : {p['sla_met_rate']*100:.1f}%")
            lines.append(f"  Fairness (Gini)   : {p['gini']:.4f}")
            lines.append('')

    lines.append('ARCHITECTURE:')
    lines.append('-' * 40)
    lines.append('Policy         : CUCB-OTA (Causal Uplift Contextual Bandit + Optimal Transport)')
    lines.append('Constraint opt.: Lagrangian dual variables (λ_aht, λ_sla, λ_fairness)')
    lines.append('Assignment     : Wasserstein optimal transport matching')
    lines.append('ML backbone    : LightGBM causal uplift + XGBoost capacity models')
    lines.append('=' * 80)

    content = '\n'.join(lines)
    return jsonify({'report': content, 'source': 'live'})


@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start simulation"""
    data = request.json
    n_batches = data.get('n_batches', 50)
    policy_name = data.get('policy', 'CUCB-OTA')
    
    state = simulator_state
    
    if state['is_running']:
        return jsonify({'error': 'Simulation already running'}), 400
    
    # Initialize simulator if needed
    if state['simulator'] is None:
        state['simulator'] = RoutingSimulator(
            state['data_generator'],
            state['uplift_model'],
            state['capacity_model']
        )
    
    state['is_running'] = True
    state['current_policy'] = policy_name
    
    # Run simulation in background thread
    def run_sim():
        from simulation.simulator import cucb_ota_policy, fcfs_policy, skill_based_greedy_policy
        
        policies = {
            'CUCB-OTA': cucb_ota_policy,
            'FCFS': fcfs_policy,
            'Skill-Greedy': skill_based_greedy_policy
        }
        
        policy_fn = policies.get(policy_name, cucb_ota_policy)
        
        for batch_id in range(n_batches):
            if not state['is_running']:
                break
            
            # Generate batch
            batch_size = max(1, min(state['data_generator'].generate_customer_batch(1).shape[0], 100))
            customers = state['data_generator'].generate_customer_batch(batch_size)
            
            # Compute assignments
            assignments = policy_fn(customers, state['simulator'].agents, state['scorer'])
            
            # Simulate outcomes
            outcomes = []
            for c_idx, a_idx in assignments:
                customer = customers.iloc[c_idx]
                agent = state['simulator'].agents.iloc[a_idx]
                outcome = state['data_generator'].simulate_interaction_outcome(customer, agent)
                outcomes.append(outcome)
                
                # Update agent load
                channel = customer['channel']
                state['simulator'].agents.at[agent.name, f'load_{channel}'] += 1
                state['simulator'].agents.at[agent.name, 'current_load'] += 1
            
            # Record metrics
            state['simulator'].metrics.record_batch(
                batch_id, assignments, customers, state['simulator'].agents, outcomes, {}
            )
            
            # Emit update via WebSocket
            summary = state['simulator'].metrics.get_summary_stats()
            socketio.emit('metrics_update', {
                'csat': summary.get('overall_avg_csat', 0.0),
                'aht': summary.get('overall_avg_aht', 0.0),
                'sla_met_rate': summary.get('overall_sla_met_rate', 0.0),
                'gini': summary.get('avg_gini', 0.0),
                'batch_id': batch_id
            })
            
            time.sleep(0.1)  # Small delay for real-time feel
        
        state['is_running'] = False
        socketio.emit('simulation_complete', {'message': 'Simulation finished'})
    
    thread = threading.Thread(target=run_sim, daemon=True)
    thread.start()
    
    return jsonify({'message': 'Simulation started', 'batches': n_batches})


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop running simulation"""
    simulator_state['is_running'] = False
    return jsonify({'message': 'Simulation stopped'})


@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Get simulation status"""
    return jsonify({
        'is_running': simulator_state['is_running'],
        'current_policy': simulator_state['current_policy']
    })


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('connected', {'message': 'Connected to SQRS API'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print('Client disconnected')


if __name__ == '__main__':
    print("\n" + "="*60)
    print("SQRS API Server Starting...")
    print("="*60)
    print(f"API will be available at: http://localhost:5000")
    print("="*60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)

