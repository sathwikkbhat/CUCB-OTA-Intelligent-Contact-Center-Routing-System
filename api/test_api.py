#!/usr/bin/env python3
"""
Quick test script to verify API functionality
Run this before starting the API server to ensure everything works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all imports"""
    print("="*60)
    print("Testing Imports")
    print("="*60)
    
    try:
        from config import config
        print("✓ config")
        
        from data.synthetic_data import SyntheticDataGenerator, data_gen
        print("✓ data.synthetic_data")
        
        from models.uplift_model import UpliftModel, CapacityModel
        print("✓ models.uplift_model")
        
        from routing.scoring import RoutingScorer
        print("✓ routing.scoring")
        
        from routing.assignment import AssignmentSolver
        print("✓ routing.assignment")
        
        from simulation.simulator import RoutingSimulator, cucb_ota_policy, fcfs_policy, skill_based_greedy_policy
        print("✓ simulation.simulator")
        
        from evaluation.metrics import MetricsTracker
        print("✓ evaluation.metrics")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_initialization():
    """Test model initialization"""
    print("\n" + "="*60)
    print("Testing Model Initialization")
    print("="*60)
    
    try:
        from data.synthetic_data import data_gen
        from models.uplift_model import UpliftModel, CapacityModel
        
        print(f"Data generator: {len(data_gen.agents)} agents")
        print(f"Historical data: {len(data_gen.historical_data)} interactions")
        
        uplift_model = UpliftModel()
        capacity_model = CapacityModel()
        print("✓ Models created")
        
        print("\nTraining models...")
        uplift_model.train(data_gen.historical_data, data_gen.agents)
        capacity_model.train(data_gen.historical_data, data_gen.agents)
        print("✓ Models trained")
        
        return True, uplift_model, capacity_model
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_simulator():
    """Test simulator"""
    print("\n" + "="*60)
    print("Testing Simulator")
    print("="*60)
    
    try:
        from data.synthetic_data import data_gen
        from models.uplift_model import UpliftModel, CapacityModel
        from simulation.simulator import RoutingSimulator, cucb_ota_policy
        
        uplift_model = UpliftModel()
        capacity_model = CapacityModel()
        uplift_model.train(data_gen.historical_data, data_gen.agents)
        capacity_model.train(data_gen.historical_data, data_gen.agents)
        
        simulator = RoutingSimulator(data_gen, uplift_model, capacity_model)
        print(f"✓ Simulator created with {len(simulator.agents)} agents")
        
        # Test a single batch
        print("\nTesting single batch...")
        customers = data_gen.generate_customer_batch(10)
        print(f"Generated {len(customers)} customers")
        
        assignments = cucb_ota_policy(customers, simulator.agents, simulator.scorer)
        print(f"✓ Made {len(assignments)} assignments")
        
        # Simulate outcomes
        outcomes = []
        for c_idx, a_idx in assignments:
            customer = customers.iloc[c_idx]
            agent = simulator.agents.iloc[a_idx]
            outcome = data_gen.simulate_interaction_outcome(customer, agent)
            outcomes.append(outcome)
        
        print(f"✓ Simulated {len(outcomes)} outcomes")
        if len(outcomes) > 0:
            avg_csat = sum(o['csat'] for o in outcomes) / len(outcomes)
            avg_aht = sum(o['aht'] for o in outcomes) / len(outcomes)
            print(f"  Avg CSAT: {avg_csat:.3f}")
            print(f"  Avg AHT: {avg_aht:.2f} min")
        
        return True
    except Exception as e:
        print(f"✗ Simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_imports():
    """Test Flask API imports"""
    print("\n" + "="*60)
    print("Testing API Dependencies")
    print("="*60)
    
    try:
        import flask
        print("✓ Flask available")
    except ImportError:
        print("⚠ Flask not installed (run: pip install flask flask-cors flask-socketio)")
        return False
    
    try:
        import flask_cors
        print("✓ flask-cors available")
    except ImportError:
        print("⚠ flask-cors not installed")
        return False
    
    try:
        import flask_socketio
        print("✓ flask-socketio available")
    except ImportError:
        print("⚠ flask-socketio not installed")
        return False
    
    return True


def main():
    print("\n" + "="*60)
    print("SQRS API Test Suite")
    print("="*60)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed!")
        return False
    
    # Test initialization
    success, uplift_model, capacity_model = test_initialization()
    if not success:
        print("\n✗ Initialization tests failed!")
        return False
    
    # Test simulator
    if not test_simulator():
        print("\n✗ Simulator tests failed!")
        return False
    
    # Test API dependencies
    api_ready = test_api_imports()
    
    print("\n" + "="*60)
    if api_ready:
        print("✅ All tests passed! API should work correctly.")
    else:
        print("⚠️  Core functionality works, but install Flask dependencies:")
        print("   pip install flask flask-cors flask-socketio")
    print("="*60)
    print("\nYou can now start the API server with:")
    print("  python api/app.py")
    print("\n" + "="*60)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

