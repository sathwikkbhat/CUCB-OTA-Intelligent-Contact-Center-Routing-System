"""
Test suite for batch prediction functionality.
Verifies that batch predictions produce the same results as sequential predictions.
"""
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.uplift_model import UpliftModel, CapacityModel
from config import config


def test_batch_uplift_predictions():
    """Test that batch uplift predictions match sequential predictions"""
    print("Testing batch uplift predictions...")
    
    model = UpliftModel()
    # Don't set is_trained - test with cold start heuristic
    
    # Create sample data
    customers = pd.DataFrame({
        'skill_needed': ['technical', 'billing', 'sales'],
        'priority': ['high', 'medium', 'low'],
        'complexity': [0.7, 0.5, 0.3],
        'wait_time': [1.0, 2.0, 0.5],
        'channel': ['voice', 'chat', 'email']
    })
    
    agents = pd.DataFrame({
        'skill_technical': [0.9, 0.5, 0.2],
        'skill_billing': [0.3, 0.9, 0.4],
        'skill_sales': [0.2, 0.4, 0.9],
        'avg_csat': [4.5, 4.2, 4.8],
        'avg_aht': [6.0, 5.5, 7.0],
        'experience_years': [5, 3, 8],
        'current_load': [1, 0, 2],
        'preferred_channels': [['voice'], ['chat'], ['email']]
    })
    
    customer_indices = np.array([0, 1, 2])
    agent_indices = np.array([0, 1, 2])
    
    # Batch prediction
    batch_uplifts, batch_uncertainties = model.predict_uplift_batch(
        customers, agents, customer_indices, agent_indices, exploration=False
    )
    
    # Sequential predictions
    sequential_results = []
    for i in range(len(customer_indices)):
        uplift, uncertainty = model.predict_uplift(
            customers.iloc[customer_indices[i]],
            agents.iloc[agent_indices[i]],
            exploration=False
        )
        sequential_results.append((uplift, uncertainty))
    
    sequential_uplifts = np.array([r[0] for r in sequential_results])
    sequential_uncertainties = np.array([r[1] for r in sequential_results])
    
    # Verify results match (within numerical precision)
    assert np.allclose(batch_uplifts, sequential_uplifts, rtol=1e-10), \
        f"Uplifts don't match: batch={batch_uplifts}, sequential={sequential_uplifts}"
    
    assert np.allclose(batch_uncertainties, sequential_uncertainties, rtol=1e-10), \
        f"Uncertainties don't match: batch={batch_uncertainties}, sequential={sequential_uncertainties}"
    
    print("✅ Batch uplift predictions match sequential predictions")


def test_batch_aht_predictions():
    """Test that batch AHT predictions match sequential predictions"""
    print("Testing batch AHT predictions...")
    
    model = CapacityModel()
    # Don't set is_trained - test with cold start heuristic
    
    # Create sample data
    customers = pd.DataFrame({
        'skill_needed': ['technical', 'billing'],
        'complexity': [0.7, 0.5],
        'channel': ['voice', 'chat']
    })
    
    agents = pd.DataFrame({
        'skill_technical': [0.9, 0.5],
        'skill_billing': [0.3, 0.9],
        'avg_aht': [6.0, 5.5],
        'experience_years': [5, 3]
    })
    
    customer_indices = np.array([0, 1])
    agent_indices = np.array([0, 1])
    
    # Batch prediction
    batch_ahts = model.predict_aht_batch(
        customers, agents, customer_indices, agent_indices
    )
    
    # Sequential predictions
    sequential_ahts = np.array([
        model.predict_aht(
            customers.iloc[customer_indices[i]],
            agents.iloc[agent_indices[i]]
        )
        for i in range(len(customer_indices))
    ])
    
    # Verify results match
    assert np.allclose(batch_ahts, sequential_ahts, rtol=1e-10), \
        f"AHTs don't match: batch={batch_ahts}, sequential={sequential_ahts}"
    
    print("✅ Batch AHT predictions match sequential predictions")


def test_batch_capacity_checks():
    """Test that batch capacity checks work correctly"""
    print("Testing batch capacity checks...")
    
    model = CapacityModel()
    
    # Create agents with different load states
    agents = pd.DataFrame({
        'load_voice': [0, 1, 0],
        'load_chat': [0, 0, 2],  # Changed to make test clearer
        'load_email': [0, 0, 0]
    })
    
    agent_indices = np.array([0, 1, 2])
    channels = np.array(['voice', 'voice', 'chat'])
    
    # Batch capacity check
    capacity_mask = model.check_capacity_batch(agents, agent_indices, channels)
    
    # Agent 0: load_voice=0, requesting voice -> should have capacity (0 < 1)
    assert capacity_mask[0] == True, f"Agent 0 should have capacity for voice (mask={capacity_mask})"
    
    # Agent 1: load_voice=1, requesting voice -> should NOT have capacity (1 >= 1)
    assert capacity_mask[1] == False, f"Agent 1 should NOT have capacity for voice (mask={capacity_mask})"
    
    # Agent 2: load_chat=2, requesting chat -> should have capacity (2 < 3)
    assert capacity_mask[2] == True, f"Agent 2 should have capacity for chat (mask={capacity_mask})"
    
    print("✅ Batch capacity checks working correctly")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Batch Prediction Tests")
    print("=" * 60)
    
    try:
        test_batch_uplift_predictions()
        test_batch_aht_predictions()
        test_batch_capacity_checks()
        print("\n" + "=" * 60)
        print("✅ All batch prediction tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
