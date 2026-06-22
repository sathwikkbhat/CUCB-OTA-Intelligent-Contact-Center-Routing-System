import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict

class OffPolicyEvaluator:
    """
    Off-Policy Evaluation for routing policies
    Estimates performance of new policies using logged interaction data
    """
    
    def __init__(self):
        self.clear_logs()
    
    def clear_logs(self):
        """Clear all logged interactions"""
        self.logged_interactions = []
    
    def log_interaction(self, 
                       customer: pd.Series,
                       agent: pd.Series,
                       assignment_prob: float,
                       outcome: Dict):
        """
        Log an interaction for OPE
        
        Args:
            customer: Customer features
            agent: Agent features
            assignment_prob: Probability of this assignment under logging policy
            outcome: Outcome dict with csat, aht, sla_met, etc.
        """
        self.logged_interactions.append({
            'customer': customer.to_dict(),
            'agent': agent.to_dict(),
            'assignment_prob': assignment_prob,
            'outcome': outcome
        })
    
    def estimate_policy_value(self, 
                              policy_fn: callable,
                              discount_factor: float = 1.0) -> Dict:
        """
        Estimate value of a new policy using importance sampling
        
        Args:
            policy_fn: Policy function to evaluate
            discount_factor: Discount factor for future rewards
        
        Returns:
            Dict with estimated metrics
        """
        if len(self.logged_interactions) == 0:
            return {}
        
        # Simple importance sampling estimator
        weighted_csat = []
        weighted_aht = []
        weighted_sla = []
        
        for log_entry in self.logged_interactions:
            # Compute importance weight (simplified)
            # In practice, would compute actual policy probability
            importance_weight = 1.0 / log_entry['assignment_prob']
            
            weighted_csat.append(log_entry['outcome']['csat'] * importance_weight)
            weighted_aht.append(log_entry['outcome']['aht'] * importance_weight)
            weighted_sla.append(log_entry['outcome']['sla_met'] * importance_weight)
        
        # Normalize weights
        total_weight = sum(1.0 / log['assignment_prob'] for log in self.logged_interactions)
        
        return {
            'estimated_csat': np.sum(weighted_csat) / total_weight if total_weight > 0 else 0,
            'estimated_aht': np.sum(weighted_aht) / total_weight if total_weight > 0 else 0,
            'estimated_sla_rate': np.sum(weighted_sla) / total_weight if total_weight > 0 else 0,
            'n_samples': len(self.logged_interactions)
        }

