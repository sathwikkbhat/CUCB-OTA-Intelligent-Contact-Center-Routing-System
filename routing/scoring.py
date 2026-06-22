import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from models.uplift_model import UpliftModel, CapacityModel
from config import config

class RoutingScorer:
    """
    Computes Routing Score (RS) for each (customer, agent) pair
    RS = predicted_uplift - Σ(lambda_i * penalty_i)
    """
    def __init__(self, uplift_model: UpliftModel, capacity_model: CapacityModel):
        self.uplift_model = uplift_model
        self.capacity_model = capacity_model
        
        # Lagrangian multipliers (dual variables)
        self.lambda_aht = config.LAMBDA_INIT
        self.lambda_sla = config.LAMBDA_INIT
        self.lambda_fairness = config.LAMBDA_INIT
    
    def compute_routing_matrix(self, 
                               customers: pd.DataFrame, 
                               agents: pd.DataFrame,
                               exploration: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Compute K x M routing score matrix using vectorized operations for performance.
        
        Optimizations:
        - Vectorized feature extraction where possible
        - Pre-computed lookups for SLA thresholds and skill indices
        - Reduced function call overhead
        
        Returns: (RS_matrix, metadata_dict)
        """
        num_customers = len(customers)
        num_agents = len(agents)
        
        # Initialize matrices
        RS_matrix = np.full((num_customers, num_agents), -np.inf, dtype=np.float64)
        metadata = {
            'uplift': np.zeros((num_customers, num_agents)),
            'aht_penalty': np.zeros((num_customers, num_agents)),
            'capacity_mask': np.zeros((num_customers, num_agents), dtype=bool),
            'skill_match': np.zeros((num_customers, num_agents))
        }
        
        # Pre-compute frequently accessed values
        agent_current_loads = agents['current_load'].values  # Vectorized access
        agent_skill_columns = [f'skill_{skill}' for skill in config.SKILLS]
        
        # Pre-compute channel-to-SLA-threshold mapping for all customers
        customer_channels = customers['channel'].values
        customer_skill_needed = customers['skill_needed'].values
        sla_thresholds = np.array([config.SLA_THRESHOLDS[ch] for ch in customer_channels])
        
        # Vectorized fairness penalty (agent loads normalized)
        fairness_normalizer = config.NUM_AGENTS * config.FAIRNESS_NORMALIZATION_FACTOR
        agent_fairness_penalties = agent_current_loads / fairness_normalizer
        
        # OPTIMIZED: Use batch predictions instead of sequential calls
        # Create all (customer, agent) pairs upfront
        customer_indices_all = np.repeat(np.arange(num_customers), num_agents)
        agent_indices_all = np.tile(np.arange(num_agents), num_customers)
        customer_channels_all = np.repeat(customer_channels, num_agents)
        
        # Batch capacity check (much faster than individual checks)
        capacity_mask_flat = self.capacity_model.check_capacity_batch(
            agents, agent_indices_all, customer_channels_all
        )
        metadata['capacity_mask'] = capacity_mask_flat.reshape(num_customers, num_agents)
        
        # Vectorized skill matching: for each customer, get skill matches across all agents
        for customer_idx in range(num_customers):
            customer_skill = customer_skill_needed[customer_idx]
            agent_skill_matches = agents[f'skill_{customer_skill}'].values
            metadata['skill_match'][customer_idx, :] = agent_skill_matches
        
        # Filter to only valid pairs (where capacity is available)
        valid_mask = capacity_mask_flat
        valid_customer_indices = customer_indices_all[valid_mask]
        valid_agent_indices = agent_indices_all[valid_mask]
        
        if len(valid_customer_indices) == 0:
            # No valid assignments
            return RS_matrix, metadata
        
        # BATCH PREDICTIONS: Process all valid pairs at once (HUGE performance win!)
        # This replaces K×M sequential model calls with 2 batch calls
        uplift_values, _ = self.uplift_model.predict_uplift_batch(
            customers, agents, valid_customer_indices, valid_agent_indices, exploration
        )
        
        aht_values = self.capacity_model.predict_aht_batch(
            customers, agents, valid_customer_indices, valid_agent_indices
        )
        
        # Compute penalties and routing scores for valid pairs (vectorized)
        valid_customer_channels = customer_channels[valid_customer_indices]
        valid_sla_thresholds = sla_thresholds[valid_customer_indices]
        valid_agent_loads = agent_fairness_penalties[valid_agent_indices]
        
        # AHT penalty: violation above threshold
        aht_penalties = np.maximum(0.0, aht_values - config.MAX_AHT_MINUTES)
        
        # SLA penalty: binary (1 if violation, 0 otherwise)
        sla_penalties = (aht_values > valid_sla_thresholds).astype(float)
        
        # Routing scores for valid pairs (vectorized computation)
        routing_scores = (
            uplift_values
            - self.lambda_aht * aht_penalties
            - self.lambda_sla * sla_penalties
            - self.lambda_fairness * valid_agent_loads
        )
        
        # Vectorized storage of results (much faster than loop)
        metadata['uplift'][valid_customer_indices, valid_agent_indices] = uplift_values
        metadata['aht_penalty'][valid_customer_indices, valid_agent_indices] = aht_penalties
        RS_matrix[valid_customer_indices, valid_agent_indices] = routing_scores
        
        return RS_matrix, metadata
    
    def update_dual_variables(self, realized_metrics: Dict):
        """
        Update Lagrangian multipliers based on constraint violations
        λ ← max(0, λ + η * (constraint_violation - budget))
        """
        lr = config.LAMBDA_LR
        
        # AHT constraint
        aht_violation = realized_metrics['avg_aht'] - config.MAX_AHT_MINUTES
        self.lambda_aht = np.clip(
            self.lambda_aht + lr * aht_violation,
            0, config.LAMBDA_MAX
        )
        
        # SLA constraint
        sla_violation = config.MAX_SLA_VIOLATION_RATE - realized_metrics['sla_met_rate']
        self.lambda_sla = np.clip(
            self.lambda_sla + lr * sla_violation,
            0, config.LAMBDA_MAX
        )
        
        # Fairness constraint
        fairness_violation = realized_metrics['gini'] - config.FAIRNESS_GINI_THRESHOLD
        self.lambda_fairness = np.clip(
            self.lambda_fairness + lr * fairness_violation,
            0, config.LAMBDA_MAX
        )
    
    def get_dual_state(self) -> Dict:
        """Return current dual variables"""
        return {
            'lambda_aht': self.lambda_aht,
            'lambda_sla': self.lambda_sla,
            'lambda_fairness': self.lambda_fairness
        }

