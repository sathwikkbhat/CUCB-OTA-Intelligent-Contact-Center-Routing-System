import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from tqdm import tqdm
import time

from config import config
from data.synthetic_data import SyntheticDataGenerator
from models.uplift_model import UpliftModel, CapacityModel
from routing.scoring import RoutingScorer
from routing.assignment import AssignmentSolver
from evaluation.metrics import MetricsTracker
from evaluation.ope import OffPolicyEvaluator


class RoutingSimulator:
    """
    Main simulation engine for routing policies
    """
    
    def __init__(self, 
                 data_generator: SyntheticDataGenerator,
                 uplift_model: UpliftModel,
                 capacity_model: CapacityModel):
        self.data_gen = data_generator
        self.uplift_model = uplift_model
        self.capacity_model = capacity_model
        
        self.agents = data_generator.agents.copy()
        self._initialize_agent_loads()
        
        self.scorer = RoutingScorer(uplift_model, capacity_model)
        self.metrics = MetricsTracker()
        self.ope = OffPolicyEvaluator()
        
        self.batch_counter = 0
    
    def _initialize_agent_loads(self):
        """Initialize load tracking for each agent"""
        for channel in config.CHANNELS:
            self.agents[f'load_{channel}'] = 0
        self.agents['current_load'] = 0
    
    def run_simulation(self, 
                      n_batches: int,
                      policy_fn: Callable,
                      policy_name: str = "CUCB-OTA") -> Dict:
        """
        Run routing simulation for n_batches
        
        Args:
            n_batches: Number of customer batches to process
            policy_fn: Function that takes (customers, agents, scorer) -> assignments
            policy_name: Name for logging
        
        Returns:
            Summary metrics dict
        """
        print(f"\n{'='*60}")
        print(f"Running {policy_name} Simulation")
        print(f"{'='*60}")
        
        for batch_id in tqdm(range(n_batches), desc=f"{policy_name} Batches"):
            # Generate arriving customers
            batch_size = np.random.poisson(config.NUM_CUSTOMERS_PER_BATCH)
            batch_size = max(1, min(batch_size, 100))  # Clamp to reasonable range
            
            customers = self.data_gen.generate_customer_batch(batch_size)
            
            # Use agents directly (policies should not modify the DataFrame)
            # Avoid unnecessary copy for performance
            available_agents = self.agents  # Reference, not copy
            
            # Compute assignments using policy
            assignments = policy_fn(customers, available_agents, self.scorer)
            
            # Print assignments for first few batches (to show in terminal)
            if batch_id < 3:  # Print first 3 batches only
                tqdm.write(f"\n  Batch {batch_id} Assignments ({len(assignments)} total):")
                for c_idx, a_idx in assignments:
                    customer = customers.iloc[c_idx]
                    agent = available_agents.iloc[a_idx]
                    tqdm.write(f"    Customer {customer['customer_id']} → Agent {agent['agent_id']} "
                               f"| Channel: {customer['channel']:<6} | Skill: {customer['skill_needed']:<10}")
            
            # Simulate outcomes
            outcomes = []
            for c_idx, a_idx in assignments:
                customer = customers.iloc[c_idx]
                agent = available_agents.iloc[a_idx]
                
                outcome = self.data_gen.simulate_interaction_outcome(customer, agent)
                outcomes.append(outcome)
                
                # Update agent load
                channel = customer['channel']
                self.agents.at[agent.name, f'load_{channel}'] += 1
                self.agents.at[agent.name, 'current_load'] += 1
                self.agents.at[agent.name, 'total_assignments'] += 1
                
                # Log for OPE
                assignment_prob = 1.0 / len(available_agents)  # Uniform for simplicity
                self.ope.log_interaction(customer, agent, assignment_prob, outcome)
            
            # Decay agent loads (simulate completion)
            self._decay_agent_loads()
            
            # Record metrics
            self.metrics.record_batch(
                batch_id, assignments, customers, available_agents, 
                outcomes, {}
            )
            
            # Update dual variables every N batches
            if batch_id % config.LOG_INTERVAL == 0 and batch_id > 0:
                summary = self.metrics.get_summary_stats()
                realized_metrics = {
                    'avg_aht': summary.get('overall_avg_aht', 5),
                    'sla_met_rate': summary.get('overall_sla_met_rate', 0.9),
                    'gini': summary.get('avg_gini', 0.2)
                }
                self.scorer.update_dual_variables(realized_metrics)
                
                # Print progress
                if batch_id % (config.LOG_INTERVAL * 2) == 0:
                    dual_state = self.scorer.get_dual_state()
                    print(f"\nBatch {batch_id}: CSAT={summary['overall_avg_csat']:.3f}, "
                          f"AHT={summary['overall_avg_aht']:.2f}, "
                          f"SLA={summary['overall_sla_met_rate']:.2%}, "
                          f"λ_aht={dual_state['lambda_aht']:.3f}")
        
        # Final summary
        final_summary = self.metrics.get_summary_stats()
        final_summary['policy_name'] = policy_name
        final_summary['dual_variables'] = self.scorer.get_dual_state()
        
        print(f"\n{policy_name} Summary:")
        print(f"  Total Assignments: {final_summary['total_assignments']}")
        print(f"  Avg CSAT: {final_summary['overall_avg_csat']:.4f}")
        print(f"  Avg AHT: {final_summary['overall_avg_aht']:.2f} min")
        print(f"  SLA Met Rate: {final_summary['overall_sla_met_rate']:.2%}")
        print(f"  Fairness (Gini): {final_summary['avg_gini']:.3f}")
        
        return final_summary
    
    def _decay_agent_loads(self, decay_rate: float = 0.1):
        """
        Simulate interactions completing (decay agent loads)
        """
        for channel in config.CHANNELS:
            self.agents[f'load_{channel}'] = np.maximum(
                0, 
                self.agents[f'load_{channel}'] - np.random.poisson(decay_rate, len(self.agents))
            )
        
        self.agents['current_load'] = sum(
            self.agents[f'load_{channel}'] for channel in config.CHANNELS
        )
    
    def reset(self):
        """Reset simulation state"""
        self.agents = self.data_gen.agents.copy()
        self._initialize_agent_loads()
        self.metrics.reset()
        self.ope.clear_logs()
        self.batch_counter = 0


# Policy implementations

def cucb_ota_policy(customers: pd.DataFrame, 
                   agents: pd.DataFrame, 
                   scorer: RoutingScorer) -> List[Tuple[int, int]]:
    """
    CUCB-OTA policy: Constrained Uplift Contextual Bandit + Optimal Transport Assignment
    """
    # Compute routing scores with exploration
    RS_matrix, metadata = scorer.compute_routing_matrix(
        customers, agents, exploration=True
    )
    
    # Solve assignment
    assignments = AssignmentSolver.hybrid_solve(RS_matrix, customers, agents)
    
    return assignments


def fcfs_policy(customers: pd.DataFrame,
               agents: pd.DataFrame,
               scorer: RoutingScorer) -> List[Tuple[int, int]]:
    """
    First-Come-First-Served baseline policy.
    
    Routes customers in arrival order to the first available agent with capacity.
    Optimized to use integer indices instead of iterrows().
    
    Returns: List of (customer_idx, agent_idx) assignment tuples
    """
    assignments = []
    num_agents = len(agents)
    
    # Sort customers by arrival time and get sorted indices
    sorted_indices = customers.sort_values('arrival_time').index.tolist()
    
    # Track agent availability using boolean array (faster than dict)
    agent_available = np.ones(num_agents, dtype=bool)
    
    # Pre-extract customer channels for faster access
    customer_channels = customers['channel'].values
    
    for customer_idx in sorted_indices:
        customer = customers.iloc[customer_idx]
        customer_channel = customer_channels[customer_idx]
        
        # Find first available agent with capacity
        for agent_idx in range(num_agents):
            if not agent_available[agent_idx]:
                continue
            
            agent = agents.iloc[agent_idx]
            
            # Check capacity
            if scorer.capacity_model.check_capacity(agent, customer_channel):
                assignments.append((customer_idx, agent_idx))
                agent_available[agent_idx] = False
                break
    
    return assignments


def skill_based_greedy_policy(customers: pd.DataFrame,
                             agents: pd.DataFrame,
                             scorer: RoutingScorer) -> List[Tuple[int, int]]:
    """
    Greedy skill-matching baseline policy.
    
    Routes each customer to the agent with highest skill match for the required skill,
    subject to capacity constraints.
    
    Optimized to use integer indices and vectorized skill matching.
    
    Returns: List of (customer_idx, agent_idx) assignment tuples
    """
    assignments = []
    num_customers = len(customers)
    num_agents = len(agents)
    
    # Track agent availability using boolean array
    agent_available = np.ones(num_agents, dtype=bool)
    
    # Pre-extract customer data for faster access
    customer_skill_needed = customers['skill_needed'].values
    customer_channels = customers['channel'].values
    
    # Pre-extract agent skill columns for vectorized operations
    skill_columns = [f'skill_{skill}' for skill in config.SKILLS]
    
    for customer_idx in range(num_customers):
        customer = customers.iloc[customer_idx]
        required_skill = customer_skill_needed[customer_idx]
        customer_channel = customer_channels[customer_idx]
        
        best_agent_idx = None
        best_skill_match = -1.0
        
        # Find agent with best skill match among available agents
        for agent_idx in range(num_agents):
            if not agent_available[agent_idx]:
                continue
            
            agent = agents.iloc[agent_idx]
            
            # Check capacity constraint
            if not scorer.capacity_model.check_capacity(agent, customer_channel):
                continue
            
            # Get skill match score (vectorized access)
            skill_match_score = agent[f'skill_{required_skill}']
            
            if skill_match_score > best_skill_match:
                best_skill_match = skill_match_score
                best_agent_idx = agent_idx
        
        # Assign to best available agent
        if best_agent_idx is not None:
            assignments.append((customer_idx, best_agent_idx))
            agent_available[best_agent_idx] = False
    
    return assignments

