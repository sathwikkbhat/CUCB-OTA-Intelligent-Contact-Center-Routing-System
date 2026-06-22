import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
import pandas as pd
from config import config


class AssignmentSolver:
    """
    Solves optimal assignment using Hungarian algorithm
    Handles one-to-one (voice) and one-to-many (chat/email) assignments
    """
    
    @staticmethod
    def solve_hungarian(RS_matrix: np.ndarray, 
                       customers: pd.DataFrame,
                       agents: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Solve optimal assignment using Hungarian algorithm (Kuhn-Munkres).
        
        Optimized version with efficient filtering and vectorized operations.
        
        Args:
            RS_matrix: Routing score matrix (num_customers x num_agents)
            customers: Customer DataFrame (used for type consistency)
            agents: Agent DataFrame (used for type consistency)
        
        Returns: List of (customer_idx, agent_idx) assignment tuples
        """
        num_customers, num_agents = RS_matrix.shape
        
        # Hungarian algorithm minimizes cost, so negate routing scores
        cost_matrix = -RS_matrix.copy()
        
        # Replace -inf with very large cost (infeasible assignments)
        large_cost_value = 1e9
        cost_matrix[np.isinf(cost_matrix)] = large_cost_value
        
        # Solve optimal assignment using Hungarian algorithm
        customer_indices, agent_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out invalid assignments (where original RS was -inf)
        # Use vectorized boolean indexing for efficiency
        if len(customer_indices) > 0:
            valid_mask = RS_matrix[customer_indices, agent_indices] > -np.inf
            valid_customer_indices = customer_indices[valid_mask]
            valid_agent_indices = agent_indices[valid_mask]
            # Return as list of tuples
            assignments = list(zip(valid_customer_indices, valid_agent_indices))
        else:
            assignments = []
        
        return assignments
    
    @staticmethod
    def solve_greedy_with_capacity(RS_matrix: np.ndarray,
                                   customers: pd.DataFrame,
                                   agents: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Greedy assignment respecting multi-channel capacity.
        Optimized version using vectorized operations and efficient data structures.
        
        Used for channels like chat/email where agents can handle multiple interactions.
        
        Returns: List of (customer_idx, agent_idx) assignment tuples
        """
        num_customers, num_agents = RS_matrix.shape
        assignments = []
        
        # Pre-extract customer channels (vectorized)
        customer_channels = customers['channel'].values
        
        # Initialize agent capacity tracking efficiently
        # Use numpy array instead of nested dict for better performance
        agent_capacity_matrix = np.zeros((num_agents, len(config.CHANNELS)), dtype=np.int32)
        channel_to_index = {ch: idx for idx, ch in enumerate(config.CHANNELS)}
        
        for agent_idx in range(num_agents):
            for channel_idx, channel in enumerate(config.CHANNELS):
                agent_capacity_matrix[agent_idx, channel_idx] = config.CAPACITY_RULES[channel]
        
        # Create sorted list of valid (RS_score, customer_idx, agent_idx) tuples
        # Use numpy boolean masking for efficiency
        valid_mask = RS_matrix > -np.inf
        customer_indices, agent_indices = np.where(valid_mask)
        routing_scores = RS_matrix[valid_mask]
        
        # Sort by routing score (descending) - use argsort for efficiency
        sorted_indices = np.argsort(routing_scores)[::-1]
        
        assigned_customers = set()
        
        # Process assignments in order of decreasing routing score
        for idx in sorted_indices:
            customer_idx = customer_indices[idx]
            agent_idx = agent_indices[idx]
            
            # Skip if customer already assigned
            if customer_idx in assigned_customers:
                continue
            
            # Get channel index for this customer
            customer_channel = customer_channels[customer_idx]
            channel_idx = channel_to_index[customer_channel]
            
            # Check if agent has capacity for this channel
            if agent_capacity_matrix[agent_idx, channel_idx] > 0:
                assignments.append((customer_idx, agent_idx))
                assigned_customers.add(customer_idx)
                agent_capacity_matrix[agent_idx, channel_idx] -= 1
        
        return assignments
    
    @staticmethod
    def hybrid_solve(RS_matrix: np.ndarray,
                    customers: pd.DataFrame,
                    agents: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Hybrid assignment approach: Hungarian algorithm for voice (one-to-one),
        greedy algorithm for chat/email (one-to-many).
        
        Optimizations:
        - Avoids unnecessary DataFrame copies
        - Efficient boolean masking
        - Direct index mapping
        
        Args:
            RS_matrix: Routing score matrix
            customers: Customer DataFrame with 'channel' column
            agents: Agent DataFrame
        
        Returns: List of (customer_idx, agent_idx) assignment tuples
        """
        customer_channels = customers['channel'].values
        voice_mask = customer_channels == 'voice'
        other_mask = ~voice_mask
        
        assignments = []
        
        # Hungarian algorithm for voice calls (one-to-one assignment)
        if np.any(voice_mask):
            voice_RS_submatrix = RS_matrix[voice_mask, :]
            voice_assignments = AssignmentSolver.solve_hungarian(
                voice_RS_submatrix, customers, agents
            )
            
            # Map submatrix indices back to original DataFrame indices
            voice_original_indices = np.where(voice_mask)[0]
            for submatrix_c_idx, agent_idx in voice_assignments:
                original_customer_idx = voice_original_indices[submatrix_c_idx]
                assignments.append((original_customer_idx, agent_idx))
        
        # Greedy algorithm for chat/email (one-to-many assignment)
        if np.any(other_mask):
            other_RS_submatrix = RS_matrix[other_mask, :]
            other_customers_subset = customers[other_mask].reset_index(drop=True)
            
            # Note: Capacity updates are handled internally in greedy solver
            # No need to copy agents DataFrame - capacity is tracked per-channel
            other_assignments = AssignmentSolver.solve_greedy_with_capacity(
                other_RS_submatrix, other_customers_subset, agents
            )
            
            # Map submatrix indices back to original DataFrame indices
            other_original_indices = np.where(other_mask)[0]
            for submatrix_c_idx, agent_idx in other_assignments:
                original_customer_idx = other_original_indices[submatrix_c_idx]
                assignments.append((original_customer_idx, agent_idx))
        
        return assignments

