import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from config import config

class SyntheticDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.agents = self._generate_agents()
        self.historical_data = self._generate_historical_interactions()
    
    def _generate_agents(self) -> pd.DataFrame:
        """Generate agent profiles with skills, capacity, and historical performance"""
        agents = []
        for i in range(config.NUM_AGENTS):
            # Skill distribution
            primary_skill = np.random.choice(config.SKILLS)
            skill_level = np.random.choice(config.SKILL_LEVELS)
            
            # Skill vector (proficiency 0-1 for each skill)
            skills = {s: np.random.beta(2, 5) for s in config.SKILLS}
            skills[primary_skill] = np.random.beta(5, 2)  # Higher for primary
            
            # Capacity preferences
            preferred_channels = np.random.choice(
                config.CHANNELS, 
                size=np.random.randint(1, len(config.CHANNELS)+1),
                replace=False
            ).tolist()
            
            # Historical performance
            avg_csat = np.random.beta(8, 2)  # Skewed toward high satisfaction
            avg_aht = np.random.gamma(5, 1.2)  # Average handle time in minutes
            
            agents.append({
                'agent_id': f'A{i:03d}',
                'primary_skill': primary_skill,
                'skill_level': skill_level,
                **{f'skill_{s}': skills[s] for s in config.SKILLS},
                'preferred_channels': preferred_channels,
                'avg_csat': avg_csat,
                'avg_aht': avg_aht,
                'experience_years': np.random.uniform(0.5, 10),
                'current_load': 0,
                'total_assignments': 0
            })
        
        return pd.DataFrame(agents)
    
    def _generate_historical_interactions(self, n=5000) -> pd.DataFrame:
        """Generate historical interaction data for training uplift models"""
        interactions = []
        
        for _ in range(n):
            agent = self.agents.sample(1).iloc[0]
            channel = np.random.choice(config.CHANNELS)
            customer_skill = np.random.choice(config.SKILLS)
            customer_priority = np.random.choice(['low', 'medium', 'high', 'vip'])
            
            # Simulated outcome based on match quality
            skill_match = agent[f'skill_{customer_skill}']
            priority_boost = {'low': 0, 'medium': 0.05, 'high': 0.1, 'vip': 0.15}[customer_priority]
            
            # CSAT outcome (0-1)
            base_csat = agent['avg_csat']
            csat = np.clip(
                base_csat * skill_match + priority_boost + np.random.normal(0, 0.1),
                0, 1
            )
            
            # AHT outcome
            aht = agent['avg_aht'] * (1.2 - skill_match) + np.random.gamma(2, 0.5)
            
            # SLA met (depends on AHT and agent efficiency)
            sla_threshold = {'voice': 5, 'chat': 3, 'email': 24*60}[channel]
            sla_met = 1 if aht < sla_threshold else 0
            
            interactions.append({
                'agent_id': agent['agent_id'],
                'channel': channel,
                'customer_skill': customer_skill,
                'customer_priority': customer_priority,
                'skill_match': skill_match,
                'csat': csat,
                'aht': aht,
                'sla_met': sla_met,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 90))
            })
        
        return pd.DataFrame(interactions)
    
    def generate_customer_batch(self, batch_size: int) -> pd.DataFrame:
        """Generate a batch of arriving customers"""
        customers = []
        
        for i in range(batch_size):
            channel = np.random.choice(config.CHANNELS, p=[0.5, 0.3, 0.2])
            skill_needed = np.random.choice(config.SKILLS)
            priority = np.random.choice(['low', 'medium', 'high', 'vip'], p=[0.4, 0.3, 0.2, 0.1])
            
            # Complexity score
            complexity = np.random.beta(2, 5)
            
            # Wait time so far
            wait_time = np.random.exponential(2)
            
            customers.append({
                'customer_id': f'C{np.random.randint(10000, 99999)}',
                'channel': channel,
                'skill_needed': skill_needed,
                'priority': priority,
                'complexity': complexity,
                'wait_time': wait_time,
                'arrival_time': pd.Timestamp.now()
            })
        
        return pd.DataFrame(customers)
    
    def simulate_interaction_outcome(self, customer: pd.Series, agent: pd.Series) -> Dict:
        """Simulate the outcome of routing a customer to an agent"""
        skill_match = agent[f'skill_{customer["skill_needed"]}']
        
        # True CSAT (unknown in real-time, observed later)
        priority_boost = {'low': 0, 'medium': 0.05, 'high': 0.1, 'vip': 0.15}[customer['priority']]
        wait_penalty = -0.01 * customer['wait_time']
        
        csat = np.clip(
            agent['avg_csat'] * skill_match + 
            priority_boost + 
            wait_penalty + 
            np.random.normal(0, 0.1),
            0, 1
        )
        
        # AHT
        aht = agent['avg_aht'] * (1.3 - skill_match) * (1 + customer['complexity']) + \
              np.random.gamma(2, 0.5)
        
        # SLA
        sla_threshold = {'voice': 5, 'chat': 3, 'email': 24*60}[customer['channel']]
        sla_met = 1 if aht < sla_threshold else 0
        
        return {
            'csat': csat,
            'aht': aht,
            'sla_met': sla_met,
            'skill_match': skill_match
        }
    
    def get_agent_availability(self, agent: pd.Series, channel: str) -> int:
        """Check how many more interactions of this channel the agent can handle"""
        current_channel_load = agent.get(f'load_{channel}', 0)
        max_capacity = config.CAPACITY_RULES[channel]
        
        # Check cross-channel constraints
        for other_channel in config.CHANNELS:
            if other_channel != channel:
                other_load = agent.get(f'load_{other_channel}', 0)
                if other_load > 0:
                    # Agent is busy on another channel
                    max_capacity = min(
                        max_capacity,
                        config.CROSS_CHANNEL_CAPACITY[other_channel][channel]
                    )
        
        return max(0, max_capacity - current_channel_load)

# Instantiate generator
data_gen = SyntheticDataGenerator()

