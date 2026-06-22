import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import json
from datetime import datetime

class MetricsTracker:
    """
    Track and aggregate routing metrics over time
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear all metrics"""
        self.batch_metrics = []
        self.assignment_history = []
        self.agent_workload = defaultdict(lambda: {
            'assignments': 0,
            'total_aht': 0,
            'total_csat': 0,
            'channels': defaultdict(int)
        })
    
    def record_batch(self, 
                    batch_id: int,
                    assignments: List[Tuple[int, int]],
                    customers: pd.DataFrame,
                    agents: pd.DataFrame,
                    outcomes: List[Dict],
                    metadata: Dict):
        """
        Record metrics for a routing batch
        
        Args:
            batch_id: Batch identifier
            assignments: List of (customer_idx, agent_idx) tuples
            customers: Customer DataFrame
            agents: Agent DataFrame
            outcomes: List of outcome dicts from simulation
            metadata: Additional metadata (RS scores, etc.)
        """
        if len(assignments) == 0:
            return
        
        # Aggregate outcomes
        csat_scores = [o['csat'] for o in outcomes]
        aht_values = [o['aht'] for o in outcomes]
        sla_met = [o['sla_met'] for o in outcomes]
        skill_matches = [o['skill_match'] for o in outcomes]
        
        # Compute batch metrics
        batch_metric = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'n_assignments': len(assignments),
            'n_customers': len(customers),
            'n_agents': len(agents),
            'avg_csat': np.mean(csat_scores),
            'std_csat': np.std(csat_scores),
            'avg_aht': np.mean(aht_values),
            'std_aht': np.std(aht_values),
            'sla_met_rate': np.mean(sla_met),
            'avg_skill_match': np.mean(skill_matches),
            'min_csat': np.min(csat_scores),
            'max_csat': np.max(csat_scores),
        }
        
        # Channel breakdown
        channel_counts = customers.iloc[[c for c, _ in assignments]]['channel'].value_counts()
        for channel, count in channel_counts.items():
            batch_metric[f'assignments_{channel}'] = count
        
        # Fairness: Gini coefficient of agent workload
        agent_loads = [agents.iloc[a]['current_load'] for _, a in assignments]
        batch_metric['gini_coefficient'] = self._compute_gini(agent_loads)
        
        self.batch_metrics.append(batch_metric)
        
        # Update agent workload tracking
        for (c_idx, a_idx), outcome in zip(assignments, outcomes):
            agent_id = agents.iloc[a_idx]['agent_id']
            channel = customers.iloc[c_idx]['channel']
            
            self.agent_workload[agent_id]['assignments'] += 1
            self.agent_workload[agent_id]['total_aht'] += outcome['aht']
            self.agent_workload[agent_id]['total_csat'] += outcome['csat']
            self.agent_workload[agent_id]['channels'][channel] += 1
            
            self.assignment_history.append({
                'batch_id': batch_id,
                'customer_id': customers.iloc[c_idx]['customer_id'],
                'agent_id': agent_id,
                'channel': channel,
                'csat': outcome['csat'],
                'aht': outcome['aht'],
                'sla_met': outcome['sla_met']
            })
    
    def _compute_gini(self, values: List[float]) -> float:
        """
        Compute Gini coefficient (inequality measure)
        0 = perfect equality, 1 = perfect inequality
        """
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(values)
        
        # Handle case where all values are zero
        if np.sum(sorted_values) == 0:
            return 0.0
        
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        return max(0, gini)
    
    def get_summary_stats(self) -> Dict:
        """
        Compute overall summary statistics
        """
        if len(self.batch_metrics) == 0:
            return {}
        
        df = pd.DataFrame(self.batch_metrics)
        
        summary = {
            'total_batches': len(df),
            'total_assignments': df['n_assignments'].sum(),
            'overall_avg_csat': df['avg_csat'].mean(),
            'overall_avg_aht': df['avg_aht'].mean(),
            'overall_sla_met_rate': df['sla_met_rate'].mean(),
            'avg_gini': df['gini_coefficient'].mean(),
            'csat_improvement_trend': self._compute_trend(df['avg_csat'].values),
        }
        
        # Agent fairness stats
        agent_assignment_counts = [w['assignments'] for w in self.agent_workload.values()]
        if len(agent_assignment_counts) > 0:
            summary['agent_load_std'] = np.std(agent_assignment_counts)
            summary['agent_load_gini'] = self._compute_gini(agent_assignment_counts)
        
        return summary
    
    def _compute_trend(self, values: np.ndarray) -> float:
        """
        Simple linear trend: positive = improving
        """
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return metrics as DataFrame"""
        return pd.DataFrame(self.batch_metrics)
    
    def get_agent_workload_dataframe(self) -> pd.DataFrame:
        """Return agent workload summary"""
        records = []
        for agent_id, stats in self.agent_workload.items():
            if stats['assignments'] > 0:
                records.append({
                    'agent_id': agent_id,
                    'total_assignments': stats['assignments'],
                    'avg_aht': stats['total_aht'] / stats['assignments'],
                    'avg_csat': stats['total_csat'] / stats['assignments'],
                    **{f'channel_{k}': v for k, v in stats['channels'].items()}
                })
        return pd.DataFrame(records)
    
    def save_to_file(self, filepath: str):
        """Save metrics to JSON file"""
        data = {
            'batch_metrics': self.batch_metrics,
            'summary': self.get_summary_stats(),
            'agent_workload': dict(self.agent_workload)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✓ Metrics saved to {filepath}")

