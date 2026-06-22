import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from config import config

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ResultsVisualizer:
    """
    Generate visualizations comparing routing policies
    """
    
    @staticmethod
    def plot_policy_comparison(results: Dict[str, Dict], 
                              save_path: str = None):
        """
        Create comprehensive comparison plots
        
        Args:
            results: Dict mapping policy_name -> metrics_dict
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Routing Policy Comparison', fontsize=16, fontweight='bold')
        
        policies = list(results.keys())
        
        # 1. CSAT Comparison
        ax = axes[0, 0]
        csat_values = [results[p]['overall_avg_csat'] for p in policies]
        bars = ax.bar(policies, csat_values, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_ylabel('Average CSAT', fontweight='bold')
        ax.set_title('Customer Satisfaction')
        ax.set_ylim([min(csat_values)*0.95, max(csat_values)*1.05])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. AHT Comparison
        ax = axes[0, 1]
        aht_values = [results[p]['overall_avg_aht'] for p in policies]
        ax.bar(policies, aht_values, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_ylabel('Avg Handle Time (min)', fontweight='bold')
        ax.set_title('Efficiency (Lower is Better)')
        ax.axhline(y=config.MAX_AHT_MINUTES, color='r', linestyle='--', 
                  label='SLA Threshold', linewidth=2)
        ax.legend()
        
        # 3. SLA Met Rate
        ax = axes[0, 2]
        sla_values = [results[p]['overall_sla_met_rate'] for p in policies]
        ax.bar(policies, sla_values, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_ylabel('SLA Met Rate', fontweight='bold')
        ax.set_title('Service Level Agreement')
        ax.set_ylim([0, 1])
        ax.axhline(y=1-config.MAX_SLA_VIOLATION_RATE, color='r', 
                  linestyle='--', label='Target', linewidth=2)
        ax.legend()
        
        # 4. Fairness (Gini)
        ax = axes[1, 0]
        gini_values = [results[p]['avg_gini'] for p in policies]
        ax.bar(policies, gini_values, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_ylabel('Gini Coefficient', fontweight='bold')
        ax.set_title('Agent Load Fairness (Lower is Better)')
        ax.set_ylim([0, max(gini_values)*1.2])
        ax.axhline(y=config.FAIRNESS_GINI_THRESHOLD, color='r', 
                  linestyle='--', label='Threshold', linewidth=2)
        ax.legend()
        
        # 5. Total Assignments
        ax = axes[1, 1]
        assign_values = [results[p]['total_assignments'] for p in policies]
        ax.bar(policies, assign_values, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_ylabel('Total Assignments', fontweight='bold')
        ax.set_title('Throughput')
        
        # 6. Improvement Summary Table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate improvement vs FCFS
        if 'FCFS' in policies:
            baseline = results['FCFS']
            table_data = []
            
            for policy in policies:
                if policy == 'FCFS':
                    continue
                
                csat_imp = ((results[policy]['overall_avg_csat'] - 
                           baseline['overall_avg_csat']) / 
                           baseline['overall_avg_csat']) * 100
                
                aht_imp = ((baseline['overall_avg_aht'] - 
                          results[policy]['overall_avg_aht']) / 
                          baseline['overall_avg_aht']) * 100
                
                sla_imp = ((results[policy]['overall_sla_met_rate'] - 
                          baseline['overall_sla_met_rate']) / 
                          baseline['overall_sla_met_rate']) * 100
                
                table_data.append([
                    policy,
                    f"+{csat_imp:.1f}%",
                    f"+{aht_imp:.1f}%",
                    f"+{sla_imp:.1f}%"
                ])
            
            table = ax.table(cellText=table_data,
                           colLabels=['Policy', 'CSAT↑', 'AHT↓', 'SLA↑'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.3, 0.2, 0.2, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            ax.set_title('Improvement vs FCFS Baseline', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_convergence(metrics_df: pd.DataFrame, 
                        policy_name: str,
                        save_path: str = None):
        """
        Plot metric convergence over batches
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{policy_name} - Convergence Over Time', 
                    fontsize=14, fontweight='bold')
        
        # Rolling average window
        window = 10
        
        # CSAT
        ax = axes[0, 0]
        ax.plot(metrics_df['batch_id'], 
               metrics_df['avg_csat'].rolling(window).mean(),
               linewidth=2, color='#2ecc71')
        ax.fill_between(metrics_df['batch_id'],
                       metrics_df['avg_csat'].rolling(window).mean() - 
                       metrics_df['std_csat'].rolling(window).mean(),
                       metrics_df['avg_csat'].rolling(window).mean() + 
                       metrics_df['std_csat'].rolling(window).mean(),
                       alpha=0.3, color='#2ecc71')
        ax.set_xlabel('Batch')
        ax.set_ylabel('CSAT')
        ax.set_title('Customer Satisfaction')
        ax.grid(True, alpha=0.3)
        
        # AHT
        ax = axes[0, 1]
        ax.plot(metrics_df['batch_id'], 
               metrics_df['avg_aht'].rolling(window).mean(),
               linewidth=2, color='#3498db')
        ax.axhline(y=config.MAX_AHT_MINUTES, color='r', linestyle='--', 
                  label='Threshold', linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Handle Time (min)')
        ax.set_title('Average Handle Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # SLA Met Rate
        ax = axes[1, 0]
        ax.plot(metrics_df['batch_id'], 
               metrics_df['sla_met_rate'].rolling(window).mean(),
               linewidth=2, color='#9b59b6')
        ax.axhline(y=1-config.MAX_SLA_VIOLATION_RATE, color='r', 
                  linestyle='--', label='Target', linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('SLA Met Rate')
        ax.set_title('Service Level Agreement')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Fairness (Gini)
        ax = axes[1, 1]
        ax.plot(metrics_df['batch_id'], 
               metrics_df['gini_coefficient'].rolling(window).mean(),
               linewidth=2, color='#e67e22')
        ax.axhline(y=config.FAIRNESS_GINI_THRESHOLD, color='r', 
                  linestyle='--', label='Threshold', linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Gini Coefficient')
        ax.set_title('Agent Load Fairness')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Convergence plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_agent_workload(workload_df: pd.DataFrame,
                           save_path: str = None):
        """
        Visualize agent workload distribution
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Agent Workload Analysis', fontsize=14, fontweight='bold')
        
        # Total assignments distribution
        ax = axes[0]
        workload_df = workload_df.sort_values('total_assignments', ascending=False)
        ax.bar(range(len(workload_df)), workload_df['total_assignments'],
              color='#3498db', alpha=0.7)
        ax.set_xlabel('Agent (sorted by load)', fontweight='bold')
        ax.set_ylabel('Total Assignments', fontweight='bold')
        ax.set_title('Assignment Distribution Across Agents')
        ax.axhline(y=workload_df['total_assignments'].mean(), 
                  color='r', linestyle='--', 
                  label=f'Mean: {workload_df["total_assignments"].mean():.1f}',
                  linewidth=2)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # CSAT vs AHT scatter
        ax = axes[1]
        scatter = ax.scatter(workload_df['avg_aht'], 
                           workload_df['avg_csat'],
                           s=workload_df['total_assignments']*10,
                           c=workload_df['total_assignments'],
                           cmap='viridis',
                           alpha=0.6)
        ax.set_xlabel('Average Handle Time (min)', fontweight='bold')
        ax.set_ylabel('Average CSAT', fontweight='bold')
        ax.set_title('Agent Performance: CSAT vs Efficiency')
        plt.colorbar(scatter, ax=ax, label='Total Assignments')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Workload plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def print_summary_table(results: Dict[str, Dict]):
        """
        Print formatted comparison table
        """
        print("\n" + "="*80)
        print(" "*25 + "ROUTING POLICY COMPARISON")
        print("="*80)
        
        # Header
        print(f"{'Metric':<30} | " + " | ".join([f"{p:^15}" for p in results.keys()]))
        print("-"*80)
        
        # Metrics
        metrics = [
            ('Avg CSAT', 'overall_avg_csat', '{:.4f}', 'higher'),
            ('Avg AHT (min)', 'overall_avg_aht', '{:.2f}', 'lower'),
            ('SLA Met Rate', 'overall_sla_met_rate', '{:.2%}', 'higher'),
            ('Fairness (Gini)', 'avg_gini', '{:.3f}', 'lower'),
            ('Total Assignments', 'total_assignments', '{:,}', 'higher'),
            ('CSAT Trend', 'csat_improvement_trend', '{:.5f}', 'higher')
        ]
        
        for metric_name, key, fmt, direction in metrics:
            values = []
            for policy in results.keys():
                val = results[policy].get(key, 0)
                values.append(val)
            
            # Highlight best
            if direction == 'higher':
                best_idx = values.index(max(values))
            else:
                best_idx = values.index(min(values))
            
            row = f"{metric_name:<30} | "
            for i, (policy, val) in enumerate(zip(results.keys(), values)):
                formatted = fmt.format(val)
                if i == best_idx:
                    formatted = f"**{formatted}**"
                row += f"{formatted:^15} | "
            
            print(row)
        
        print("="*80)
        
        # Calculate improvements
        if 'FCFS' in results:
            print("\nIMPROVEMENT OVER FCFS BASELINE:")
            print("-"*80)
            
            baseline = results['FCFS']
            
            for policy in results.keys():
                if policy == 'FCFS':
                    continue
                
                print(f"\n{policy}:")
                
                csat_imp = ((results[policy]['overall_avg_csat'] - 
                           baseline['overall_avg_csat']) / 
                           baseline['overall_avg_csat']) * 100
                print(f"  CSAT Uplift:        +{csat_imp:.2f}%")
                
                aht_imp = ((baseline['overall_avg_aht'] - 
                          results[policy]['overall_avg_aht']) / 
                          baseline['overall_avg_aht']) * 100
                print(f"  AHT Reduction:      +{aht_imp:.2f}%")
                
                sla_imp = ((results[policy]['overall_sla_met_rate'] - 
                          baseline['overall_sla_met_rate']) / 
                          baseline['overall_sla_met_rate']) * 100
                print(f"  SLA Improvement:    +{sla_imp:.2f}%")
                
                fairness_imp = ((baseline['avg_gini'] - 
                               results[policy]['avg_gini']) / 
                               baseline['avg_gini']) * 100
                print(f"  Fairness Gain:      +{fairness_imp:.2f}%")
            
            print("="*80)

