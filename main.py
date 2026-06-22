#!/usr/bin/env python3
"""
Smart Queue Routing System (SQRS) - Hackathon POC
CUCB-OTA vs Baseline Policies

Usage: python main.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

# Import all modules
from config import config
from data.synthetic_data import SyntheticDataGenerator, data_gen
from models.uplift_model import UpliftModel, CapacityModel
from routing.scoring import RoutingScorer
from routing.assignment import AssignmentSolver
from evaluation.metrics import MetricsTracker
from evaluation.ope import OffPolicyEvaluator
from simulation.simulator import (
    RoutingSimulator, 
    cucb_ota_policy, 
    fcfs_policy, 
    skill_based_greedy_policy
)
from simulation.visualizer import ResultsVisualizer


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║          SMART QUEUE ROUTING SYSTEM (SQRS)                     ║
    ║      CUCB-OTA: Causal Uplift Contextual Bandit +               ║
    ║            Optimal Transport Assignment                        ║
    ║                                                                ║
    ║  Problem: Tetherfi - AI-Driven Queue Routing                   ║
    ║  Team: Fullstack Alchemists                                    ║
    ║  Hackathon: HACKOTSAVA 2025                                    ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_problem_statement():
    """Print clear problem statement explanation"""
    print("\n" + "="*60)
    print("PROBLEM STATEMENT")
    print("="*60)
   
    print("="*60 + "\n")


def print_config():
    """Print configuration summary"""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Agents:              {config.NUM_AGENTS}")
    print(f"Customers/Batch:     {config.NUM_CUSTOMERS_PER_BATCH}")
    print(f"Channels:            {', '.join(config.CHANNELS)}")
    print(f"Skills:              {', '.join(config.SKILLS)}")
    print(f"Capacity Rules:      {config.CAPACITY_RULES}")
    print(f"Max AHT:             {config.MAX_AHT_MINUTES} min")
    print(f"Max SLA Violation:   {config.MAX_SLA_VIOLATION_RATE:.1%}")
    print(f"Fairness Threshold:  {config.FAIRNESS_GINI_THRESHOLD}")
    print("="*60 + "\n")


def train_models(data_generator: SyntheticDataGenerator):
    """
    Train uplift and capacity models on historical data
    """
    print("\n" + "="*60)
    print("STEP 1: MODEL TRAINING")
    print("="*60)
    
    # Initialize models
    uplift_model = UpliftModel()
    capacity_model = CapacityModel()
    
    # Get historical data
    historical_data = data_generator.historical_data
    agents_df = data_generator.agents
    
    print(f"Historical data: {len(historical_data)} interactions")
    print(f"Agent pool: {len(agents_df)} agents\n")
    
    # Train
    uplift_model.train(historical_data, agents_df)
    capacity_model.train(historical_data, agents_df)
    
    print("\n✓ All models trained successfully!\n")
    
    return uplift_model, capacity_model


def run_experiments(data_generator: SyntheticDataGenerator,
                   uplift_model: UpliftModel,
                   capacity_model: CapacityModel,
                   n_batches: int = 100):
    """
    Run all policy experiments
    """
    print("\n" + "="*60)
    print("STEP 2: RUNNING EXPERIMENTS")
    print("="*60)
    
    policies = [
        ("CUCB-OTA", cucb_ota_policy),
        ("FCFS", fcfs_policy),
        ("Skill-Greedy", skill_based_greedy_policy)
    ]
    
    results = {}
    simulators = {}
    
    for policy_name, policy_fn in policies:
        print(f"\n{'='*60}")
        print(f"Testing: {policy_name}")
        print('='*60)
        
        # Create fresh simulator
        simulator = RoutingSimulator(data_generator, uplift_model, capacity_model)
        
        # Run simulation
        summary = simulator.run_simulation(n_batches, policy_fn, policy_name)
        
        results[policy_name] = summary
        simulators[policy_name] = simulator
    
    return results, simulators


def generate_visualizations(results: Dict, simulators: Dict):
    """
    Generate all comparison plots and tables
    """
    print("\n" + "="*60)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualizer = ResultsVisualizer()
    
    # 1. Policy comparison
    print("\nGenerating policy comparison plot...")
    visualizer.plot_policy_comparison(
        results, 
        save_path='data/logs/policy_comparison.png'
    )
    
    # 2. Convergence plots for CUCB-OTA
    print("Generating convergence plots...")
    cucb_metrics = simulators['CUCB-OTA'].metrics.get_dataframe()
    visualizer.plot_convergence(
        cucb_metrics,
        'CUCB-OTA',
        save_path='data/logs/cucb_convergence.png'
    )
    
    # 3. Agent workload analysis
    print("Generating workload analysis...")
    workload_df = simulators['CUCB-OTA'].metrics.get_agent_workload_dataframe()
    visualizer.plot_agent_workload(
        workload_df,
        save_path='data/logs/agent_workload.png'
    )
    
    # 4. Print summary table
    visualizer.print_summary_table(results)
    
    print("\n✓ All visualizations generated!")


def run_ope_analysis(simulators: Dict):
    """
    Run Off-Policy Evaluation comparison
    """
    print("\n" + "="*60)
    print("STEP 4: OFF-POLICY EVALUATION")
    print("="*60)
    
    # Get logged data from CUCB-OTA
    ope = simulators['CUCB-OTA'].ope
    
    if len(ope.logged_interactions) == 0:
        print("No logged data available for OPE")
        return
    
    print(f"\nAnalyzing {len(ope.logged_interactions)} logged interactions...")
    
    # Estimate policy value using importance sampling
    ope_results = ope.estimate_policy_value(None)
    
    print("\nOff-Policy Estimates:")
    print(f"  Estimated CSAT:     {ope_results.get('estimated_csat', 0):.4f}")
    print(f"  Estimated AHT:      {ope_results.get('estimated_aht', 0):.2f} min")
    print(f"  Estimated SLA Rate: {ope_results.get('estimated_sla_rate', 0):.2%}")
    print(f"  Sample Size:        {ope_results.get('n_samples', 0):,}")
    
    # Save OPE logs as DataFrame
    if len(ope.logged_interactions) > 0:
        logs_data = []
        for log in ope.logged_interactions:
            logs_data.append({
                'customer_id': log['customer'].get('customer_id', ''),
                'agent_id': log['agent'].get('agent_id', ''),
                'channel': log['customer'].get('channel', ''),
                'skill_needed': log['customer'].get('skill_needed', ''),
                'priority': log['customer'].get('priority', ''),
                'csat': log['outcome']['csat'],
                'aht': log['outcome']['aht'],
                'sla_met': log['outcome']['sla_met'],
                'assignment_prob': log['assignment_prob']
            })
        
        logs_df = pd.DataFrame(logs_data)
        logs_df.to_csv('data/logs/ope_interactions.csv', index=False)
        print("\n✓ OPE analysis complete. Logs saved to data/logs/ope_interactions.csv")


def save_final_report(results: Dict, simulators: Dict):
    """
    Save comprehensive final report
    """
    print("\n" + "="*60)
    print("STEP 5: GENERATING FINAL REPORT")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'data/logs/final_report_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" "*20 + "SMART QUEUE ROUTING SYSTEM - FINAL REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        f.write(f"Agents:              {config.NUM_AGENTS}\n")
        f.write(f"Channels:            {', '.join(config.CHANNELS)}\n")
        f.write(f"Skills:              {', '.join(config.SKILLS)}\n")
        f.write(f"Capacity Rules:      {config.CAPACITY_RULES}\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-"*80 + "\n")
        
        for policy_name, summary in results.items():
            f.write(f"\n{policy_name}:\n")
            f.write(f"  Total Assignments:   {summary['total_assignments']:,}\n")
            f.write(f"  Avg CSAT:            {summary['overall_avg_csat']:.4f}\n")
            f.write(f"  Avg AHT:             {summary['overall_avg_aht']:.2f} min\n")
            f.write(f"  SLA Met Rate:        {summary['overall_sla_met_rate']:.2%}\n")
            f.write(f"  Fairness (Gini):     {summary['avg_gini']:.3f}\n")
            
            if 'dual_variables' in summary:
                dv = summary['dual_variables']
                f.write(f"  Dual Variables:\n")
                f.write(f"    λ_aht:      {dv['lambda_aht']:.3f}\n")
                f.write(f"    λ_sla:      {dv['lambda_sla']:.3f}\n")
                f.write(f"    λ_fairness: {dv['lambda_fairness']:.3f}\n")
        
        if 'FCFS' in results:
            f.write("\n\nIMPROVEMENT OVER FCFS:\n")
            f.write("-"*80 + "\n")
            
            baseline = results['FCFS']
            for policy in results.keys():
                if policy == 'FCFS':
                    continue
                
                f.write(f"\n{policy}:\n")
                
                csat_imp = ((results[policy]['overall_avg_csat'] - 
                           baseline['overall_avg_csat']) / 
                           baseline['overall_avg_csat']) * 100
                f.write(f"  CSAT Uplift:      +{csat_imp:.2f}%\n")
                
                aht_imp = ((baseline['overall_avg_aht'] - 
                          results[policy]['overall_avg_aht']) / 
                          baseline['overall_avg_aht']) * 100
                f.write(f"  AHT Reduction:    +{aht_imp:.2f}%\n")
                
                sla_imp = ((results[policy]['overall_sla_met_rate'] - 
                          baseline['overall_sla_met_rate']) / 
                          baseline['overall_sla_met_rate']) * 100
                f.write(f"  SLA Improvement:  +{sla_imp:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Final report saved to {report_path}")
    
    # Also save metrics DataFrames
    for policy_name, simulator in simulators.items():
        metrics_df = simulator.metrics.get_dataframe()
        metrics_path = f'data/logs/{policy_name}_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ {policy_name} metrics saved to {metrics_path}")


def main():
    """
    Main execution flow
    """
    print_banner()
    print_problem_statement()
    print_config()
    
    # Generate data
    print("Initializing data generator...")
    data_generator = data_gen
    print(f"✓ Generated {len(data_generator.agents)} agents")
    print(f"✓ Generated {len(data_generator.historical_data)} historical interactions\n")
    
    # Train models
    uplift_model, capacity_model = train_models(data_generator)
    
    # Run experiments (adjust n_batches for longer runs)
    n_batches = 150  # Increase for more data
    results, simulators = run_experiments(
        data_generator, 
        uplift_model, 
        capacity_model,
        n_batches=n_batches
    )
    
    # Generate visualizations
    generate_visualizations(results, simulators)
    
    # Run OPE analysis
    run_ope_analysis(simulators)
    
    # Save final report
    save_final_report(results, simulators)
    
    # Final summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    
    if 'CUCB-OTA' in results and 'FCFS' in results:
        cucb = results['CUCB-OTA']
        fcfs = results['FCFS']
        
        csat_gain = ((cucb['overall_avg_csat'] - fcfs['overall_avg_csat']) / 
                     fcfs['overall_avg_csat']) * 100
        
        print(f"\n✓ CUCB-OTA achieved {csat_gain:+.2f}% CSAT improvement over FCFS")
        print(f"✓ Maintained SLA compliance: {cucb['overall_sla_met_rate']:.1%}")
        print(f"✓ Fairness (Gini): {cucb['avg_gini']:.3f}")
        print(f"✓ All visualizations saved to data/logs/")
    
    print("\n" + "="*60)
    print("Thank you for using SQRS!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

