"""
Evaluation Metrics Module
Calculates and reports all evaluation metrics for the optimization
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INTERMEDIATE_DIR, OUTPUT_DIR, ANALYSIS_DIR


def calculate_primary_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate primary evaluation metrics
    
    Primary metric: Total units successfully picked before cutoff
    """
    summary = results['summary']
    assignments = results['assignments']
    
    # Primary: Units picked on time
    units_on_time = summary['total_units_on_time']
    total_units = summary['total_units_picked']
    
    # Late units
    units_late = summary['total_units_late']
    
    # Effectiveness rate
    effectiveness_rate = units_on_time / total_units if total_units > 0 else 0
    
    return {
        'total_units_picked_on_time': units_on_time,
        'total_units_picked': total_units,
        'units_late': units_late,
        'effectiveness_rate': effectiveness_rate,
        'effectiveness_percentage': f"{effectiveness_rate * 100:.2f}%"
    }


def calculate_order_completion_metrics(
    results: Dict[str, Any],
    original_data_path: str = None
) -> Dict[str, Any]:
    """
    Calculate secondary metric: Number of completed orders
    """
    assignments = results['assignments']
    picklists = results['picklists']
    
    # Track order fulfillment
    order_fulfillment = defaultdict(lambda: {
        'total_items': 0,
        'picked_items': 0,
        'picked_on_time': 0,
        'picked_late': 0,
        'total_units': 0,
        'picked_units': 0,
        'units_on_time': 0
    })
    
    # Aggregate from picklists
    for i, assignment in enumerate(assignments):
        picklist = picklists[i]
        on_time = assignment['on_time']
        
        for item in picklist['items']:
            order_id = item['order_id']
            order_fulfillment[order_id]['picked_items'] += 1
            order_fulfillment[order_id]['picked_units'] += item['quantity']
            
            if on_time:
                order_fulfillment[order_id]['picked_on_time'] += 1
                order_fulfillment[order_id]['units_on_time'] += item['quantity']
            else:
                order_fulfillment[order_id]['picked_late'] += 1
    
    # Calculate completed orders
    # An order is "complete" if all its items were picked on time
    complete_orders = 0
    partial_orders = 0
    incomplete_orders = 0
    
    for order_id, data in order_fulfillment.items():
        if data['picked_on_time'] > 0 and data['picked_late'] == 0:
            complete_orders += 1
        elif data['picked_on_time'] > 0:
            partial_orders += 1
        else:
            incomplete_orders += 1
    
    total_orders = len(order_fulfillment)
    
    return {
        'complete_orders': complete_orders,
        'partial_orders': partial_orders,
        'incomplete_orders': incomplete_orders,
        'total_orders_processed': total_orders,
        'order_completion_rate': complete_orders / total_orders if total_orders > 0 else 0,
        'order_completion_percentage': f"{complete_orders / total_orders * 100:.2f}%" if total_orders > 0 else "0%"
    }


def calculate_wasted_effort_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate wasted picking effort (late picklists)
    """
    assignments = results['assignments']
    
    late_picklists = [a for a in assignments if not a['on_time']]
    
    wasted_units = sum(a['units_count'] for a in late_picklists)
    wasted_weight = sum(a['weight_kg'] for a in late_picklists)
    wasted_picklists = len(late_picklists)
    
    total_units = sum(a['units_count'] for a in assignments)
    total_picklists = len(assignments)
    
    return {
        'wasted_picklists': wasted_picklists,
        'wasted_units': wasted_units,
        'wasted_weight_kg': round(wasted_weight, 2),
        'waste_rate_picklists': wasted_picklists / total_picklists if total_picklists > 0 else 0,
        'waste_rate_units': wasted_units / total_units if total_units > 0 else 0,
        'waste_percentage': f"{wasted_units / total_units * 100:.2f}%" if total_units > 0 else "0%"
    }


def calculate_picker_utilization_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate picker utilization metrics
    """
    picker_util = results['picker_utilization']
    
    # Overall metrics
    active_pickers = sum(1 for p in picker_util if p['picklists_assigned'] > 0)
    total_pickers = len(picker_util)
    
    utilizations = [p['utilization'] for p in picker_util if p['picklists_assigned'] > 0]
    
    avg_utilization = np.mean(utilizations) if utilizations else 0
    min_utilization = np.min(utilizations) if utilizations else 0
    max_utilization = np.max(utilizations) if utilizations else 0
    std_utilization = np.std(utilizations) if utilizations else 0
    
    # Shift-level breakdown
    shift_util = defaultdict(list)
    for p in picker_util:
        if p['picklists_assigned'] > 0:
            shift_util[p['shift']].append(p['utilization'])
    
    shift_summary = {}
    for shift, utils in shift_util.items():
        shift_summary[shift] = {
            'avg_utilization': np.mean(utils),
            'active_pickers': len(utils),
            'total_picklists': sum(p['picklists_assigned'] for p in picker_util if p['shift'] == shift)
        }
    
    return {
        'active_pickers': active_pickers,
        'total_pickers': total_pickers,
        'picker_activation_rate': active_pickers / total_pickers if total_pickers > 0 else 0,
        'avg_utilization': round(avg_utilization, 4),
        'min_utilization': round(min_utilization, 4),
        'max_utilization': round(max_utilization, 4),
        'std_utilization': round(std_utilization, 4),
        'utilization_percentage': f"{avg_utilization * 100:.2f}%",
        'shift_breakdown': shift_summary
    }


def calculate_scalability_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate scalability and runtime metrics
    """
    summary = results['summary']
    assignments = results['assignments']
    
    # Calculate average processing metrics
    total_picklists = len(assignments)
    total_duration = sum(a['end_time'] - a['start_time'] for a in assignments)
    
    avg_picklist_duration = total_duration / total_picklists if total_picklists > 0 else 0
    
    # Estimate throughput
    max_time = max(a['end_time'] for a in assignments) if assignments else 0
    throughput_units_per_hour = (summary['total_units_picked'] / max_time * 60) if max_time > 0 else 0
    throughput_picklists_per_hour = (total_picklists / max_time * 60) if max_time > 0 else 0
    
    return {
        'total_picklists_created': total_picklists,
        'total_processing_time_minutes': round(max_time, 2),
        'avg_picklist_duration_minutes': round(avg_picklist_duration, 2),
        'throughput_units_per_hour': round(throughput_units_per_hour, 0),
        'throughput_picklists_per_hour': round(throughput_picklists_per_hour, 2)
    }


def calculate_priority_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate metrics broken down by priority
    """
    # This would need priority info in assignments
    # For now, calculate zone-level metrics
    assignments = results['assignments']
    
    zone_metrics = defaultdict(lambda: {
        'picklists': 0,
        'units': 0,
        'on_time_units': 0,
        'late_units': 0
    })
    
    for assignment in assignments:
        zone = assignment['zone']
        zone_metrics[zone]['picklists'] += 1
        zone_metrics[zone]['units'] += assignment['units_count']
        if assignment['on_time']:
            zone_metrics[zone]['on_time_units'] += assignment['units_count']
        else:
            zone_metrics[zone]['late_units'] += assignment['units_count']
    
    # Calculate effectiveness per zone
    zone_effectiveness = {}
    for zone, data in zone_metrics.items():
        effectiveness = data['on_time_units'] / data['units'] if data['units'] > 0 else 0
        zone_effectiveness[zone] = {
            'picklists': data['picklists'],
            'total_units': data['units'],
            'on_time_units': data['on_time_units'],
            'effectiveness': round(effectiveness, 4)
        }
    
    return {
        'zone_metrics': zone_effectiveness,
        'num_zones': len(zone_metrics)
    }


def generate_evaluation_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report
    """
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'processing_date': results['summary']['processing_date']
    }
    
    # Calculate all metrics
    report['primary_metrics'] = calculate_primary_metrics(results)
    report['order_completion'] = calculate_order_completion_metrics(results)
    report['wasted_effort'] = calculate_wasted_effort_metrics(results)
    report['picker_utilization'] = calculate_picker_utilization_metrics(results)
    report['scalability'] = calculate_scalability_metrics(results)
    report['zone_breakdown'] = calculate_priority_metrics(results)
    
    return report


def print_evaluation_report(report: Dict[str, Any]):
    """
    Print formatted evaluation report
    """
    print("\n" + "="*70)
    print("WAREHOUSE PICKLIST OPTIMIZATION - EVALUATION REPORT")
    print("="*70)
    print(f"Processing Date: {report['processing_date']}")
    print(f"Report Generated: {report['timestamp']}")
    print("="*70)
    
    # Primary Metrics
    print("\n📊 PRIMARY METRIC: Units Picked Before Cutoff")
    print("-"*50)
    pm = report['primary_metrics']
    print(f"  ✅ Units Picked On-Time: {pm['total_units_picked_on_time']:,}")
    print(f"  📦 Total Units Picked: {pm['total_units_picked']:,}")
    print(f"  ⚡ Effectiveness Rate: {pm['effectiveness_percentage']}")
    
    # Order Completion
    print("\n📋 SECONDARY METRIC: Order Completion")
    print("-"*50)
    oc = report['order_completion']
    print(f"  ✅ Complete Orders: {oc['complete_orders']:,}")
    print(f"  ⚠️  Partial Orders: {oc['partial_orders']:,}")
    print(f"  ❌ Incomplete Orders: {oc['incomplete_orders']:,}")
    print(f"  📊 Completion Rate: {oc['order_completion_percentage']}")
    
    # Wasted Effort
    print("\n🗑️  WASTED EFFORT (Late Picklists)")
    print("-"*50)
    we = report['wasted_effort']
    print(f"  ⏰ Late Picklists: {we['wasted_picklists']:,}")
    print(f"  📦 Wasted Units: {we['wasted_units']:,}")
    print(f"  ⚖️  Wasted Weight: {we['wasted_weight_kg']:.2f} kg")
    print(f"  📉 Waste Rate: {we['waste_percentage']}")
    
    # Picker Utilization
    print("\n👷 PICKER UTILIZATION")
    print("-"*50)
    pu = report['picker_utilization']
    print(f"  👥 Active Pickers: {pu['active_pickers']} / {pu['total_pickers']}")
    print(f"  📊 Average Utilization: {pu['utilization_percentage']}")
    print(f"  📈 Max Utilization: {pu['max_utilization']*100:.1f}%")
    print(f"  📉 Min Utilization: {pu['min_utilization']*100:.1f}%")
    
    # Scalability
    print("\n⚡ SCALABILITY & THROUGHPUT")
    print("-"*50)
    sc = report['scalability']
    print(f"  📋 Total Picklists: {sc['total_picklists_created']:,}")
    print(f"  ⏱️  Total Time: {sc['total_processing_time_minutes']:.0f} minutes")
    print(f"  🚀 Throughput: {sc['throughput_units_per_hour']:.0f} units/hour")
    print(f"  📊 Avg Picklist Duration: {sc['avg_picklist_duration_minutes']:.1f} min")
    
    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70 + "\n")


def save_evaluation_report(
    report: Dict[str, Any],
    output_path: str = None
) -> str:
    """
    Save evaluation report to JSON file
    """
    if output_path is None:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        output_path = os.path.join(ANALYSIS_DIR, "evaluation_report.json")
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Saved evaluation report to {output_path}")
    return output_path


def run_evaluation(results_path: str = None) -> Dict[str, Any]:
    """
    Run full evaluation on optimization results
    """
    if results_path is None:
        results_path = os.path.join(INTERMEDIATE_DIR, "optimization_results.json")
    
    # Load results
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Generate report
    report = generate_evaluation_report(results)
    
    # Print report
    print_evaluation_report(report)
    
    # Save report
    save_evaluation_report(report)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate optimization results")
    parser.add_argument("--input", "-i", type=str,
                       default=os.path.join(INTERMEDIATE_DIR, "optimization_results.json"),
                       help="Path to optimization results JSON")
    
    args = parser.parse_args()
    
    run_evaluation(args.input)
