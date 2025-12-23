"""
Output Generator Module
Generates picklist CSV files and summary in the required format
"""

import pandas as pd
import os
import json
from typing import Dict, List, Any
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INTERMEDIATE_DIR, OUTPUT_DIR


def generate_picklist_csv(
    picklist_data: Dict,
    output_dir: str,
    date_str: str
) -> str:
    """
    Generate individual picklist CSV file
    
    Output format:
    - Filename: {date}_{Picklist_no}.csv
    - Columns: SKU, Store, Bin, Bin Rank
    """
    picklist_id = picklist_data['picklist_id']
    items = picklist_data['items']
    
    # Extract picklist number from ID
    picklist_no = picklist_id.split('_')[-1]
    
    # Create DataFrame for output
    rows = []
    for item in items:
        rows.append({
            'SKU': item['sku'],
            'Store': item['store_id'],
            'Bin': item['bin'],
            'Bin Rank': item['bin_rank']
        })
    
    df = pd.DataFrame(rows)
    
    # Create filename
    filename = f"{date_str}_{picklist_no}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save CSV
    df.to_csv(filepath, index=False)
    
    return filepath


def generate_summary_csv(
    picklists: List[Dict],
    output_dir: str,
    date_str: str
) -> str:
    """
    Generate summary CSV file
    
    Output format:
    - Filename: Summary.csv
    - Columns: Picklist_date, picklist_no, picklist_type, stores_in_picklist
    """
    rows = []
    
    for picklist in picklists:
        picklist_id = picklist['picklist_id']
        picklist_no = picklist_id.split('_')[-1]
        
        # Get unique stores in picklist
        stores = list(set(item['store_id'] for item in picklist['items']))
        stores_str = ','.join(map(str, sorted(stores)))
        
        rows.append({
            'Picklist_date': date_str,
            'picklist_no': picklist_no,
            'picklist_type': picklist['picklist_type'],
            'stores_in_picklist': stores_str
        })
    
    df = pd.DataFrame(rows)
    
    # Save summary
    filepath = os.path.join(output_dir, 'Summary.csv')
    df.to_csv(filepath, index=False)
    
    return filepath


def generate_detailed_summary(
    results: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Generate a detailed summary with additional metrics
    """
    summary = results['summary']
    assignments = results['assignments']
    picker_util = results['picker_utilization']
    
    # Create detailed summary dataframe
    detail_rows = []
    
    for assignment in assignments:
        # Calculate time in HH:MM format from picking start (9PM)
        start_hours = int(assignment['start_time'] // 60)
        start_mins = int(assignment['start_time'] % 60)
        end_hours = int(assignment['end_time'] // 60)
        end_mins = int(assignment['end_time'] % 60)
        
        # Convert to actual time (9PM + offset)
        actual_start_hour = (21 + start_hours) % 24
        actual_end_hour = (21 + end_hours) % 24
        
        detail_rows.append({
            'picklist_id': assignment['picklist_id'],
            'zone': assignment['zone'],
            'picker_id': assignment['picker_id'],
            'start_time': f"{actual_start_hour:02d}:{start_mins:02d}",
            'end_time': f"{actual_end_hour:02d}:{end_mins:02d}",
            'duration_mins': round(assignment['end_time'] - assignment['start_time'], 2),
            'items_count': assignment['items_count'],
            'units_count': assignment['units_count'],
            'weight_kg': round(assignment['weight_kg'], 2),
            'num_orders': len(assignment['orders']),
            'num_stores': len(assignment['stores']),
            'picklist_type': assignment['picklist_type'],
            'is_fragile': assignment['is_fragile'],
            'cutoff_time_mins': assignment['cutoff_time'],
            'on_time': assignment['on_time']
        })
    
    df = pd.DataFrame(detail_rows)
    
    # Save detailed summary
    filepath = os.path.join(output_dir, 'Detailed_Summary.csv')
    df.to_csv(filepath, index=False)
    
    return filepath


def generate_picker_summary(
    picker_utilization: List[Dict],
    output_dir: str
) -> str:
    """
    Generate picker utilization summary
    """
    df = pd.DataFrame(picker_utilization)
    
    # Calculate shift-level aggregates
    shift_summary = df.groupby('shift').agg({
        'picker_id': 'count',
        'picklists_assigned': ['sum', 'mean'],
        'utilization': 'mean'
    }).round(3)
    shift_summary.columns = ['num_pickers', 'total_picklists', 'avg_picklists_per_picker', 'avg_utilization']
    
    # Save picker detail
    picker_filepath = os.path.join(output_dir, 'Picker_Detail.csv')
    df.to_csv(picker_filepath, index=False)
    
    # Save shift summary
    shift_filepath = os.path.join(output_dir, 'Shift_Summary.csv')
    shift_summary.to_csv(shift_filepath)
    
    return picker_filepath


def generate_zone_summary(
    picklists: List[Dict],
    output_dir: str
) -> str:
    """
    Generate zone-level summary
    """
    zone_data = {}
    
    for picklist in picklists:
        zone = picklist['zone']
        if zone not in zone_data:
            zone_data[zone] = {
                'num_picklists': 0,
                'total_units': 0,
                'total_weight_kg': 0,
                'unique_orders': set(),
                'unique_stores': set(),
                'fragile_picklists': 0
            }
        
        zone_data[zone]['num_picklists'] += 1
        zone_data[zone]['total_units'] += picklist['total_units']
        zone_data[zone]['total_weight_kg'] += picklist['total_weight_kg']
        zone_data[zone]['unique_orders'].update(item['order_id'] for item in picklist['items'])
        zone_data[zone]['unique_stores'].update(item['store_id'] for item in picklist['items'])
        if picklist['is_fragile']:
            zone_data[zone]['fragile_picklists'] += 1
    
    rows = []
    for zone, data in zone_data.items():
        rows.append({
            'zone': zone,
            'num_picklists': data['num_picklists'],
            'total_units': data['total_units'],
            'total_weight_kg': round(data['total_weight_kg'], 2),
            'num_unique_orders': len(data['unique_orders']),
            'num_unique_stores': len(data['unique_stores']),
            'fragile_picklists': data['fragile_picklists']
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('total_units', ascending=False)
    
    filepath = os.path.join(output_dir, 'Zone_Summary.csv')
    df.to_csv(filepath, index=False)
    
    return filepath


def generate_all_outputs(results_path: str = None, output_dir: str = None) -> Dict[str, str]:
    """
    Generate all output files from optimization results
    
    Args:
        results_path: Path to optimization results JSON
        output_dir: Directory for output files
        
    Returns:
        Dictionary with paths to generated files
    """
    if results_path is None:
        results_path = os.path.join(INTERMEDIATE_DIR, "optimization_results.json")
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading optimization results from {results_path}...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    date_str = results['summary']['processing_date']
    picklists = results['picklists']
    
    generated_files = {}
    
    # Create date-specific subdirectory for picklists
    picklist_dir = os.path.join(output_dir, date_str.replace('-', '_'))
    os.makedirs(picklist_dir, exist_ok=True)
    
    # Generate individual picklist CSVs
    print("Generating picklist CSV files...")
    for picklist in tqdm(picklists, desc="Creating picklists"):
        generate_picklist_csv(picklist, picklist_dir, date_str.replace('-', '_'))
    
    generated_files['picklist_dir'] = picklist_dir
    
    # Generate summary CSV
    print("Generating summary files...")
    summary_path = generate_summary_csv(picklists, output_dir, date_str.replace('-', '_'))
    generated_files['summary'] = summary_path
    
    # Generate detailed summary
    detail_path = generate_detailed_summary(results, output_dir)
    generated_files['detailed_summary'] = detail_path
    
    # Generate picker summary
    picker_path = generate_picker_summary(results['picker_utilization'], output_dir)
    generated_files['picker_summary'] = picker_path
    
    # Generate zone summary
    zone_path = generate_zone_summary(picklists, output_dir)
    generated_files['zone_summary'] = zone_path
    
    print(f"\nGenerated {len(picklists)} picklist files in {picklist_dir}")
    print(f"Generated summary files in {output_dir}")
    
    return generated_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate picklist output files")
    parser.add_argument("--input", "-i", type=str,
                       default=os.path.join(INTERMEDIATE_DIR, "optimization_results.json"),
                       help="Path to optimization results JSON")
    parser.add_argument("--output", "-o", type=str,
                       default=OUTPUT_DIR,
                       help="Output directory for CSV files")
    
    args = parser.parse_args()
    
    generate_all_outputs(args.input, args.output)
