"""
Picklist Optimization Engine
Main optimization logic for creating and scheduling picklists
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import heapq

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MAX_ITEMS_PER_PICKLIST, MAX_WEIGHT_PER_PICKLIST_KG, MAX_WEIGHT_FRAGILE_KG,
    PRIORITY_ORDER, PRIORITY_CUTOFFS, PICKER_SHIFTS, PICKING_START_TIME,
    INTERMEDIATE_DIR
)

from .time_calculator import (
    PicklistItem, Picklist, calculate_picklist_duration,
    calculate_picklist_duration_estimate, optimize_bin_order
)


@dataclass
class Picker:
    """Represents a warehouse picker"""
    picker_id: int
    shift_name: str
    shift_start: float  # Minutes from midnight
    shift_end: float
    current_time: float = 0  # Current available time
    assigned_picklists: List[str] = field(default_factory=list)
    
    @property
    def available_time(self) -> float:
        """Remaining time in shift"""
        return max(0, self.shift_end - self.current_time)
    
    def can_complete_before(self, duration: float, cutoff: float) -> bool:
        """Check if picker can complete a task before cutoff"""
        return self.current_time + duration <= cutoff


@dataclass 
class PicklistAssignment:
    """Tracks a picklist assignment"""
    picklist_id: str
    picker_id: int
    zone: str
    start_time: float
    end_time: float
    cutoff_time: float
    items_count: int
    units_count: int
    weight_kg: float
    orders: List[str]
    stores: List[int]
    is_fragile: bool
    on_time: bool
    picklist_type: str


class PicklistOptimizer:
    """
    Main optimization engine for warehouse picklist scheduling
    
    Strategy:
    1. Group items by zone (zone constraint)
    2. Sort by priority (P1 > P2 > ... > P9)
    3. Within each priority, create picklists respecting capacity limits
    4. Assign picklists to available pickers based on shift windows
    5. Track completion times vs cutoffs
    """
    
    def __init__(self, items_data: List[Dict], processing_date: str):
        """
        Initialize optimizer with item data
        
        Args:
            items_data: List of item dictionaries from preprocessed data
            processing_date: Date string for the processing day
        """
        self.items_df = pd.DataFrame(items_data)
        self.processing_date = processing_date
        
        # Initialize pickers
        self.pickers = self._initialize_pickers()
        
        # Track assignments
        self.assignments: List[PicklistAssignment] = []
        self.picklists: List[Picklist] = []
        self.picklist_counter = 0
        
        # Track metrics
        self.total_units_picked = 0
        self.total_units_on_time = 0
        self.total_units_late = 0
        
    def _initialize_pickers(self) -> List[Picker]:
        """Initialize picker pool based on shift configuration"""
        pickers = []
        picker_id = 0
        
        # Convert picking start time to minutes from midnight (9PM = 21:00 = 1260 min)
        picking_start_minutes = PICKING_START_TIME.hour * 60 + PICKING_START_TIME.minute
        
        for shift_name, shift_config in PICKER_SHIFTS.items():
            start_time = shift_config['start']
            end_time = shift_config['end']
            num_pickers = shift_config['num_pickers']
            
            # Convert to minutes from midnight
            start_minutes = start_time.hour * 60 + start_time.minute
            end_minutes = end_time.hour * 60 + end_time.minute
            
            # Handle overnight shifts (end time wraps to next day)
            if end_minutes < start_minutes:
                end_minutes += 24 * 60
            
            # Adjust to picking start time reference (9PM = 0)
            # For shifts starting before 9PM on the same operational day
            start_relative = start_minutes - picking_start_minutes
            end_relative = end_minutes - picking_start_minutes
            
            # If shift starts before picking time (e.g., 8PM shift when picking is 9PM)
            # The picker can start at time 0 (when picking begins)
            # If shift ends before picking starts, skip this shift for this day
            if end_relative <= 0:
                # Shift ends before picking starts - not available
                continue
            
            # Shift start is before picking start - can begin at picking start (time 0)
            if start_relative < 0:
                start_relative = 0
            
            for _ in range(num_pickers):
                picker = Picker(
                    picker_id=picker_id,
                    shift_name=shift_name,
                    shift_start=start_relative,
                    shift_end=end_relative,
                    current_time=max(0, start_relative)  # Can't start before picking time
                )
                pickers.append(picker)
                picker_id += 1
        
        print(f"Initialized {len(pickers)} pickers across {len(PICKER_SHIFTS)} shifts")
        return pickers
    
    def _get_available_picker(self, duration: float, cutoff: float) -> Optional[Picker]:
        """
        Find an available picker who can complete the task before cutoff
        Uses earliest available time heuristic
        """
        available_pickers = [
            p for p in self.pickers 
            if p.current_time <= cutoff - duration and 
               p.current_time < p.shift_end and
               p.can_complete_before(duration, cutoff)
        ]
        
        if not available_pickers:
            return None
        
        # Return picker with earliest available time
        return min(available_pickers, key=lambda p: p.current_time)
    
    def _get_any_available_picker(self, duration: float) -> Optional[Picker]:
        """Find any available picker (when cutoff can't be met)"""
        available_pickers = [
            p for p in self.pickers 
            if p.current_time + duration <= p.shift_end
        ]
        
        if not available_pickers:
            return None
            
        return min(available_pickers, key=lambda p: p.current_time)
    
    def _create_picklist_id(self) -> str:
        """Generate unique picklist ID"""
        self.picklist_counter += 1
        return f"PL_{self.processing_date}_{self.picklist_counter:05d}"
    
    def _group_items_by_zone_priority(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Group items by zone and priority for processing
        Returns nested dict: zone -> priority -> items DataFrame
        """
        grouped = {}
        
        for zone in self.items_df['zone'].unique():
            zone_items = self.items_df[self.items_df['zone'] == zone]
            grouped[zone] = {}
            
            for priority in PRIORITY_ORDER:
                priority_items = zone_items[zone_items['priority'] == priority]
                if len(priority_items) > 0:
                    grouped[zone][priority] = priority_items
        
        return grouped
    
    def _create_picklists_for_zone_priority(
        self, 
        items_df: pd.DataFrame, 
        zone: str, 
        priority: str,
        is_fragile_zone: bool = False
    ) -> List[Picklist]:
        """
        Create picklists for a specific zone-priority combination
        Respects capacity constraints (items, weight)
        """
        picklists = []
        
        # Get max weight based on fragile status
        max_weight = MAX_WEIGHT_FRAGILE_KG if is_fragile_zone else MAX_WEIGHT_PER_PICKLIST_KG
        
        # Sort items by store_id and bin_rank for efficient picking
        items_df = items_df.sort_values(['store_id', 'bin_rank'])
        
        current_items = []
        current_units = 0
        current_weight = 0.0
        current_orders = set()
        current_stores = set()
        
        # Get pods_per_picklist constraint for this zone
        pods_per_picklist = items_df['pods_per_picklist'].iloc[0] if len(items_df) > 0 else 100
        
        for _, row in items_df.iterrows():
            item_units = row['quantity']
            item_weight = row['weight_kg'] * item_units
            item_fragile = row['is_fragile'] or is_fragile_zone
            
            # Check if adding this item would exceed limits
            would_exceed_items = current_units + item_units > MAX_ITEMS_PER_PICKLIST
            would_exceed_weight = current_weight + item_weight > max_weight
            would_exceed_pods = len(current_stores | {row['store_id']}) > pods_per_picklist
            
            if would_exceed_items or would_exceed_weight or would_exceed_pods:
                # Finalize current picklist if not empty
                if current_items:
                    picklist = Picklist(
                        picklist_id=self._create_picklist_id(),
                        zone=zone,
                        items=current_items
                    )
                    picklists.append(picklist)
                
                # Start new picklist
                current_items = []
                current_units = 0
                current_weight = 0.0
                current_orders = set()
                current_stores = set()
            
            # Add item to current picklist
            picklist_item = PicklistItem(
                order_id=row['order_id'],
                store_id=row['store_id'],
                sku=row['sku'],
                zone=zone,
                bin_location=row['bin'],
                bin_rank=row['bin_rank'],
                quantity=row['quantity'],
                weight_kg=row['weight_kg'],
                is_fragile=item_fragile
            )
            current_items.append(picklist_item)
            current_units += item_units
            current_weight += item_weight
            current_orders.add(row['order_id'])
            current_stores.add(row['store_id'])
        
        # Don't forget the last picklist
        if current_items:
            picklist = Picklist(
                picklist_id=self._create_picklist_id(),
                zone=zone,
                items=optimize_bin_order(current_items)
            )
            picklists.append(picklist)
        
        return picklists
    
    def _assign_picklist_to_picker(
        self, 
        picklist: Picklist, 
        cutoff_minutes: float
    ) -> Optional[PicklistAssignment]:
        """
        Assign a picklist to an available picker
        """
        duration = calculate_picklist_duration(picklist)
        
        # Try to find a picker who can complete before cutoff
        picker = self._get_available_picker(duration, cutoff_minutes)
        on_time = True
        
        if picker is None:
            # Try to find any available picker (late completion)
            picker = self._get_any_available_picker(duration)
            on_time = False
        
        if picker is None:
            # No picker available at all
            return None
        
        # Create assignment
        start_time = picker.current_time
        end_time = start_time + duration
        
        assignment = PicklistAssignment(
            picklist_id=picklist.picklist_id,
            picker_id=picker.picker_id,
            zone=picklist.zone,
            start_time=start_time,
            end_time=end_time,
            cutoff_time=cutoff_minutes,
            items_count=len(picklist.items),
            units_count=picklist.total_units,
            weight_kg=picklist.total_weight_kg,
            orders=list(set(item.order_id for item in picklist.items)),
            stores=list(set(item.store_id for item in picklist.items)),
            is_fragile=picklist.is_fragile,
            on_time=end_time <= cutoff_minutes,
            picklist_type=picklist.picklist_type
        )
        
        # Update picker state
        picker.current_time = end_time
        picker.assigned_picklists.append(picklist.picklist_id)
        
        # Update picklist
        picklist.picker_id = picker.picker_id
        picklist.start_time = start_time
        picklist.end_time = end_time
        
        return assignment
    
    def optimize(self) -> Dict[str, Any]:
        """
        Main optimization routine
        
        Returns:
            Dictionary with optimization results
        """
        print("Starting optimization...")
        
        # Group items
        zone_priority_groups = self._group_items_by_zone_priority()
        
        # Process by priority (highest first)
        for priority in tqdm(PRIORITY_ORDER, desc="Processing priorities"):
            cutoff_minutes = PRIORITY_CUTOFFS[priority].total_seconds() / 60
            
            # Process each zone for this priority
            for zone, priority_items in zone_priority_groups.items():
                if priority not in priority_items:
                    continue
                
                items_df = priority_items[priority]
                
                # Check if zone is fragile
                is_fragile_zone = items_df['is_fragile'].any()
                
                # Create picklists
                zone_picklists = self._create_picklists_for_zone_priority(
                    items_df, zone, priority, is_fragile_zone
                )
                
                # Assign picklists to pickers
                for picklist in zone_picklists:
                    assignment = self._assign_picklist_to_picker(picklist, cutoff_minutes)
                    
                    if assignment:
                        self.assignments.append(assignment)
                        self.picklists.append(picklist)
                        
                        # Update metrics
                        self.total_units_picked += assignment.units_count
                        if assignment.on_time:
                            self.total_units_on_time += assignment.units_count
                        else:
                            self.total_units_late += assignment.units_count
        
        # Generate results
        results = self._generate_results()
        print(f"Optimization complete: {len(self.picklists)} picklists created")
        
        return results
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate optimization results summary"""
        
        # Calculate picker utilization
        picker_utilization = []
        for picker in self.pickers:
            shift_duration = picker.shift_end - picker.shift_start
            work_time = picker.current_time - picker.shift_start
            utilization = work_time / shift_duration if shift_duration > 0 else 0
            picker_utilization.append({
                'picker_id': picker.picker_id,
                'shift': picker.shift_name,
                'picklists_assigned': len(picker.assigned_picklists),
                'utilization': min(1.0, max(0, utilization))
            })
        
        # Calculate order completion
        order_items = defaultdict(lambda: {'total': 0, 'picked': 0, 'on_time': 0})
        for assignment in self.assignments:
            for order_id in assignment.orders:
                order_items[order_id]['picked'] += 1
                if assignment.on_time:
                    order_items[order_id]['on_time'] += 1
        
        # Count from original data
        for _, row in self.items_df.iterrows():
            order_items[row['order_id']]['total'] += 1
        
        completed_orders = sum(
            1 for oid, data in order_items.items() 
            if data['picked'] == data['total'] and data['on_time'] == data['total']
        )
        
        results = {
            'summary': {
                'total_picklists': len(self.picklists),
                'total_units_picked': self.total_units_picked,
                'total_units_on_time': self.total_units_on_time,
                'total_units_late': self.total_units_late,
                'on_time_rate': self.total_units_on_time / max(1, self.total_units_picked),
                'completed_orders': completed_orders,
                'total_orders': len(order_items),
                'order_completion_rate': completed_orders / max(1, len(order_items)),
                'avg_picker_utilization': np.mean([p['utilization'] for p in picker_utilization]),
                'processing_date': self.processing_date
            },
            'assignments': [asdict(a) if hasattr(a, '__dataclass_fields__') else a.__dict__ 
                          for a in self.assignments],
            'picker_utilization': picker_utilization,
            'picklists': [
                {
                    'picklist_id': p.picklist_id,
                    'zone': p.zone,
                    'total_units': p.total_units,
                    'total_weight_kg': p.total_weight_kg,
                    'unique_bins': p.unique_bins,
                    'unique_orders': p.unique_orders,
                    'unique_stores': p.unique_stores,
                    'is_fragile': p.is_fragile,
                    'picklist_type': p.picklist_type,
                    'picker_id': p.picker_id,
                    'start_time': p.start_time,
                    'end_time': p.end_time,
                    'items': [
                        {
                            'order_id': item.order_id,
                            'store_id': item.store_id,
                            'sku': item.sku,
                            'bin': item.bin_location,
                            'bin_rank': item.bin_rank,
                            'quantity': item.quantity,
                            'weight_kg': item.weight_kg,
                            'is_fragile': item.is_fragile
                        }
                        for item in p.items
                    ]
                }
                for p in self.picklists
            ]
        }
        
        return results


def run_optimization(processed_data_path: str = None) -> Dict[str, Any]:
    """
    Run the optimization pipeline
    
    Args:
        processed_data_path: Path to preprocessed JSON data
        
    Returns:
        Optimization results dictionary
    """
    if processed_data_path is None:
        processed_data_path = os.path.join(INTERMEDIATE_DIR, "processed_orders.json")
    
    # Load processed data
    print(f"Loading processed data from {processed_data_path}...")
    with open(processed_data_path, 'r') as f:
        data = json.load(f)
    
    # Create optimizer
    optimizer = PicklistOptimizer(
        items_data=data['items'],
        processing_date=data['processing_date']
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save results
    output_path = os.path.join(INTERMEDIATE_DIR, "optimization_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved optimization results to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    summary = results['summary']
    print(f"Total Picklists: {summary['total_picklists']:,}")
    print(f"Total Units Picked: {summary['total_units_picked']:,}")
    print(f"Units On-Time: {summary['total_units_on_time']:,} ({summary['on_time_rate']*100:.1f}%)")
    print(f"Units Late: {summary['total_units_late']:,}")
    print(f"Completed Orders: {summary['completed_orders']:,} / {summary['total_orders']:,}")
    print(f"Order Completion Rate: {summary['order_completion_rate']*100:.1f}%")
    print(f"Avg Picker Utilization: {summary['avg_picker_utilization']*100:.1f}%")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run picklist optimization")
    parser.add_argument("--input", "-i", type=str, 
                       default=os.path.join(INTERMEDIATE_DIR, "processed_orders.json"),
                       help="Path to processed data JSON")
    
    args = parser.parse_args()
    
    run_optimization(args.input)
