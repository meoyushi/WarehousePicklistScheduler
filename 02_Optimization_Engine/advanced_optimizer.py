"""
Advanced Picklist Optimization Engine - Hackathon Winning Strategy
Optimized for:
1. Maximum units picked before cutoff (PRIMARY)
2. Maximum completed orders (SECONDARY)
3. Zero wasted effort (no late picklists)
4. Maximum picker utilization
5. Fast runtime
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
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
    INTERMEDIATE_DIR, START_TO_ZONE_TIME, ZONE_TO_STAGING_TIME,
    INTRA_ZONE_BIN_TRAVEL_TIME, SKU_PICKUP_TIME_PER_UNIT, UNLOADING_TIME_PER_ORDER
)


@dataclass
class Picker:
    """Represents a warehouse picker"""
    picker_id: int
    shift_name: str
    shift_start: float
    shift_end: float
    current_time: float = 0
    assigned_picklists: List[str] = field(default_factory=list)
    total_units_picked: int = 0
    
    @property
    def available_time(self) -> float:
        return max(0, self.shift_end - self.current_time)
    
    def can_complete_by(self, duration: float, deadline: float) -> bool:
        return self.current_time + duration <= deadline


@dataclass
class PicklistCandidate:
    """A candidate picklist with estimated metrics"""
    zone: str
    priority: str
    cutoff_minutes: float
    items: List[Dict]
    total_units: int
    total_weight: float
    num_orders: int
    num_stores: int
    num_bins: int
    is_fragile: bool
    estimated_duration: float
    value_score: float  # Higher = better candidate
    
    def __lt__(self, other):
        # For heap: higher value_score = higher priority
        return self.value_score > other.value_score


class AdvancedPicklistOptimizer:
    """
    Advanced optimization engine using greedy heuristics optimized for hackathon metrics.
    
    Strategy:
    1. ONLY create picklists that can complete before their cutoff (zero waste)
    2. Prioritize by: earliest cutoff first, then by units/time ratio
    3. Group items to maximize complete order fulfillment
    4. Use bin-rank ordering to minimize travel time
    5. Pack picklists efficiently to maximize throughput
    """
    
    def __init__(self, items_data: List[Dict], processing_date: str):
        self.items_df = pd.DataFrame(items_data)
        self.processing_date = processing_date
        self.pickers = self._initialize_pickers()
        
        # Results tracking
        self.assignments = []
        self.picklists = []
        self.picklist_counter = 0
        
        # Metrics
        self.total_units_picked = 0
        self.total_units_on_time = 0
        self.unassigned_items = []
        
    def _initialize_pickers(self) -> List[Picker]:
        """Initialize pickers available during picking hours (9PM onwards)"""
        pickers = []
        picker_id = 0
        picking_start_minutes = PICKING_START_TIME.hour * 60 + PICKING_START_TIME.minute
        
        for shift_name, shift_config in PICKER_SHIFTS.items():
            start_time = shift_config['start']
            end_time = shift_config['end']
            num_pickers = shift_config['num_pickers']
            
            start_minutes = start_time.hour * 60 + start_time.minute
            end_minutes = end_time.hour * 60 + end_time.minute
            
            if end_minutes < start_minutes:
                end_minutes += 24 * 60
                
            start_relative = start_minutes - picking_start_minutes
            end_relative = end_minutes - picking_start_minutes
            
            # Skip shifts that end before picking starts
            if end_relative <= 0:
                continue
            
            # Adjust start to picking time if shift starts earlier
            if start_relative < 0:
                start_relative = 0
            
            for _ in range(num_pickers):
                pickers.append(Picker(
                    picker_id=picker_id,
                    shift_name=shift_name,
                    shift_start=start_relative,
                    shift_end=end_relative,
                    current_time=start_relative
                ))
                picker_id += 1
        
        # Sort pickers by shift end time (use those ending soonest first for early priorities)
        pickers.sort(key=lambda p: (p.shift_start, p.shift_end))
        print(f"Initialized {len(pickers)} pickers for picking window")
        return pickers
    
    def _estimate_picklist_duration(self, num_units: int, num_bins: int, num_orders: int) -> float:
        """
        Estimate picklist completion time in minutes
        
        Components:
        - Start to zone: 2 min
        - Bin travel: 0.5 min per bin transition
        - Picking: 5 sec (0.083 min) per unit
        - Zone to staging: 2 min
        - Unloading: 0.5 min per order
        """
        travel_to_zone = START_TO_ZONE_TIME
        bin_travel = max(0, num_bins - 1) * INTRA_ZONE_BIN_TRAVEL_TIME
        picking_time = num_units * SKU_PICKUP_TIME_PER_UNIT
        return_travel = ZONE_TO_STAGING_TIME
        unloading = num_orders * UNLOADING_TIME_PER_ORDER
        
        return travel_to_zone + bin_travel + picking_time + return_travel + unloading
    
    def _get_best_available_picker(self, duration: float, deadline: float) -> Optional[Picker]:
        """
        Find the best picker who can complete the task before deadline.
        Prioritize pickers whose shift ends soonest to maximize overall utilization.
        """
        candidates = []
        for p in self.pickers:
            if p.current_time + duration <= min(deadline, p.shift_end):
                # Score: prefer pickers with less remaining time (use them efficiently)
                slack = p.shift_end - (p.current_time + duration)
                candidates.append((slack, p.current_time, p))
        
        if not candidates:
            return None
        
        # Sort by slack time (ascending) then by current time (ascending)
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]
    
    def _create_optimized_picklists_for_zone_priority(
        self,
        items_df: pd.DataFrame,
        zone: str,
        priority: str,
        cutoff_minutes: float
    ) -> List[PicklistCandidate]:
        """
        Create optimized picklists for a zone-priority combination.
        
        Strategy:
        - Group by store to maximize complete order fulfillment
        - Order bins by rank to minimize travel
        - Pack efficiently within weight/item limits
        - Only create picklists that can feasibly complete before cutoff
        """
        candidates = []
        is_fragile = items_df['is_fragile'].any()
        max_weight = MAX_WEIGHT_FRAGILE_KG if is_fragile else MAX_WEIGHT_PER_PICKLIST_KG
        
        # Get pods_per_picklist constraint
        pods_limit = items_df['pods_per_picklist'].iloc[0] if len(items_df) > 0 else 100
        
        # Sort by store_id (group orders) then bin_rank (optimize path)
        items_df = items_df.sort_values(['store_id', 'bin_rank'])
        
        current_items = []
        current_units = 0
        current_weight = 0.0
        current_stores = set()
        current_orders = set()
        current_bins = set()
        
        def finalize_picklist():
            nonlocal current_items, current_units, current_weight, current_stores, current_orders, current_bins
            
            if not current_items:
                return
            
            duration = self._estimate_picklist_duration(
                current_units, len(current_bins), len(current_orders)
            )
            
            # Value score: units per minute (efficiency)
            # Bonus for completing full orders
            value_score = current_units / max(duration, 0.1)
            
            candidates.append(PicklistCandidate(
                zone=zone,
                priority=priority,
                cutoff_minutes=cutoff_minutes,
                items=current_items.copy(),
                total_units=current_units,
                total_weight=current_weight,
                num_orders=len(current_orders),
                num_stores=len(current_stores),
                num_bins=len(current_bins),
                is_fragile=is_fragile,
                estimated_duration=duration,
                value_score=value_score
            ))
            
            current_items = []
            current_units = 0
            current_weight = 0.0
            current_stores = set()
            current_orders = set()
            current_bins = set()
        
        for _, row in items_df.iterrows():
            item_units = row['quantity']
            item_weight = row['weight_kg'] * item_units
            
            # Check constraints
            would_exceed_items = current_units + item_units > MAX_ITEMS_PER_PICKLIST
            would_exceed_weight = current_weight + item_weight > max_weight
            would_exceed_stores = len(current_stores | {row['store_id']}) > pods_limit
            
            if would_exceed_items or would_exceed_weight or would_exceed_stores:
                finalize_picklist()
            
            current_items.append({
                'order_id': row['order_id'],
                'store_id': row['store_id'],
                'sku': row['sku'],
                'bin': row['bin'],
                'bin_rank': row['bin_rank'],
                'quantity': row['quantity'],
                'weight_kg': row['weight_kg'],
                'is_fragile': row['is_fragile']
            })
            current_units += item_units
            current_weight += item_weight
            current_stores.add(row['store_id'])
            current_orders.add(row['order_id'])
            current_bins.add(row['bin'])
        
        finalize_picklist()
        return candidates
    
    def _assign_picklist(self, candidate: PicklistCandidate) -> Optional[Dict]:
        """Assign a picklist candidate to an available picker"""
        picker = self._get_best_available_picker(
            candidate.estimated_duration, 
            candidate.cutoff_minutes
        )
        
        if picker is None:
            return None
        
        self.picklist_counter += 1
        picklist_id = f"PL_{self.processing_date}_{self.picklist_counter:05d}"
        
        start_time = picker.current_time
        end_time = start_time + candidate.estimated_duration
        
        # Update picker
        picker.current_time = end_time
        picker.assigned_picklists.append(picklist_id)
        picker.total_units_picked += candidate.total_units
        
        # Determine picklist type
        unique_skus = len(set(item['sku'] for item in candidate.items))
        if candidate.is_fragile:
            picklist_type = "fragile"
        elif unique_skus == 1:
            picklist_type = "bulk"
        else:
            picklist_type = "multi_order"
        
        assignment = {
            'picklist_id': picklist_id,
            'picker_id': picker.picker_id,
            'zone': candidate.zone,
            'priority': candidate.priority,
            'start_time': start_time,
            'end_time': end_time,
            'cutoff_time': candidate.cutoff_minutes,
            'items_count': len(candidate.items),
            'units_count': candidate.total_units,
            'weight_kg': candidate.total_weight,
            'orders': list(set(item['order_id'] for item in candidate.items)),
            'stores': list(set(item['store_id'] for item in candidate.items)),
            'is_fragile': candidate.is_fragile,
            'on_time': True,  # We only assign if it can complete on time
            'picklist_type': picklist_type,
            'items': candidate.items
        }
        
        return assignment
    
    def optimize(self) -> Dict[str, Any]:
        """
        Main optimization routine - GREEDY approach optimized for hackathon metrics
        
        Strategy:
        1. Process priorities in order (P1 first - earliest cutoff)
        2. For each priority, create all possible picklist candidates
        3. Score candidates by efficiency (units/time) 
        4. Greedily assign best candidates that fit before cutoff
        5. NEVER assign late picklists (zero waste)
        """
        print("Starting advanced optimization...")
        print(f"Total items: {len(self.items_df):,}")
        print(f"Total units: {self.items_df['quantity'].sum():,}")
        
        all_candidates = []
        
        # Phase 1: Generate all picklist candidates grouped by priority
        print("\nPhase 1: Generating picklist candidates...")
        for priority in tqdm(PRIORITY_ORDER, desc="Processing priorities"):
            cutoff_minutes = PRIORITY_CUTOFFS[priority].total_seconds() / 60
            priority_items = self.items_df[self.items_df['priority'] == priority]
            
            if len(priority_items) == 0:
                continue
            
            # Process each zone
            for zone in priority_items['zone'].unique():
                zone_items = priority_items[priority_items['zone'] == zone]
                candidates = self._create_optimized_picklists_for_zone_priority(
                    zone_items, zone, priority, cutoff_minutes
                )
                all_candidates.extend(candidates)
        
        print(f"Generated {len(all_candidates)} picklist candidates")
        
        # Phase 2: Sort candidates by priority (cutoff time) then by value score
        # Earlier cutoff = higher priority
        all_candidates.sort(key=lambda c: (c.cutoff_minutes, -c.value_score))
        
        # Phase 3: Greedy assignment
        print("\nPhase 2: Assigning picklists to pickers...")
        assigned_count = 0
        skipped_count = 0
        
        for candidate in tqdm(all_candidates, desc="Assigning picklists"):
            assignment = self._assign_picklist(candidate)
            
            if assignment:
                self.assignments.append(assignment)
                self.total_units_picked += candidate.total_units
                self.total_units_on_time += candidate.total_units
                assigned_count += 1
            else:
                skipped_count += 1
                # Track unassigned for analysis
                for item in candidate.items:
                    self.unassigned_items.append(item)
        
        print(f"\nAssigned: {assigned_count} picklists")
        print(f"Skipped (no picker available before cutoff): {skipped_count} picklists")
        
        return self._generate_results()
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate optimization results"""
        
        # Calculate picker utilization
        picker_utilization = []
        for picker in self.pickers:
            shift_duration = picker.shift_end - picker.shift_start
            if shift_duration > 0:
                work_time = picker.current_time - picker.shift_start
                utilization = min(1.0, work_time / shift_duration)
            else:
                utilization = 0
            
            picker_utilization.append({
                'picker_id': picker.picker_id,
                'shift': picker.shift_name,
                'picklists_assigned': len(picker.assigned_picklists),
                'units_picked': picker.total_units_picked,
                'utilization': utilization
            })
        
        # Calculate order completion
        order_items_total = defaultdict(int)
        order_items_picked = defaultdict(int)
        
        for _, row in self.items_df.iterrows():
            order_items_total[row['order_id']] += 1
        
        for assignment in self.assignments:
            for item in assignment['items']:
                order_items_picked[item['order_id']] += 1
        
        complete_orders = sum(
            1 for oid in order_items_total 
            if order_items_picked.get(oid, 0) == order_items_total[oid]
        )
        
        # Build picklists for output
        picklists_output = []
        for assignment in self.assignments:
            picklists_output.append({
                'picklist_id': assignment['picklist_id'],
                'zone': assignment['zone'],
                'total_units': assignment['units_count'],
                'total_weight_kg': assignment['weight_kg'],
                'unique_bins': len(set(item['bin'] for item in assignment['items'])),
                'unique_orders': len(assignment['orders']),
                'unique_stores': len(assignment['stores']),
                'is_fragile': assignment['is_fragile'],
                'picklist_type': assignment['picklist_type'],
                'picker_id': assignment['picker_id'],
                'start_time': assignment['start_time'],
                'end_time': assignment['end_time'],
                'items': assignment['items']
            })
        
        total_possible_units = int(self.items_df['quantity'].sum())
        
        results = {
            'summary': {
                'total_picklists': len(self.assignments),
                'total_units_picked': self.total_units_picked,
                'total_units_on_time': self.total_units_on_time,
                'total_units_late': 0,  # We never assign late picklists!
                'total_possible_units': total_possible_units,
                'fulfillment_rate': self.total_units_on_time / total_possible_units if total_possible_units > 0 else 0,
                'on_time_rate': 1.0,  # All assigned picklists are on-time by design
                'completed_orders': complete_orders,
                'total_orders': len(order_items_total),
                'order_completion_rate': complete_orders / len(order_items_total) if order_items_total else 0,
                'avg_picker_utilization': np.mean([p['utilization'] for p in picker_utilization]),
                'wasted_effort_units': 0,  # Zero waste!
                'processing_date': self.processing_date
            },
            'assignments': self.assignments,
            'picker_utilization': picker_utilization,
            'picklists': picklists_output
        }
        
        return results


def run_advanced_optimization(processed_data_path: str = None) -> Dict[str, Any]:
    """Run the advanced optimization pipeline"""
    if processed_data_path is None:
        processed_data_path = os.path.join(INTERMEDIATE_DIR, "processed_orders.json")
    
    print(f"Loading processed data from {processed_data_path}...")
    with open(processed_data_path, 'r') as f:
        data = json.load(f)
    
    optimizer = AdvancedPicklistOptimizer(
        items_data=data['items'],
        processing_date=data['processing_date']
    )
    
    results = optimizer.optimize()
    
    # Save results
    output_path = os.path.join(INTERMEDIATE_DIR, "optimization_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved optimization results to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("🏆 OPTIMIZATION SUMMARY - HACKATHON METRICS")
    print("="*70)
    s = results['summary']
    print(f"📊 PRIMARY: Units Picked On-Time: {s['total_units_on_time']:,} / {s['total_possible_units']:,} ({s['fulfillment_rate']*100:.2f}%)")
    print(f"📋 SECONDARY: Complete Orders: {s['completed_orders']:,} / {s['total_orders']:,} ({s['order_completion_rate']*100:.2f}%)")
    print(f"🗑️  Wasted Effort: {s['wasted_effort_units']:,} units (ZERO WASTE!)")
    print(f"👷 Picker Utilization: {s['avg_picker_utilization']*100:.1f}%")
    print(f"📦 Total Picklists: {s['total_picklists']:,}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_advanced_optimization()
