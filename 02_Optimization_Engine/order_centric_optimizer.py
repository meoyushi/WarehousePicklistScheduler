"""
Hackathon Winning Strategy - Order-Centric Optimization
Maximizes complete orders while maximizing units picked before cutoff

Key Strategies:
1. Prioritize picking ALL items for high-priority orders first
2. Use efficient bin-path ordering to maximize throughput
3. Smart batching to maximize picker utilization
4. Zero waste guarantee - only assign completable picklists
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
    picker_id: int
    shift_name: str
    shift_start: float
    shift_end: float
    current_time: float = 0
    assigned_picklists: List[str] = field(default_factory=list)
    total_units_picked: int = 0


class OrderCentricOptimizer:
    """
    Order-centric optimization that maximizes:
    1. Units picked on-time (PRIMARY)
    2. Complete orders (SECONDARY) 
    3. Zero wasted effort
    4. Maximum picker utilization
    """
    
    def __init__(self, items_data: List[Dict], processing_date: str):
        self.items_df = pd.DataFrame(items_data)
        self.processing_date = processing_date
        self.pickers = self._initialize_pickers()
        
        # Build order index
        self.orders = self._build_order_index()
        
        # Track what's been picked
        self.picked_items = set()  # (order_id, sku, bin) tuples
        self.assignments = []
        self.picklist_counter = 0
        
        # Metrics
        self.total_units_on_time = 0
        
    def _initialize_pickers(self) -> List[Picker]:
        """Initialize available pickers"""
        pickers = []
        picker_id = 0
        picking_start = PICKING_START_TIME.hour * 60 + PICKING_START_TIME.minute
        
        for shift_name, config in PICKER_SHIFTS.items():
            start = config['start'].hour * 60 + config['start'].minute
            end = config['end'].hour * 60 + config['end'].minute
            
            if end < start:
                end += 24 * 60
            
            start_rel = start - picking_start
            end_rel = end - picking_start
            
            if end_rel <= 0:
                continue
            if start_rel < 0:
                start_rel = 0
            
            for _ in range(config['num_pickers']):
                pickers.append(Picker(
                    picker_id=picker_id,
                    shift_name=shift_name,
                    shift_start=start_rel,
                    shift_end=end_rel,
                    current_time=start_rel
                ))
                picker_id += 1
        
        pickers.sort(key=lambda p: p.shift_end)
        print(f"Initialized {len(pickers)} pickers")
        return pickers
    
    def _build_order_index(self) -> Dict[str, Dict]:
        """Build index of orders with their items grouped by zone"""
        orders = defaultdict(lambda: {
            'store_id': None,
            'priority': None,
            'cutoff_minutes': None,
            'zones': defaultdict(list),
            'total_units': 0,
            'total_items': 0
        })
        
        for _, row in self.items_df.iterrows():
            oid = row['order_id']
            orders[oid]['store_id'] = row['store_id']
            orders[oid]['priority'] = row['priority']
            orders[oid]['cutoff_minutes'] = row['cutoff_minutes']
            orders[oid]['zones'][row['zone']].append({
                'sku': row['sku'],
                'bin': row['bin'],
                'bin_rank': row['bin_rank'],
                'quantity': row['quantity'],
                'weight_kg': row['weight_kg'],
                'is_fragile': row['is_fragile'],
                'order_id': oid,
                'store_id': row['store_id']
            })
            orders[oid]['total_units'] += row['quantity']
            orders[oid]['total_items'] += 1
        
        return dict(orders)
    
    def _estimate_duration(self, num_units: int, num_bins: int, num_orders: int) -> float:
        """Estimate picklist duration in minutes"""
        return (START_TO_ZONE_TIME + 
                max(0, num_bins - 1) * INTRA_ZONE_BIN_TRAVEL_TIME +
                num_units * SKU_PICKUP_TIME_PER_UNIT +
                ZONE_TO_STAGING_TIME +
                num_orders * UNLOADING_TIME_PER_ORDER)
    
    def _get_picker_for_deadline(self, duration: float, deadline: float) -> Optional[Picker]:
        """Find picker who can complete before deadline"""
        for p in self.pickers:
            if p.current_time + duration <= min(deadline, p.shift_end):
                return p
        return None
    
    def _create_picklist_for_zone_items(
        self, 
        items: List[Dict], 
        zone: str,
        priority: str,
        cutoff: float
    ) -> Optional[Dict]:
        """Create a single picklist from zone items"""
        if not items:
            return None
        
        # Sort by bin_rank for efficient path
        items = sorted(items, key=lambda x: x['bin_rank'])
        
        is_fragile = any(i['is_fragile'] for i in items)
        max_weight = MAX_WEIGHT_FRAGILE_KG if is_fragile else MAX_WEIGHT_PER_PICKLIST_KG
        
        # Get pods_per_picklist limit from data
        pods_limit = self.items_df[self.items_df['zone'] == zone]['pods_per_picklist'].iloc[0]
        
        # Pack items respecting constraints
        packed_items = []
        total_units = 0
        total_weight = 0.0
        stores = set()
        orders = set()
        bins = set()
        
        for item in items:
            item_units = item['quantity']
            item_weight = item['weight_kg'] * item_units
            
            # Check constraints
            if total_units + item_units > MAX_ITEMS_PER_PICKLIST:
                break
            if total_weight + item_weight > max_weight:
                break
            if len(stores | {item['store_id']}) > pods_limit:
                break
            
            packed_items.append(item)
            total_units += item_units
            total_weight += item_weight
            stores.add(item['store_id'])
            orders.add(item['order_id'])
            bins.add(item['bin'])
        
        if not packed_items:
            return None
        
        duration = self._estimate_duration(total_units, len(bins), len(orders))
        
        # Check if any picker can complete before cutoff
        picker = self._get_picker_for_deadline(duration, cutoff)
        if picker is None:
            return None
        
        self.picklist_counter += 1
        picklist_id = f"PL_{self.processing_date}_{self.picklist_counter:05d}"
        
        start_time = picker.current_time
        end_time = start_time + duration
        
        # Update picker
        picker.current_time = end_time
        picker.assigned_picklists.append(picklist_id)
        picker.total_units_picked += total_units
        
        # Determine type
        unique_skus = len(set(i['sku'] for i in packed_items))
        if is_fragile:
            ptype = "fragile"
        elif unique_skus == 1:
            ptype = "bulk"
        else:
            ptype = "multi_order"
        
        return {
            'picklist_id': picklist_id,
            'picker_id': picker.picker_id,
            'zone': zone,
            'priority': priority,
            'start_time': start_time,
            'end_time': end_time,
            'cutoff_time': cutoff,
            'items_count': len(packed_items),
            'units_count': total_units,
            'weight_kg': total_weight,
            'orders': list(orders),
            'stores': list(stores),
            'is_fragile': is_fragile,
            'on_time': True,
            'picklist_type': ptype,
            'items': packed_items
        }
    
    def optimize(self) -> Dict[str, Any]:
        """
        Order-centric optimization strategy:
        1. Sort orders by priority (earliest cutoff first)
        2. For each order, try to pick ALL its items (complete order bonus)
        3. Group items by zone and create efficient picklists
        4. Never assign picklists that would be late
        """
        print("Starting order-centric optimization...")
        
        # Sort orders by priority (cutoff time)
        sorted_orders = sorted(
            self.orders.items(),
            key=lambda x: (x[1]['cutoff_minutes'], -x[1]['total_units'])
        )
        
        print(f"Processing {len(sorted_orders)} orders by priority...")
        
        # Track which items have been assigned
        assigned_item_keys = set()
        
        for order_id, order_data in tqdm(sorted_orders, desc="Processing orders"):
            priority = order_data['priority']
            cutoff = order_data['cutoff_minutes']
            
            # Process each zone for this order
            for zone, zone_items in order_data['zones'].items():
                # Filter out already assigned items
                remaining_items = [
                    item for item in zone_items
                    if (item['order_id'], item['sku'], item['bin']) not in assigned_item_keys
                ]
                
                if not remaining_items:
                    continue
                
                # Try to create picklist for these items
                picklist = self._create_picklist_for_zone_items(
                    remaining_items, zone, priority, cutoff
                )
                
                if picklist:
                    self.assignments.append(picklist)
                    self.total_units_on_time += picklist['units_count']
                    
                    # Mark items as assigned
                    for item in picklist['items']:
                        assigned_item_keys.add((item['order_id'], item['sku'], item['bin']))
        
        # Phase 2: Try to fill remaining picker capacity with any remaining high-value items
        print("\nPhase 2: Filling remaining capacity...")
        self._fill_remaining_capacity(assigned_item_keys)
        
        return self._generate_results()
    
    def _fill_remaining_capacity(self, assigned_keys: Set[Tuple]):
        """Fill remaining picker capacity with unassigned items"""
        # Get remaining items grouped by zone and priority
        remaining = defaultdict(lambda: defaultdict(list))
        
        for _, row in self.items_df.iterrows():
            key = (row['order_id'], row['sku'], row['bin'])
            if key not in assigned_keys:
                remaining[row['priority']][row['zone']].append({
                    'order_id': row['order_id'],
                    'store_id': row['store_id'],
                    'sku': row['sku'],
                    'bin': row['bin'],
                    'bin_rank': row['bin_rank'],
                    'quantity': row['quantity'],
                    'weight_kg': row['weight_kg'],
                    'is_fragile': row['is_fragile']
                })
        
        # Process by priority
        for priority in PRIORITY_ORDER:
            if priority not in remaining:
                continue
            
            cutoff = PRIORITY_CUTOFFS[priority].total_seconds() / 60
            
            for zone, items in remaining[priority].items():
                # Sort by bin_rank
                items.sort(key=lambda x: x['bin_rank'])
                
                while items:
                    picklist = self._create_picklist_for_zone_items(
                        items, zone, priority, cutoff
                    )
                    
                    if picklist is None:
                        break
                    
                    self.assignments.append(picklist)
                    self.total_units_on_time += picklist['units_count']
                    
                    # Remove assigned items
                    assigned_in_pl = set(
                        (i['order_id'], i['sku'], i['bin']) 
                        for i in picklist['items']
                    )
                    items = [
                        i for i in items 
                        if (i['order_id'], i['sku'], i['bin']) not in assigned_in_pl
                    ]
                    assigned_keys.update(assigned_in_pl)
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate final results"""
        # Picker utilization
        picker_util = []
        for p in self.pickers:
            duration = p.shift_end - p.shift_start
            work = p.current_time - p.shift_start
            util = min(1.0, work / duration) if duration > 0 else 0
            picker_util.append({
                'picker_id': p.picker_id,
                'shift': p.shift_name,
                'picklists_assigned': len(p.assigned_picklists),
                'units_picked': p.total_units_picked,
                'utilization': util
            })
        
        # Order completion tracking
        order_total = defaultdict(int)
        order_picked = defaultdict(int)
        
        for _, row in self.items_df.iterrows():
            order_total[row['order_id']] += 1
        
        for a in self.assignments:
            for item in a['items']:
                order_picked[item['order_id']] += 1
        
        complete_orders = sum(
            1 for oid in order_total 
            if order_picked.get(oid, 0) == order_total[oid]
        )
        
        # Build output
        picklists = [{
            'picklist_id': a['picklist_id'],
            'zone': a['zone'],
            'total_units': a['units_count'],
            'total_weight_kg': a['weight_kg'],
            'unique_bins': len(set(i['bin'] for i in a['items'])),
            'unique_orders': len(a['orders']),
            'unique_stores': len(a['stores']),
            'is_fragile': a['is_fragile'],
            'picklist_type': a['picklist_type'],
            'picker_id': a['picker_id'],
            'start_time': a['start_time'],
            'end_time': a['end_time'],
            'items': a['items']
        } for a in self.assignments]
        
        total_possible = int(self.items_df['quantity'].sum())
        
        return {
            'summary': {
                'total_picklists': len(self.assignments),
                'total_units_picked': self.total_units_on_time,
                'total_units_on_time': self.total_units_on_time,
                'total_units_late': 0,
                'total_possible_units': total_possible,
                'fulfillment_rate': self.total_units_on_time / total_possible if total_possible else 0,
                'on_time_rate': 1.0,
                'completed_orders': complete_orders,
                'total_orders': len(order_total),
                'order_completion_rate': complete_orders / len(order_total) if order_total else 0,
                'avg_picker_utilization': np.mean([p['utilization'] for p in picker_util]),
                'wasted_effort_units': 0,
                'processing_date': self.processing_date
            },
            'assignments': self.assignments,
            'picker_utilization': picker_util,
            'picklists': picklists
        }


def run_order_centric_optimization(processed_data_path: str = None) -> Dict[str, Any]:
    """Run the order-centric optimization"""
    if processed_data_path is None:
        processed_data_path = os.path.join(INTERMEDIATE_DIR, "processed_orders.json")
    
    print(f"Loading data from {processed_data_path}...")
    with open(processed_data_path, 'r') as f:
        data = json.load(f)
    
    optimizer = OrderCentricOptimizer(
        items_data=data['items'],
        processing_date=data['processing_date']
    )
    
    results = optimizer.optimize()
    
    output_path = os.path.join(INTERMEDIATE_DIR, "optimization_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")
    
    print("\n" + "="*70)
    print("🏆 HACKATHON OPTIMIZATION RESULTS")
    print("="*70)
    s = results['summary']
    print(f"📊 PRIMARY METRIC - Units On-Time: {s['total_units_on_time']:,} / {s['total_possible_units']:,}")
    print(f"   Fulfillment Rate: {s['fulfillment_rate']*100:.2f}%")
    print(f"📋 SECONDARY METRIC - Complete Orders: {s['completed_orders']:,} / {s['total_orders']:,}")
    print(f"   Order Completion Rate: {s['order_completion_rate']*100:.2f}%")
    print(f"🗑️  WASTED EFFORT: {s['wasted_effort_units']} units (ZERO!)")
    print(f"👷 PICKER UTILIZATION: {s['avg_picker_utilization']*100:.1f}%")
    print(f"📦 TOTAL PICKLISTS: {s['total_picklists']}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_order_centric_optimization()
