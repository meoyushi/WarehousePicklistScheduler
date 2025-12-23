"""
Time Calculator Module for Warehouse Picklist Optimization
Calculates picking times based on warehouse layout and constraints
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    START_TO_ZONE_TIME,
    INTRA_ZONE_BIN_TRAVEL_TIME,
    SKU_PICKUP_TIME_PER_UNIT,
    UNLOADING_TIME_PER_ORDER,
    ZONE_TO_STAGING_TIME
)


@dataclass
class PicklistItem:
    """Represents an item in a picklist"""
    order_id: str
    store_id: int
    sku: int
    zone: str
    bin_location: str
    bin_rank: int
    quantity: int
    weight_kg: float
    is_fragile: bool


@dataclass
class Picklist:
    """Represents a complete picklist"""
    picklist_id: str
    zone: str
    items: List[PicklistItem]
    picker_id: int = None
    start_time: float = None  # Minutes from picking start
    end_time: float = None
    
    @property
    def total_units(self) -> int:
        return sum(item.quantity for item in self.items)
    
    @property
    def total_weight_kg(self) -> float:
        return sum(item.weight_kg * item.quantity for item in self.items)
    
    @property
    def unique_bins(self) -> int:
        return len(set(item.bin_location for item in self.items))
    
    @property
    def unique_orders(self) -> int:
        return len(set(item.order_id for item in self.items))
    
    @property
    def unique_stores(self) -> int:
        return len(set(item.store_id for item in self.items))
    
    @property
    def is_fragile(self) -> bool:
        return any(item.is_fragile for item in self.items)
    
    @property
    def is_bulk(self) -> bool:
        """Bulk picklist: all items are same SKU"""
        return len(set(item.sku for item in self.items)) == 1
    
    @property
    def picklist_type(self) -> str:
        if self.is_fragile:
            return "fragile"
        elif self.is_bulk:
            return "bulk"
        else:
            return "multi_order"


def calculate_bin_travel_time(bin_ranks: List[int]) -> float:
    """
    Calculate travel time between bins based on bin ranks
    Assumes bins are visited in order of their ranks
    """
    if len(bin_ranks) <= 1:
        return 0
    
    # Sort bins by rank for optimal path
    sorted_ranks = sorted(set(bin_ranks))
    
    # Travel time = number of bin-to-bin transitions * time per transition
    num_transitions = len(sorted_ranks) - 1
    return num_transitions * INTRA_ZONE_BIN_TRAVEL_TIME


def calculate_picking_time(items: List[PicklistItem]) -> float:
    """
    Calculate total picking time for items
    Time = SKU pickup time per unit * total units
    """
    total_units = sum(item.quantity for item in items)
    return total_units * SKU_PICKUP_TIME_PER_UNIT


def calculate_unloading_time(items: List[PicklistItem]) -> float:
    """
    Calculate unloading time at staging area
    Time = 30 seconds per order (unique order in picklist)
    """
    unique_orders = len(set(item.order_id for item in items))
    return unique_orders * UNLOADING_TIME_PER_ORDER


def calculate_picklist_duration(picklist: Picklist) -> float:
    """
    Calculate total duration for completing a picklist
    
    Components:
    1. Start to Zone: 2 minutes
    2. Intra-zone bin travel: 30 sec per bin-to-bin
    3. SKU pickup: 5 sec per unit
    4. Zone to Staging: 2 minutes
    5. Unloading: 30 sec per order
    """
    items = picklist.items
    
    # 1. Travel to zone
    start_to_zone = START_TO_ZONE_TIME
    
    # 2. Intra-zone travel
    bin_ranks = [item.bin_rank for item in items]
    bin_travel = calculate_bin_travel_time(bin_ranks)
    
    # 3. Picking time
    picking_time = calculate_picking_time(items)
    
    # 4. Return to staging
    zone_to_staging = ZONE_TO_STAGING_TIME
    
    # 5. Unloading
    unloading_time = calculate_unloading_time(items)
    
    total_duration = (
        start_to_zone + 
        bin_travel + 
        picking_time + 
        zone_to_staging + 
        unloading_time
    )
    
    return total_duration


def calculate_picklist_duration_estimate(
    num_items: int,
    num_bins: int,
    num_orders: int,
    total_units: int
) -> float:
    """
    Estimate picklist duration without creating full Picklist object
    Useful for optimization planning
    """
    start_to_zone = START_TO_ZONE_TIME
    bin_travel = max(0, num_bins - 1) * INTRA_ZONE_BIN_TRAVEL_TIME
    picking_time = total_units * SKU_PICKUP_TIME_PER_UNIT
    zone_to_staging = ZONE_TO_STAGING_TIME
    unloading_time = num_orders * UNLOADING_TIME_PER_ORDER
    
    return start_to_zone + bin_travel + picking_time + zone_to_staging + unloading_time


def estimate_zone_clearing_time(
    zone_data: Dict,
    max_items_per_picklist: int = 2000,
    max_weight_per_picklist: float = 200
) -> Tuple[int, float]:
    """
    Estimate number of picklists and total time to clear a zone
    
    Returns:
        Tuple of (estimated_picklists, estimated_total_time)
    """
    total_units = zone_data.get('total_units', 0)
    total_weight = zone_data.get('total_weight_kg', 0)
    num_bins = zone_data.get('num_unique_skus', 0)  # Approximate bins by unique SKUs
    num_orders = zone_data.get('num_orders', 0)
    
    # Estimate picklists needed (limited by items or weight)
    picklists_by_items = math.ceil(total_units / max_items_per_picklist)
    picklists_by_weight = math.ceil(total_weight / max_weight_per_picklist)
    estimated_picklists = max(picklists_by_items, picklists_by_weight)
    
    # Estimate average duration per picklist
    avg_items = total_units / max(1, estimated_picklists)
    avg_bins = num_bins / max(1, estimated_picklists)
    avg_orders = num_orders / max(1, estimated_picklists)
    
    avg_duration = calculate_picklist_duration_estimate(
        int(avg_items), int(avg_bins), int(avg_orders), int(avg_items)
    )
    
    return estimated_picklists, estimated_picklists * avg_duration


def check_picklist_completion_before_cutoff(
    picklist_start_time: float,
    picklist_duration: float,
    cutoff_time: float
) -> bool:
    """
    Check if a picklist can be completed before its cutoff
    
    Args:
        picklist_start_time: Minutes from picking start (9PM)
        picklist_duration: Duration in minutes
        cutoff_time: Cutoff time in minutes from picking start
    
    Returns:
        True if picklist completes before cutoff
    """
    completion_time = picklist_start_time + picklist_duration
    return completion_time <= cutoff_time


def optimize_bin_order(items: List[PicklistItem]) -> List[PicklistItem]:
    """
    Optimize the order of items within a picklist based on bin ranks
    This minimizes travel time within a zone
    """
    return sorted(items, key=lambda x: x.bin_rank)


if __name__ == "__main__":
    # Test time calculations
    print("Time Calculator Test")
    print("=" * 40)
    
    # Create sample items
    sample_items = [
        PicklistItem(
            order_id="O1", store_id=1, sku=100, zone="A",
            bin_location="A-1-1", bin_rank=1, quantity=10,
            weight_kg=0.5, is_fragile=False
        ),
        PicklistItem(
            order_id="O1", store_id=1, sku=101, zone="A",
            bin_location="A-1-2", bin_rank=2, quantity=5,
            weight_kg=0.3, is_fragile=False
        ),
        PicklistItem(
            order_id="O2", store_id=2, sku=102, zone="A",
            bin_location="A-2-1", bin_rank=10, quantity=3,
            weight_kg=1.0, is_fragile=False
        ),
    ]
    
    picklist = Picklist(
        picklist_id="PL001",
        zone="A",
        items=sample_items
    )
    
    duration = calculate_picklist_duration(picklist)
    print(f"Sample Picklist Duration: {duration:.2f} minutes")
    print(f"  - Total units: {picklist.total_units}")
    print(f"  - Total weight: {picklist.total_weight_kg:.2f} kg")
    print(f"  - Unique bins: {picklist.unique_bins}")
    print(f"  - Unique orders: {picklist.unique_orders}")
    print(f"  - Picklist type: {picklist.picklist_type}")
