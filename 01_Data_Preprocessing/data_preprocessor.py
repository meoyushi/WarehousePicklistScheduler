"""
Data Preprocessing Module for Warehouse Picklist Optimization
Handles loading, cleaning, and transforming raw order data
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    INPUT_DATA_FILE, INTERMEDIATE_DIR, FRAGILE_ZONE_KEYWORDS,
    PRIORITY_CUTOFFS, PICKING_START_TIME, GRAMS_TO_KG, PRIORITY_ORDER
)


@dataclass
class SKUItem:
    """Represents a single SKU item in an order"""
    sku: int
    order_qty: int
    zone: str
    bin_location: str
    bin_rank: int
    weight_kg: float
    length_cm: float
    width_cm: float
    height_cm: float
    floor: str
    rack: str
    aisle: str
    is_fragile: bool


@dataclass
class Order:
    """Represents a customer order"""
    order_id: str
    store_id: int
    location_code: str
    pod_priority: str
    cutoff_time: timedelta
    order_date: datetime
    dt: datetime
    pods_per_picklist_zone: int
    items: List[SKUItem]
    
    @property
    def total_weight_kg(self) -> float:
        return sum(item.weight_kg * item.order_qty for item in self.items)
    
    @property
    def total_units(self) -> int:
        return sum(item.order_qty for item in self.items)
    
    @property
    def zones(self) -> set:
        return set(item.zone for item in self.items)
    
    @property
    def has_fragile(self) -> bool:
        return any(item.is_fragile for item in self.items)


def identify_fragile(zone: str, order_tag: str = "Normal") -> bool:
    """
    Identify if a zone/item is fragile based on zone name or order tag
    """
    zone_upper = str(zone).upper() if pd.notna(zone) else ""
    
    # Check for fragile keywords in zone
    for keyword in FRAGILE_ZONE_KEYWORDS:
        if keyword.upper() in zone_upper:
            return True
    
    # Check order tag
    if order_tag and "fragile" in str(order_tag).lower():
        return True
        
    return False


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Load raw data from Excel file and perform initial preprocessing
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_excel(data_path)
    
    print(f"Loaded {len(df):,} rows")
    
    # Handle missing values
    df['floor'] = df['floor'].fillna('UNKNOWN')
    df['rack'] = df['rack'].fillna('UNKNOWN')
    df['aisle'] = df['aisle'].fillna('UNKNOWN')
    df['weight_in_grams'] = df['weight_in_grams'].fillna(0)
    df['length_in_cm'] = df['length_in_cm'].fillna(0)
    df['width_in_cm'] = df['width_in_cm'].fillna(0)
    df['height_in_cm'] = df['height_in_cm'].fillna(0)
    
    # Convert weight to kg
    df['weight_kg'] = df['weight_in_grams'] / GRAMS_TO_KG
    
    # Identify fragile items
    df['is_fragile'] = df.apply(
        lambda row: identify_fragile(row['zone'], row.get('order_tag', 'Normal')), 
        axis=1
    )
    
    # Add cutoff time based on priority
    df['cutoff_time'] = df['pod_priority'].map(PRIORITY_CUTOFFS)
    
    # Add priority rank for sorting
    df['priority_rank'] = df['pod_priority'].map({p: i for i, p in enumerate(PRIORITY_ORDER)})
    
    print(f"Data preprocessing complete")
    print(f"  - Unique orders: {df['order_id'].nunique():,}")
    print(f"  - Unique stores: {df['store_id'].nunique():,}")
    print(f"  - Unique zones: {df['zone'].nunique()}")
    print(f"  - Fragile items: {df['is_fragile'].sum():,}")
    
    return df


def aggregate_orders(df: pd.DataFrame) -> Dict[str, Order]:
    """
    Aggregate SKU-level data into Order objects
    """
    print("Aggregating orders...")
    orders: Dict[str, Order] = {}
    
    # Group by order_id
    grouped = df.groupby('order_id')
    
    for order_id, group in tqdm(grouped, desc="Processing orders"):
        first_row = group.iloc[0]
        
        items = []
        for _, row in group.iterrows():
            item = SKUItem(
                sku=int(row['sku']),
                order_qty=int(row['order_qty']),
                zone=str(row['zone']),
                bin_location=str(row['bin']),
                bin_rank=int(row['bin_rank']),
                weight_kg=float(row['weight_kg']),
                length_cm=float(row['length_in_cm']),
                width_cm=float(row['width_in_cm']),
                height_cm=float(row['height_in_cm']),
                floor=str(row['floor']),
                rack=str(row['rack']),
                aisle=str(row['aisle']),
                is_fragile=bool(row['is_fragile'])
            )
            items.append(item)
        
        order = Order(
            order_id=str(order_id),
            store_id=int(first_row['store_id']),
            location_code=str(first_row['location_code']),
            pod_priority=str(first_row['pod_priority']),
            cutoff_time=first_row['cutoff_time'],
            order_date=first_row['order_date'],
            dt=first_row['dt'],
            pods_per_picklist_zone=int(first_row['pods_per_picklist_in_that_zone']),
            items=items
        )
        
        orders[order_id] = order
    
    print(f"Aggregated {len(orders):,} orders")
    return orders


def group_items_by_zone(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group all items by zone for zone-wise picklist generation
    """
    return {zone: group for zone, group in df.groupby('zone')}


def create_zone_item_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of items per zone
    """
    zone_summary = df.groupby('zone').agg({
        'order_id': 'nunique',
        'sku': 'nunique',
        'order_qty': 'sum',
        'weight_kg': lambda x: (x * df.loc[x.index, 'order_qty']).sum(),
        'is_fragile': 'any',
        'store_id': 'nunique'
    }).rename(columns={
        'order_id': 'num_orders',
        'sku': 'num_unique_skus',
        'order_qty': 'total_units',
        'weight_kg': 'total_weight_kg',
        'is_fragile': 'has_fragile',
        'store_id': 'num_stores'
    })
    
    return zone_summary.reset_index()


def prepare_optimization_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare data in format suitable for optimization engine
    """
    print("Preparing optimization data...")
    
    # Create order-SKU-zone mapping
    order_items = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing items"):
        order_items.append({
            'order_id': str(row['order_id']),
            'store_id': int(row['store_id']),
            'sku': int(row['sku']),
            'zone': str(row['zone']),
            'bin': str(row['bin']),
            'bin_rank': int(row['bin_rank']),
            'quantity': int(row['order_qty']),
            'weight_kg': float(row['weight_kg']),
            'is_fragile': bool(row['is_fragile']),
            'priority': str(row['pod_priority']),
            'priority_rank': int(row['priority_rank']),
            'cutoff_minutes': row['cutoff_time'].total_seconds() / 60,
            'floor': str(row['floor']),
            'rack': str(row['rack']),
            'aisle': str(row['aisle']),
            'pods_per_picklist': int(row['pods_per_picklist_in_that_zone'])
        })
    
    # Create zone summary
    zone_summary = create_zone_item_summary(df)
    
    optimization_data = {
        'items': order_items,
        'zones': df['zone'].unique().tolist(),
        'zone_summary': zone_summary.to_dict('records'),
        'total_orders': df['order_id'].nunique(),
        'total_stores': df['store_id'].nunique(),
        'total_units': int(df['order_qty'].sum()),
        'processing_date': str(df['dt'].iloc[0].date())
    }
    
    return optimization_data


def save_processed_data(optimization_data: Dict[str, Any], output_path: str):
    """
    Save processed data to JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(optimization_data, f, indent=2, default=str)
    
    print(f"Saved processed data to {output_path}")


def run_preprocessing(data_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main preprocessing pipeline
    """
    if data_path is None:
        data_path = INPUT_DATA_FILE
    
    if output_path is None:
        output_path = os.path.join(INTERMEDIATE_DIR, "processed_orders.json")
    
    # Load and preprocess
    df = load_and_preprocess_data(data_path)
    
    # Prepare optimization data
    optimization_data = prepare_optimization_data(df)
    
    # Save to file
    save_processed_data(optimization_data, output_path)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total Orders: {optimization_data['total_orders']:,}")
    print(f"Total Stores: {optimization_data['total_stores']:,}")
    print(f"Total Units: {optimization_data['total_units']:,}")
    print(f"Total Zones: {len(optimization_data['zones'])}")
    print(f"Processing Date: {optimization_data['processing_date']}")
    print("="*60)
    
    return optimization_data


if __name__ == "__main__":
    # Run preprocessing from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess warehouse order data")
    parser.add_argument("--input", "-i", type=str, default=INPUT_DATA_FILE,
                       help="Path to input Excel file")
    parser.add_argument("--output", "-o", type=str, 
                       default=os.path.join(INTERMEDIATE_DIR, "processed_orders.json"),
                       help="Path to output JSON file")
    
    args = parser.parse_args()
    
    run_preprocessing(args.input, args.output)
