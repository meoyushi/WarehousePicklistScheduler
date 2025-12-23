"""
Fragile Item Identifier Module
Identifies fragile items based on zone names and item characteristics
"""

import pandas as pd
from typing import List, Set

# Keywords that indicate fragile zones/items
FRAGILE_KEYWORDS = [
    "FRAGILE",
    "GLASS",
    "BREAKABLE",
    "DELICATE",
    "FD",  # Fragile Department
]

# Known fragile zones (can be extended)
FRAGILE_ZONES = {
    "FRAGILE_FD",
    "FRAGILE_G1",
    "FRAGILE_G2",
    "FRAGILE_G3",
}

# SKU categories that are typically fragile
FRAGILE_CATEGORIES = [
    "glass",
    "ceramic",
    "porcelain",
    "crystal",
    "bottle",
    "jar",
]


def is_fragile_zone(zone: str) -> bool:
    """
    Check if a zone is designated as fragile
    """
    if pd.isna(zone):
        return False
    
    zone_upper = str(zone).upper()
    
    # Check direct match
    if zone_upper in FRAGILE_ZONES:
        return True
    
    # Check keyword presence
    for keyword in FRAGILE_KEYWORDS:
        if keyword.upper() in zone_upper:
            return True
    
    return False


def is_fragile_item(row: pd.Series) -> bool:
    """
    Determine if an item is fragile based on various attributes
    """
    # Check zone
    if is_fragile_zone(row.get('zone', '')):
        return True
    
    # Check order tag
    order_tag = str(row.get('order_tag', '')).lower()
    if 'fragile' in order_tag:
        return True
    
    # Check bin location for fragile indicators
    bin_loc = str(row.get('bin', '')).upper()
    if 'FRAG' in bin_loc or 'FD' in bin_loc:
        return True
    
    return False


def identify_fragile_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fragile flag to dataframe
    """
    df = df.copy()
    df['is_fragile'] = df.apply(is_fragile_item, axis=1)
    return df


def get_fragile_zones(df: pd.DataFrame) -> Set[str]:
    """
    Get all zones that contain fragile items
    """
    fragile_zones = set()
    
    for zone in df['zone'].unique():
        if is_fragile_zone(zone):
            fragile_zones.add(zone)
    
    return fragile_zones


def fragile_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for fragile items
    """
    df_with_fragile = identify_fragile_items(df)
    
    fragile_df = df_with_fragile[df_with_fragile['is_fragile']]
    
    return {
        'total_fragile_items': len(fragile_df),
        'fragile_zones': list(get_fragile_zones(df)),
        'fragile_orders': fragile_df['order_id'].nunique() if len(fragile_df) > 0 else 0,
        'fragile_units': int(fragile_df['order_qty'].sum()) if len(fragile_df) > 0 else 0,
        'fragile_weight_kg': float(fragile_df['weight_in_grams'].sum() / 1000) if len(fragile_df) > 0 else 0,
    }


if __name__ == "__main__":
    # Test with sample data
    import sys
    import os
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import INPUT_DATA_FILE
    
    df = pd.read_excel(f"../{INPUT_DATA_FILE}")
    
    summary = fragile_summary(df)
    print("Fragile Item Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
