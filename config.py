"""
Warehouse Picklist Optimization - Configuration Module
Contains all constants and configuration parameters
"""

from datetime import datetime, timedelta
from typing import Dict

# =============================================================================
# TIME CONSTANTS (in minutes)
# =============================================================================
START_TO_ZONE_TIME = 2  # Starting point to zone
INTRA_ZONE_BIN_TRAVEL_TIME = 0.5  # 30 seconds = 0.5 minutes
SKU_PICKUP_TIME_PER_UNIT = 5 / 60  # 5 seconds per unit = 5/60 minutes
UNLOADING_TIME_PER_ORDER = 0.5  # 30 seconds per order
ZONE_TO_STAGING_TIME = 2  # Zone to staging area

# =============================================================================
# PICKLIST CONSTRAINTS
# =============================================================================
MAX_ITEMS_PER_PICKLIST = 2000  # Maximum units per picklist
MAX_WEIGHT_PER_PICKLIST_KG = 200  # Maximum weight in kg for normal picklist
MAX_WEIGHT_FRAGILE_KG = 50  # Maximum weight in kg for fragile picklist

# =============================================================================
# PRIORITY CUTOFF TIMES
# Picking starts at 9PM everyday
# =============================================================================
PICKING_START_TIME = datetime.strptime("21:00", "%H:%M")  # 9 PM

# Cutoff times for each priority (next day for those after midnight)
PRIORITY_CUTOFFS: Dict[str, timedelta] = {
    "P1": timedelta(hours=2, minutes=30),   # 11:30 PM (same day) = 2.5 hrs from 9PM
    "P2": timedelta(hours=5),               # 02:00 AM (next day) = 5 hrs from 9PM
    "P3": timedelta(hours=7),               # 04:00 AM (next day) = 7 hrs from 9PM
    "P4": timedelta(hours=9),               # 06:00 AM (next day) = 9 hrs from 9PM
    "P5": timedelta(hours=10),              # 07:00 AM (next day) = 10 hrs from 9PM
    "P6": timedelta(hours=12),              # 09:00 AM (next day) = 12 hrs from 9PM
    "P9": timedelta(hours=14),              # 11:00 AM (next day) = 14 hrs from 9PM
}

# Priority order (higher priority first)
PRIORITY_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P9"]

# =============================================================================
# PICKER SHIFTS
# =============================================================================
PICKER_SHIFTS = {
    "Morning Shift": {
        "start": datetime.strptime("08:00", "%H:%M"),
        "end": datetime.strptime("17:00", "%H:%M"),
        "num_pickers": 40
    },
    "General Shift": {
        "start": datetime.strptime("10:00", "%H:%M"),
        "end": datetime.strptime("19:00", "%H:%M"),
        "num_pickers": 30
    },
    "Night Shift 1": {
        "start": datetime.strptime("20:00", "%H:%M"),
        "end": datetime.strptime("05:00", "%H:%M") + timedelta(days=1),  # Next day
        "num_pickers": 45
    },
    "Night Shift 2": {
        "start": datetime.strptime("21:00", "%H:%M"),
        "end": datetime.strptime("07:00", "%H:%M") + timedelta(days=1),  # Next day
        "num_pickers": 35
    },
}

# =============================================================================
# ZONE TYPES FOR IDENTIFYING FRAGILE ZONES
# =============================================================================
FRAGILE_ZONE_KEYWORDS = ["FRAGILE", "FD"]  # FD likely means Fragile Department

# =============================================================================
# FILE PATHS
# =============================================================================
INPUT_DATA_FILE = "picklist_creation_data_for_hackathon_with_order_date.xlsx"
INTERMEDIATE_DIR = "03_Intermediate_Artifacts"
OUTPUT_DIR = "04_Output_Picklists"
ANALYSIS_DIR = "05_Ananlysis_Insights"

# =============================================================================
# WEIGHT CONVERSION
# =============================================================================
GRAMS_TO_KG = 1000
