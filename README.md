# WarehousePicklistScheduler

# 🚀 Warehouse Picklist Optimization Engine

A **Constraint Programming (CP)** solution for high-throughput warehouse fulfillment that maximizes SKU units picked before loading deadlines using **Google OR-Tools** and **C++** for performance.

---

## 📋 Table of Contents
- [Problem Overview](#problem-overview)
- [Solution Architecture](#solution-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Algorithm Design](#algorithm-design)
- [Output Format](#output-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Features](#key-features)
- [Team & Contact](#team--contact)

---

## 🎯 Problem Overview

Quick-commerce warehouses process hundreds to thousands of orders per hour under strict loading schedules. This system optimizes:
- **Picklist creation** across multiple zones
- **Picker scheduling** within shift constraints
- **Partial fulfillment** to maximize units picked before cutoff
- **Constraint satisfaction** (weight limits, zone restrictions, fragile items)

### Key Constraints
- ✅ One zone per picklist
- ✅ Max 2000 items OR 200kg per picklist (50kg for fragile)
- ✅ Orders can be split across multiple picklists
- ✅ Only items picked before cutoff contribute value
- ✅ Picker shift windows must be respected

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────┐
│      Python: Data Preprocessing                 │
│      - Load CSV, clean data                     │
│      - Map priorities to cutoffs                │
│      - Calculate weights & fragility            │
│      - Export JSON for C++                      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│      C++: Constraint Programming Engine         │
│      - OR-Tools CP-SAT Solver                   │
│      - Zone-based batching                      │
│      - Time window enforcement                  │
│      - Picker assignment                        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│      Python: Analysis & Visualization           │
│      - Calculate metrics                        │
│      - Picker utilization heatmaps              │
│      - Fulfillment rate analysis                │
└─────────────────────────────────────────────────┘
```

**Tech Stack:**
- **Python 3.9+**: Data preprocessing, analysis
- **C++17**: Core optimization engine
- **Google OR-Tools**: CP-SAT constraint solver
- **CMake**: Build system
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

---

## 📁 Project Structure

```
/Warehouse-Optimization-Hackathon
│
├── 📁 01_Data_Preprocessing (Python)
│   ├── Preprocessing_Notebook.ipynb       # Main preprocessing pipeline
│   ├── fragile_identifier.py              # Logic to tag fragile SKUs
│   ├── config.py                          # Constants (cutoffs, weights)
│   └── /Input_Data
│       └── orders_dataset.csv             # Raw order data (download first)
│
├── 📁 02_Optimization_Engine (C++)
│   ├── main.cpp                           # Entry point
│   ├── cp_solver_logic.cpp                # OR-Tools CP-SAT implementation
│   ├── constraints.hpp                    # Constraint definitions
│   ├── time_calculator.cpp                # Travel & pickup time logic
│   ├── types.hpp                          # Data structures
│   └── CMakeLists.txt                     # Build configuration
│
├── 📁 03_Intermediate_Artifacts
│   └── processed_orders.json              # Python → C++ handoff file
│
├── 📁 04_Output_Results
│   ├── /Picklists
│   │   └── {date}_{Picklist_no}.csv      # Individual picklist outputs
│   └── Summary.csv                        # Aggregated metrics
│
├── 📁 05_Analysis_Insights (Python)
│   └── Evaluation_Metrics.ipynb           # Results analysis
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
└── run_pipeline.sh                        # Automated execution script
```

---

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **C++ Compiler**: g++ 9+ or clang++ 10+ (C++17 support)
- **CMake**: 3.15 or higher
- **Git**: For cloning repository

### Check Your Setup
```bash
python3 --version    # Should be 3.9+
g++ --version        # Should support C++17
cmake --version      # Should be 3.15+
```

---

## 💿 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-team/warehouse-optimization-hackathon.git
cd warehouse-optimization-hackathon
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `ortools>=9.8.0` - Constraint programming solver
- `jupyter>=1.0.0` - Notebook environment
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization

### 3. Download Dataset
Place the provided `orders_dataset.csv` in:
```
01_Data_Preprocessing/Input_Data/orders_dataset.csv
```

### 4. Build C++ Engine
```bash
cd 02_Optimization_Engine
mkdir build && cd build
cmake ..
make
```

This creates the executable: `build/optimizer`

---

## ⚡ Quick Start

### Option 1: Run Complete Pipeline (Recommended)
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will:
1. ✅ Preprocess data (Python)
2. ✅ Build C++ optimizer
3. ✅ Run constraint solver
4. ✅ Generate picklists
5. ✅ Validate outputs

### Option 2: Manual Step-by-Step

#### Step 1: Preprocess Data
```bash
cd 01_Data_Preprocessing
jupyter notebook Preprocessing_Notebook.ipynb
```
Run all cells to generate `processed_orders.json`

#### Step 2: Run Optimizer
```bash
cd ../02_Optimization_Engine/build
./optimizer ../../03_Intermediate_Artifacts/processed_orders.json
```

#### Step 3: Analyze Results
```bash
cd ../../05_Analysis_Insights
jupyter notebook Evaluation_Metrics.ipynb
```

---

## 📖 Detailed Usage

### Data Preprocessing (`01_Data_Preprocessing`)

**Input:** `orders_dataset.csv` with columns:
- `dt`, `order_id`, `sku`, `order_qty`, `zone`, `bin_rank`
- `Weight`, `pod_priority`, `store_id`, etc.

**Process:**
1. **Clean Data**: Remove duplicates, handle nulls
2. **Map Deadlines**: Convert P1-P9 priorities to timestamps
3. **Calculate Weights**: `total_weight = Weight × order_qty`
4. **Tag Fragile Items**: Using `fragile_identifier.py`
5. **Export JSON**: Structured format for C++

**Output:** `03_Intermediate_Artifacts/processed_orders.json`

```json
{
  "orders": [
    {
      "order_id": "ORD001",
      "store_id": "STORE_A",
      "cutoff_time": "2024-01-16T02:00:00",
      "priority_score": 2,
      "skus": [
        {
          "sku_id": "SKU123",
          "quantity": 5,
          "zone": "Zone_A",
          "bin_rank": 10,
          "weight": 1.5,
          "is_fragile": false
        }
      ]
    }
  ]
}
```

---

### Optimization Engine (`02_Optimization_Engine`)

**Core Algorithm:**
1. **Decision Variables**: Binary assignments `x_ij` (SKU i in picklist j)
2. **Constraints**:
   - Zone constraint: All items in picklist from same zone
   - Capacity: ≤2000 items, ≤200kg (or 50kg fragile)
   - Time windows: Complete before cutoff
   - Picker availability: Within shift hours
3. **Objective**: Maximize Σ(units picked before cutoff)
4. **Solver**: Google OR-Tools CP-SAT

**Key Files:**
- `types.hpp`: Data structures (Order, SKU, Picklist, Picker)
- `constraints.hpp`: Constraint definitions
- `cp_solver_logic.cpp`: OR-Tools integration
- `time_calculator.cpp`: Travel time computations

**Time Calculations:**
```cpp
Total Time = 2min (start→zone) 
           + 0.5min × (bins - 1) 
           + (5sec × units) 
           + 0.5min × orders 
           + 2min (zone→staging)
```

---

### Analysis & Metrics (`05_Analysis_Insights`)

**Key Metrics Calculated:**
1. **Primary**: Total units picked before cutoff
2. **Secondary**: Number of completed orders
3. Wasted picking effort (late picklists)
4. Picker utilization rate
5. Zone efficiency analysis

**Visualizations:**
- Picker workload heatmap
- Cumulative fulfillment over time
- Priority vs completion rate
- Zone bottleneck analysis

---

## 📤 Output Format

### Picklist Files: `{date}_{Picklist_no}.csv`
```csv
SKU,Store,Bin,Bin Rank
SKU123,STORE_A,B05,10
SKU456,STORE_A,B12,15
SKU789,STORE_B,B03,8
```

### Summary File: `Summary.csv`
```csv
Picklist_date,picklist_no,picklist_type,stores_in_picklist
2024-01-15,PL001,multi_order,"STORE_A,STORE_B"
2024-01-15,PL002,fragile,STORE_C
2024-01-15,PL003,bulk,STORE_A
```

**Picklist Types:**
- `bulk`: All same SKU
- `multi_order`: Multiple different SKUs
- `fragile`: Contains fragile items (50kg limit)

---

## 📊 Evaluation Metrics

### Primary Metric
**Units Fulfilled Before Cutoff:**
```
Score = Σ(units where finish_time ≤ cutoff_time)
```

### Secondary Metrics
- **Order Completion Rate**: % of orders fully picked
- **Wasted Effort**: Units picked after cutoff
- **Picker Utilization**: Active time / total shift time
- **Average Picklist Load**: Items per picklist
- **Zone Balance**: Work distribution across zones

### Bonus Points
- ✨ Priority handling (P1 > P2 > P3...)
- ✨ Runtime efficiency (<5 min for 10K orders)
- ✨ Scalability to 100K+ orders

---

## 🌟 Key Features

### 1. Constraint Programming Approach
- **Exact satisfaction** of all hard constraints
- **Provably optimal** or near-optimal solutions
- Handles complex interdependencies

### 2. Zone-Based Batching
- Respects warehouse layout
- Minimizes picker travel time
- Enforces `pods_per_picklist_in_that_zone`

### 3. Fragile Item Handling
- Automatic detection
- Separate 50kg weight limit
- Special picklist flagging

### 4. Time-Aware Scheduling
- Picker shift windows
- Cutoff time enforcement
- Real-time feasibility checking

### 5. Partial Fulfillment
- Orders can split across picklists
- Every unit counts toward score
- No binary "all or nothing"

---

## 🧪 Testing & Validation

### Unit Tests (Optional)
```bash
cd 02_Optimization_Engine/build
make test
```

### Validation Script
```bash
python 05_Analysis_Insights/validate_output.py
```

Checks:
- ✅ All picklists respect zone constraint
- ✅ No capacity violations
- ✅ CSV format compliance
- ✅ No duplicate SKU assignments

---

## 🐛 Troubleshooting

### Issue: CMake can't find OR-Tools
**Solution:**
```bash
pip install ortools
export ORTOOLS_DIR=$(python -c "import ortools; print(ortools.__path__[0])")
```

### Issue: C++ compilation errors
**Solution:** Ensure C++17 support:
```bash
g++ --version  # Should be 9.0+
# In CMakeLists.txt, verify: set(CMAKE_CXX_STANDARD 17)
```

### Issue: "processed_orders.json not found"
**Solution:** Run preprocessing notebook first:
```bash
cd 01_Data_Preprocessing
jupyter notebook Preprocessing_Notebook.ipynb
```

### Issue: Solver times out
**Solution:** Reduce problem size or add time limit:
```cpp
solver.parameters()->set_max_time_in_seconds(300.0);
```

---

## 📚 References

- [Google OR-Tools Documentation](https://developers.google.com/optimization)
- [CP-SAT Solver Guide](https://developers.google.com/optimization/cp/cp_solver)
- Problem Statement: `docs/problem_statement.md`

---

## 🏆 Team & Contact

**Team Name:** Diamond

**Members:**
- Aayushi Thakre - Data Preprocessing & ML
- Anupam Singh - C++ Optimization Engine
- Prachi - Analysis & Visualization

**Submission Date:** 24th December 2025

---

## 📝 License

This project is developed for the Warehouse Optimization Hackathon by Swiggy at IIITDMJ.

---

## 🙏 Acknowledgments

- Google OR-Tools team for the amazing CP-SAT solver
- Hackathon organizers for the challenging problem

---

**⭐ If you found this useful, please star the repository!**

```bash
# Happy Optimizing! 🚀📦
```
