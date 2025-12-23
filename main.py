"""
Main Pipeline Script for Warehouse Picklist Optimization
Orchestrates the entire optimization workflow
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import INPUT_DATA_FILE, INTERMEDIATE_DIR, OUTPUT_DIR, ANALYSIS_DIR

# Import modules
from importlib import import_module


def run_pipeline(
    input_file: str = None,
    skip_preprocessing: bool = False,
    skip_optimization: bool = False,
    skip_output: bool = False,
    skip_evaluation: bool = False
):
    """
    Run the complete picklist optimization pipeline
    
    Steps:
    1. Data Preprocessing - Load and transform raw order data
    2. Optimization - Create and schedule picklists
    3. Output Generation - Generate CSV files
    4. Evaluation - Calculate and report metrics
    """
    
    print("\n" + "="*70)
    print("🏭 WAREHOUSE PICKLIST OPTIMIZATION PIPELINE")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    if input_file is None:
        input_file = INPUT_DATA_FILE
    
    # Ensure directories exist
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # Step 1: Data Preprocessing
    if not skip_preprocessing:
        print("\n" + "="*50)
        print("📥 STEP 1: DATA PREPROCESSING")
        print("="*50)
        
        from importlib import import_module
        preprocessor = import_module("01_Data_Preprocessing.data_preprocessor")
        
        processed_data = preprocessor.run_preprocessing(
            data_path=input_file,
            output_path=os.path.join(INTERMEDIATE_DIR, "processed_orders.json")
        )
        print("✅ Preprocessing complete\n")
    else:
        print("⏭️  Skipping preprocessing\n")
    
    # Step 2: Optimization
    if not skip_optimization:
        print("\n" + "="*50)
        print("🔧 STEP 2: PICKLIST OPTIMIZATION")
        print("="*50)
        
        optimizer_module = import_module("02_Optimization_Engine.optimizer")
        
        results = optimizer_module.run_optimization(
            processed_data_path=os.path.join(INTERMEDIATE_DIR, "processed_orders.json")
        )
        print("✅ Optimization complete\n")
    else:
        print("⏭️  Skipping optimization\n")
    
    # Step 3: Output Generation
    if not skip_output:
        print("\n" + "="*50)
        print("📤 STEP 3: OUTPUT GENERATION")
        print("="*50)
        
        output_generator = import_module("03_Output_Generator.output_generator")
        
        generated_files = output_generator.generate_all_outputs(
            results_path=os.path.join(INTERMEDIATE_DIR, "optimization_results.json"),
            output_dir=OUTPUT_DIR
        )
        print("✅ Output generation complete\n")
    else:
        print("⏭️  Skipping output generation\n")
    
    # Step 4: Evaluation
    if not skip_evaluation:
        print("\n" + "="*50)
        print("📊 STEP 4: EVALUATION")
        print("="*50)
        
        evaluation = import_module("05_Ananlysis_Insights.evaluation_metrics")
        
        report = evaluation.run_evaluation(
            results_path=os.path.join(INTERMEDIATE_DIR, "optimization_results.json")
        )
        print("✅ Evaluation complete\n")
    else:
        print("⏭️  Skipping evaluation\n")
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE")
    print("="*70)
    print(f"Total Runtime: {elapsed_time:.2f} seconds")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutput Locations:")
    print(f"  📁 Intermediate Files: {INTERMEDIATE_DIR}/")
    print(f"  📁 Picklist CSVs: {OUTPUT_DIR}/")
    print(f"  📁 Analysis Reports: {ANALYSIS_DIR}/")
    print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Warehouse Picklist Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py -i data.xlsx       # Run with custom input file
  python main.py --skip-preprocess  # Skip preprocessing step
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=INPUT_DATA_FILE,
        help="Path to input Excel file"
    )
    
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing step (use existing processed data)"
    )
    
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip optimization step (use existing results)"
    )
    
    parser.add_argument(
        "--skip-output",
        action="store_true",
        help="Skip output generation step"
    )
    
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        input_file=args.input,
        skip_preprocessing=args.skip_preprocess,
        skip_optimization=args.skip_optimization,
        skip_output=args.skip_output,
        skip_evaluation=args.skip_evaluation
    )


if __name__ == "__main__":
    main()
