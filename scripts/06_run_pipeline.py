"""
Master Pipeline Script
Execute complete steel production analysis workflow
"""

import sys
import os
import time
import importlib.util
from pathlib import Path

def print_banner(text: str):
    """Print formatted banner."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def load_module_from_file(file_path: str, module_name: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_script(script_name: str, description: str):
    """Run a script module and handle errors."""
    print_banner(f"PHASE: {description}")
    start_time = time.time()
    
    try:
        # Get the scripts directory
        base_path = Path(__file__).parent
        script_path = base_path / f"{script_name}.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Load and execute the module
        module = load_module_from_file(str(script_path), script_name)
        
        # Execute main function
        module.main()
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"\n✗ Error in {description}:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Execute complete analysis pipeline."""
    print_banner("STEEL PRODUCTION ANALYSIS PIPELINE")
    print("Starting complete analysis workflow...")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    pipeline_start = time.time()
    
    # Define pipeline stages
    stages = [
        ("01_data_loading", "Data Loading & Statistics"),
        ("02_data_preprocessing", "Data Preprocessing & Validation"),
        ("03_eda", "Exploratory Data Analysis"),
        ("04_model_training", "Model Training & Evaluation"),
        ("05_results_analysis", "Results Analysis & Visualization")
    ]
    
    # Execute each stage
    results = []
    for script_name, description in stages:
        success = run_script(script_name, description)
        results.append((description, success))
        
        if not success:
            print("\n⚠ Pipeline stopped due to error")
            break
    
    # Print summary
    print_banner("PIPELINE EXECUTION SUMMARY")
    
    for description, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    total_time = time.time() - pipeline_start
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if all(success for _, success in results):
        print("\n🎉 Pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Review results in 'results/' directory")
        print("2. Check visualizations in 'figures/' directory")
        print("3. Read analysis summary in 'results/analysis_summary.txt'")
        print("4. Create your project report using the generated outputs")
    else:
        print("\n⚠ Pipeline completed with errors. Please check the logs above.")

if __name__ == "__main__":
    main()