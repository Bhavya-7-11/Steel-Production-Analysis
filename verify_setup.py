"""
Setup Verification Script
Run this before executing the main pipeline to verify everything is ready
"""

import sys
import os
from pathlib import Path

def print_status(message, status):
    """Print formatted status message."""
    icon = "✓" if status else "✗"
    color = "green" if status else "red"
    print(f"{icon} {message}")

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    compatible = version.major == 3 and version.minor >= 8
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro}", compatible)
    return compatible

def check_packages():
    """Check if all required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'tensorflow', 'jupyter'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"Package '{package}' installed", True)
        except ImportError:
            print_status(f"Package '{package}' NOT installed", False)
            all_installed = False
    
    return all_installed

def check_directories():
    """Check if required directories exist."""
    base_path = Path(__file__).parent
    required_dirs = ['data', 'scripts', 'results', 'figures']
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        exists = dir_path.exists()
        print_status(f"Directory '{dir_name}/' exists", exists)
        if not exists:
            all_exist = False
    
    return all_exist

def check_data_files():
    """Check if data files exist."""
    base_path = Path(__file__).parent
    data_files = [
        'data/normalized_train_data.csv',
        'data/normalized_test_data.csv'
    ]
    
    all_exist = True
    for file_path in data_files:
        full_path = base_path / file_path
        exists = full_path.exists()
        if exists:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print_status(f"Data file '{file_path}' exists ({size_mb:.2f} MB)", True)
        else:
            print_status(f"Data file '{file_path}' NOT FOUND", False)
            all_exist = False
    
    return all_exist

def check_scripts():
    """Check if all script files exist."""
    base_path = Path(__file__).parent
    script_files = [
        'scripts/01_data_loading.py',
        'scripts/02_data_preprocessing.py',
        'scripts/03_eda.py',
        'scripts/04_model_training.py',
        'scripts/05_results_analysis.py',
        'scripts/06_run_pipeline.py'
    ]
    
    all_exist = True
    for script_path in script_files:
        full_path = base_path / script_path
        exists = full_path.exists()
        print_status(f"Script '{script_path}' exists", exists)
        if not exists:
            all_exist = False
    
    return all_exist

def check_memory():
    """Check available system memory."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        sufficient = available_gb >= 2
        print_status(f"Available RAM: {available_gb:.2f} GB (recommended: 4+ GB)", sufficient)
        return sufficient
    except ImportError:
        print_status("Could not check memory (psutil not installed)", None)
        return True

def check_disk_space():
    """Check available disk space."""
    try:
        import shutil
        base_path = Path(__file__).parent
        total, used, free = shutil.disk_usage(base_path)
        free_gb = free / (1024**3)
        sufficient = free_gb >= 1
        print_status(f"Available disk space: {free_gb:.2f} GB", sufficient)
        return sufficient
    except Exception:
        print_status("Could not check disk space", None)
        return True

def print_recommendations(checks):
    """Print recommendations based on check results."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if not checks['python']:
        print("⚠ CRITICAL: Upgrade Python to version 3.8 or higher")
    
    if not checks['packages']:
        print("⚠ Install missing packages: pip install -r requirements.txt")
    
    if not checks['directories']:
        print("⚠ Create missing directories:")
        print("  mkdir scripts results figures results\\model_predictions")
    
    if not checks['data']:
        print("⚠ CRITICAL: Data files not found!")
        print("  Download data and place in 'data/' directory")
    
    if not checks['scripts']:
        print("⚠ Create missing script files from provided code")
    
    if all(checks.values()):
        print("\n✓ All checks passed! You're ready to run the pipeline.")
        print("\nNext step: python scripts/06_run_pipeline.py")
    else:
        print("\n⚠ Some checks failed. Please fix issues above before running.")

def main():
    """Run all verification checks."""
    print("="*60)
    print("STEEL PRODUCTION ANALYSIS - SETUP VERIFICATION")
    print("="*60)
    print()
    
    checks = {}
    
    print("--- Python Environment ---")
    checks['python'] = check_python_version()
    print()
    
    print("--- Required Packages ---")
    checks['packages'] = check_packages()
    print()
    
    print("--- Project Structure ---")
    checks['directories'] = check_directories()
    print()
    
    print("--- Data Files ---")
    checks['data'] = check_data_files()
    print()
    
    print("--- Script Files ---")
    checks['scripts'] = check_scripts()
    print()
    
    print("--- System Resources ---")
    checks['memory'] = check_memory()
    checks['disk'] = check_disk_space()
    
    print_recommendations(checks)
    
    # Return exit code
    return 0 if all(checks.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)