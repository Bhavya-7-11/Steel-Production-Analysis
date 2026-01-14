"""
Path Configuration Update Script
Run this to update all paths to Mac filesystem
"""

import os
import re

# New base path for Mac
MAC_BASE_PATH = "/Users/bhavyabansal/Downloads/steel_production_analysis"

def update_script_paths(script_path: str):
    """Update paths in a Python script file"""
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace Windows-style paths
    content = re.sub(
        r'C:\\Projects\\Assignment1\\steel_production_analysis',
        MAC_BASE_PATH,
        content
    )
    
    # Update base_path detection to use absolute path
    old_pattern = r"base_path = os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)"
    new_pattern = f'base_path = "{MAC_BASE_PATH}"'
    content = re.sub(old_pattern, new_pattern, content)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated: {script_path}")

def main():
    """Update all script files"""
    scripts_dir = os.path.join(MAC_BASE_PATH, "scripts")
    
    scripts = [
        "01_data_loading.py",
        "02_data_preprocessing.py",
        "03_eda.py",
        "04_model_training.py",
        "05_results_analysis.py",
        "06_run_pipeline.py"
    ]
    
    print("Updating script paths for Mac filesystem...")
    print(f"Base path: {MAC_BASE_PATH}\n")
    
    for script in scripts:
        script_path = os.path.join(scripts_dir, script)
        if os.path.exists(script_path):
            update_script_paths(script_path)
        else:
            print(f"⚠ Not found: {script_path}")
    
    print("\n✓ All paths updated successfully!")
    print(f"\nYou can now run:")
    print(f"cd {scripts_dir}")
    print(f"python 06_run_pipeline.py")

if __name__ == "__main__":
    main()