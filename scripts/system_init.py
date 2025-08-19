#!/usr/bin/env python3
"""System Initialization Script"""
import sys
from pathlib import Path

def check_system():
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
        
    # Check directories
    required_dirs = ['src/system', 'configs', 'models', 'data/cleaned']
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
            
    print("✅ System ready")
    return True

if __name__ == "__main__":
    sys.exit(0 if check_system() else 1)
