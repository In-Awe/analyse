#!/usr/bin/env python3
"""Phase 5 Integrity Checker - Validates implementation"""
import os
import json
import sys
from pathlib import Path
from datetime import datetime

class Phase5IntegrityChecker:
    def __init__(self):
        self.project_root = Path(".")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "phase": 5,
            "checks": {},
            "missing_files": [],
            "errors": [],
            "warnings": []
        }
    
    def check_phase5(self):
        """Check Phase 5 specific files"""
        print("\n[PHASE 5] Checking High-Performance Architecture...")
        print("-" * 40)
        
        required_files = {
            # Core modules
            "src/system/market_data.py": "file",
            "src/system/signal_service.py": "file",
            "src/system/risk_manager.py": "file",
            "src/system/executor.py": "file",
            "src/system/rate_limiter.py": "file",
            "src/system/monitoring.py": "file",
            
            # Documentation
            "docs/architecture.md": "file",
            "docs/binance_integration.md": "file",
            
            # Scripts
            "scripts/replay_market.py": "file",
            "scripts/main.py": "file",
            "scripts/system_init.py": "file",
            "scripts/generate_test_data.py": "file",
            
            # Configs
            "configs/risk_limits.yaml": "file",
            "configs/features.yaml": "file",
            "configs/trade_logic.yaml": "file",
            "configs/monitoring.yaml": "file",
            
            # Tests
            "tests/unit/test_market_data.py": "file",
            "tests/unit/test_risk_manager.py": "file",
            "tests/integration/test_integration.py": "file",
            
            # Docker
            "Dockerfile": "file",
            "docker-compose.yml": "file",
            
            # CI/CD
            ".github/workflows/lint.yml": "file",
            ".github/workflows/tests.yml": "file",
            
            # Environment
            "requirements.txt": "file",
            ".env.example": "file",
            ".gitignore": "file"
        }
        
        total = len(required_files)
        present = 0
        
        for file_path, file_type in required_files.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                present += 1
                print(f"  ✓ {file_path}")
            else:
                self.results["missing_files"].append(file_path)
                print(f"  ✗ {file_path} [MISSING]")
        
        completion = (present / total * 100) if total > 0 else 0
        print(f"\n  Phase 5 Completion: {completion:.1f}% ({present}/{total} files)")
        
        self.results["checks"]["Phase 5"] = {
            "total": total,
            "present": present,
            "missing": total - present,
            "completion_pct": completion
        }
        
        # Save report
        report_path = Path("artifacts/phase5_integrity_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nReport saved to: {report_path}")
        
        if completion < 30:
            print("\n⚠️  CRITICAL: Phase 5 is severely incomplete!")
            print("   Run the build script again.")
            return False
        elif completion < 90:
            print("\n⚠️  WARNING: Phase 5 is partially complete.")
            print("   Some files are missing. Check the report for details.")
            return False
        else:
            print("\n✅ SUCCESS: Phase 5 is complete!")
            print("   Ready for Phase 6: Out-of-Sample Testing")
            return True
    
    def run(self):
        print("=" * 80)
        print("PHASE 5 INTEGRITY CHECK")
        print("=" * 80)
        return self.check_phase5()

if __name__ == "__main__":
    checker = Phase5IntegrityChecker()
    success = checker.run()
    sys.exit(0 if success else 1)
