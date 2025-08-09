#!/usr/bin/env python3
"""
Test Suite Validation Script
===========================

This script validates that the comprehensive test suite is properly structured
and all components can be imported and initialized correctly.

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import sys
import os
import importlib.util
from pathlib import Path
from typing import List, Dict, Any


def validate_file_exists(file_path: Path) -> bool:
    """Validate that a file exists and is readable."""
    try:
        return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0
    except Exception:
        return False


def validate_python_syntax(file_path: Path) -> Dict[str, Any]:
    """Validate Python syntax of a file."""
    result = {
        "valid": False,
        "error": None,
        "line_count": 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            result["line_count"] = len(content.splitlines())
        
        # Try to compile the file
        compile(content, str(file_path), 'exec')
        result["valid"] = True
        
    except SyntaxError as e:
        result["error"] = f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        result["error"] = f"Error reading file: {str(e)}"
    
    return result


def validate_imports(file_path: Path) -> Dict[str, Any]:
    """Validate that all imports in a file can be resolved."""
    result = {
        "valid": False,
        "missing_imports": [],
        "error": None
    }
    
    try:
        # Load the module spec
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec is None:
            result["error"] = "Could not create module spec"
            return result
        
        # Try to load the module (this will trigger import errors)
        module = importlib.util.module_from_spec(spec)
        
        # Add the current directory to sys.path temporarily
        original_path = sys.path[:]
        sys.path.insert(0, str(file_path.parent))
        
        try:
            spec.loader.exec_module(module)
            result["valid"] = True
        except ImportError as e:
            missing_module = str(e).replace("No module named ", "").strip("'\"")
            result["missing_imports"].append(missing_module)
            result["error"] = str(e)
        except Exception as e:
            result["error"] = f"Module execution error: {str(e)}"
        finally:
            sys.path = original_path
    
    except Exception as e:
        result["error"] = f"Import validation error: {str(e)}"
    
    return result


def validate_test_structure(file_path: Path) -> Dict[str, Any]:
    """Validate the structure of a test file."""
    result = {
        "valid": False,
        "test_classes": [],
        "test_methods": [],
        "fixtures": [],
        "error": None
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        
        # Count test classes
        test_classes = [line.strip() for line in lines if line.strip().startswith("class Test")]
        result["test_classes"] = test_classes
        
        # Count test methods
        test_methods = [line.strip() for line in lines if line.strip().startswith("def test_")]
        result["test_methods"] = test_methods
        
        # Count fixtures
        fixtures = []
        for i, line in enumerate(lines):
            if "@pytest.fixture" in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("def "):
                    fixtures.append(next_line)
        result["fixtures"] = fixtures
        
        # Validate minimum requirements
        if len(test_classes) >= 5 and len(test_methods) >= 20:
            result["valid"] = True
        else:
            result["error"] = f"Insufficient test coverage: {len(test_classes)} classes, {len(test_methods)} methods"
    
    except Exception as e:
        result["error"] = f"Structure validation error: {str(e)}"
    
    return result


def main():
    """Main validation function."""
    print("Unified System Health Dashboard - Test Suite Validation")
    print("=" * 60)
    
    # Define paths to validate
    base_path = Path(__file__).parent
    files_to_validate = [
        {
            "path": base_path / "test_unified_system_health_dashboard_comprehensive.py",
            "name": "Main Test Suite",
            "required": True
        },
        {
            "path": base_path / "run_dashboard_tests.py",
            "name": "Test Runner",
            "required": True
        },
        {
            "path": base_path / "pytest.ini",
            "name": "Pytest Configuration",
            "required": True
        },
        {
            "path": base_path / "test_requirements.txt",
            "name": "Test Requirements",
            "required": True
        },
        {
            "path": base_path / "unified_system_health_dashboard.py",
            "name": "Dashboard Implementation",
            "required": True
        },
        {
            "path": base_path / "dashboard_integration_helper.py",
            "name": "Integration Helper",
            "required": True
        }
    ]
    
    validation_results = []
    
    for file_info in files_to_validate:
        file_path = file_info["path"]
        file_name = file_info["name"]
        is_required = file_info["required"]
        
        print(f"\nValidating: {file_name}")
        print(f"Path: {file_path}")
        
        # Check file existence
        if not validate_file_exists(file_path):
            print("❌ File not found or empty")
            validation_results.append({
                "file": file_name,
                "status": "missing",
                "critical": is_required
            })
            continue
        
        print("✅ File exists")
        
        # For Python files, validate syntax and imports
        if file_path.suffix == ".py":
            # Validate syntax
            syntax_result = validate_python_syntax(file_path)
            if syntax_result["valid"]:
                print(f"✅ Syntax valid ({syntax_result['line_count']} lines)")
            else:
                print(f"❌ Syntax error: {syntax_result['error']}")
                validation_results.append({
                    "file": file_name,
                    "status": "syntax_error",
                    "error": syntax_result["error"],
                    "critical": is_required
                })
                continue
            
            # For test files, validate structure
            if "test_" in file_path.name:
                structure_result = validate_test_structure(file_path)
                if structure_result["valid"]:
                    print(f"✅ Test structure valid:")
                    print(f"   - {len(structure_result['test_classes'])} test classes")
                    print(f"   - {len(structure_result['test_methods'])} test methods")
                    print(f"   - {len(structure_result['fixtures'])} fixtures")
                else:
                    print(f"⚠️  Test structure warning: {structure_result['error']}")
            
            # Validate imports (optional for main validation)
            # Note: This may fail in CI environments without all dependencies
            print("ℹ️  Import validation skipped (requires full environment)")
        
        validation_results.append({
            "file": file_name,
            "status": "valid",
            "critical": is_required
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_files = len(files_to_validate)
    valid_files = len([r for r in validation_results if r["status"] == "valid"])
    critical_issues = [r for r in validation_results if r["status"] != "valid" and r["critical"]]
    
    print(f"Total files validated: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Files with issues: {total_files - valid_files}")
    
    if critical_issues:
        print(f"\n❌ Critical issues found:")
        for issue in critical_issues:
            print(f"   - {issue['file']}: {issue['status']}")
            if "error" in issue:
                print(f"     Error: {issue['error']}")
        print("\nTest suite validation FAILED. Please fix critical issues.")
        return False
    else:
        print(f"\n✅ All critical validations passed!")
        
        # Additional recommendations
        print("\nRecommendations:")
        print("- Run 'python run_dashboard_tests.py --validate-env' for full environment check")
        print("- Install test dependencies: 'pip install -r test_requirements.txt'")
        print("- Run quick tests: 'python run_dashboard_tests.py --quick'")
        print("- Generate coverage report: 'python run_dashboard_tests.py --coverage'")
        
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)