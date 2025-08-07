#!/usr/bin/env python3
"""
Master Comprehensive Error Handling Validation for Clinical Metabolomics Oracle.

This is the master validation script that runs all error handling tests and provides
a final production readiness assessment. It integrates all testing components:

1. End-to-End Error Handling Validation
2. Logging System Validation  
3. Performance Under Stress Testing
4. Memory Leak Detection
5. Circuit Breaker Behavior Validation
6. Recovery Mechanism Testing
7. Resource Exhaustion Simulation

The script provides a comprehensive report that can be used to determine if the
error handling system is ready for production deployment.

Usage:
    python comprehensive_error_handling_master_validation.py [options]

Options:
    --quick              Run abbreviated tests (faster execution)
    --output-dir DIR     Directory for test results (default: ./master_validation_results)
    --verbose           Enable verbose logging
    --skip-slow         Skip slow/long-running tests
    --parallel          Run tests in parallel where possible
    --report-format     Report format: json, html, text (default: text)

Exit Codes:
    0: All validations passed - system ready for production
    1: Some validations failed - issues need to be addressed
    2: Critical failures - system not ready for production

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import test modules
sys.path.append(str(Path(__file__).parent))

try:
    from test_error_handling_e2e_validation import (
        ErrorHandlingE2EValidator, ComprehensiveErrorHandlingTestRunner
    )
    from test_logging_validation import LoggingValidationScenarios
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Please ensure all test dependencies are available.")
    sys.exit(2)


class MasterValidationResult:
    """Represents the overall validation result."""
    
    def __init__(self):
        """Initialize result container."""
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.validation_components: Dict[str, Dict[str, Any]] = {}
        self.overall_status = "UNKNOWN"
        self.critical_failures: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        self.production_readiness_score = 0.0
        self.execution_summary: Dict[str, Any] = {}
    
    @property
    def duration_seconds(self) -> float:
        """Get total validation duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def add_component_result(self, component: str, result: Dict[str, Any]) -> None:
        """Add result for a validation component."""
        self.validation_components[component] = result
    
    def calculate_production_readiness(self) -> None:
        """Calculate overall production readiness score."""
        if not self.validation_components:
            self.production_readiness_score = 0.0
            self.overall_status = "FAIL"
            return
        
        # Component weights (higher weight = more critical for production readiness)
        component_weights = {
            "e2e_validation": 0.35,          # Core functionality - highest weight
            "logging_validation": 0.20,      # Critical for debugging
            "memory_leak_detection": 0.25,   # Critical for stability
            "circuit_breaker_validation": 0.10,  # Important for resilience
            "performance_validation": 0.10   # Important but less critical
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for component, result in self.validation_components.items():
            weight = component_weights.get(component, 0.1)  # Default weight for unknown components
            
            # Get component score
            if "overall_pass_rate" in result:
                component_score = result["overall_pass_rate"]
            elif "overall_status" in result:
                component_score = 1.0 if result["overall_status"] == "PASS" else 0.0
            elif "passed" in result:
                component_score = 1.0 if result["passed"] else 0.0
            else:
                component_score = 0.5  # Unknown status
            
            total_weighted_score += component_score * weight
            total_weight += weight
        
        self.production_readiness_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine overall status
        if self.production_readiness_score >= 0.90:
            self.overall_status = "PRODUCTION_READY"
        elif self.production_readiness_score >= 0.80:
            self.overall_status = "CONDITIONALLY_READY"  # Ready with monitoring
        elif self.production_readiness_score >= 0.60:
            self.overall_status = "NEEDS_IMPROVEMENT"
        else:
            self.overall_status = "NOT_READY"
        
        # Check for critical failures
        critical_threshold = 0.7
        for component, result in self.validation_components.items():
            component_score = self._get_component_score(result)
            
            if component_score < critical_threshold:
                if component in ["e2e_validation", "memory_leak_detection"]:
                    # These are critical components
                    self.critical_failures.append(f"Critical component '{component}' failed validation")
                    if self.overall_status in ["PRODUCTION_READY", "CONDITIONALLY_READY"]:
                        self.overall_status = "NEEDS_IMPROVEMENT"
                else:
                    self.warnings.append(f"Component '{component}' has low validation score")
    
    def _get_component_score(self, result: Dict[str, Any]) -> float:
        """Extract score from component result."""
        if "overall_pass_rate" in result:
            return result["overall_pass_rate"]
        elif "overall_status" in result:
            return 1.0 if result["overall_status"] == "PASS" else 0.0
        elif "passed" in result:
            return 1.0 if result["passed"] else 0.0
        return 0.5
    
    def finalize(self) -> None:
        """Finalize the validation result."""
        self.end_time = datetime.now()
        self.calculate_production_readiness()
        
        # Generate execution summary
        self.execution_summary = {
            "total_duration_seconds": self.duration_seconds,
            "components_tested": len(self.validation_components),
            "production_readiness_score": self.production_readiness_score,
            "overall_status": self.overall_status,
            "critical_failures_count": len(self.critical_failures),
            "warnings_count": len(self.warnings)
        }


class MasterErrorHandlingValidator:
    """Master validator that orchestrates all error handling tests."""
    
    def __init__(self, 
                 output_dir: Path,
                 quick_mode: bool = False,
                 skip_slow: bool = False,
                 parallel: bool = False,
                 logger: Optional[logging.Logger] = None):
        """Initialize master validator."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.quick_mode = quick_mode
        self.skip_slow = skip_slow
        self.parallel = parallel
        
        self.logger = logger or self._setup_logging()
        self.result = MasterValidationResult()
        
        # Test component instances will be created with temporary directories
        self.temp_dirs: List[Path] = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for master validator."""
        logger = logging.getLogger("MasterErrorHandlingValidator")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / f"master_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def run_e2e_validation(self) -> Dict[str, Any]:
        """Run end-to-end error handling validation."""
        self.logger.info("Starting E2E error handling validation...")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                self.temp_dirs.append(temp_path)
                
                validator = ErrorHandlingE2EValidator(temp_path, self.logger)
                
                # Determine test durations based on mode
                if self.quick_mode:
                    durations = {
                        "basic_error_injection": 20.0,
                        "resource_pressure": 30.0,
                        "circuit_breaker": 40.0,
                        "checkpoint_corruption": 20.0,
                        "memory_leak_detection": 60.0 if not self.skip_slow else 30.0
                    }
                else:
                    durations = {
                        "basic_error_injection": 45.0,
                        "resource_pressure": 60.0,
                        "circuit_breaker": 90.0,
                        "checkpoint_corruption": 45.0,
                        "memory_leak_detection": 120.0 if not self.skip_slow else 60.0
                    }
                
                results = {}
                
                # Run scenarios
                scenarios_to_run = [
                    ("basic_error_injection", validator.run_basic_error_injection_scenario),
                    ("resource_pressure", validator.run_resource_pressure_scenario),
                    ("circuit_breaker", validator.run_circuit_breaker_scenario),
                    ("checkpoint_corruption", validator.run_checkpoint_corruption_scenario)
                ]
                
                if not self.skip_slow:
                    scenarios_to_run.append(
                        ("memory_leak_detection", validator.run_memory_leak_detection_scenario)
                    )
                
                for scenario_name, scenario_func in scenarios_to_run:
                    try:
                        self.logger.info(f"Running {scenario_name} scenario...")
                        duration = durations.get(scenario_name, 60.0)
                        result = scenario_func(duration)
                        results[scenario_name] = result
                        
                        status = result.get("validation_status", "UNKNOWN")
                        self.logger.info(f"{scenario_name}: {status}")
                        
                    except Exception as e:
                        self.logger.error(f"Error in {scenario_name}: {e}")
                        results[scenario_name] = {
                            "scenario": scenario_name,
                            "validation_status": "ERROR",
                            "error": str(e)
                        }
                
                # Generate comprehensive report
                validator.validation_results = results
                report = validator.generate_comprehensive_report()
                
                return report
                
        except Exception as e:
            self.logger.error(f"E2E validation failed: {e}")
            return {
                "validation_report": {
                    "overall_status": "ERROR",
                    "error_message": str(e)
                },
                "scenario_results": {}
            }
    
    def run_logging_validation(self) -> Dict[str, Any]:
        """Run logging system validation."""
        self.logger.info("Starting logging system validation...")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                self.temp_dirs.append(temp_path)
                
                scenarios = LoggingValidationScenarios(temp_path)
                report = scenarios.generate_logging_validation_report()
                
                status = report["logging_validation_report"]["overall_status"]
                self.logger.info(f"Logging validation: {status}")
                
                return report
                
        except Exception as e:
            self.logger.error(f"Logging validation failed: {e}")
            return {
                "logging_validation_report": {
                    "overall_status": "ERROR",
                    "error_message": str(e)
                },
                "scenario_results": {}
            }
    
    def run_performance_stress_test(self) -> Dict[str, Any]:
        """Run performance validation under stress conditions."""
        self.logger.info("Starting performance stress test...")
        
        try:
            import psutil
            import threading
            
            # Monitor system resources during test
            resource_samples = []
            monitoring_active = True
            
            def monitor_resources():
                while monitoring_active:
                    try:
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / (1024 * 1024)
                        cpu_percent = process.cpu_percent()
                        
                        resource_samples.append({
                            "timestamp": time.time(),
                            "memory_mb": memory_mb,
                            "cpu_percent": cpu_percent
                        })
                        
                        time.sleep(1.0)  # Sample every second
                    except Exception:
                        break
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
            monitor_thread.start()
            
            start_time = time.time()
            
            # Run a simplified stress test
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                validator = ErrorHandlingE2EValidator(temp_path, self.logger)
                
                # Run multiple scenarios in quick succession to stress system
                stress_duration = 30.0 if self.quick_mode else 60.0
                
                results = []
                end_time = start_time + stress_duration
                
                while time.time() < end_time:
                    try:
                        # Run a quick basic error injection
                        result = validator.run_basic_error_injection_scenario(10.0)
                        results.append(result)
                        
                        # Small delay between runs
                        time.sleep(2.0)
                        
                    except Exception as e:
                        self.logger.warning(f"Error during stress test iteration: {e}")
                        break
            
            # Stop monitoring
            monitoring_active = False
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)
            
            # Analyze results
            total_duration = time.time() - start_time
            successful_runs = len([r for r in results if r.get("validation_status") == "PASS"])
            total_runs = len(results)
            
            # Analyze resource usage
            if resource_samples:
                memory_values = [s["memory_mb"] for s in resource_samples]
                cpu_values = [s["cpu_percent"] for s in resource_samples]
                
                avg_memory = sum(memory_values) / len(memory_values)
                max_memory = max(memory_values)
                avg_cpu = sum(cpu_values) / len(cpu_values)
                max_cpu = max(cpu_values)
                
                # Memory leak detection (simple)
                if len(memory_values) > 10:
                    initial_memory = sum(memory_values[:5]) / 5  # First 5 samples
                    final_memory = sum(memory_values[-5:]) / 5   # Last 5 samples
                    memory_increase = final_memory - initial_memory
                    memory_leak_suspected = memory_increase > 50.0  # More than 50MB increase
                else:
                    memory_leak_suspected = False
            else:
                avg_memory = max_memory = avg_cpu = max_cpu = 0
                memory_leak_suspected = False
            
            # Determine if performance test passed
            success_rate = successful_runs / total_runs if total_runs > 0 else 0
            performance_acceptable = (
                success_rate >= 0.7 and
                max_memory < 2048 and  # Less than 2GB
                not memory_leak_suspected
            )
            
            return {
                "performance_stress_test": {
                    "overall_status": "PASS" if performance_acceptable else "FAIL",
                    "duration_seconds": total_duration,
                    "total_runs": total_runs,
                    "successful_runs": successful_runs,
                    "success_rate": success_rate,
                    "resource_usage": {
                        "avg_memory_mb": avg_memory,
                        "max_memory_mb": max_memory,
                        "avg_cpu_percent": avg_cpu,
                        "max_cpu_percent": max_cpu
                    },
                    "memory_leak_suspected": memory_leak_suspected,
                    "performance_acceptable": performance_acceptable
                }
            }
            
        except Exception as e:
            self.logger.error(f"Performance stress test failed: {e}")
            return {
                "performance_stress_test": {
                    "overall_status": "ERROR",
                    "error_message": str(e)
                }
            }
    
    def run_parallel_validations(self) -> Dict[str, Any]:
        """Run validations in parallel where possible."""
        self.logger.info("Running validations in parallel...")
        
        results = {}
        
        # Define validation functions that can run in parallel
        validation_functions = [
            ("logging_validation", self.run_logging_validation),
            ("performance_stress_test", self.run_performance_stress_test)
        ]
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit parallel tasks
            future_to_name = {
                executor.submit(func): name for name, func in validation_functions
            }
            
            # E2E validation runs separately as it's the most resource intensive
            e2e_result = self.run_e2e_validation()
            results["e2e_validation"] = e2e_result
            
            # Collect parallel results
            for future in concurrent.futures.as_completed(future_to_name, timeout=300):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[name] = result
                except Exception as e:
                    self.logger.error(f"Parallel validation {name} failed: {e}")
                    results[name] = {
                        "overall_status": "ERROR",
                        "error_message": str(e)
                    }
        
        return results
    
    def run_sequential_validations(self) -> Dict[str, Any]:
        """Run all validations sequentially."""
        self.logger.info("Running validations sequentially...")
        
        results = {}
        
        # Run validations in order of importance
        validation_sequence = [
            ("e2e_validation", self.run_e2e_validation),
            ("logging_validation", self.run_logging_validation),
            ("performance_stress_test", self.run_performance_stress_test)
        ]
        
        for name, func in validation_sequence:
            try:
                self.logger.info(f"Starting {name}...")
                result = func()
                results[name] = result
                
                status = self._extract_status(result)
                self.logger.info(f"Completed {name}: {status}")
                
            except Exception as e:
                self.logger.error(f"Validation {name} failed: {e}")
                results[name] = {
                    "overall_status": "ERROR",
                    "error_message": str(e)
                }
        
        return results
    
    def _extract_status(self, result: Dict[str, Any]) -> str:
        """Extract status from validation result."""
        if "validation_report" in result:
            return result["validation_report"].get("overall_status", "UNKNOWN")
        elif "logging_validation_report" in result:
            return result["logging_validation_report"].get("overall_status", "UNKNOWN")
        elif "performance_stress_test" in result:
            return result["performance_stress_test"].get("overall_status", "UNKNOWN")
        else:
            return result.get("overall_status", "UNKNOWN")
    
    def run_comprehensive_validation(self) -> MasterValidationResult:
        """Run comprehensive validation of all error handling components."""
        self.logger.info("Starting comprehensive error handling validation...")
        self.logger.info(f"Mode: {'Quick' if self.quick_mode else 'Full'}")
        self.logger.info(f"Parallel execution: {self.parallel}")
        self.logger.info(f"Skip slow tests: {self.skip_slow}")
        
        try:
            # Run validations
            if self.parallel:
                validation_results = self.run_parallel_validations()
            else:
                validation_results = self.run_sequential_validations()
            
            # Add results to master result
            for component, result in validation_results.items():
                self.result.add_component_result(component, result)
            
            # Generate recommendations based on results
            self.result.recommendations = self._generate_master_recommendations(validation_results)
            
        except Exception as e:
            self.logger.error(f"Critical error during validation: {e}")
            self.result.critical_failures.append(f"Critical validation error: {e}")
            
        finally:
            self.result.finalize()
        
        return self.result
    
    def _generate_master_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate master recommendations based on all validation results."""
        recommendations = []
        
        # Analyze E2E validation
        e2e_result = results.get("e2e_validation", {})
        if e2e_result.get("validation_report", {}).get("overall_status") != "PASS":
            recommendations.append("E2E error handling validation failed - review error handling implementation")
        
        # Analyze logging validation
        logging_result = results.get("logging_validation", {})
        if logging_result.get("logging_validation_report", {}).get("overall_status") != "PASS":
            recommendations.append("Logging validation failed - improve structured logging and correlation ID usage")
        
        # Analyze performance results
        perf_result = results.get("performance_stress_test", {})
        if not perf_result.get("performance_stress_test", {}).get("performance_acceptable", False):
            recommendations.append("Performance under stress is not acceptable - optimize resource usage")
        
        # Check for memory leaks
        if perf_result.get("performance_stress_test", {}).get("memory_leak_suspected", False):
            recommendations.append("CRITICAL: Potential memory leak detected - investigate memory management")
        
        # Add production readiness recommendations
        if self.result.production_readiness_score < 0.9:
            if self.result.production_readiness_score >= 0.8:
                recommendations.append("System is conditionally ready - deploy with enhanced monitoring")
            else:
                recommendations.append("System needs improvement before production deployment")
        
        return recommendations
    
    def save_results(self, format_type: str = "json") -> Path:
        """Save validation results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == "json":
            filename = f"master_validation_results_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert result to dictionary for JSON serialization
            result_dict = {
                "master_validation_result": {
                    "start_time": self.result.start_time.isoformat(),
                    "end_time": self.result.end_time.isoformat() if self.result.end_time else None,
                    "duration_seconds": self.result.duration_seconds,
                    "overall_status": self.result.overall_status,
                    "production_readiness_score": self.result.production_readiness_score,
                    "critical_failures": self.result.critical_failures,
                    "warnings": self.result.warnings,
                    "recommendations": self.result.recommendations,
                    "execution_summary": self.result.execution_summary
                },
                "component_results": self.result.validation_components
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
        
        else:  # Default to text format
            filename = f"master_validation_results_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(self._generate_text_report())
        
        return filepath
    
    def _generate_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = [
            "="*80,
            "COMPREHENSIVE ERROR HANDLING VALIDATION REPORT",
            "="*80,
            "",
            f"Validation completed: {self.result.end_time}",
            f"Duration: {self.result.duration_seconds:.1f} seconds",
            f"Overall Status: {self.result.overall_status}",
            f"Production Readiness Score: {self.result.production_readiness_score:.1%}",
            ""
        ]
        
        # Component results
        lines.append("COMPONENT VALIDATION RESULTS:")
        lines.append("-" * 50)
        
        for component, result in self.result.validation_components.items():
            status = self._extract_status(result)
            lines.append(f"{component:<25} {status}")
        
        lines.append("")
        
        # Critical failures
        if self.result.critical_failures:
            lines.append("CRITICAL FAILURES:")
            lines.append("-" * 30)
            for failure in self.result.critical_failures:
                lines.append(f"  - {failure}")
            lines.append("")
        
        # Warnings
        if self.result.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 20)
            for warning in self.result.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        # Recommendations
        if self.result.recommendations:
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 30)
            for i, rec in enumerate(self.result.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        # Final assessment
        lines.extend([
            "="*80,
            "FINAL ASSESSMENT",
            "="*80
        ])
        
        if self.result.overall_status == "PRODUCTION_READY":
            lines.append("✓ SYSTEM IS READY FOR PRODUCTION DEPLOYMENT")
            lines.append("  All critical validations passed. The error handling system")
            lines.append("  demonstrates robust behavior under various failure conditions.")
        elif self.result.overall_status == "CONDITIONALLY_READY":
            lines.append("⚠ SYSTEM IS CONDITIONALLY READY FOR PRODUCTION")
            lines.append("  Most validations passed, but deploy with enhanced monitoring")
            lines.append("  and be prepared to address any issues that arise.")
        elif self.result.overall_status == "NEEDS_IMPROVEMENT":
            lines.append("⚠ SYSTEM NEEDS IMPROVEMENT BEFORE PRODUCTION")
            lines.append("  Several validations failed. Address the identified issues")
            lines.append("  before considering production deployment.")
        else:
            lines.append("✗ SYSTEM IS NOT READY FOR PRODUCTION")
            lines.append("  Critical validations failed. Significant improvements")
            lines.append("  required before production deployment is advisable.")
        
        lines.append("="*80)
        
        return "\n".join(lines)


def main():
    """Main entry point for master validation."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive error handling validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--quick", action="store_true",
        help="Run abbreviated tests for faster execution"
    )
    
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./master_validation_results"),
        help="Directory for test results"
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--skip-slow", action="store_true",
        help="Skip slow/long-running tests"
    )
    
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run tests in parallel where possible"
    )
    
    parser.add_argument(
        "--report-format", choices=["json", "text"], default="text",
        help="Report output format"
    )
    
    args = parser.parse_args()
    
    # Create master validator
    validator = MasterErrorHandlingValidator(
        output_dir=args.output_dir,
        quick_mode=args.quick,
        skip_slow=args.skip_slow,
        parallel=args.parallel
    )
    
    # Run comprehensive validation
    print("Starting comprehensive error handling validation...")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    try:
        result = validator.run_comprehensive_validation()
        
        # Save results
        result_file = validator.save_results(args.report_format)
        print(f"\nResults saved to: {result_file}")
        
        # Print summary
        if args.report_format == "text":
            print("\n" + validator._generate_text_report())
        else:
            print(f"\nOverall Status: {result.overall_status}")
            print(f"Production Readiness: {result.production_readiness_score:.1%}")
        
        # Return appropriate exit code
        if result.overall_status == "PRODUCTION_READY":
            return 0
        elif result.overall_status == "CONDITIONALLY_READY":
            return 0  # Allow deployment with monitoring
        elif result.overall_status == "NEEDS_IMPROVEMENT":
            return 1
        else:
            return 2  # Critical failure
    
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\nCritical error during validation: {e}")
        if args.verbose:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())