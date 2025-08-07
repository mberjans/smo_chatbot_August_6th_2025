#!/usr/bin/env python3
"""
Standalone Error Handling Validation Runner for Clinical Metabolomics Oracle.

This script provides a standalone way to run comprehensive error handling validation
without requiring pytest. It can be executed directly to validate that the error
handling implementation is ready for production use.

Usage:
    python run_error_handling_validation.py [--quick] [--output-dir DIR] [--verbose]

Options:
    --quick         Run abbreviated tests (faster execution)
    --output-dir    Directory to save test results (default: ./validation_results)
    --verbose       Enable verbose logging
    --scenarios     Comma-separated list of specific scenarios to run

Scenarios:
    basic_error_injection    - Test basic error handling with various failure types
    resource_pressure        - Test behavior under resource constraints
    circuit_breaker         - Test circuit breaker patterns under sustained failures
    checkpoint_corruption   - Test checkpoint/resume with corruption simulation
    memory_leak_detection   - Test for memory leaks during repeated failures
    all                     - Run all scenarios (default)

Example:
    # Quick validation (recommended for CI/CD)
    python run_error_handling_validation.py --quick
    
    # Full validation with verbose output
    python run_error_handling_validation.py --verbose
    
    # Run specific scenarios
    python run_error_handling_validation.py --scenarios basic_error_injection,circuit_breaker

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from test_error_handling_e2e_validation import (
        ComprehensiveErrorHandlingTestRunner,
        ErrorHandlingE2EValidator
    )
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Please ensure you're running from the correct directory and all dependencies are installed.")
    sys.exit(1)


class ValidationCLI:
    """Command-line interface for error handling validation."""
    
    def __init__(self):
        """Initialize CLI."""
        self.available_scenarios = {
            "basic_error_injection": "Test basic error handling with various failure types",
            "resource_pressure": "Test behavior under resource constraints", 
            "circuit_breaker": "Test circuit breaker patterns under sustained failures",
            "checkpoint_corruption": "Test checkpoint/resume with corruption simulation",
            "memory_leak_detection": "Test for memory leaks during repeated failures"
        }
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Run comprehensive error handling validation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --quick
  %(prog)s --verbose --output-dir ./test_results  
  %(prog)s --scenarios basic_error_injection,circuit_breaker
            """
        )
        
        parser.add_argument(
            "--quick", 
            action="store_true",
            help="Run abbreviated tests for faster execution"
        )
        
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("./validation_results"),
            help="Directory to save test results (default: ./validation_results)"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true", 
            help="Enable verbose logging"
        )
        
        parser.add_argument(
            "--scenarios",
            type=str,
            default="all",
            help="Comma-separated list of scenarios to run (default: all)"
        )
        
        parser.add_argument(
            "--list-scenarios",
            action="store_true",
            help="List available scenarios and exit"
        )
        
        parser.add_argument(
            "--json-output",
            action="store_true",
            help="Output results in JSON format only (suppress formatted output)"
        )
        
        parser.add_argument(
            "--timeout",
            type=int,
            default=600,  # 10 minutes
            help="Maximum time in seconds for validation (default: 600)"
        )
        
        return parser.parse_args()
    
    def list_scenarios(self) -> None:
        """List available test scenarios."""
        print("Available validation scenarios:")
        print("-" * 50)
        
        for scenario, description in self.available_scenarios.items():
            print(f"{scenario:25} - {description}")
        
        print(f"{'all':25} - Run all scenarios")
    
    def validate_scenarios(self, scenarios_arg: str) -> List[str]:
        """
        Validate and parse scenarios argument.
        
        Args:
            scenarios_arg: Comma-separated scenario names
            
        Returns:
            List of valid scenario names
        """
        if scenarios_arg.lower() == "all":
            return list(self.available_scenarios.keys())
        
        requested_scenarios = [s.strip() for s in scenarios_arg.split(",")]
        valid_scenarios = []
        invalid_scenarios = []
        
        for scenario in requested_scenarios:
            if scenario in self.available_scenarios:
                valid_scenarios.append(scenario)
            else:
                invalid_scenarios.append(scenario)
        
        if invalid_scenarios:
            print(f"Error: Invalid scenarios: {', '.join(invalid_scenarios)}")
            print("Use --list-scenarios to see available options")
            sys.exit(1)
        
        return valid_scenarios
    
    def setup_logging(self, verbose: bool, output_dir: Path) -> logging.Logger:
        """
        Setup logging configuration.
        
        Args:
            verbose: Enable verbose logging
            output_dir: Directory for log files
            
        Returns:
            Configured logger
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        log_file = output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        return root_logger
    
    def run_individual_scenario(self, 
                               validator: ErrorHandlingE2EValidator,
                               scenario: str,
                               quick_mode: bool) -> dict:
        """
        Run a single validation scenario.
        
        Args:
            validator: Error handling validator instance
            scenario: Scenario name to run
            quick_mode: Whether to run in quick mode
            
        Returns:
            Scenario results
        """
        durations = {
            "basic_error_injection": 30.0 if quick_mode else 60.0,
            "resource_pressure": 45.0 if quick_mode else 90.0,
            "circuit_breaker": 60.0 if quick_mode else 120.0,
            "checkpoint_corruption": 30.0 if quick_mode else 60.0,
            "memory_leak_detection": 90.0 if quick_mode else 180.0
        }
        
        duration = durations.get(scenario, 60.0)
        
        logging.info(f"Running scenario: {scenario} (duration: {duration}s)")
        
        if scenario == "basic_error_injection":
            return validator.run_basic_error_injection_scenario(duration)
        elif scenario == "resource_pressure":
            return validator.run_resource_pressure_scenario(duration)
        elif scenario == "circuit_breaker":
            return validator.run_circuit_breaker_scenario(duration)
        elif scenario == "checkpoint_corruption":
            return validator.run_checkpoint_corruption_scenario(duration)
        elif scenario == "memory_leak_detection":
            return validator.run_memory_leak_detection_scenario(duration)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def run_validation(self, args: argparse.Namespace) -> int:
        """
        Run the validation with given arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        # Setup logging
        logger = self.setup_logging(args.verbose, args.output_dir)
        
        # Validate scenarios
        scenarios_to_run = self.validate_scenarios(args.scenarios)
        
        logger.info("Starting Error Handling Validation")
        logger.info(f"Mode: {'Quick' if args.quick else 'Full'}")
        logger.info(f"Scenarios: {', '.join(scenarios_to_run)}")
        logger.info(f"Output directory: {args.output_dir}")
        
        start_time = time.time()
        
        try:
            # Create validator with temporary directory
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                validator = ErrorHandlingE2EValidator(Path(temp_dir), logger)
                
                # Run scenarios individually with timeout protection
                all_results = {}
                
                for scenario in scenarios_to_run:
                    scenario_start = time.time()
                    
                    # Check overall timeout
                    if time.time() - start_time > args.timeout:
                        logger.warning(f"Overall timeout ({args.timeout}s) reached, skipping remaining scenarios")
                        break
                    
                    try:
                        result = self.run_individual_scenario(validator, scenario, args.quick)
                        all_results[scenario] = result
                        
                        scenario_duration = time.time() - scenario_start
                        status = result.get("validation_status", "UNKNOWN")
                        
                        logger.info(f"Scenario {scenario}: {status} (completed in {scenario_duration:.1f}s)")
                        
                    except Exception as e:
                        logger.error(f"Error in scenario {scenario}: {e}", exc_info=args.verbose)
                        all_results[scenario] = {
                            "scenario": scenario,
                            "validation_status": "ERROR", 
                            "error_message": str(e),
                            "duration_seconds": time.time() - scenario_start
                        }
                
                # Update validator results and generate report
                validator.validation_results = all_results
                report = validator.generate_comprehensive_report()
                
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = args.output_dir / f"error_handling_validation_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {results_file}")
            
            # Output results
            if args.json_output:
                print(json.dumps(report, indent=2, default=str))
            else:
                self.print_results_summary(report)
            
            # Determine exit code
            overall_status = report["validation_report"]["overall_status"]
            return 0 if overall_status == "PASS" else 1
            
        except KeyboardInterrupt:
            logger.warning("Validation interrupted by user")
            return 1
            
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}", exc_info=True)
            return 1
    
    def print_results_summary(self, report: dict) -> None:
        """Print human-readable results summary."""
        validation_info = report["validation_report"]
        
        print("\n" + "="*80)
        print("ERROR HANDLING VALIDATION RESULTS")
        print("="*80)
        
        # Overall status
        status = validation_info["overall_status"]
        status_symbol = "✓" if status == "PASS" else "✗"
        
        print(f"\n{status_symbol} Overall Status: {status}")
        print(f"  Success Rate: {validation_info['overall_success_rate']:.1%}")
        print(f"  Duration: {validation_info['duration_seconds']:.1f} seconds")
        print(f"  Scenarios: {validation_info['passed_scenarios']}/{validation_info['total_scenarios']} passed")
        
        # Scenario details
        print(f"\nSCENARIO RESULTS:")
        print("-" * 50)
        
        for scenario_name, result in report["scenario_results"].items():
            scenario_status = result.get("validation_status", "UNKNOWN")
            scenario_symbol = "✓" if scenario_status == "PASS" else "✗" if scenario_status == "FAIL" else "?"
            duration = result.get("duration_seconds", 0)
            
            print(f"{scenario_symbol} {scenario_name:<25} {scenario_status:<6} ({duration:5.1f}s)")
            
            # Show key metrics for each scenario
            if scenario_name == "basic_error_injection" and "success_rate" in result:
                print(f"    Success Rate: {result['success_rate']:.1%}, "
                      f"Documents: {result.get('documents_processed', 0)}")
            
            elif scenario_name == "resource_pressure":
                degraded = result.get("degradation_triggered", False)
                reduced = result.get("batch_size_reduced", False)
                print(f"    Degradation: {'Yes' if degraded else 'No'}, "
                      f"Batch Reduced: {'Yes' if reduced else 'No'}")
            
            elif scenario_name == "circuit_breaker":
                opened = result.get("circuit_opened", False)
                recovered = result.get("circuit_recovered", False)
                print(f"    Circuit Opened: {'Yes' if opened else 'No'}, "
                      f"Recovered: {'Yes' if recovered else 'No'}")
            
            elif scenario_name == "checkpoint_corruption":
                created = result.get("checkpoints_created", 0)
                resume_rate = result.get("resume_success_rate", 0)
                print(f"    Checkpoints: {created}, Resume Rate: {resume_rate:.1%}")
            
            elif scenario_name == "memory_leak_detection":
                leak = result.get("memory_leak_detected", False)
                increase = result.get("memory_increase_mb", 0)
                print(f"    Memory Leak: {'Yes' if leak else 'No'}, "
                      f"Increase: {increase:.1f}MB")
        
        # Performance summary
        perf = report.get("performance_summary", {})
        if perf:
            print(f"\nPERFORMANCE METRICS:")
            print("-" * 30)
            print(f"  Average Memory: {perf.get('avg_memory_usage_mb', 0):.1f} MB")
            print(f"  Peak Memory:    {perf.get('max_memory_usage_mb', 0):.1f} MB") 
            print(f"  Average CPU:    {perf.get('avg_cpu_usage_percent', 0):.1f}%")
        
        # Error summary
        aggregate = report.get("aggregate_metrics", {})
        if aggregate.get("error_counts"):
            print(f"\nERROR SUMMARY:")
            print("-" * 30)
            for error_type, count in aggregate["error_counts"].items():
                print(f"  {error_type:<20}: {count}")
        
        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\nRECOMMENDATIONS:")
            print("-" * 30)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Final assessment
        print("\n" + "="*80)
        if status == "PASS":
            print("✓ VALIDATION PASSED - Error handling system is ready for production")
        else:
            print("✗ VALIDATION FAILED - Please address identified issues before production use")
        print("="*80)


def main() -> int:
    """Main entry point."""
    cli = ValidationCLI()
    args = cli.parse_arguments()
    
    if args.list_scenarios:
        cli.list_scenarios()
        return 0
    
    return cli.run_validation(args)


if __name__ == "__main__":
    sys.exit(main())