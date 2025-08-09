#!/usr/bin/env python3
"""
CMO Test Scenarios Runner
========================

A convenient utility script for running comprehensive CMO test scenarios.
This script provides easy access to all predefined scenarios and allows
for quick execution of specific test categories or individual scenarios.

Usage Examples:
    # Run all clinical scenarios
    python run_cmo_scenarios.py --category clinical
    
    # Run specific scenario by ID
    python run_cmo_scenarios.py --scenario clinical_morning_rush_60
    
    # Run all scenarios (comprehensive test suite)
    python run_cmo_scenarios.py --all
    
    # List available scenarios
    python run_cmo_scenarios.py --list
    
    # Get recommendations based on requirements
    python run_cmo_scenarios.py --recommend --users 100 --duration 20 --use-case research

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-09
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import CMO test configuration system
from cmo_test_configurations import (
    CMOTestConfigurationFactory,
    CMOTestSuiteOrchestrator,
    run_scenario_by_id,
    get_scenario_documentation,
    create_all_test_scenarios
)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cmo_test_scenarios.log')
        ]
    )


def print_scenarios_table(scenarios_doc: Dict[str, List[Dict[str, Any]]]):
    """Print scenarios in a formatted table."""
    print("\n" + "="*100)
    print(f"{'CATEGORY':<20} {'SCENARIO ID':<35} {'USERS':<6} {'DURATION':<10} {'LOAD PATTERN':<15}")
    print("="*100)
    
    for category, scenarios in scenarios_doc.items():
        for i, scenario in enumerate(scenarios):
            category_display = category.upper() if i == 0 else ""
            print(f"{category_display:<20} {scenario['id']:<35} {scenario['concurrent_users']:<6} "
                  f"{scenario['duration_minutes']}m{'':<7} {scenario['load_pattern']:<15}")
    
    print("="*100)


def print_scenario_details(scenario_doc: Dict[str, Any]):
    """Print detailed information about a specific scenario."""
    print(f"\n{scenario_doc['name']}")
    print("=" * len(scenario_doc['name']))
    print(f"ID: {scenario_doc['id']}")
    print(f"Description: {scenario_doc['description']}")
    print(f"Concurrent Users: {scenario_doc['concurrent_users']}")
    print(f"Duration: {scenario_doc['duration_minutes']} minutes")
    print(f"Load Pattern: {scenario_doc['load_pattern']}")
    print(f"User Types: {', '.join(scenario_doc['user_types'])}")
    print(f"Component Focus: {', '.join(scenario_doc['component_focus'])}")
    print("\nPerformance Targets:")
    for key, value in scenario_doc['performance_targets'].items():
        print(f"  {key}: {value}")


async def run_single_scenario(scenario_id: str) -> Dict[str, Any]:
    """Run a single scenario by ID."""
    try:
        print(f"\nStarting scenario: {scenario_id}")
        start_time = time.time()
        
        result = await run_scenario_by_id(scenario_id)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Scenario {scenario_id} completed in {duration:.1f} seconds")
        return {
            'scenario_id': scenario_id,
            'success': True,
            'duration': duration,
            'metrics': result
        }
        
    except Exception as e:
        print(f"Scenario {scenario_id} failed: {e}")
        return {
            'scenario_id': scenario_id,
            'success': False,
            'error': str(e)
        }


async def run_category_scenarios(category: str) -> Dict[str, Any]:
    """Run all scenarios in a specific category."""
    factory = CMOTestConfigurationFactory()
    scenarios = factory.get_scenarios_by_category(category)
    
    if not scenarios:
        raise ValueError(f"No scenarios found for category '{category}'")
    
    print(f"\nRunning {len(scenarios)} scenarios in category '{category}'")
    
    results = {}
    for scenario in scenarios:
        result = await run_single_scenario(scenario.scenario_id)
        results[scenario.scenario_id] = result
    
    return results


async def run_comprehensive_suite(categories: List[str] = None) -> Dict[str, Any]:
    """Run comprehensive test suite."""
    orchestrator = CMOTestSuiteOrchestrator()
    
    print(f"\nRunning comprehensive test suite...")
    if categories:
        print(f"Categories: {', '.join(categories)}")
    else:
        print("Categories: All available categories")
    
    results = await orchestrator.run_comprehensive_test_suite(
        categories=categories,
        max_concurrent_suites=2
    )
    
    return results


def get_recommendations(users: int, duration: int, use_case: str) -> List[str]:
    """Get scenario recommendations based on requirements."""
    factory = CMOTestConfigurationFactory()
    recommendations = factory.get_configuration_recommendations(
        target_users=users,
        test_duration_minutes=duration,
        primary_use_case=use_case
    )
    return recommendations


def print_results_summary(results: Dict[str, Any]):
    """Print a summary of test results."""
    if 'execution_summary' in results:
        # Comprehensive suite results
        summary = results['execution_summary']
        analysis = results.get('comprehensive_analysis', {})
        recommendations = results.get('recommendations', [])
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TEST SUITE RESULTS")
        print(f"{'='*60}")
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Categories Tested: {', '.join(summary['categories_tested'])}")
        
        if analysis.get('overall_performance'):
            perf = analysis['overall_performance']
            print(f"\nOverall Performance:")
            print(f"  Average Success Rate: {perf.get('avg_success_rate', 0):.1%}")
            print(f"  Median Response Time: {perf.get('median_response_time', 0):.0f}ms")
            print(f"  P95 Response Time: {perf.get('p95_response_time', 0):.0f}ms")
            print(f"  Max Concurrent Users: {perf.get('max_concurrent_users', 0)}")
        
        if analysis.get('cmo_specific_performance'):
            cmo_perf = analysis['cmo_specific_performance']
            print(f"\nCMO-Specific Performance:")
            print(f"  LightRAG Success Rate: {cmo_perf.get('avg_lightrag_success_rate', 0):.1%}")
            print(f"  Cache Hit Rate: {cmo_perf.get('avg_cache_hit_rate', 0):.1%}")
        
        print(f"\nSystem Stability: {analysis.get('system_stability', 'Unknown')}")
        print(f"Scalability Assessment: {analysis.get('scalability_assessment', 'Unknown')}")
        
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
    
    else:
        # Individual scenario or category results
        successful = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        
        print(f"\n{'='*60}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Scenarios: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success Rate: {successful/total:.1%}" if total > 0 else "Success Rate: N/A")
        
        # Show individual results
        for scenario_id, result in results.items():
            status = "✓" if result.get('success', False) else "✗"
            duration = result.get('duration', 0)
            print(f"  {status} {scenario_id} ({duration:.1f}s)")


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run CMO test scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cmo_scenarios.py --list
  python run_cmo_scenarios.py --scenario clinical_morning_rush_60
  python run_cmo_scenarios.py --category clinical
  python run_cmo_scenarios.py --all
  python run_cmo_scenarios.py --recommend --users 100 --duration 15 --use-case clinical
        """
    )
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--list', action='store_true',
                             help='List all available scenarios')
    action_group.add_argument('--scenario', type=str,
                             help='Run a specific scenario by ID')
    action_group.add_argument('--category', type=str,
                             choices=['clinical', 'research', 'scalability', 'component_testing'],
                             help='Run all scenarios in a category')
    action_group.add_argument('--all', action='store_true',
                             help='Run comprehensive test suite (all scenarios)')
    action_group.add_argument('--recommend', action='store_true',
                             help='Get scenario recommendations based on requirements')
    
    # Recommendation arguments
    parser.add_argument('--users', type=int, default=50,
                       help='Target number of concurrent users (for recommendations)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Target test duration in minutes (for recommendations)')
    parser.add_argument('--use-case', type=str, 
                       choices=['clinical', 'research', 'testing', 'scalability'],
                       default='clinical',
                       help='Primary use case (for recommendations)')
    
    # General arguments
    parser.add_argument('--categories', type=str, nargs='+',
                       choices=['clinical', 'research', 'scalability', 'component_testing'],
                       help='Specific categories to run (for --all option)')
    parser.add_argument('--output', type=str,
                       help='Output file path for results (JSON format)')
    parser.add_argument('--log-level', type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    parser.add_argument('--details', action='store_true',
                       help='Show detailed scenario information')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        results = None
        
        if args.list:
            # List all available scenarios
            docs = get_scenario_documentation()
            print_scenarios_table(docs)
            
            if args.details:
                print("\nDetailed Scenario Information:")
                for category, scenarios in docs.items():
                    for scenario in scenarios:
                        print_scenario_details(scenario)
            
            total_scenarios = sum(len(scenarios) for scenarios in docs.values())
            print(f"\nTotal available scenarios: {total_scenarios}")
            
        elif args.recommend:
            # Get recommendations
            recommendations = get_recommendations(args.users, args.duration, args.use_case)
            
            print(f"\nScenario Recommendations for:")
            print(f"  Target Users: {args.users}")
            print(f"  Duration: {args.duration} minutes")
            print(f"  Use Case: {args.use_case}")
            
            if recommendations:
                print(f"\nRecommended Scenarios:")
                docs = get_scenario_documentation()
                all_scenarios = {s['id']: s for scenarios in docs.values() for s in scenarios}
                
                for i, rec_id in enumerate(recommendations, 1):
                    if rec_id in all_scenarios:
                        scenario = all_scenarios[rec_id]
                        print(f"  {i}. {scenario['name']} ({rec_id})")
                        print(f"     Users: {scenario['concurrent_users']}, "
                              f"Duration: {scenario['duration_minutes']}m, "
                              f"Pattern: {scenario['load_pattern']}")
            else:
                print("No scenarios match the specified criteria.")
        
        elif args.scenario:
            # Run specific scenario
            result = await run_single_scenario(args.scenario)
            results = {args.scenario: result}
        
        elif args.category:
            # Run category scenarios
            results = await run_category_scenarios(args.category)
        
        elif args.all:
            # Run comprehensive suite
            results = await run_comprehensive_suite(args.categories)
        
        # Print results summary if we have results
        if results:
            print_results_summary(results)
            
            # Save results to file if specified
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nResults saved to: {output_path}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())