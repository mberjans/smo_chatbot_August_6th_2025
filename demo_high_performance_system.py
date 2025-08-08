#!/usr/bin/env python3
"""
High-Performance Classification System Demo

This script demonstrates the key features and usage of the high-performance
LLM-based classification system for the Clinical Metabolomics Oracle.

Features demonstrated:
    - Basic query classification with performance tracking
    - Multi-level caching effectiveness
    - Performance benchmarking
    - System validation
    - Real-time monitoring

Author: Claude Code (Anthropic)
Version: 1.0.0
Created: 2025-08-08
"""

import asyncio
import logging
import time
import sys
from pathlib import Path

# Add the lightrag_integration directory to Python path
sys.path.append(str(Path(__file__).parent / "lightrag_integration"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_basic_classification():
    """Demonstrate basic classification functionality."""
    
    print("\n" + "="*80)
    print("DEMO 1: BASIC HIGH-PERFORMANCE CLASSIFICATION")
    print("="*80)
    
    try:
        from lightrag_integration.high_performance_classification_system import (
            HighPerformanceConfig,
            high_performance_classification_context
        )
        
        # Configure for aggressive performance
        config = HighPerformanceConfig(
            target_response_time_ms=1200,  # Very aggressive target
            l1_cache_size=5000,           # Large cache
            enable_cache_warming=True,
            enable_adaptive_optimization=True
        )
        
        # Test queries representing different categories
        test_queries = [
            "What is metabolomics?",
            "LC-MS analysis for biomarker identification methods",
            "Latest research in clinical metabolomics published in 2025",
            "Relationship between glucose metabolism and insulin signaling pathways",
            "Statistical analysis approaches for metabolomics data",
        ]
        
        async with high_performance_classification_context(config) as hp_system:
            print(f"✓ High-performance system initialized with {config.target_response_time_ms}ms target")
            
            results = []
            total_time = 0
            
            for i, query in enumerate(test_queries, 1):
                print(f"\nQuery {i}: {query[:60]}...")
                
                start_time = time.time()
                result, metadata = await hp_system.classify_query_optimized(query)
                response_time = (time.time() - start_time) * 1000
                total_time += response_time
                
                results.append({
                    'query': query,
                    'category': result.category,
                    'confidence': result.confidence,
                    'response_time_ms': response_time,
                    'target_met': metadata['target_met'],
                    'cache_hit': metadata.get('cache_hit', False),
                    'optimizations': metadata.get('optimizations_applied', [])
                })
                
                # Display results
                status = "✓" if metadata['target_met'] else "⚠"
                cache_status = "HIT" if metadata.get('cache_hit', False) else "MISS"
                
                print(f"  {status} Result: {result.category} (conf: {result.confidence:.3f})")
                print(f"  {status} Time: {response_time:.1f}ms (target: {config.target_response_time_ms}ms)")
                print(f"  {status} Cache: {cache_status}")
                print(f"  {status} Optimizations: {', '.join(metadata.get('optimizations_applied', ['none']))}")
            
            # Summary statistics
            avg_time = total_time / len(test_queries)
            target_compliance = sum(1 for r in results if r['target_met']) / len(results)
            cache_hit_rate = sum(1 for r in results if r['cache_hit']) / len(results)
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Average Response Time: {avg_time:.1f}ms")
            print(f"   Target Compliance: {target_compliance:.1%}")
            print(f"   Cache Hit Rate: {cache_hit_rate:.1%}")
            print(f"   All Targets Met: {'✓ YES' if target_compliance == 1.0 else '✗ NO'}")
            
            # Get detailed system stats
            stats = hp_system.get_comprehensive_performance_stats()
            print(f"   Overall Cache Hit Rate: {stats['cache']['overall']['hit_rate']:.1%}")
            print(f"   Total Requests Processed: {stats['requests']['total']}")
            
    except ImportError as e:
        print(f"❌ Could not import high-performance components: {e}")
        print("   This is expected if running outside the package context.")


async def demo_cache_effectiveness():
    """Demonstrate cache effectiveness with repeated queries."""
    
    print("\n" + "="*80)
    print("DEMO 2: CACHE EFFECTIVENESS DEMONSTRATION")
    print("="*80)
    
    try:
        from lightrag_integration.high_performance_classification_system import (
            HighPerformanceConfig,
            high_performance_classification_context
        )
        
        config = HighPerformanceConfig(
            target_response_time_ms=1500,
            l1_cache_size=1000,
            l1_cache_ttl=600,  # 10 minutes
            enable_cache_warming=True
        )
        
        test_query = "What is the role of metabolomics in clinical diagnosis?"
        
        async with high_performance_classification_context(config) as hp_system:
            print("Testing cache performance with repeated queries...")
            
            # First request - should be cache miss
            print("\n🔍 First request (cache miss expected):")
            start_time = time.time()
            result1, metadata1 = await hp_system.classify_query_optimized(test_query)
            time1 = (time.time() - start_time) * 1000
            
            print(f"   Response time: {time1:.1f}ms")
            print(f"   Cache hit: {metadata1.get('cache_hit', False)}")
            print(f"   Category: {result1.category}")
            
            # Wait a moment to ensure cache is populated
            await asyncio.sleep(0.5)
            
            # Second request - should be cache hit
            print("\n⚡ Second request (cache hit expected):")
            start_time = time.time()
            result2, metadata2 = await hp_system.classify_query_optimized(test_query)
            time2 = (time.time() - start_time) * 1000
            
            print(f"   Response time: {time2:.1f}ms")
            print(f"   Cache hit: {metadata2.get('cache_hit', False)}")
            print(f"   Category: {result2.category}")
            
            # Performance comparison
            if time2 < time1:
                improvement = ((time1 - time2) / time1) * 100
                print(f"\n📈 CACHE PERFORMANCE:")
                print(f"   Speed improvement: {improvement:.1f}%")
                print(f"   Cache working: {'✓ YES' if metadata2.get('cache_hit', False) else '✗ NO'}")
            
            # Test multiple queries to demonstrate cache warming
            print("\n🔥 Testing cache warming with query variations:")
            variations = [
                "What is metabolomics in clinical research?",
                "Explain metabolomics for clinical applications",
                "How is metabolomics used in clinical settings?",
                "Metabolomics applications in clinical diagnosis"
            ]
            
            warm_times = []
            for var in variations:
                start_time = time.time()
                result, metadata = await hp_system.classify_query_optimized(var)
                response_time = (time.time() - start_time) * 1000
                warm_times.append(response_time)
                
                cache_indicator = "💨" if metadata.get('cache_hit', False) else "🔍"
                print(f"   {cache_indicator} {var[:45]}... = {response_time:.1f}ms")
            
            avg_warm_time = sum(warm_times) / len(warm_times)
            print(f"\n   Average variation time: {avg_warm_time:.1f}ms")
            
    except ImportError as e:
        print(f"❌ Could not import high-performance components: {e}")


async def demo_performance_benchmark():
    """Demonstrate performance benchmarking capabilities."""
    
    print("\n" + "="*80)
    print("DEMO 3: PERFORMANCE BENCHMARKING")
    print("="*80)
    
    try:
        from lightrag_integration.performance_benchmark_suite import (
            run_quick_performance_test,
            BenchmarkConfig,
            BenchmarkType,
            LoadPattern,
            run_comprehensive_benchmark
        )
        
        print("🚀 Running quick performance test...")
        
        # Quick performance test
        results = await run_quick_performance_test(
            target_response_time_ms=1500,
            total_requests=25  # Small number for demo
        )
        
        print(f"\n📊 QUICK BENCHMARK RESULTS:")
        print(f"   Performance Grade: {results.metrics.performance_grade.value}")
        print(f"   Average Response Time: {results.metrics.avg_response_time_ms:.1f}ms")
        print(f"   P95 Response Time: {results.metrics.p95_response_time_ms:.1f}ms")
        print(f"   Success Rate: {results.metrics.success_rate:.1%}")
        print(f"   Target Compliance: {results.metrics.target_compliance_rate:.1%}")
        print(f"   Cache Hit Rate: {results.metrics.cache_hit_rate:.1%}")
        print(f"   Throughput: {results.metrics.actual_throughput_rps:.1f} RPS")
        
        # Comprehensive benchmark (smaller scale for demo)
        print(f"\n🔬 Running comprehensive benchmark...")
        
        config = BenchmarkConfig(
            benchmark_name="demo_comprehensive",
            benchmark_type=BenchmarkType.LOAD,
            load_pattern=LoadPattern.CONSTANT,
            concurrent_users=5,
            total_requests=50,
            target_response_time_ms=1500,
            enable_real_time_monitoring=True,
            export_results=False,  # Don't export for demo
            generate_plots=False   # Don't generate plots for demo
        )
        
        comprehensive_results = await run_comprehensive_benchmark(config)
        
        print(f"\n📈 COMPREHENSIVE BENCHMARK RESULTS:")
        print(f"   Performance Grade: {comprehensive_results.metrics.performance_grade.value}")
        print(f"   Average Time: {comprehensive_results.metrics.avg_response_time_ms:.1f}ms")
        print(f"   Standard Deviation: {comprehensive_results.metrics.std_dev_ms:.1f}ms")
        print(f"   Min Time: {comprehensive_results.metrics.min_response_time_ms:.1f}ms")
        print(f"   Max Time: {comprehensive_results.metrics.max_response_time_ms:.1f}ms")
        print(f"   Optimization Score: {comprehensive_results.metrics.optimization_score:.1f}/100")
        
        # Show recommendations if any
        if comprehensive_results.recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(comprehensive_results.recommendations[:3], 1):
                print(f"   {i}. {rec['category'].upper()}: {rec['issue']}")
        else:
            print(f"\n✅ No optimization recommendations - system performing excellently!")
            
    except ImportError as e:
        print(f"❌ Could not import benchmark components: {e}")


async def demo_system_validation():
    """Demonstrate system validation capabilities."""
    
    print("\n" + "="*80)
    print("DEMO 4: SYSTEM VALIDATION")
    print("="*80)
    
    try:
        from lightrag_integration.validate_high_performance_system import (
            ValidationConfig,
            ValidationLevel,
            HighPerformanceSystemValidator
        )
        
        print("🔍 Running basic system validation...")
        
        # Configure validation for demo
        config = ValidationConfig(
            validation_level=ValidationLevel.BASIC,
            target_response_time_ms=1500,
            enable_stress_testing=False,  # Skip stress testing for demo
            enable_cache_testing=True,
            export_results=False
        )
        
        # Run validation
        validator = HighPerformanceSystemValidator(config)
        summary = await validator.run_validation()
        
        print(f"\n📋 VALIDATION RESULTS:")
        print(f"   Overall Grade: {summary.overall_grade}")
        print(f"   Tests Passed: {summary.passed_count}/{len(summary.results)}")
        print(f"   Success Rate: {getattr(summary, 'success_rate', 0):.1%}")
        print(f"   Duration: {getattr(summary, 'total_duration', 0):.1f}s")
        
        # Show individual test results
        print(f"\n🧪 INDIVIDUAL TEST RESULTS:")
        for result in summary.results:
            status = "✅" if result.passed else "❌"
            print(f"   {status} {result.test_name}: {result.duration_seconds:.2f}s")
            
            if not result.passed and result.error_message:
                print(f"      Error: {result.error_message}")
        
        # Show system info
        if summary.system_info:
            print(f"\n💻 SYSTEM INFO:")
            print(f"   Platform: {summary.system_info.get('platform', 'Unknown')}")
            print(f"   CPUs: {summary.system_info.get('cpu_count', 'Unknown')}")
            print(f"   Memory: {summary.system_info.get('memory_total_gb', 0):.1f}GB")
            print(f"   Target: {summary.system_info.get('target_response_time_ms', 0)}ms")
        
        # Show key recommendations
        if summary.recommendations:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for rec in summary.recommendations[:5]:
                print(f"   • {rec}")
        
    except ImportError as e:
        print(f"❌ Could not import validation components: {e}")


async def demo_monitoring_example():
    """Demonstrate real-time monitoring capabilities."""
    
    print("\n" + "="*80)
    print("DEMO 5: REAL-TIME MONITORING")
    print("="*80)
    
    try:
        from lightrag_integration.high_performance_classification_system import (
            HighPerformanceConfig,
            high_performance_classification_context
        )
        
        config = HighPerformanceConfig(
            target_response_time_ms=1500,
            enable_adaptive_optimization=True,
            monitoring_interval_seconds=5.0
        )
        
        print("📊 Demonstrating real-time performance monitoring...")
        
        async with high_performance_classification_context(config) as hp_system:
            # Simulate some load
            queries = [
                "What is metabolomics?",
                "LC-MS analysis methods",
                "Biomarker discovery techniques",
                "Pathway analysis approaches",
                "Clinical diagnosis applications",
                "Statistical analysis methods",
                "Quality control procedures",
                "Data preprocessing techniques",
                "Machine learning applications",
                "Multi-omics integration"
            ]
            
            print("   Generating load to demonstrate monitoring...")
            
            # Process queries and track performance
            for i in range(20):  # 20 requests
                query = queries[i % len(queries)]
                
                start_time = time.time()
                result, metadata = await hp_system.classify_query_optimized(query)
                response_time = (time.time() - start_time) * 1000
                
                if i % 5 == 0:  # Progress update every 5 requests
                    print(f"   Request {i+1}: {response_time:.1f}ms - {result.category}")
                
                # Brief pause between requests
                await asyncio.sleep(0.1)
            
            # Get comprehensive performance statistics
            stats = hp_system.get_comprehensive_performance_stats()
            
            print(f"\n📈 REAL-TIME PERFORMANCE STATS:")
            print(f"   Requests Processed: {stats['requests']['total']}")
            print(f"   Success Rate: {stats['requests']['success_rate']:.1%}")
            print(f"   Average Response Time: {stats['response_times']['avg_ms']:.1f}ms")
            print(f"   P95 Response Time: {stats['response_times']['p95_ms']:.1f}ms")
            
            print(f"\n💾 CACHE PERFORMANCE:")
            print(f"   L1 Cache Size: {stats['cache']['l1_cache']['size']}/{stats['cache']['l1_cache']['max_size']}")
            print(f"   L1 Hit Rate: {stats['cache']['l1_cache']['hit_rate']:.1%}")
            print(f"   Overall Hit Rate: {stats['cache']['overall']['hit_rate']:.1%}")
            
            print(f"\n🖥️  RESOURCE UTILIZATION:")
            print(f"   CPU Usage: {stats['resources']['cpu']['current_usage_percent']:.1f}%")
            print(f"   Memory Usage: {stats['resources']['memory']['current_usage_percent']:.1f}%")
            print(f"   Worker Threads: {stats['resources']['threading']['max_worker_threads']}")
            
            # Get optimization recommendations
            recommendations = hp_system.get_optimization_recommendations()
            if recommendations:
                print(f"\n🔧 OPTIMIZATION RECOMMENDATIONS:")
                for rec in recommendations[:3]:
                    print(f"   • {rec['category'].upper()}: {rec['issue']}")
            else:
                print(f"\n✅ SYSTEM OPTIMALLY CONFIGURED - No recommendations needed!")
            
    except ImportError as e:
        print(f"❌ Could not import monitoring components: {e}")


async def main():
    """Run all demonstration scenarios."""
    
    print("🚀 HIGH-PERFORMANCE CLASSIFICATION SYSTEM DEMONSTRATION")
    print("🎯 Target: Consistent <2 second response times with enterprise reliability")
    print("📅 Created: 2025-08-08")
    print("👨‍💻 Author: Claude Code (Anthropic)")
    
    # Check if we're in the right environment
    try:
        import lightrag_integration
        print("✅ Environment check passed - running full demonstrations")
        
        # Run all demonstrations
        await demo_basic_classification()
        await demo_cache_effectiveness()
        await demo_performance_benchmark()
        await demo_system_validation()
        await demo_monitoring_example()
        
    except ImportError:
        print("⚠️  Running in standalone mode - some features may be limited")
        print("   (This is normal when running the demo outside the full package)")
        
        # Run what we can
        await demo_basic_classification()
        await demo_cache_effectiveness()
    
    print("\n" + "="*80)
    print("🎉 DEMONSTRATION COMPLETED")
    print("="*80)
    print("✅ High-Performance Classification System demonstrated successfully!")
    print("📚 For more information, see HIGH_PERFORMANCE_CLASSIFICATION_README.md")
    print("🧪 To run full validation: python -m lightrag_integration.validate_high_performance_system")
    print("📊 To run benchmarks: see lightrag_integration/performance_benchmark_suite.py")
    print("="*80)


if __name__ == "__main__":
    # Run the demonstration
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()