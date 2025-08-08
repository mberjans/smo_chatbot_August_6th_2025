"""
Demonstration Script for LLM-based Query Classification

This script demonstrates how to use the LLM-powered query classification system
for the Clinical Metabolomics Oracle, showing integration with existing infrastructure
and various classification scenarios.

Usage:
    python demo_llm_classification.py

Features Demonstrated:
    - Basic LLM classification with different query types
    - Comparison with existing keyword-based classification
    - Performance monitoring and cost tracking
    - Cache efficiency and fallback mechanisms
    - Integration with existing routing infrastructure
"""

import asyncio
import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import the LLM classification modules
from llm_classification_prompts import LLMClassificationPrompts, ClassificationCategory
from llm_query_classifier import (
    LLMQueryClassifier, 
    LLMClassificationConfig, 
    LLMProvider,
    create_llm_enhanced_router,
    convert_llm_result_to_routing_prediction
)
from query_router import BiomedicalQueryRouter


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClassificationDemo:
    """Demonstration class for LLM-based query classification."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.test_queries = self._get_test_queries()
        self.results = []
    
    def _get_test_queries(self) -> List[Dict[str, Any]]:
        """Get a comprehensive set of test queries for demonstration."""
        
        return [
            # KNOWLEDGE_GRAPH queries
            {
                "query": "What is the relationship between glucose metabolism and insulin signaling in type 2 diabetes?",
                "expected_category": "KNOWLEDGE_GRAPH",
                "description": "Established biochemical relationship query"
            },
            {
                "query": "How does the citric acid cycle connect to fatty acid biosynthesis pathways?", 
                "expected_category": "KNOWLEDGE_GRAPH",
                "description": "Metabolic pathway connection query"
            },
            {
                "query": "Find biomarkers associated with Alzheimer's disease in cerebrospinal fluid metabolomics",
                "expected_category": "KNOWLEDGE_GRAPH", 
                "description": "Biomarker association query"
            },
            {
                "query": "Mechanism of action for metformin in glucose homeostasis regulation",
                "expected_category": "KNOWLEDGE_GRAPH",
                "description": "Drug mechanism query"
            },
            
            # REAL_TIME queries
            {
                "query": "What are the latest 2024 FDA approvals for metabolomics-based diagnostics?",
                "expected_category": "REAL_TIME",
                "description": "Current regulatory information query"
            },
            {
                "query": "Recent breakthrough discoveries in cancer metabolomics this year",
                "expected_category": "REAL_TIME", 
                "description": "Recent research developments query"
            },
            {
                "query": "Current clinical trials using mass spectrometry for early disease detection",
                "expected_category": "REAL_TIME",
                "description": "Ongoing trials query"
            },
            {
                "query": "New developments in AI-powered metabolomics analysis platforms in 2024",
                "expected_category": "REAL_TIME",
                "description": "Recent technology developments query"
            },
            {
                "query": "Just announced metabolomics biomarker partnerships between pharma companies",
                "expected_category": "REAL_TIME",
                "description": "Very recent news query"
            },
            
            # GENERAL queries
            {
                "query": "What is metabolomics and how does it work?",
                "expected_category": "GENERAL",
                "description": "Basic definitional query"
            },
            {
                "query": "Explain the basics of LC-MS analysis for beginners", 
                "expected_category": "GENERAL",
                "description": "Educational methodology query"
            },
            {
                "query": "How to interpret NMR spectra in metabolomics studies",
                "expected_category": "GENERAL",
                "description": "General methodology query"
            },
            {
                "query": "What are the main applications of metabolomics in healthcare?",
                "expected_category": "GENERAL",
                "description": "Broad applications query"
            },
            {
                "query": "Define biomarker and its role in personalized medicine",
                "expected_category": "GENERAL", 
                "description": "Basic definition with context"
            },
            
            # Edge cases and ambiguous queries
            {
                "query": "metabolomics cancer",
                "expected_category": "GENERAL",
                "description": "Very short, ambiguous query"
            },
            {
                "query": "How has metabolomics advanced in recent years and what pathways are most studied?",
                "expected_category": "REAL_TIME",  # Should lean toward real-time due to "recent years"
                "description": "Mixed temporal and relationship query"
            },
            {
                "query": "Latest research on glucose-insulin pathway mechanisms published in 2024",
                "expected_category": "REAL_TIME",  # Temporal wins over established knowledge
                "description": "Temporal + mechanism hybrid query"
            }
        ]
    
    async def run_demo(self, api_key: str = None) -> None:
        """
        Run the complete demonstration.
        
        Args:
            api_key: OpenAI API key (if not provided, will try environment variable)
        """
        print("=" * 80)
        print("LLM-POWERED QUERY CLASSIFICATION DEMONSTRATION")
        print("Clinical Metabolomics Oracle System")
        print("=" * 80)
        print()
        
        # Get API key
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("⚠️  WARNING: No OpenAI API key provided.")
            print("   Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            print("   Running in simulation mode with mock responses.")
            print()
            await self._run_simulation_mode()
            return
        
        # Initialize LLM classifier
        config = LLMClassificationConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",  # Fast, cost-effective model
            api_key=api_key,
            timeout_seconds=5.0,
            daily_api_budget=2.0,  # Conservative budget for demo
            enable_caching=True
        )
        
        # Create biomedical router for comparison
        biomedical_router = BiomedicalQueryRouter(logger)
        
        # Create LLM classifier
        llm_classifier = LLMQueryClassifier(config, biomedical_router, logger)
        
        print("🚀 LLM Classifier initialized successfully!")
        print(f"   Model: {config.model_name}")
        print(f"   Daily Budget: ${config.daily_api_budget}")
        print(f"   Caching: {'Enabled' if config.enable_caching else 'Disabled'}")
        print()
        
        # Run classification tests
        await self._run_classification_tests(llm_classifier, biomedical_router)
        
        # Show performance analysis
        await self._show_performance_analysis(llm_classifier)
        
        # Demonstrate optimization recommendations
        await self._show_optimization_recommendations(llm_classifier)
    
    async def _run_classification_tests(self, 
                                      llm_classifier: LLMQueryClassifier,
                                      biomedical_router: BiomedicalQueryRouter) -> None:
        """Run classification tests and compare results."""
        
        print("📊 RUNNING CLASSIFICATION TESTS")
        print("-" * 50)
        print()
        
        correct_predictions = 0
        total_predictions = len(self.test_queries)
        
        for i, test_case in enumerate(self.test_queries, 1):
            query = test_case["query"]
            expected = test_case["expected_category"]
            description = test_case["description"]
            
            print(f"Test {i}/{total_predictions}: {description}")
            print(f"Query: \"{query[:60]}{'...' if len(query) > 60 else ''}\"")
            print(f"Expected: {expected}")
            
            # Get LLM classification
            start_time = asyncio.get_event_loop().time()
            llm_result, used_llm = await llm_classifier.classify_query(query)
            llm_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get keyword-based classification for comparison
            start_time = asyncio.get_event_loop().time()
            keyword_result = biomedical_router.route_query(query)
            keyword_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Map routing decision to category for comparison
            routing_to_category = {
                "lightrag": "KNOWLEDGE_GRAPH",
                "perplexity": "REAL_TIME", 
                "either": "GENERAL",
                "hybrid": "GENERAL"
            }
            keyword_category = routing_to_category.get(keyword_result.routing_decision.value, "GENERAL")
            
            # Check if LLM prediction is correct
            llm_correct = llm_result.category == expected
            keyword_correct = keyword_category == expected
            
            if llm_correct:
                correct_predictions += 1
            
            # Store results
            self.results.append({
                "query": query,
                "expected": expected,
                "llm_result": llm_result.category,
                "llm_confidence": llm_result.confidence,
                "llm_correct": llm_correct,
                "llm_time_ms": llm_time,
                "used_llm": used_llm,
                "keyword_result": keyword_category,
                "keyword_confidence": keyword_result.confidence,
                "keyword_correct": keyword_correct,
                "keyword_time_ms": keyword_time
            })
            
            # Display results
            print(f"LLM Result: {llm_result.category} (conf: {llm_result.confidence:.3f}) "
                  f"{'✅' if llm_correct else '❌'} [{llm_time:.1f}ms] {'🤖' if used_llm else '⚡'}")
            print(f"Keyword Result: {keyword_category} (conf: {keyword_result.confidence:.3f}) "
                  f"{'✅' if keyword_correct else '❌'} [{keyword_time:.1f}ms]")
            
            if llm_result.reasoning:
                print(f"Reasoning: {llm_result.reasoning}")
            
            if llm_result.uncertainty_indicators:
                print(f"Uncertainty: {', '.join(llm_result.uncertainty_indicators)}")
            
            print()
        
        # Summary
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"🎯 CLASSIFICATION ACCURACY: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        print()
    
    async def _show_performance_analysis(self, llm_classifier: LLMQueryClassifier) -> None:
        """Show detailed performance analysis."""
        
        print("📈 PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        # Get statistics
        stats = llm_classifier.get_classification_statistics()
        
        # Classification metrics
        class_metrics = stats["classification_metrics"]
        print(f"📊 Classification Metrics:")
        print(f"   Total Classifications: {class_metrics['total_classifications']}")
        print(f"   LLM Successful: {class_metrics['llm_successful']}")
        print(f"   LLM Failures: {class_metrics['llm_failures']}")
        print(f"   Fallback Used: {class_metrics['fallback_used']}")
        print(f"   Cache Hits: {class_metrics['cache_hits']}")
        print(f"   Success Rate: {class_metrics['success_rate']:.1f}%")
        print()
        
        # Performance metrics
        perf_metrics = stats["performance_metrics"]
        print(f"⚡ Performance Metrics:")
        print(f"   Avg Response Time: {perf_metrics['avg_response_time_ms']:.1f}ms")
        print(f"   Avg Confidence: {perf_metrics['avg_confidence_score']:.3f}")
        print()
        
        # Cost metrics
        cost_metrics = stats["cost_metrics"]
        print(f"💰 Cost Metrics:")
        print(f"   Daily API Cost: ${cost_metrics['daily_api_cost']:.4f}")
        print(f"   Daily Budget: ${cost_metrics['daily_budget']:.2f}")
        print(f"   Budget Utilization: {cost_metrics['budget_utilization']:.1f}%")
        print(f"   Daily Token Usage: {cost_metrics['daily_token_usage']}")
        print(f"   Cost per Classification: ${cost_metrics['estimated_cost_per_classification']:.6f}")
        print()
        
        # Cache stats
        cache_stats = stats.get("cache_stats", {})
        if cache_stats:
            print(f"🗄️ Cache Statistics:")
            print(f"   Cache Size: {cache_stats['cache_size']}/{cache_stats['max_size']}")
            print(f"   Utilization: {cache_stats['utilization']*100:.1f}%")
            print(f"   TTL Hours: {cache_stats['ttl_hours']}")
            print()
        
        # Compare LLM vs Keyword performance
        await self._compare_methods_performance()
    
    async def _compare_methods_performance(self) -> None:
        """Compare LLM vs keyword-based classification performance."""
        
        if not self.results:
            return
        
        print("🔍 LLM vs KEYWORD COMPARISON")
        print("-" * 30)
        
        llm_correct = sum(1 for r in self.results if r["llm_correct"])
        keyword_correct = sum(1 for r in self.results if r["keyword_correct"])
        total = len(self.results)
        
        llm_avg_time = sum(r["llm_time_ms"] for r in self.results) / total
        keyword_avg_time = sum(r["keyword_time_ms"] for r in self.results) / total
        
        llm_avg_conf = sum(r["llm_confidence"] for r in self.results) / total
        keyword_avg_conf = sum(r["keyword_confidence"] for r in self.results) / total
        
        print(f"Accuracy Comparison:")
        print(f"   LLM: {llm_correct}/{total} ({llm_correct/total*100:.1f}%)")
        print(f"   Keyword: {keyword_correct}/{total} ({keyword_correct/total*100:.1f}%)")
        print()
        
        print(f"Speed Comparison:")
        print(f"   LLM: {llm_avg_time:.1f}ms average")
        print(f"   Keyword: {keyword_avg_time:.1f}ms average")
        print(f"   Speed Ratio: {llm_avg_time/keyword_avg_time:.1f}x slower")
        print()
        
        print(f"Confidence Comparison:")
        print(f"   LLM: {llm_avg_conf:.3f} average")
        print(f"   Keyword: {keyword_avg_conf:.3f} average")
        print()
        
        # Show where LLM performed better
        llm_better = [r for r in self.results if r["llm_correct"] and not r["keyword_correct"]]
        if llm_better:
            print(f"🎯 LLM Outperformed Keywords ({len(llm_better)} cases):")
            for result in llm_better[:3]:  # Show first 3
                print(f"   • \"{result['query'][:50]}...\"")
                print(f"     LLM: {result['llm_result']} ✅, Keyword: {result['keyword_result']} ❌")
        print()
    
    async def _show_optimization_recommendations(self, llm_classifier: LLMQueryClassifier) -> None:
        """Show optimization recommendations."""
        
        print("🔧 OPTIMIZATION RECOMMENDATIONS")
        print("-" * 50)
        
        optimization = llm_classifier.optimize_configuration()
        
        print(f"Overall Health: {optimization['overall_health'].upper()}")
        print()
        
        if optimization["recommendations"]:
            for i, rec in enumerate(optimization["recommendations"], 1):
                print(f"{i}. {rec['type'].upper()}: {rec['issue']}")
                print(f"   Suggestion: {rec['suggestion']}")
                print()
        else:
            print("✅ No optimization recommendations - system is performing well!")
            print()
    
    async def _run_simulation_mode(self) -> None:
        """Run demo in simulation mode without actual LLM API calls."""
        
        print("🎭 SIMULATION MODE")
        print("-" * 50)
        print("Simulating LLM responses based on prompt templates...")
        print()
        
        # Show how prompts would be constructed
        sample_queries = self.test_queries[:3]  # First 3 queries
        
        for i, test_case in enumerate(sample_queries, 1):
            query = test_case["query"]
            expected = test_case["expected_category"]
            
            print(f"Sample {i}: {expected} Query")
            print(f"Query: \"{query}\"")
            print()
            
            # Show primary prompt
            primary_prompt = LLMClassificationPrompts.build_primary_prompt(query)
            print("Primary Prompt (truncated):")
            print(primary_prompt[:300] + "..." if len(primary_prompt) > 300 else primary_prompt)
            print()
            
            # Show fallback prompt 
            fallback_prompt = LLMClassificationPrompts.build_fallback_prompt(query)
            print("Fallback Prompt:")
            print(fallback_prompt)
            print()
            
            # Estimate tokens
            token_estimate = LLMClassificationPrompts.estimate_token_usage(query, include_examples=False)
            print(f"Token Estimate: {token_estimate['primary_prompt_tokens']} + {token_estimate['estimated_response_tokens']} = {token_estimate['primary_prompt_tokens'] + token_estimate['estimated_response_tokens']} total")
            print()
            
            print("-" * 40)
            print()
        
        # Show example classifications from the prompt templates
        print("📋 EXAMPLE CLASSIFICATIONS FROM PROMPT TEMPLATES")
        print("-" * 50)
        
        for category in [ClassificationCategory.KNOWLEDGE_GRAPH, ClassificationCategory.REAL_TIME, ClassificationCategory.GENERAL]:
            examples = LLMClassificationPrompts.get_few_shot_examples(category, count=1)
            if examples:
                example = examples[0]
                print(f"{category.value} Example:")
                print(f"Query: \"{example['query']}\"")
                print(f"Classification: {example['classification']['category']}")
                print(f"Confidence: {example['classification']['confidence']}")
                print(f"Reasoning: {example['classification']['reasoning']}")
                print()


async def main():
    """Main demonstration function."""
    
    # Check for API key in environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Initialize and run demo
    demo = LLMClassificationDemo()
    await demo.run_demo(api_key)
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)
    
    if not api_key:
        print("\n💡 To run with actual LLM classifications:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   python demo_llm_classification.py")


if __name__ == "__main__":
    asyncio.run(main())