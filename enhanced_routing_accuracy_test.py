#!/usr/bin/env python3
"""
Enhanced Routing Accuracy Test for Edge Cases and Improvements
CMO-LIGHTRAG-013-T08: Address accuracy deficiencies found in initial testing

This test focuses on improving routing accuracy by testing edge cases and
identifying patterns in misclassified queries to guide system improvements.
"""

import sys
import os
import json
import time
import statistics
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict

# Add the parent directory to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
from lightrag_integration.intelligent_query_router import IntelligentQueryRouter


class EnhancedAccuracyTester:
    """Enhanced accuracy testing for edge cases and improvements"""
    
    def __init__(self):
        self.router = BiomedicalQueryRouter()
        self.intelligent_router = IntelligentQueryRouter(self.router)
        
        # Edge case test patterns based on observed failures
        self.edge_case_patterns = {
            "lightrag_misclassified": [
                # These were incorrectly classified as EITHER in initial tests
                "How do neurotransmitters interact in synaptic transmission?",
                "How does the citric acid cycle connect to oxidative phosphorylation?",
                "What are the regulatory mechanisms in steroid hormone synthesis?",
                "How do growth factors regulate cell cycle progression?",
                
                # Additional similar pathway/mechanism queries
                "What are the molecular mechanisms of protein folding?",
                "How do transcription factors regulate gene expression?",
                "What is the role of calcium signaling in muscle contraction?",
                "How do enzymes catalyze biochemical reactions?",
                "What are the steps in DNA damage repair pathways?",
                "How does the electron transport chain generate ATP?"
            ],
            "hybrid_underdetected": [
                # These should be HYBRID but were classified as single systems
                "What is the molecular basis of diabetes and recent treatment advances?",
                "How do metabolic pathways change with age and current research?",
                "What are cancer metabolism mechanisms and latest clinical trials?",
                "How does gut microbiome affect metabolism and recent findings?",
                "What are aging pathways and current longevity research?",
                "How do circadian rhythms affect metabolism and recent research?",
                "What are cellular cancer mechanisms and immunotherapy developments?",
                "How do metabolic disorders affect brain function and neurometabolism research?",
                
                # Additional hybrid test cases
                "What are the genetic mechanisms of Alzheimer's and current therapeutic trials?",
                "How does exercise affect cellular metabolism and recent performance research?",
                "What are the molecular pathways of depression and latest medication developments?",
                "How do environmental toxins affect metabolism and current regulatory updates?"
            ],
            "temporal_indicators": [
                # Test temporal detection improvements
                "What are the 2025 breakthrough discoveries in metabolomics?",
                "Latest clinical trials published this month for diabetes treatment",
                "Recent FDA approvals in the past week for cancer drugs",
                "Current research trends emerging in 2025 for personalized medicine",
                "What were the top biomedical discoveries of 2024?",
                "Newest treatment protocols released this year for autoimmune diseases"
            ],
            "confidence_calibration": [
                # Test confidence score accuracy
                "What are the key enzymes in glycolysis?",  # Should have high confidence for LIGHTRAG
                "Latest COVID-19 research published yesterday",  # Should have high confidence for PERPLEXITY
                "How does diabetes affect kidney function?",  # Should have medium confidence for EITHER
                "What are metabolic pathways involved in aging and recent longevity research breakthroughs?"  # Should have high confidence for HYBRID
            ]
        }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge case patterns to identify improvement opportunities"""
        print("="*80)
        print("ENHANCED EDGE CASE ACCURACY TESTING")
        print("="*80)
        
        pattern_results = {}
        all_improvements = []
        
        for pattern_name, queries in self.edge_case_patterns.items():
            print(f"\nTesting {pattern_name.upper()} pattern ({len(queries)} queries)...")
            
            pattern_correct = 0
            pattern_details = []
            
            for i, query in enumerate(queries, 1):
                start_time = time.time()
                
                try:
                    result = self.router.route_query(query)
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Analyze pattern-specific correctness
                    is_correct = self._analyze_pattern_correctness(pattern_name, query, result)
                    improvement_suggestions = self._generate_improvement_suggestions(pattern_name, query, result)
                    
                    if is_correct:
                        pattern_correct += 1
                    
                    test_result = {
                        'query_id': i,
                        'query': query[:100] + "..." if len(query) > 100 else query,
                        'pattern': pattern_name,
                        'routing_decision': result.routing_decision.value,
                        'confidence': result.confidence,
                        'is_correct': is_correct,
                        'processing_time_ms': processing_time,
                        'improvement_suggestions': improvement_suggestions,
                        'confidence_metrics': {
                            'overall': result.confidence_metrics.overall_confidence if result.confidence_metrics else None,
                            'temporal': result.confidence_metrics.temporal_analysis_confidence if result.confidence_metrics else None,
                            'biomedical_entities': result.confidence_metrics.biomedical_entity_count if result.confidence_metrics else 0
                        }
                    }
                    
                    pattern_details.append(test_result)
                    all_improvements.extend(improvement_suggestions)
                
                except Exception as e:
                    print(f"  Error processing query {i}: {e}")
                    pattern_details.append({
                        'query_id': i,
                        'query': query,
                        'error': str(e),
                        'is_correct': False
                    })
            
            pattern_accuracy = (pattern_correct / len(queries)) * 100
            pattern_results[pattern_name] = {
                'total_queries': len(queries),
                'correct_predictions': pattern_correct,
                'accuracy_percentage': pattern_accuracy,
                'details': pattern_details
            }
            
            print(f"  {pattern_name.upper()} Accuracy: {pattern_accuracy:.2f}% ({pattern_correct}/{len(queries)})")
        
        return {
            'pattern_results': pattern_results,
            'improvement_suggestions': self._consolidate_improvements(all_improvements),
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_pattern_correctness(self, pattern_name: str, query: str, result) -> bool:
        """Analyze if routing is correct for specific patterns"""
        
        if pattern_name == "lightrag_misclassified":
            # These should route to LIGHTRAG (pathway/mechanism queries)
            return result.routing_decision == RoutingDecision.LIGHTRAG
            
        elif pattern_name == "hybrid_underdetected":
            # These should route to HYBRID (dual system queries)
            return result.routing_decision == RoutingDecision.HYBRID
            
        elif pattern_name == "temporal_indicators":
            # These should route to PERPLEXITY (temporal queries)
            return result.routing_decision == RoutingDecision.PERPLEXITY
            
        elif pattern_name == "confidence_calibration":
            # Check if confidence aligns with expected routing strength
            if "glycolysis" in query.lower():
                return result.routing_decision == RoutingDecision.LIGHTRAG and result.confidence > 0.7
            elif "published yesterday" in query.lower():
                return result.routing_decision == RoutingDecision.PERPLEXITY and result.confidence > 0.8
            elif "diabetes affect kidney" in query.lower():
                return result.routing_decision in [RoutingDecision.EITHER, RoutingDecision.LIGHTRAG] and result.confidence > 0.3
            elif "aging and recent longevity research" in query.lower():
                return result.routing_decision == RoutingDecision.HYBRID and result.confidence > 0.6
        
        return False
    
    def _generate_improvement_suggestions(self, pattern_name: str, query: str, result) -> List[str]:
        """Generate specific improvement suggestions based on patterns"""
        suggestions = []
        
        if pattern_name == "lightrag_misclassified":
            if result.routing_decision != RoutingDecision.LIGHTRAG:
                suggestions.append(f"Enhance pathway/mechanism keyword detection for: {query[:50]}...")
                suggestions.append("Increase weight for biological process indicators")
                if result.confidence < 0.5:
                    suggestions.append("Improve confidence scoring for molecular mechanism queries")
        
        elif pattern_name == "hybrid_underdetected":
            if result.routing_decision != RoutingDecision.HYBRID:
                suggestions.append(f"Improve dual-requirement detection for: {query[:50]}...")
                suggestions.append("Enhance combination keyword pattern matching")
                if "and recent" in query.lower() or "and current" in query.lower():
                    suggestions.append("Strengthen temporal + knowledge hybrid detection")
        
        elif pattern_name == "temporal_indicators":
            if result.routing_decision != RoutingDecision.PERPLEXITY:
                suggestions.append(f"Strengthen temporal indicator detection for: {query[:50]}...")
                suggestions.append("Improve year/date pattern recognition")
                if result.confidence < 0.7:
                    suggestions.append("Boost confidence for strong temporal indicators")
        
        elif pattern_name == "confidence_calibration":
            expected_confidence = self._get_expected_confidence(query)
            if abs(result.confidence - expected_confidence) > 0.3:
                suggestions.append(f"Calibrate confidence scoring for query type: {query[:50]}...")
                suggestions.append("Adjust confidence thresholds for different query categories")
        
        return suggestions
    
    def _get_expected_confidence(self, query: str) -> float:
        """Get expected confidence for calibration testing"""
        if "glycolysis" in query.lower():
            return 0.9  # High confidence for well-known pathway
        elif "published yesterday" in query.lower():
            return 0.95  # Very high confidence for strong temporal
        elif "diabetes affect kidney" in query.lower():
            return 0.5  # Medium confidence for general medical
        elif "aging and recent longevity research" in query.lower():
            return 0.8  # High confidence for clear hybrid pattern
        return 0.5
    
    def _consolidate_improvements(self, all_improvements: List[str]) -> List[Dict[str, Any]]:
        """Consolidate and prioritize improvement suggestions"""
        suggestion_counts = defaultdict(int)
        for suggestion in all_improvements:
            suggestion_counts[suggestion] += 1
        
        prioritized_suggestions = []
        for suggestion, count in sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True):
            prioritized_suggestions.append({
                'suggestion': suggestion,
                'frequency': count,
                'priority': 'high' if count >= 3 else 'medium' if count >= 2 else 'low'
            })
        
        return prioritized_suggestions
    
    def test_confidence_accuracy_correlation(self) -> Dict[str, Any]:
        """Test correlation between confidence scores and actual accuracy"""
        print("\nTesting Confidence-Accuracy Correlation...")
        
        test_queries = [
            ("What are the metabolic pathways in glycolysis?", RoutingDecision.LIGHTRAG),
            ("Latest COVID-19 research published this week", RoutingDecision.PERPLEXITY),
            ("How does exercise affect health?", RoutingDecision.EITHER),
            ("What are cancer pathways and recent immunotherapy trials?", RoutingDecision.HYBRID)
        ]
        
        confidence_accuracy_data = []
        
        for query, expected_decision in test_queries:
            result = self.router.route_query(query)
            is_correct = (result.routing_decision == expected_decision) or \
                        (expected_decision == RoutingDecision.EITHER and 
                         result.routing_decision in [RoutingDecision.LIGHTRAG, RoutingDecision.PERPLEXITY])
            
            confidence_accuracy_data.append({
                'query': query,
                'confidence': result.confidence,
                'is_correct': is_correct,
                'expected_decision': expected_decision.value,
                'actual_decision': result.routing_decision.value
            })
        
        # Calculate correlation metrics
        correct_confidences = [item['confidence'] for item in confidence_accuracy_data if item['is_correct']]
        incorrect_confidences = [item['confidence'] for item in confidence_accuracy_data if not item['is_correct']]
        
        correlation_metrics = {
            'correct_queries_avg_confidence': statistics.mean(correct_confidences) if correct_confidences else 0,
            'incorrect_queries_avg_confidence': statistics.mean(incorrect_confidences) if incorrect_confidences else 0,
            'confidence_discrimination': 0,
            'details': confidence_accuracy_data
        }
        
        if correct_confidences and incorrect_confidences:
            correlation_metrics['confidence_discrimination'] = \
                correlation_metrics['correct_queries_avg_confidence'] - \
                correlation_metrics['incorrect_queries_avg_confidence']
        
        return correlation_metrics
    
    def generate_improvement_roadmap(self, edge_case_results: Dict, confidence_results: Dict) -> str:
        """Generate comprehensive improvement roadmap"""
        roadmap = []
        roadmap.append("="*80)
        roadmap.append("ROUTING ACCURACY IMPROVEMENT ROADMAP")
        roadmap.append(f"Generated: {datetime.now().isoformat()}")
        roadmap.append("="*80)
        
        # Pattern-specific accuracy analysis
        roadmap.append("\n1. PATTERN-SPECIFIC ACCURACY ANALYSIS:")
        roadmap.append("="*50)
        
        for pattern_name, results in edge_case_results['pattern_results'].items():
            accuracy = results['accuracy_percentage']
            status = "✓ GOOD" if accuracy >= 80 else "⚠ NEEDS WORK" if accuracy >= 60 else "✗ CRITICAL"
            roadmap.append(f"{pattern_name.upper()}: {accuracy:.1f}% {status}")
        
        # Priority improvements
        roadmap.append("\n2. PRIORITY IMPROVEMENTS:")
        roadmap.append("="*50)
        
        high_priority = [s for s in edge_case_results['improvement_suggestions'] if s['priority'] == 'high']
        if high_priority:
            roadmap.append("HIGH PRIORITY:")
            for suggestion in high_priority[:5]:  # Top 5
                roadmap.append(f"  • {suggestion['suggestion']} (frequency: {suggestion['frequency']})")
        
        medium_priority = [s for s in edge_case_results['improvement_suggestions'] if s['priority'] == 'medium']
        if medium_priority:
            roadmap.append("\nMEDIUM PRIORITY:")
            for suggestion in medium_priority[:3]:  # Top 3
                roadmap.append(f"  • {suggestion['suggestion']} (frequency: {suggestion['frequency']})")
        
        # Confidence calibration
        roadmap.append(f"\n3. CONFIDENCE CALIBRATION:")
        roadmap.append("="*50)
        roadmap.append(f"Correct predictions avg confidence: {confidence_results['correct_queries_avg_confidence']:.3f}")
        roadmap.append(f"Incorrect predictions avg confidence: {confidence_results['incorrect_queries_avg_confidence']:.3f}")
        roadmap.append(f"Confidence discrimination: {confidence_results['confidence_discrimination']:.3f}")
        
        if confidence_results['confidence_discrimination'] < 0.2:
            roadmap.append("⚠ Poor confidence calibration - correct predictions should have higher confidence")
        else:
            roadmap.append("✓ Good confidence discrimination between correct/incorrect predictions")
        
        # Implementation recommendations
        roadmap.append(f"\n4. IMPLEMENTATION RECOMMENDATIONS:")
        roadmap.append("="*50)
        roadmap.append("A. Immediate Actions (Week 1):")
        roadmap.append("   - Enhance keyword detection for pathway/mechanism queries")
        roadmap.append("   - Improve temporal indicator pattern matching")
        roadmap.append("   - Calibrate confidence thresholds per category")
        
        roadmap.append("\nB. Short-term Improvements (2-4 weeks):")
        roadmap.append("   - Implement hybrid query detection improvements")
        roadmap.append("   - Add biomedical entity count weighting")
        roadmap.append("   - Create confidence score validation framework")
        
        roadmap.append("\nC. Long-term Enhancements (1-2 months):")
        roadmap.append("   - Machine learning model fine-tuning")
        roadmap.append("   - Advanced pattern recognition for complex queries")
        roadmap.append("   - Continuous accuracy monitoring system")
        
        # Success metrics
        roadmap.append(f"\n5. SUCCESS METRICS:")
        roadmap.append("="*50)
        roadmap.append("Target Goals:")
        roadmap.append("  • Overall routing accuracy: >90%")
        roadmap.append("  • LIGHTRAG pathway queries: >85%")
        roadmap.append("  • HYBRID detection: >75%")
        roadmap.append("  • Temporal queries: >90%")
        roadmap.append("  • Confidence discrimination: >0.25")
        
        return "\n".join(roadmap)


def main():
    """Main execution function for enhanced accuracy testing"""
    print("Initializing Enhanced Routing Accuracy Tester...")
    
    try:
        tester = EnhancedAccuracyTester()
        
        # Run edge case testing
        print("Running edge case pattern testing...")
        edge_case_results = tester.test_edge_cases()
        
        # Run confidence-accuracy correlation testing
        print("Running confidence-accuracy correlation testing...")
        confidence_results = tester.test_confidence_accuracy_correlation()
        
        # Generate improvement roadmap
        roadmap = tester.generate_improvement_roadmap(edge_case_results, confidence_results)
        print(roadmap)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save edge case results
        edge_case_file = f"enhanced_accuracy_edge_cases_{timestamp}.json"
        with open(edge_case_file, 'w') as f:
            json.dump(edge_case_results, f, indent=2, default=str)
        
        # Save confidence results
        confidence_file = f"confidence_accuracy_correlation_{timestamp}.json"
        with open(confidence_file, 'w') as f:
            json.dump(confidence_results, f, indent=2, default=str)
        
        # Save roadmap
        roadmap_file = f"routing_improvement_roadmap_{timestamp}.txt"
        with open(roadmap_file, 'w') as f:
            f.write(roadmap)
        
        print(f"\nDetailed results saved:")
        print(f"- Edge cases: {edge_case_file}")
        print(f"- Confidence correlation: {confidence_file}")
        print(f"- Improvement roadmap: {roadmap_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during enhanced testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)