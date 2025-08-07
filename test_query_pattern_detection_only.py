#!/usr/bin/env python3
"""
Simplified test script for enhanced biomedical query pattern detection.

This script tests only the query pattern detection methods without full RAG initialization.
"""

import sys
import json
from pathlib import Path

# Add the lightrag_integration directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_pattern_detection():
    """Test the enhanced query pattern detection logic directly."""
    
    # Import the class and inspect its biomedical_params
    try:
        from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
        print("✅ Successfully imported ClinicalMetabolomicsRAG")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return
    
    # Create a mock instance to access the biomedical_params
    class MockRAG:
        def __init__(self):
            # Copy the biomedical configuration from the actual class
            temp_instance = type('temp', (), {})()
            temp_instance._initialize_biomedical_params = ClinicalMetabolomicsRAG._initialize_biomedical_params.__get__(temp_instance, type(temp_instance))
            self.biomedical_params = temp_instance._initialize_biomedical_params()
            
        def _detect_query_pattern(self, query: str):
            """Simplified version of pattern detection for testing."""
            import re
            from typing import Dict, Tuple, List
            
            query_lower = query.lower()
            optimization_params = self.biomedical_params.get('query_optimization', {})
            
            # Pattern detection with confidence scoring
            pattern_matches: List[Tuple[str, float, str]] = []
            
            # Priority-ordered pattern configurations
            pattern_configs = [
                'metabolite_identification', 'clinical_diagnostic', 'therapeutic_target',
                'pathway_analysis', 'biomarker_discovery', 'comparative_analysis',
                'disease_association'
            ]
            
            # Check each pattern type with confidence scoring
            for pattern_type in pattern_configs:
                if pattern_type not in optimization_params:
                    continue
                    
                config = optimization_params[pattern_type]
                patterns = config.get('query_patterns', [])
                confidence_threshold = config.get('confidence_threshold', 0.7)
                
                pattern_score = 0.0
                matched_patterns = []
                
                for pattern in patterns:
                    try:
                        if re.search(pattern, query_lower):
                            # Simple pattern specificity score
                            pattern_specificity = min(len(pattern) * 0.05, 0.8) + 0.2
                            pattern_score += pattern_specificity
                            matched_patterns.append(pattern)
                            
                    except re.error as e:
                        continue
                
                # Normalize score
                if matched_patterns:
                    match_bonus = min(len(matched_patterns) * 0.1, 0.3)
                    final_score = min(pattern_score + match_bonus, 1.0)
                    
                    if final_score >= confidence_threshold:
                        pattern_matches.append((pattern_type, final_score, ', '.join(matched_patterns[:2])))
            
            # Check platform-specific patterns
            platform_configs = optimization_params.get('platform_specific', {})
            for platform_type, config in platform_configs.items():
                patterns = config.get('query_patterns', [])
                platform_score = 0.0
                matched_patterns = []
                
                for pattern in patterns:
                    try:
                        if re.search(pattern, query_lower):
                            pattern_specificity = min(len(pattern) * 0.05, 0.8) + 0.2
                            platform_score += pattern_specificity
                            matched_patterns.append(pattern)
                    except re.error:
                        continue
                
                if matched_patterns and platform_score >= 0.6:
                    match_bonus = min(len(matched_patterns) * 0.1, 0.2)
                    final_score = min(platform_score + match_bonus, 1.0)
                    pattern_matches.append((f"platform_specific.{platform_type}", final_score, 
                                          ', '.join(matched_patterns[:2])))
            
            # Select best match
            if pattern_matches:
                pattern_matches.sort(key=lambda x: x[1], reverse=True)
                return pattern_matches[0][0], pattern_matches[0][1], pattern_matches[0][2]
            
            return None, 0.0, ""
    
    # Initialize mock RAG for testing
    mock_rag = MockRAG()
    print("✅ Created mock RAG instance for testing")
    
    # Test queries for different clinical metabolomics patterns
    test_queries = {
        # Metabolite identification queries
        "metabolite_identification": [
            "What is the chemical structure of glucose?",
            "Identify the metabolite at m/z 180.063",
            "What is creatinine and what does it indicate?",
            "Chemical structure of cholesterol",
            "MS/MS identification of amino acid peaks"
        ],
        
        # Pathway analysis queries  
        "pathway_analysis": [
            "Explain the glycolysis pathway",
            "How does the TCA cycle regulate metabolism?", 
            "Pathway analysis of fatty acid oxidation",
            "Metabolic network interactions in diabetes",
            "Pentose phosphate pathway regulation"
        ],
        
        # Biomarker discovery queries
        "biomarker_discovery": [
            "Biomarker discovery for diabetes",
            "Prognostic biomarkers in cancer",
            "Diagnostic biomarker validation methods",
            "Metabolomic biomarker panel development",
            "ROC analysis of biomarker performance"
        ],
        
        # Disease association queries
        "disease_association": [
            "Diabetes metabolomics associations",
            "Cancer metabolite dysregulation", 
            "Disease mechanisms in Alzheimer's",
            "Metabolic dysfunction in obesity",
            "Cardiovascular disease metabolomics"
        ],
        
        # Clinical diagnostic queries (new)
        "clinical_diagnostic": [
            "Clinical diagnosis using metabolomics",
            "Point of care metabolite testing",
            "Diagnostic accuracy of metabolomic tests",
            "Clinical utility of biomarker panels",
            "Patient screening protocols"
        ],
        
        # Therapeutic target queries (new)
        "therapeutic_target": [
            "Drug target identification in metabolism",
            "Therapeutic targets for diabetes",
            "Enzyme targets for drug development", 
            "Molecular docking of metabolites",
            "Structure-activity relationships"
        ],
        
        # Platform-specific queries
        "platform_specific": [
            "LC-MS analysis of plasma metabolites",
            "GC-MS volatile metabolite profiling",
            "NMR spectroscopy of urine samples",
            "Targeted metabolomics using MRM",
            "Untargeted metabolomic profiling methods"
        ],
        
        # Uncertain queries (should not match or have low confidence)
        "uncertain": [
            "Tell me about metabolism",
            "What is metabolomics?",
            "Research methods in biochemistry",
            "Laboratory equipment maintenance"
        ]
    }
    
    print("\n🔬 Testing Enhanced Biomedical Query Pattern Detection")
    print("=" * 70)
    
    results = {}
    total_correct = 0
    total_queries = 0
    
    for category, queries in test_queries.items():
        print(f"\n📋 Testing {category.upper()} queries:")
        print("-" * 50)
        
        category_results = []
        
        for query in queries:
            detected_pattern, confidence, matched = mock_rag._detect_query_pattern(query)
            
            # Determine if detection was correct
            if category == "platform_specific":
                correct = detected_pattern and detected_pattern.startswith('platform_specific.')
            elif category == "uncertain":
                correct = detected_pattern is None or confidence < 0.7
            else:
                correct = detected_pattern == category
            
            status = "✅" if correct else "❌"
            total_correct += 1 if correct else 0
            total_queries += 1
            
            print(f"{status} Query: {query[:50]}{'...' if len(query) > 50 else ''}")
            print(f"   Pattern: {detected_pattern or 'None'}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Matched: {matched[:50]}{'...' if len(matched) > 50 else ''}")
            print()
            
            category_results.append({
                'query': query,
                'detected_pattern': detected_pattern,
                'confidence': confidence,
                'matched_patterns': matched,
                'correct': correct
            })
        
        results[category] = category_results
    
    # Summary statistics
    print("\n📊 DETECTION ACCURACY SUMMARY")
    print("=" * 50)
    
    for category, category_results in results.items():
        correct_count = sum(1 for r in category_results if r.get('correct', False))
        total_count = len(category_results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        print(f"{category:20s}: {correct_count:2d}/{total_count:2d} ({accuracy:5.1f}%)")
    
    overall_accuracy = (total_correct / total_queries * 100) if total_queries > 0 else 0
    print(f"{'OVERALL':20s}: {total_correct:2d}/{total_queries:2d} ({overall_accuracy:5.1f}%)")
    
    # Save detailed results
    output_file = Path(__file__).parent / "query_pattern_detection_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_correct': total_correct,
                'total_queries': total_queries,
                'overall_accuracy': overall_accuracy
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    # Test specific enhancement features
    print("\n🎯 ENHANCED FEATURES VERIFICATION")
    print("=" * 40)
    
    enhancement_tests = [
        ("LC-MS analysis of glucose", "Should detect platform_specific.lc_ms"),
        ("Clinical biomarker validation study", "Should detect clinical_diagnostic"),
        ("Drug target enzyme identification", "Should detect therapeutic_target"),
        ("Compare metabolomic study results", "Should detect comparative_analysis"),
        ("Metabolic pathway network analysis", "Should detect pathway_analysis"),
        ("Random text about science", "Should not detect specific pattern")
    ]
    
    for query, expectation in enhancement_tests:
        detected_pattern, confidence, matched = mock_rag._detect_query_pattern(query)
        print(f"Query: {query}")
        print(f"  Result: {detected_pattern or 'None'} (confidence: {confidence:.3f})")
        print(f"  Expected: {expectation}")
        print()
    
    print("🏁 Enhanced query pattern detection testing completed!")
    return results


if __name__ == "__main__":
    test_pattern_detection()