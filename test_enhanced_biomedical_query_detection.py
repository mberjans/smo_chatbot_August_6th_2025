#!/usr/bin/env python3
"""
Test script for enhanced biomedical query mode selection logic.

This script demonstrates the improved query pattern detection and mode routing
with confidence scoring and clinical context awareness.
"""

import sys
import os
import json
from pathlib import Path

# Add the lightrag_integration directory to the path
sys.path.insert(0, str(Path(__file__).parent / "lightrag_integration"))

try:
    from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
    from lightrag_integration.progress_config import ProgressTrackingConfig
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running from the correct directory")
    
    # Fallback: try adding both current and parent directories to path
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    for dir_path in [current_dir, parent_dir]:
        lightrag_path = dir_path / "lightrag_integration" 
        if lightrag_path.exists():
            sys.path.insert(0, str(dir_path))
            break
    
    try:
        from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
        from lightrag_integration.progress_config import ProgressTrackingConfig
        print("✅ Successfully imported after path adjustment")
    except ImportError as e2:
        print(f"❌ Still failed after path adjustment: {e2}")
        sys.exit(1)


def test_enhanced_query_detection():
    """Test the enhanced biomedical query pattern detection system."""
    
    # Initialize the RAG system with minimal config for testing
    try:
        config = ProgressTrackingConfig()
        rag = ClinicalMetabolomicsRAG(
            kb_directory="./test_kb_directory",  # Will use default if doesn't exist
            config=config
        )
        print("✅ Successfully initialized ClinicalMetabolomicsRAG")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    # Test queries for different clinical metabolomics patterns
    test_queries = {
        # Metabolite identification queries
        "metabolite_identification": [
            "What is the chemical structure of glucose?",
            "Identify the metabolite at m/z 180.063",
            "What is creatinine and what does it indicate?",
            "Chemical structure of cholesterol",
            "MS/MS identification of amino acid peaks",
            "Molecular formula of fatty acid C18:1"
        ],
        
        # Pathway analysis queries  
        "pathway_analysis": [
            "Explain the glycolysis pathway",
            "How does the TCA cycle regulate metabolism?",
            "Pathway analysis of fatty acid oxidation",
            "Metabolic network interactions in diabetes",
            "Pentose phosphate pathway regulation",
            "Cholesterol biosynthesis pathway steps"
        ],
        
        # Biomarker discovery queries
        "biomarker_discovery": [
            "Biomarker discovery for diabetes",
            "Prognostic biomarkers in cancer",
            "Diagnostic biomarker validation methods",
            "Metabolomic biomarker panel development",
            "ROC analysis of biomarker performance",
            "Biomarker screening in cardiovascular disease"
        ],
        
        # Disease association queries
        "disease_association": [
            "Diabetes metabolomics associations",
            "Cancer metabolite dysregulation",
            "Disease mechanisms in Alzheimer's",
            "Metabolic dysfunction in obesity",
            "Cardiovascular disease metabolomics",
            "Kidney disease biofluid analysis"
        ],
        
        # Clinical diagnostic queries (new)
        "clinical_diagnostic": [
            "Clinical diagnosis using metabolomics",
            "Point of care metabolite testing",
            "Diagnostic accuracy of metabolomic tests",
            "Clinical utility of biomarker panels",
            "Patient screening protocols",
            "Therapeutic monitoring strategies"
        ],
        
        # Therapeutic target queries (new)
        "therapeutic_target": [
            "Drug target identification in metabolism",
            "Therapeutic targets for diabetes",
            "Enzyme targets for drug development",
            "Molecular docking of metabolites",
            "Structure-activity relationships",
            "Druggable metabolic pathways"
        ],
        
        # Platform-specific queries
        "platform_specific": [
            "LC-MS analysis of plasma metabolites",
            "GC-MS volatile metabolite profiling", 
            "NMR spectroscopy of urine samples",
            "Targeted metabolomics using MRM",
            "Untargeted metabolomic profiling methods",
            "Mass spectrometry metabolite annotation"
        ],
        
        # Mixed/uncertain queries (should use fallback)
        "uncertain": [
            "Tell me about metabolism",
            "What is metabolomics?",
            "Research methods in biochemistry",
            "Laboratory equipment maintenance",
            "Data analysis software recommendations"
        ]
    }
    
    print("\n🔬 Testing Enhanced Biomedical Query Pattern Detection")
    print("=" * 70)
    
    results = {}
    
    for category, queries in test_queries.items():
        print(f"\n📋 Testing {category.upper()} queries:")
        print("-" * 50)
        
        category_results = []
        
        for query in queries:
            try:
                # Test the enhanced smart query parameter detection
                params = rag.get_smart_query_params(query)
                
                detected_pattern = params.get('_pattern_detected')
                suggested_mode = params.get('_suggested_mode')
                confidence_level = params.get('_confidence_level')
                clinical_context = params.get('_has_clinical_terms')
                clinical_boost = params.get('_clinical_context_boost', False)
                
                # Determine if detection was correct
                if category == "platform_specific":
                    correct = detected_pattern and detected_pattern.startswith('platform_specific.')
                elif category == "uncertain":
                    correct = detected_pattern is None or confidence_level in ['hybrid_fallback', 'specified_fallback']
                else:
                    correct = detected_pattern == category
                
                status = "✅" if correct else "❌"
                
                print(f"{status} Query: {query[:50]}{'...' if len(query) > 50 else ''}")
                print(f"   Pattern: {detected_pattern or 'None'}")
                print(f"   Mode: {suggested_mode or 'Default'}")
                print(f"   Confidence: {confidence_level}")
                print(f"   Clinical: {clinical_context} {'(boosted)' if clinical_boost else ''}")
                print(f"   Tokens: {params.get('max_total_tokens')}, K: {params.get('top_k')}")
                print()
                
                category_results.append({
                    'query': query,
                    'detected_pattern': detected_pattern,
                    'suggested_mode': suggested_mode,
                    'confidence_level': confidence_level,
                    'clinical_context': clinical_context,
                    'clinical_boost': clinical_boost,
                    'correct': correct,
                    'tokens': params.get('max_total_tokens'),
                    'top_k': params.get('top_k')
                })
                
            except Exception as e:
                print(f"❌ Error processing query '{query[:30]}...': {e}")
                category_results.append({
                    'query': query,
                    'error': str(e),
                    'correct': False
                })
        
        results[category] = category_results
    
    # Summary statistics
    print("\n📊 DETECTION ACCURACY SUMMARY")
    print("=" * 50)
    
    total_correct = 0
    total_queries = 0
    
    for category, category_results in results.items():
        correct_count = sum(1 for r in category_results if r.get('correct', False))
        total_count = len(category_results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        total_correct += correct_count
        total_queries += total_count
        
        print(f"{category:20s}: {correct_count:2d}/{total_count:2d} ({accuracy:5.1f}%)")
    
    overall_accuracy = (total_correct / total_queries * 100) if total_queries > 0 else 0
    print(f"{'OVERALL':20s}: {total_correct:2d}/{total_queries:2d} ({overall_accuracy:5.1f}%)")
    
    # Save detailed results
    output_file = Path(__file__).parent / "enhanced_biomedical_query_detection_results.json"
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
    
    # Test mode routing logic specifically
    print("\n🎯 MODE ROUTING VERIFICATION")
    print("=" * 40)
    
    mode_tests = [
        ("What is glucose structure?", "local", "metabolite_identification"),
        ("Explain TCA cycle pathway", "global", "pathway_analysis"), 
        ("Find diabetes biomarkers", "hybrid", "biomarker_discovery"),
        ("Clinical diagnosis using LC-MS", "hybrid", "clinical_diagnostic"),
        ("Drug targets for cancer", "global", "therapeutic_target"),
        ("Compare metabolomic studies", "global", "comparative_analysis")
    ]
    
    for query, expected_mode, expected_pattern in mode_tests:
        params = rag.get_smart_query_params(query)
        actual_mode = params.get('_suggested_mode')
        actual_pattern = params.get('_pattern_detected')
        
        mode_correct = actual_mode == expected_mode
        pattern_correct = actual_pattern == expected_pattern
        
        status_mode = "✅" if mode_correct else "❌"
        status_pattern = "✅" if pattern_correct else "❌"
        
        print(f"Query: {query}")
        print(f"  {status_pattern} Pattern: {actual_pattern} (expected: {expected_pattern})")
        print(f"  {status_mode} Mode: {actual_mode} (expected: {expected_mode})")
        print()
    
    print("🏁 Enhanced biomedical query detection testing completed!")
    return results


if __name__ == "__main__":
    test_enhanced_query_detection()