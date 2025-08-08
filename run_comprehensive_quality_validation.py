#!/usr/bin/env python3
"""
Comprehensive Quality Validation and Benchmarking System Runner.

This script executes the integrated quality workflow to perform comprehensive
quality assessment tests and calculate relevance scores for the Clinical
Metabolomics Oracle system.
"""

import asyncio
import sys
import json
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Add lightrag_integration to path
sys.path.insert(0, str(Path(__file__).parent / 'lightrag_integration'))

try:
    from integrated_quality_workflow import IntegratedQualityWorkflow
except ImportError as e:
    print(f"Error importing IntegratedQualityWorkflow: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def run_quality_assessment():
    print('=' * 80)
    print('COMPREHENSIVE QUALITY VALIDATION AND BENCHMARKING SYSTEM')
    print('=' * 80)
    
    try:
        # Initialize the integrated quality workflow
        print('\n1. Initializing Integrated Quality Workflow...')
        workflow = IntegratedQualityWorkflow()
        
        # Test queries and responses for validation
        test_cases = [
            {
                'query': 'What are the clinical applications of metabolomics in personalized medicine?',
                'response': '''Metabolomics has several important clinical applications in personalized medicine. 
                First, it enables biomarker discovery for disease diagnosis and prognosis. LC-MS and GC-MS platforms 
                are used to analyze metabolite profiles in patient samples. Studies show that metabolomic signatures 
                can predict treatment responses and identify patients who may benefit from specific therapies. 
                Research indicates that metabolomics-based approaches show promise for precision medicine applications 
                in cancer, cardiovascular disease, and metabolic disorders.''',
                'source_documents': ['Metabolomics research paper 1', 'Clinical study on biomarkers'],
                'expected_concepts': ['metabolomics', 'personalized medicine', 'biomarker', 'clinical']
            },
            {
                'query': 'How does LC-MS work in metabolomics analysis?',
                'response': '''Liquid chromatography-mass spectrometry (LC-MS) is a powerful analytical technique 
                used extensively in metabolomics. The LC component separates metabolites based on their chemical 
                properties, while MS identifies and quantifies them based on mass-to-charge ratios. Modern LC-MS 
                systems can detect thousands of metabolites in a single analysis, providing comprehensive metabolic 
                profiles. High-resolution mass spectrometry enables accurate mass measurements for confident 
                metabolite identification.''',
                'source_documents': ['LC-MS methodology paper', 'Analytical chemistry guide'],
                'expected_concepts': ['LC-MS', 'mass spectrometry', 'metabolomics', 'analytical']
            },
            {
                'query': 'What are the main challenges in metabolomics data analysis?',
                'response': '''Metabolomics data analysis faces several key challenges. Data preprocessing is 
                complex due to batch effects, missing values, and normalization requirements. Statistical analysis 
                is complicated by high-dimensional data with relatively small sample sizes. Metabolite identification 
                remains difficult due to incomplete databases and structural complexity. Integration of metabolomics 
                with other omics data presents computational challenges. Reproducibility across different laboratories 
                and platforms is an ongoing concern.''',
                'source_documents': ['Data analysis methods paper', 'Challenges in metabolomics review'],
                'expected_concepts': ['data analysis', 'metabolomics', 'challenges', 'statistics']
            }
        ]
        
        print(f'\n2. Running Comprehensive Quality Assessment on {len(test_cases)} test cases...')
        
        results = []
        total_scores = []
        relevance_scores = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f'\n   Test Case {i}: {test_case["query"][:50]}...')
            
            # Perform comprehensive assessment
            result = await workflow.assess_comprehensive_quality(
                query=test_case['query'],
                response=test_case['response'],
                source_documents=test_case['source_documents'],
                expected_concepts=test_case['expected_concepts']
            )
            
            results.append(result)
            total_scores.append(result.overall_quality_score)
            
            # Extract relevance score if available
            if result.relevance_assessment and 'overall_score' in result.relevance_assessment:
                relevance_scores.append(result.relevance_assessment['overall_score'])
            
            print(f'     ✓ Overall Quality: {result.overall_quality_score:.1f}/100 ({result.quality_grade})')
            print(f'     ✓ Processing Time: {result.processing_time_ms:.2f}ms')
            print(f'     ✓ Components Used: {len(result.components_used)}')
            print(f'     ✓ Confidence: {result.assessment_confidence:.1f}/100')
        
        # Calculate overall metrics
        avg_quality_score = sum(total_scores) / len(total_scores) if total_scores else 0
        avg_relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        print('\n' + '=' * 80)
        print('COMPREHENSIVE QUALITY ASSESSMENT RESULTS')
        print('=' * 80)
        
        print(f'\nOVERALL SYSTEM PERFORMANCE:')
        print(f'  • Total Test Cases Processed: {len(results)}')
        print(f'  • Average Overall Quality Score: {avg_quality_score:.1f}/100')
        print(f'  • Average Relevance Score: {avg_relevance_score:.1f}/100')
        print(f'  • Quality Grade: {"Excellent" if avg_quality_score >= 90 else "Good" if avg_quality_score >= 80 else "Acceptable" if avg_quality_score >= 70 else "Marginal" if avg_quality_score >= 60 else "Poor"}')
        
        # Check 80% threshold requirement
        threshold_met = avg_relevance_score >= 80.0
        print(f'\nRELEVANCE SCORE THRESHOLD VALIDATION:')
        print(f'  • Required Threshold: ≥80%')
        print(f'  • Achieved Score: {avg_relevance_score:.1f}%')
        print(f'  • Threshold Status: {"✓ PASSED" if threshold_met else "✗ FAILED"}')
        
        if threshold_met:
            print(f'  • Result: System meets the >80% relevance score requirement!')
        else:
            print(f'  • Result: System needs improvement to meet the >80% threshold.')
        
        print(f'\nDETAILED COMPONENT ANALYSIS:')
        
        # Analyze component usage
        all_components = set()
        component_usage = defaultdict(int)
        
        for result in results:
            for component in result.components_used:
                all_components.add(component)
                component_usage[component] += 1
        
        print(f'  • Available Components: {len(all_components)}')
        for component, usage in component_usage.items():
            print(f'    - {component}: Used in {usage}/{len(results)} tests ({usage/len(results)*100:.1f}%)')
        
        print(f'\nPERFORMANCE METRICS:')
        processing_times = [r.processing_time_ms for r in results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        print(f'  • Average Processing Time: {avg_processing_time:.2f}ms')
        print(f'  • Processing Performance: {"Excellent" if avg_processing_time <= 1000 else "Good" if avg_processing_time <= 3000 else "Acceptable" if avg_processing_time <= 5000 else "Slow"}')
        
        # Quality insights summary
        print(f'\nQUALITY INSIGHTS SUMMARY:')
        all_strengths = []
        all_improvements = []
        
        for result in results:
            all_strengths.extend(result.strength_areas)
            all_improvements.extend(result.improvement_areas)
        
        # Count most common strengths and improvements
        strength_counts = defaultdict(int)
        improvement_counts = defaultdict(int)
        
        for strength in all_strengths:
            strength_counts[strength] += 1
        for improvement in all_improvements:
            improvement_counts[improvement] += 1
        
        print(f'  • Common Strengths:')
        for strength, count in sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f'    - {strength} (identified in {count} cases)')
        
        print(f'  • Areas for Improvement:')
        for improvement, count in sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f'    - {improvement} (identified in {count} cases)')
        
        print('\n' + '=' * 80)
        print('VALIDATION SUMMARY')
        print('=' * 80)
        
        validation_results = {
            'overall_quality_score': avg_quality_score,
            'relevance_score': avg_relevance_score,
            'threshold_80_percent_met': threshold_met,
            'total_test_cases': len(results),
            'components_active': len(all_components),
            'average_processing_time_ms': avg_processing_time,
            'validation_timestamp': datetime.now().isoformat(),
            'detailed_results': []
        }
        
        for i, result in enumerate(results, 1):
            validation_results['detailed_results'].append({
                'test_case': i,
                'query_preview': test_cases[i-1]['query'][:50] + '...',
                'overall_quality_score': result.overall_quality_score,
                'quality_grade': result.quality_grade,
                'processing_time_ms': result.processing_time_ms,
                'components_used': result.components_used,
                'assessment_confidence': result.assessment_confidence
            })
        
        print(f'\n✓ Comprehensive quality validation completed successfully')
        print(f'✓ Average quality score: {avg_quality_score:.1f}/100')
        print(f'✓ Relevance threshold (>80%): {"PASSED" if threshold_met else "FAILED"} ({avg_relevance_score:.1f}%)')
        print(f'✓ System components: {len(all_components)} active')
        print(f'✓ Performance: {avg_processing_time:.2f}ms average processing time')
        
        # Save detailed results to file
        results_file = '/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/comprehensive_quality_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print(f'\n📄 Detailed results saved to: {results_file}')
        
        # Get performance statistics from workflow
        perf_stats = workflow.get_performance_statistics()
        if perf_stats.get('status') != 'no_data':
            print(f'\nWORKFLOW PERFORMANCE STATISTICS:')
            print(f'  • Total Assessments: {perf_stats.get("total_assessments", 0)}')
            print(f'  • Average Quality Score: {perf_stats.get("avg_quality_score", 0):.1f}')
            print(f'  • Median Processing Time: {perf_stats.get("median_processing_time_ms", 0):.1f}ms')
        
        return validation_results
        
    except Exception as e:
        print(f'\n❌ Error during quality assessment: {str(e)}')
        import traceback
        print('\nFull traceback:')
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the quality assessment
    result = asyncio.run(run_quality_assessment())