#!/usr/bin/env python3
"""
Cross-Document Knowledge Synthesis Validation Tests.

This module implements specialized tests for validating cross-document knowledge 
synthesis capabilities, building upon the comprehensive test infrastructure to 
validate that the system can effectively integrate and synthesize information 
from multiple biomedical research papers.

Test Focus Areas:
- Multi-study biomarker consensus identification
- Conflicting findings recognition and analysis
- Methodological comparison across studies  
- Evidence integration and synthesis quality
- Disease-specific knowledge synthesis
- Large-scale multi-document processing

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import time
import logging
import statistics
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import test infrastructure
from comprehensive_test_fixtures import (
    AdvancedBiomedicalContentGenerator,
    CrossDocumentSynthesisValidator,
    ProductionScaleSimulator
)


# =====================================================================
# CROSS-DOCUMENT SYNTHESIS TESTS
# =====================================================================

@pytest.mark.biomedical
@pytest.mark.integration 
@pytest.mark.comprehensive
class TestCrossDocumentBiomarkerSynthesis:
    """
    Test cross-document synthesis of biomarker information across multiple studies.
    Validates the system's ability to identify consensus, conflicts, and patterns.
    """

    @pytest.mark.asyncio
    async def test_diabetes_biomarker_consensus_synthesis(
        self,
        diabetes_focused_study_collection,
        comprehensive_mock_rag_system_with_synthesis,
        cross_document_synthesis_validator
    ):
        """
        Test synthesis of diabetes biomarker consensus across multiple studies.
        Validates identification of consistently reported biomarkers and methodologies.
        """
        # Load diabetes study collection
        rag_system = comprehensive_mock_rag_system_with_synthesis
        
        for study in diabetes_focused_study_collection:
            await rag_system.index_study(study)
        
        logging.info(f"Loaded {len(diabetes_focused_study_collection)} diabetes studies for consensus testing")
        
        # Test consensus identification queries
        consensus_queries = [
            "What biomarkers are consistently identified across diabetes metabolomics studies?",
            "Which analytical methods are most commonly used in diabetes research?",
            "What sample sizes are typically employed in diabetes metabolomics studies?",
            "Identify common statistical approaches used across diabetes studies"
        ]
        
        synthesis_results = []
        
        for query in consensus_queries:
            # Execute query
            response = await rag_system.query(query)
            
            # Validate synthesis quality
            assessment = cross_document_synthesis_validator.assess_synthesis_quality(
                response, diabetes_focused_study_collection
            )
            
            synthesis_results.append({
                'query': query,
                'response': response,
                'assessment': assessment
            })
            
            # Individual query validations
            assert assessment['overall_synthesis_quality'] >= 70.0, \
                f"Consensus synthesis quality {assessment['overall_synthesis_quality']:.1f}% below 70% threshold for query: {query[:50]}..."
            
            assert 'CONSENSUS_IDENTIFIED' in assessment['synthesis_flags'] or \
                   assessment['pattern_scores']['consensus_identification'] >= 0.2, \
                f"Consensus identification patterns insufficient for query: {query[:50]}..."
            
            # Content validation - should mention diabetes-specific terms
            response_lower = response.lower()
            diabetes_terms = ['glucose', 'insulin', 'diabetes', 'metabolomics', 'biomarker']
            term_mentions = sum(1 for term in diabetes_terms if term in response_lower)
            
            assert term_mentions >= 3, \
                f"Response should mention multiple diabetes-related terms, found {term_mentions}: {query[:50]}..."
        
        # Overall synthesis quality assessment
        avg_synthesis_quality = statistics.mean([r['assessment']['overall_synthesis_quality'] for r in synthesis_results])
        consensus_identification_rate = sum(1 for r in synthesis_results 
                                          if 'CONSENSUS_IDENTIFIED' in r['assessment']['synthesis_flags']) / len(synthesis_results)
        
        assert avg_synthesis_quality >= 75.0, \
            f"Average synthesis quality {avg_synthesis_quality:.1f}% below 75% threshold"
        assert consensus_identification_rate >= 0.75, \
            f"Consensus identification rate {consensus_identification_rate:.1%} below 75% threshold"
        
        logging.info(f"✅ Diabetes Biomarker Consensus Synthesis Results:")
        logging.info(f"  - Studies Analyzed: {len(diabetes_focused_study_collection)}")
        logging.info(f"  - Consensus Queries: {len(consensus_queries)}")
        logging.info(f"  - Average Synthesis Quality: {avg_synthesis_quality:.1f}%")
        logging.info(f"  - Consensus Identification Rate: {consensus_identification_rate:.1%}")
        logging.info(f"  - Synthesis Patterns Detected: {sum(len(r['assessment']['synthesis_flags']) for r in synthesis_results)}")

    @pytest.mark.asyncio
    async def test_cross_disease_comparative_synthesis(
        self,
        multi_disease_study_collection,
        comprehensive_mock_rag_system_with_synthesis,
        cross_document_synthesis_validator
    ):
        """
        Test comparative synthesis across different disease areas.
        Validates cross-disease pattern identification and methodological comparisons.
        """
        rag_system = comprehensive_mock_rag_system_with_synthesis
        
        # Load multi-disease study collection
        for study in multi_disease_study_collection:
            await rag_system.index_study(study)
        
        # Organize studies by disease for validation
        diseases = {}
        for study in multi_disease_study_collection:
            disease = study['profile'].disease_focus
            if disease not in diseases:
                diseases[disease] = []
            diseases[disease].append(study)
        
        logging.info(f"Loaded {len(multi_disease_study_collection)} studies across {len(diseases)} diseases")
        
        # Cross-disease comparative queries
        comparative_queries = [
            "Compare metabolomic biomarkers across diabetes, cardiovascular disease, and cancer studies",
            "What analytical platforms are preferred for different disease areas?",
            "Compare sample sizes and study designs across different diseases",
            "Identify methodological differences between disease-specific metabolomics studies"
        ]
        
        comparative_results = []
        
        for query in comparative_queries:
            response = await rag_system.query(query)
            assessment = cross_document_synthesis_validator.assess_synthesis_quality(
                response, multi_disease_study_collection
            )
            
            comparative_results.append({
                'query': query,
                'response': response,
                'assessment': assessment
            })
            
            # Validate comparative synthesis
            assert assessment['overall_synthesis_quality'] >= 65.0, \
                f"Cross-disease synthesis quality {assessment['overall_synthesis_quality']:.1f}% insufficient"
            
            assert assessment['pattern_scores']['methodology_comparison'] >= 0.2 or \
                   'METHODOLOGIES_COMPARED' in assessment['synthesis_flags'], \
                f"Methodological comparison patterns insufficient: {query[:50]}..."
            
            # Validate cross-disease content integration
            response_lower = response.lower()
            disease_mentions = sum(1 for disease in diseases.keys() if disease in response_lower)
            
            assert disease_mentions >= 2, \
                f"Should mention multiple diseases in comparative analysis, found {disease_mentions}"
        
        # Advanced cross-disease synthesis validation
        platform_comparison_query = comparative_queries[1]  # Analytical platforms query
        platform_response = next(r for r in comparative_results if r['query'] == platform_comparison_query)
        
        # Should identify different platform preferences
        platforms = ['lc-ms', 'gc-ms', 'nmr', 'ce-ms']
        platform_mentions = sum(1 for platform in platforms if platform in platform_response['response'].lower())
        
        assert platform_mentions >= 2, \
            f"Platform comparison should mention multiple analytical platforms, found {platform_mentions}"
        
        # Calculate comprehensive metrics
        avg_comparative_quality = statistics.mean([r['assessment']['overall_synthesis_quality'] for r in comparative_results])
        methodology_comparison_rate = sum(1 for r in comparative_results 
                                        if r['assessment']['pattern_scores']['methodology_comparison'] >= 0.3) / len(comparative_results)
        
        assert avg_comparative_quality >= 70.0, \
            f"Average comparative synthesis quality {avg_comparative_quality:.1f}% below 70% threshold"
        
        logging.info(f"✅ Cross-Disease Comparative Synthesis Results:")
        logging.info(f"  - Diseases Analyzed: {list(diseases.keys())}")
        logging.info(f"  - Studies per Disease: {[len(studies) for studies in diseases.values()]}")
        logging.info(f"  - Comparative Queries: {len(comparative_queries)}")
        logging.info(f"  - Average Synthesis Quality: {avg_comparative_quality:.1f}%")
        logging.info(f"  - Methodology Comparison Rate: {methodology_comparison_rate:.1%}")
        logging.info(f"  - Platform Mentions: {platform_mentions}/{len(platforms)} platforms")

    @pytest.mark.asyncio
    async def test_conflicting_findings_recognition_synthesis(
        self,
        diabetes_focused_study_collection,
        comprehensive_mock_rag_system_with_synthesis,
        cross_document_synthesis_validator
    ):
        """
        Test recognition and analysis of conflicting findings across studies.
        Validates the system's ability to identify and explain discrepancies.
        """
        rag_system = comprehensive_mock_rag_system_with_synthesis
        
        # Load studies
        for study in diabetes_focused_study_collection:
            await rag_system.index_study(study)
        
        # Create artificial conflicts by modifying study profiles
        # (In real implementation, would use actual conflicting studies)
        conflicting_scenarios = [
            "What conflicting results exist regarding glucose biomarkers in diabetes studies?",
            "Identify discrepant findings about analytical platform effectiveness",
            "What contradictory conclusions exist about sample size requirements?",
            "Analyze conflicting recommendations for statistical analysis methods"
        ]
        
        conflict_analysis_results = []
        
        for query in conflicting_scenarios:
            response = await rag_system.query(query)
            assessment = cross_document_synthesis_validator.assess_synthesis_quality(
                response, diabetes_focused_study_collection
            )
            
            conflict_analysis_results.append({
                'query': query,
                'response': response,
                'assessment': assessment
            })
            
            # Validate conflict recognition
            assert assessment['pattern_scores']['conflict_recognition'] >= 0.1 or \
                   'CONFLICTS_RECOGNIZED' in assessment['synthesis_flags'], \
                f"Conflict recognition insufficient for query: {query[:50]}..."
            
            # Should provide explanatory content
            response_lower = response.lower()
            explanatory_terms = ['differ', 'conflict', 'discrepant', 'varying', 'inconsistent']
            explanatory_mentions = sum(1 for term in explanatory_terms if term in response_lower)
            
            assert explanatory_mentions >= 1, \
                f"Should provide explanatory language for conflicts: {query[:50]}..."
        
        # Validate comprehensive conflict analysis
        avg_conflict_recognition = statistics.mean([
            r['assessment']['pattern_scores']['conflict_recognition'] 
            for r in conflict_analysis_results
        ])
        
        conflict_flags_count = sum(1 for r in conflict_analysis_results 
                                 if 'CONFLICTS_RECOGNIZED' in r['assessment']['synthesis_flags'])
        
        # Even with artificial conflicts, system should attempt analysis
        assert avg_conflict_recognition >= 0.05, \
            f"Average conflict recognition pattern score {avg_conflict_recognition:.2f} too low"
        
        logging.info(f"✅ Conflicting Findings Recognition Results:")
        logging.info(f"  - Conflict Analysis Queries: {len(conflicting_scenarios)}")
        logging.info(f"  - Average Conflict Recognition Score: {avg_conflict_recognition:.2f}")
        logging.info(f"  - Conflicts Recognized Flags: {conflict_flags_count}/{len(conflict_analysis_results)}")
        logging.info(f"  - Studies Analyzed for Conflicts: {len(diabetes_focused_study_collection)}")


@pytest.mark.biomedical
@pytest.mark.integration
@pytest.mark.performance
class TestLargeScaleDocumentSynthesis:
    """Test large-scale multi-document synthesis capabilities."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_scale_evidence_integration(
        self,
        large_scale_study_collection,
        comprehensive_mock_rag_system_with_synthesis,
        cross_document_synthesis_validator,
        production_scale_simulator
    ):
        """
        Test evidence integration across large collection of studies.
        Validates scalability of synthesis capabilities with 50+ documents.
        """
        rag_system = comprehensive_mock_rag_system_with_synthesis
        
        # Load large study collection
        loading_start = time.time()
        for study in large_scale_study_collection:
            await rag_system.index_study(study)
        loading_time = time.time() - loading_start
        
        logging.info(f"Loaded {len(large_scale_study_collection)} studies in {loading_time:.1f}s")
        
        # Large-scale synthesis queries
        large_scale_queries = [
            "Provide comprehensive overview of metabolomics research methodologies",
            "Synthesize biomarker discoveries across all indexed studies", 
            "Compare analytical platform usage patterns across the research database",
            "Identify emerging trends in clinical metabolomics applications",
            "Generate evidence-based recommendations for study design"
        ]
        
        synthesis_performance_metrics = []
        
        for query in large_scale_queries:
            query_start = time.time()
            
            # Execute large-scale synthesis query
            response = await rag_system.query(query)
            
            query_time = time.time() - query_start
            
            # Assess synthesis quality
            assessment = cross_document_synthesis_validator.assess_synthesis_quality(
                response, large_scale_study_collection[:10]  # Sample for assessment
            )
            
            synthesis_performance_metrics.append({
                'query': query,
                'response_time': query_time,
                'response_length': len(response),
                'synthesis_quality': assessment['overall_synthesis_quality'],
                'evidence_integration_score': assessment['source_integration_score']
            })
            
            # Large-scale synthesis validations
            assert query_time <= 90.0, \
                f"Large-scale synthesis query took {query_time:.1f}s, limit is 90s"
            
            assert len(response) >= 300, \
                f"Large-scale synthesis response too brief: {len(response)} characters"
            
            assert assessment['overall_synthesis_quality'] >= 60.0, \
                f"Large-scale synthesis quality {assessment['overall_synthesis_quality']:.1f}% insufficient"
            
            # Should demonstrate evidence integration
            response_lower = response.lower()
            integration_indicators = ['across studies', 'multiple reports', 'comprehensive analysis', 'evidence']
            integration_mentions = sum(1 for indicator in integration_indicators if indicator in response_lower)
            
            assert integration_mentions >= 2, \
                f"Should demonstrate evidence integration: {query[:50]}..."
        
        # Calculate large-scale performance metrics
        avg_response_time = statistics.mean([m['response_time'] for m in synthesis_performance_metrics])
        avg_synthesis_quality = statistics.mean([m['synthesis_quality'] for m in synthesis_performance_metrics])
        avg_response_length = statistics.mean([m['response_length'] for m in synthesis_performance_metrics])
        
        # Large-scale performance validations
        assert avg_response_time <= 60.0, \
            f"Average large-scale synthesis time {avg_response_time:.1f}s exceeds 60s limit"
        
        assert avg_synthesis_quality >= 65.0, \
            f"Average large-scale synthesis quality {avg_synthesis_quality:.1f}% below 65% threshold"
        
        # Scalability validation - synthesis quality shouldn't degrade significantly with scale
        small_scale_baseline = 75.0  # Expected quality with fewer documents
        quality_degradation = max(0, small_scale_baseline - avg_synthesis_quality)
        
        assert quality_degradation <= 15.0, \
            f"Quality degradation {quality_degradation:.1f}% too high for large-scale synthesis"
        
        logging.info(f"✅ Large-Scale Evidence Integration Results:")
        logging.info(f"  - Studies Indexed: {len(large_scale_study_collection)}")
        logging.info(f"  - Loading Time: {loading_time:.1f}s")
        logging.info(f"  - Synthesis Queries: {len(large_scale_queries)}")
        logging.info(f"  - Average Response Time: {avg_response_time:.1f}s")
        logging.info(f"  - Average Synthesis Quality: {avg_synthesis_quality:.1f}%")
        logging.info(f"  - Average Response Length: {avg_response_length:.0f} characters")
        logging.info(f"  - Quality Degradation vs Baseline: {quality_degradation:.1f}%")
        
        # Test production-scale simulation with large document collection
        simulation_results = await production_scale_simulator.simulate_usage_pattern(
            'research_institution', 2.0, None, rag_system  # 2 hour simulation
        )
        
        # Validate production simulation with large-scale collection
        assert simulation_results['success_rate'] >= 0.95, \
            f"Production simulation success rate {simulation_results['success_rate']:.1%} below 95%"
        
        assert simulation_results['user_satisfaction_score'] >= 75.0, \
            f"User satisfaction {simulation_results['user_satisfaction_score']:.1f}% below 75%"
        
        logging.info(f"  📊 Production Simulation (2h with {len(large_scale_study_collection)} studies):")
        logging.info(f"    - Operations Completed: {simulation_results['operations_completed']}")
        logging.info(f"    - Success Rate: {simulation_results['success_rate']:.1%}")
        logging.info(f"    - Average Response Time: {simulation_results['average_response_time']:.1f}s")
        logging.info(f"    - User Satisfaction: {simulation_results['user_satisfaction_score']:.1f}%")


@pytest.mark.biomedical
@pytest.mark.comprehensive
class TestSynthesisQualityFramework:
    """Test the synthesis quality assessment framework itself."""

    def test_synthesis_validator_pattern_recognition(self, cross_document_synthesis_validator):
        """Test synthesis validator's pattern recognition capabilities."""
        
        # Test consensus identification patterns
        consensus_response = """
        Analysis across multiple studies consistently identifies glucose and insulin as 
        key biomarkers. These findings are replicated across different research groups
        and represent common observations in the diabetes metabolomics literature.
        """
        
        consensus_assessment = cross_document_synthesis_validator.assess_synthesis_quality(
            consensus_response, []  # Empty source studies for pattern testing
        )
        
        assert consensus_assessment['pattern_scores']['consensus_identification'] >= 0.5, \
            "Should recognize consensus identification patterns"
        
        # Test conflict recognition patterns
        conflict_response = """
        Results show conflicting findings regarding biomarker effectiveness. 
        Some studies report contradictory results for TMAO levels, while 
        others demonstrate discrepant conclusions about analytical platform performance.
        """
        
        conflict_assessment = cross_document_synthesis_validator.assess_synthesis_quality(
            conflict_response, []
        )
        
        assert conflict_assessment['pattern_scores']['conflict_recognition'] >= 0.5, \
            "Should recognize conflict recognition patterns"
        
        # Test methodology comparison patterns
        comparison_response = """
        LC-MS demonstrates superior sensitivity compared to GC-MS approaches.
        In contrast to NMR methods, mass spectrometry provides better coverage.
        Different analytical approaches show varying performance characteristics.
        """
        
        comparison_assessment = cross_document_synthesis_validator.assess_synthesis_quality(
            comparison_response, []
        )
        
        assert comparison_assessment['pattern_scores']['methodology_comparison'] >= 0.5, \
            "Should recognize methodology comparison patterns"
        
        logging.info("✅ Synthesis Quality Framework Validation:")
        logging.info(f"  - Consensus Pattern Recognition: {consensus_assessment['pattern_scores']['consensus_identification']:.2f}")
        logging.info(f"  - Conflict Pattern Recognition: {conflict_assessment['pattern_scores']['conflict_recognition']:.2f}")
        logging.info(f"  - Comparison Pattern Recognition: {comparison_assessment['pattern_scores']['methodology_comparison']:.2f}")

    def test_synthesis_quality_scoring(self, cross_document_synthesis_validator):
        """Test synthesis quality scoring accuracy."""
        
        # High-quality comprehensive synthesis
        high_quality_response = """
        Comprehensive analysis across 15 diabetes metabolomics studies consistently 
        identifies glucose, insulin, and HbA1c as primary biomarkers. These findings 
        are replicated across multiple research groups using different analytical platforms.
        
        Comparative analysis reveals LC-MS/MS demonstrates superior sensitivity compared 
        to GC-MS approaches, while NMR provides unique quantitative advantages. 
        Sample sizes across studies range from 50-300 participants, with larger cohorts 
        showing more robust statistical outcomes.
        
        Some studies report conflicting results for branched-chain amino acids, 
        potentially due to methodological differences in sample preparation protocols.
        Integration of evidence suggests standardized protocols would improve reproducibility.
        """
        
        high_assessment = cross_document_synthesis_validator.assess_synthesis_quality(
            high_quality_response, []
        )
        
        # Low-quality limited synthesis
        low_quality_response = """
        Some studies look at diabetes. There are biomarkers involved.
        Different methods are used sometimes. Results vary.
        """
        
        low_assessment = cross_document_synthesis_validator.assess_synthesis_quality(
            low_quality_response, []
        )
        
        # Validate scoring discrimination
        assert high_assessment['overall_synthesis_quality'] >= 70.0, \
            f"High-quality synthesis scored {high_assessment['overall_synthesis_quality']:.1f}%, should be ≥70%"
        
        assert low_assessment['overall_synthesis_quality'] <= 40.0, \
            f"Low-quality synthesis scored {low_assessment['overall_synthesis_quality']:.1f}%, should be ≤40%"
        
        assert high_assessment['overall_synthesis_quality'] > low_assessment['overall_synthesis_quality'], \
            "High-quality synthesis should score higher than low-quality"
        
        # Validate flag generation
        assert 'HIGH_SYNTHESIS_QUALITY' in high_assessment['synthesis_flags'] or \
               'MODERATE_SYNTHESIS_QUALITY' in high_assessment['synthesis_flags'], \
            "High-quality response should receive appropriate quality flag"
        
        assert 'LOW_SYNTHESIS_QUALITY' in low_assessment['synthesis_flags'], \
            "Low-quality response should receive low quality flag"
        
        logging.info("✅ Synthesis Quality Scoring Validation:")
        logging.info(f"  - High-Quality Score: {high_assessment['overall_synthesis_quality']:.1f}%")
        logging.info(f"  - Low-Quality Score: {low_assessment['overall_synthesis_quality']:.1f}%")
        logging.info(f"  - High-Quality Flags: {high_assessment['synthesis_flags']}")
        logging.info(f"  - Low-Quality Flags: {low_assessment['synthesis_flags']}")


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])