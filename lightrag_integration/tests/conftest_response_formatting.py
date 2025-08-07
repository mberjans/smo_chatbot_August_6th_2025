#!/usr/bin/env python3
"""
Pytest configuration and fixtures for response formatting tests.

This module provides shared fixtures and configuration for comprehensive
response formatting tests. It includes mock data, test utilities, and
performance monitoring fixtures.
"""

import pytest
import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Configure test logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_biomedical_entities():
    """Mock biomedical entities for testing entity extraction."""
    return {
        'metabolites': [
            'glucose', 'fructose', 'sucrose', 'lactose', 'pyruvate', 'lactate',
            'citrate', 'succinate', 'acetyl-CoA', 'NADH', 'ATP', 'creatinine',
            '1,5-anhydroglucitol', '2-hydroxybutyrate', 'beta-hydroxybutyrate'
        ],
        'proteins': [
            'hexokinase', 'pyruvate kinase', 'citrate synthase', 'insulin',
            'glucagon', 'HbA1c', 'albumin', 'transferrin', 'ferritin',
            'alpha-ketoglutarate dehydrogenase', 'phosphofructokinase'
        ],
        'pathways': [
            'glycolysis', 'gluconeogenesis', 'TCA cycle', 'pentose phosphate pathway',
            'fatty acid oxidation', 'amino acid metabolism', 'purine metabolism',
            'pyrimidine metabolism', 'cholesterol biosynthesis', 'ketogenesis'
        ],
        'diseases': [
            'diabetes mellitus', 'type 2 diabetes', 'metabolic syndrome',
            'cardiovascular disease', 'obesity', 'insulin resistance',
            'diabetic nephropathy', 'diabetic retinopathy', 'hyperlipidemia'
        ]
    }


@pytest.fixture
def mock_statistical_data():
    """Mock statistical data for testing statistical extraction."""
    return [
        {
            'type': 'p_value',
            'value': 0.001,
            'text': 'p < 0.001',
            'context': 'glucose levels comparison',
            'raw_text': 'glucose levels were significantly elevated (p < 0.001)'
        },
        {
            'type': 'mean_sd',
            'value': {'mean': 15.2, 'sd': 3.4},
            'text': '15.2 ± 3.4 mmol/L',
            'context': 'glucose concentration',
            'raw_text': 'glucose concentrations were 15.2 ± 3.4 mmol/L'
        },
        {
            'type': 'confidence_interval',
            'value': {'lower': 1.8, 'upper': 2.9},
            'text': '95% CI: 1.8-2.9',
            'context': 'fold change analysis',
            'raw_text': '2.3-fold increase (95% CI: 1.8-2.9)'
        },
        {
            'type': 'sensitivity_specificity',
            'value': {'sensitivity': 87, 'specificity': 82},
            'text': 'sensitivity: 87%, specificity: 82%',
            'context': 'diagnostic performance',
            'raw_text': '1,5-anhydroglucitol (sensitivity: 87%, specificity: 82%)'
        }
    ]


@pytest.fixture
def mock_citations():
    """Mock citation data for testing citation processing."""
    return [
        {
            'type': 'journal_article',
            'title': 'Metabolomics in Type 2 Diabetes',
            'authors': ['Smith, J.', 'Doe, A.'],
            'journal': 'Nature Medicine',
            'year': 2023,
            'volume': '29',
            'pages': '123-135',
            'doi': '10.1038/s41591-023-02234-x',
            'pmid': '37123456',
            'text': 'Smith et al. (2023). Metabolomics in diabetes. Nature Medicine 29:123-135.',
            'credibility_score': 0.95
        },
        {
            'type': 'journal_article',
            'title': 'Glycolysis Disruption in T2DM',
            'authors': ['Johnson, A.', 'Brown, B.'],
            'journal': 'Diabetes Care',
            'year': 2022,
            'volume': '45',
            'issue': '8',
            'pages': '1234-1245',
            'pmid': '35123456',
            'text': 'Johnson A, Brown B. Glycolysis disruption in T2DM. Diabetes Care. 2022;45(8):1234-1245.',
            'credibility_score': 0.88
        },
        {
            'type': 'journal_article',
            'title': 'Clinical Metabolomics Applications',
            'authors': ['Lee, K.H.', 'Park, J.W.'],
            'journal': 'Cell Metabolism',
            'year': 2023,
            'volume': '37',
            'issue': '4',
            'pages': '789-802',
            'pmcid': 'PMC9876543',
            'text': 'Lee KH, Park JW. Clinical metabolomics applications. Cell Metab. 2023;37(4):789-802.',
            'credibility_score': 0.92
        }
    ]


@pytest.fixture
def performance_monitor():
    """Fixture for monitoring test performance."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.metrics = {}
        
        def start(self):
            self.start_time = time.time()
        
        def end(self):
            self.end_time = time.time()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        def record_metric(self, name: str, value: Any):
            self.metrics[name] = value
        
        def get_metrics(self) -> Dict[str, Any]:
            metrics = self.metrics.copy()
            if self.elapsed():
                metrics['elapsed_time'] = self.elapsed()
            return metrics
    
    return PerformanceMonitor()


@pytest.fixture
def mock_validator_results():
    """Mock validator results for testing validation functionality."""
    return {
        'good_response': {
            'validation_passed': True,
            'quality_score': 0.85,
            'validation_results': {
                'scientific_accuracy': {
                    'score': 0.9,
                    'details': {
                        'metabolite_validation': 'passed',
                        'pathway_validation': 'passed',
                        'statistical_validation': 'passed'
                    }
                },
                'completeness': {
                    'score': 0.8,
                    'missing_elements': [],
                    'coverage_assessment': 'comprehensive'
                },
                'clarity': {
                    'score': 0.85,
                    'readability_score': 0.8,
                    'structure_score': 0.9
                },
                'hallucination_check': {
                    'risk_level': 'low',
                    'absolute_claims_detected': 0,
                    'unsourced_specific_claims': 0
                },
                'confidence_assessment': {
                    'overall_confidence': 0.82,
                    'statistical_confidence': 0.85,
                    'source_confidence': 0.8
                }
            },
            'recommendations': []
        },
        'poor_response': {
            'validation_passed': False,
            'quality_score': 0.25,
            'validation_results': {
                'scientific_accuracy': {
                    'score': 0.2,
                    'details': {
                        'metabolite_validation': 'failed',
                        'pathway_validation': 'questionable',
                        'statistical_validation': 'failed'
                    }
                },
                'completeness': {
                    'score': 0.3,
                    'missing_elements': ['proper_citations', 'statistical_support'],
                    'coverage_assessment': 'incomplete'
                },
                'clarity': {
                    'score': 0.4,
                    'readability_score': 0.3,
                    'structure_score': 0.5
                },
                'hallucination_check': {
                    'risk_level': 'high',
                    'absolute_claims_detected': 5,
                    'unsourced_specific_claims': 8
                },
                'confidence_assessment': {
                    'overall_confidence': 0.15,
                    'statistical_confidence': 0.1,
                    'source_confidence': 0.2
                }
            },
            'recommendations': [
                'Add proper citations and source references',
                'Reduce absolute claims and add uncertainty expressions',
                'Include statistical support for claims',
                'Improve scientific accuracy of biomedical statements'
            ]
        }
    }


@pytest.fixture
def sample_clinical_responses():
    """Sample clinical responses for testing various scenarios."""
    return {
        'diabetes_metabolomics': """
        Metabolomics analysis of type 2 diabetes patients reveals significant alterations in glucose metabolism.
        
        Key Findings:
        - Glucose levels were elevated (mean: 15.2 ± 3.4 mmol/L, p < 0.001, n=156)
        - Insulin resistance markers showed 2.3-fold increase (95% CI: 1.8-2.9)
        - HbA1c levels were significantly higher in patients (8.4 ± 1.2%, p < 0.001)
        
        Metabolic Pathways:
        The glycolysis pathway showed marked disruption with reduced pyruvate levels
        and increased lactate/pyruvate ratio (p = 0.003). Hexokinase enzyme activity
        was decreased by 45% in muscle tissue samples.
        
        Clinical Biomarkers:
        Several metabolites show diagnostic potential:
        - 1,5-anhydroglucitol: sensitivity 87%, specificity 82%
        - 2-hydroxybutyrate: AUC = 0.84 (p < 0.001)
        - BCAA ratio significantly altered
        
        These findings are supported by multiple studies (Smith et al., 2023; Johnson & Brown, 2022).
        """,
        
        'cardiovascular_metabolomics': """
        Cardiovascular disease patients exhibit distinct metabolomic signatures in plasma samples.
        
        Lipid Metabolism:
        - Cholesterol levels elevated (6.8 ± 1.2 mmol/L vs 4.2 ± 0.8 mmol/L controls, p < 0.001)
        - LDL/HDL ratio increased 2.1-fold (95% CI: 1.7-2.6)
        - Fatty acid oxidation pathway disrupted
        
        Inflammatory Markers:
        C-reactive protein levels were significantly elevated (>3.0 mg/L in 78% of patients).
        Correlation with metabolic dysfunction was strong (r = 0.76, p < 0.001).
        
        Prognostic Implications:
        These metabolic alterations correlate with cardiovascular outcomes and may
        serve as early warning biomarkers for disease progression.
        """,
        
        'cancer_metabolomics': """
        Tumor metabolism shows characteristic alterations in energy production pathways.
        
        Warburg Effect:
        - Glucose uptake increased 3.5-fold in tumor tissue
        - Lactate production elevated despite adequate oxygen
        - Pyruvate kinase M2 isoform predominant
        
        Amino Acid Metabolism:
        - Glutamine consumption rate 4x higher than normal cells
        - Serine biosynthesis pathway upregulated
        - Methionine cycle alterations observed
        
        Therapeutic Targets:
        These metabolic dependencies offer potential therapeutic vulnerabilities
        for targeted intervention strategies.
        """,
        
        'empty_response': "",
        
        'minimal_response': "Glucose is a sugar.",
        
        'problematic_response': """
        Diabetes always causes complete metabolic failure in 100% of all cases.
        Every patient will definitely die within exactly 30 days.
        Glucose levels are precisely 999999 mg/L in every single case.
        This treatment eliminates all symptoms forever with zero side effects.
        No research is needed because everything is perfectly understood.
        """
    }


@pytest.fixture
def test_configurations():
    """Various test configurations for formatters and validators."""
    return {
        'formatter_minimal': {
            'extract_entities': False,
            'format_statistics': False,
            'process_sources': False,
            'structure_sections': False
        },
        'formatter_comprehensive': {
            'extract_entities': True,
            'format_statistics': True,
            'process_sources': True,
            'structure_sections': True,
            'add_clinical_indicators': True,
            'highlight_metabolites': True,
            'format_pathways': True,
            'max_entity_extraction': 50,
            'validate_scientific_accuracy': True,
            'enhanced_citation_processing': True
        },
        'formatter_fast': {
            'extract_entities': True,
            'format_statistics': True,
            'process_sources': False,
            'max_entity_extraction': 10,
            'validate_scientific_accuracy': False
        },
        'validator_strict': {
            'enabled': True,
            'performance_mode': 'comprehensive',
            'quality_gate_enabled': True,
            'thresholds': {
                'minimum_quality_score': 0.8,
                'scientific_confidence_threshold': 0.85,
                'completeness_threshold': 0.7,
                'clarity_threshold': 0.75
            }
        },
        'validator_lenient': {
            'enabled': True,
            'performance_mode': 'fast',
            'quality_gate_enabled': False,
            'thresholds': {
                'minimum_quality_score': 0.4,
                'scientific_confidence_threshold': 0.5,
                'completeness_threshold': 0.3,
                'clarity_threshold': 0.4
            }
        }
    }


@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing external service integration."""
    return {
        'doi_validation_success': {
            'status': 'valid',
            'title': 'Metabolomics in Type 2 Diabetes',
            'authors': ['Smith, J.', 'Doe, A.'],
            'journal': 'Nature Medicine',
            'year': 2023
        },
        'doi_validation_failure': {
            'status': 'invalid',
            'error': 'DOI not found'
        },
        'pmid_validation_success': {
            'status': 'valid',
            'title': 'Glycolysis Disruption in T2DM',
            'authors': ['Johnson, A.', 'Brown, B.'],
            'journal': 'Diabetes Care'
        }
    }


@pytest.fixture(autouse=True)
def suppress_logging():
    """Suppress logging during tests to reduce noise."""
    logging.getLogger().setLevel(logging.CRITICAL)
    yield
    logging.getLogger().setLevel(logging.WARNING)


class TestUtilities:
    """Utility class for common test operations."""
    
    @staticmethod
    def assert_valid_response_structure(response: Dict[str, Any]):
        """Assert that a response has the expected structure."""
        required_fields = ['formatted_content', 'entities', 'statistics', 'sources']
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
        
        assert isinstance(response['formatted_content'], str)
        assert isinstance(response['entities'], dict)
        assert isinstance(response['statistics'], list)
        assert isinstance(response['sources'], list)
    
    @staticmethod
    def assert_valid_validation_result(result: Dict[str, Any]):
        """Assert that a validation result has the expected structure."""
        required_fields = ['validation_passed', 'quality_score', 'validation_results']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        assert isinstance(result['validation_passed'], bool)
        assert isinstance(result['quality_score'], (int, float))
        assert 0 <= result['quality_score'] <= 1
        assert isinstance(result['validation_results'], dict)
    
    @staticmethod
    def count_entities_in_response(response: Dict[str, Any]) -> int:
        """Count total number of extracted entities."""
        entities = response.get('entities', {})
        return sum(len(entity_list) for entity_list in entities.values())
    
    @staticmethod
    def count_statistics_in_response(response: Dict[str, Any]) -> int:
        """Count total number of extracted statistics."""
        return len(response.get('statistics', []))
    
    @staticmethod
    def extract_p_values(response: Dict[str, Any]) -> List[float]:
        """Extract p-values from response statistics."""
        statistics = response.get('statistics', [])
        p_values = []
        for stat in statistics:
            if stat.get('type') == 'p_value' and 'value' in stat:
                p_values.append(stat['value'])
        return p_values


@pytest.fixture
def test_utilities():
    """Provide test utilities for test methods."""
    return TestUtilities()


# Performance testing utilities

@pytest.fixture
def performance_thresholds():
    """Performance thresholds for various operations."""
    return {
        'entity_extraction': 5.0,  # seconds
        'statistical_formatting': 3.0,  # seconds
        'response_validation': 10.0,  # seconds
        'complete_formatting': 15.0,  # seconds
        'large_response_processing': 30.0,  # seconds
        'concurrent_operations': 20.0  # seconds
    }


# Error simulation fixtures

@pytest.fixture
def error_simulation():
    """Utilities for simulating various error conditions."""
    class ErrorSimulation:
        @staticmethod
        def create_memory_pressure():
            """Simulate memory pressure conditions."""
            # Create large objects to simulate memory pressure
            large_data = ['x' * 1000000 for _ in range(100)]
            return large_data
        
        @staticmethod
        def create_timeout_condition():
            """Simulate timeout conditions."""
            import time
            time.sleep(0.1)  # Small delay to simulate processing time
        
        @staticmethod
        def create_network_error():
            """Simulate network error conditions."""
            from unittest.mock import Mock
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = Exception("Network error")
            return mock_response
    
    return ErrorSimulation()


if __name__ == "__main__":
    print("Response formatting test configuration loaded successfully.")