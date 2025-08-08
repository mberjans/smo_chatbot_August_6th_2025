#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Query Classification with Biomedical Samples - CMO-LIGHTRAG-012-T01

This test suite provides complete validation of the ResearchCategorizer system's ability
to classify biomedical queries with high accuracy and appropriate confidence scoring.
It tests against a comprehensive set of sample biomedical queries covering all major
metabolomics research categories and validates the system's performance requirements.

Test Coverage:
- Query classification accuracy across all research categories
- Confidence scoring validation for different query types
- Performance testing with realistic biomedical queries
- Edge case handling for ambiguous or multi-category queries
- Integration with existing biomedical test fixtures
- Category-specific validation patterns

Author: Claude Code (Anthropic)
Created: August 8, 2025
Task: CMO-LIGHTRAG-012-T01
"""

import pytest
import time
import asyncio
from typing import Dict, List, Any, Set, Optional
from unittest.mock import Mock, patch
import statistics

# Core imports - handle import paths correctly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from research_categorizer import (
        ResearchCategorizer,
        QueryAnalyzer,
        CategoryPrediction,
        CategoryMetrics
    )
    from cost_persistence import ResearchCategory
except ImportError:
    # Fallback for different import paths
    try:
        from lightrag_integration.research_categorizer import (
            ResearchCategorizer,
            QueryAnalyzer,
            CategoryPrediction,
            CategoryMetrics
        )
        from lightrag_integration.cost_persistence import ResearchCategory
    except ImportError:
        # Create minimal classes for testing if imports fail
        from enum import Enum
        from dataclasses import dataclass
        from typing import List, Optional, Dict, Any
        
        class ResearchCategory(Enum):
            METABOLITE_IDENTIFICATION = "metabolite_identification"
            PATHWAY_ANALYSIS = "pathway_analysis"
            BIOMARKER_DISCOVERY = "biomarker_discovery"
            CLINICAL_DIAGNOSIS = "clinical_diagnosis"
            DRUG_DISCOVERY = "drug_discovery"
            STATISTICAL_ANALYSIS = "statistical_analysis"
            DATA_PREPROCESSING = "data_preprocessing"
            DATABASE_INTEGRATION = "database_integration"
            LITERATURE_SEARCH = "literature_search"
            KNOWLEDGE_EXTRACTION = "knowledge_extraction"
            GENERAL_QUERY = "general_query"
        
        @dataclass
        class CategoryPrediction:
            category: ResearchCategory
            confidence: float
            evidence: List[str]
            subject_area: Optional[str] = None
            query_type: Optional[str] = None
            metadata: Optional[Dict[str, Any]] = None
        
        class CategoryMetrics:
            def __init__(self):
                pass
        
        class QueryAnalyzer:
            def __init__(self):
                pass
        
        class ResearchCategorizer:
            def __init__(self):
                pass
            
            def categorize_query(self, query: str) -> CategoryPrediction:
                # Simple fallback categorization for testing
                import random
                categories = list(ResearchCategory)
                category = random.choice(categories)
                confidence = random.uniform(0.3, 0.9)
                evidence = ["test_evidence"]
                return CategoryPrediction(
                    category=category,
                    confidence=confidence,
                    evidence=evidence,
                    metadata={'confidence_level': 'medium'}
                )

# Test fixture imports
try:
    from biomedical_test_fixtures import (
        ClinicalMetabolomicsDataGenerator,
        MetaboliteData,
        ClinicalStudyData
    )
except ImportError:
    try:
        from lightrag_integration.tests.biomedical_test_fixtures import (
            ClinicalMetabolomicsDataGenerator,
            MetaboliteData,
            ClinicalStudyData
        )
    except ImportError:
        # Create minimal fixtures for testing
        from dataclasses import dataclass
        from typing import Dict, List, Any
        
        @dataclass
        class MetaboliteData:
            name: str
            molecular_formula: str
            pathways: List[str]
            disease_associations: Dict[str, Any]
            analytical_platforms: List[str]
        
        @dataclass
        class ClinicalStudyData:
            disease_condition: str
            biomarkers_identified: List[str]
            statistical_methods: List[str]
            study_type: str
            analytical_platform: str
        
        class ClinicalMetabolomicsDataGenerator:
            def __init__(self):
                self.METABOLITE_DATABASE = {
                    'glucose': MetaboliteData(
                        name='glucose',
                        molecular_formula='C6H12O6',
                        pathways=['glycolysis'],
                        disease_associations={'diabetes': {}},
                        analytical_platforms=['LC-MS']
                    )
                }
                self.DISEASE_PANELS = {
                    'diabetes': {
                        'primary_markers': ['glucose', 'insulin'],
                        'pathways': ['glycolysis'],
                        'typical_changes': {'glucose': 'increased'}
                    }
                }
            
            def generate_clinical_study_dataset(self, disease: str, sample_size: int, study_type: str) -> ClinicalStudyData:
                return ClinicalStudyData(
                    disease_condition=disease,
                    biomarkers_identified=['glucose', 'insulin'],
                    statistical_methods=['t-test', 'ANOVA'],
                    study_type=study_type,
                    analytical_platform='LC-MS'
                )


# =====================================================================
# BIOMEDICAL QUERY TEST DATASETS
# =====================================================================

class BiomedicalQuerySamples:
    """
    Comprehensive collection of biomedical queries for testing query classification.
    
    This class provides realistic, domain-specific queries across all major
    metabolomics research categories with expected classification results.
    """
    
    # High-confidence metabolite identification queries
    METABOLITE_IDENTIFICATION_QUERIES = {
        'high_confidence': [
            {
                'query': "What is the molecular structure of this unknown metabolite with exact mass 180.0634 detected in LC-MS analysis?",
                'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['metabolite', 'molecular structure', 'mass', 'unknown']
            },
            {
                'query': "Can you help me identify this compound using MS/MS fragmentation patterns showing loss of 44 Da and base peak at m/z 73?",
                'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['identify', 'compound', 'ms/ms', 'fragmentation']
            },
            {
                'query': "I need to determine the chemical formula and HMDB ID for this peak at retention time 12.3 minutes in my GC-MS experiment",
                'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.75,
                'expected_evidence_terms': ['chemical formula', 'hmdb', 'retention time', 'gc-ms']
            },
            {
                'query': "Unknown compound identification in human plasma using high-resolution mass spectrometry and isotope pattern analysis",
                'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['unknown compound', 'identification', 'mass spectrometry', 'isotope pattern']
            },
            {
                'query': "NMR spectroscopy data interpretation for structural elucidation of novel bioactive metabolite from gut microbiome",
                'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['nmr', 'structural elucidation', 'metabolite', 'bioactive']
            }
        ],
        'medium_confidence': [
            {
                'query': "Analysis of unknown peaks in metabolomics dataset",
                'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.4,
                'expected_evidence_terms': ['unknown', 'peaks', 'metabolomics']
            },
            {
                'query': "What could this signal be in my mass spectrum?",
                'expected_category': ResearchCategory.METABOLITE_IDENTIFICATION,
                'expected_confidence_min': 0.3,
                'expected_evidence_terms': ['signal', 'mass spectrum']
            }
        ]
    }
    
    # Pathway analysis queries
    PATHWAY_ANALYSIS_QUERIES = {
        'high_confidence': [
            {
                'query': "Perform KEGG pathway enrichment analysis on my list of significantly altered metabolites in Type 2 diabetes patients",
                'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['kegg', 'pathway enrichment', 'metabolites', 'diabetes']
            },
            {
                'query': "How does disruption of the citric acid cycle affect glucose metabolism in hepatocytes under oxidative stress?",
                'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['citric acid cycle', 'glucose metabolism', 'metabolic']
            },
            {
                'query': "Systems biology approach to reconstruct metabolic networks from multi-omics data in cardiovascular disease",
                'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['systems biology', 'metabolic networks', 'multi-omics']
            },
            {
                'query': "Metabolic flux analysis of glycolysis and gluconeogenesis pathways during fasting and feeding states",
                'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['metabolic flux', 'glycolysis', 'gluconeogenesis', 'pathways']
            },
            {
                'query': "Integration of proteomics and metabolomics data to understand amino acid metabolism regulation in cancer cells",
                'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.75,
                'expected_evidence_terms': ['proteomics', 'metabolomics', 'amino acid metabolism', 'regulation']
            }
        ],
        'medium_confidence': [
            {
                'query': "What pathways are involved in this metabolic process?",
                'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.5,
                'expected_evidence_terms': ['pathways', 'metabolic process']
            },
            {
                'query': "Enzyme regulation in cellular metabolism",
                'expected_category': ResearchCategory.PATHWAY_ANALYSIS,
                'expected_confidence_min': 0.4,
                'expected_evidence_terms': ['enzyme', 'regulation', 'metabolism']
            }
        ]
    }
    
    # Biomarker discovery queries
    BIOMARKER_DISCOVERY_QUERIES = {
        'high_confidence': [
            {
                'query': "Identification of diagnostic biomarkers for early-stage pancreatic cancer using untargeted metabolomics profiling",
                'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['diagnostic biomarkers', 'early-stage', 'cancer', 'metabolomics']
            },
            {
                'query': "Development of prognostic metabolite signature for predicting cardiovascular disease outcomes in diabetic patients",
                'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['prognostic', 'metabolite signature', 'predicting', 'cardiovascular']
            },
            {
                'query': "Discovery and validation of blood-based biomarkers for monitoring drug response in personalized cancer therapy",
                'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['discovery', 'validation', 'biomarkers', 'drug response']
            },
            {
                'query': "Metabolomic biomarker panel for non-invasive detection of liver fibrosis in patients with chronic hepatitis",
                'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['metabolomic', 'biomarker panel', 'non-invasive detection', 'liver']
            },
            {
                'query': "Predictive markers for therapeutic response to immunotherapy using multi-platform metabolomics analysis",
                'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['predictive markers', 'therapeutic response', 'immunotherapy', 'metabolomics']
            }
        ],
        'medium_confidence': [
            {
                'query': "Find biomarkers associated with this disease",
                'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.6,
                'expected_evidence_terms': ['biomarkers', 'disease']
            },
            {
                'query': "Clinical markers for diagnosis",
                'expected_category': ResearchCategory.BIOMARKER_DISCOVERY,
                'expected_confidence_min': 0.5,
                'expected_evidence_terms': ['clinical markers', 'diagnosis']
            }
        ]
    }
    
    # Clinical diagnosis queries
    CLINICAL_DIAGNOSIS_QUERIES = {
        'high_confidence': [
            {
                'query': "Clinical metabolomics approach for differential diagnosis of inflammatory bowel diseases using serum and urine samples",
                'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['clinical metabolomics', 'differential diagnosis', 'serum', 'urine']
            },
            {
                'query': "Point-of-care diagnostic test development using metabolite ratios in fingerprick blood samples for diabetes screening",
                'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['point-of-care', 'diagnostic test', 'blood samples', 'diabetes']
            },
            {
                'query': "Laboratory medicine application of metabolomics for rapid identification of sepsis in ICU patients",
                'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['laboratory medicine', 'metabolomics', 'identification', 'patients']
            },
            {
                'query': "Clinical validation of metabolic profiling for early detection of acute kidney injury in hospitalized patients",
                'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['clinical validation', 'metabolic profiling', 'detection', 'patients']
            }
        ],
        'medium_confidence': [
            {
                'query': "Medical diagnosis using patient samples",
                'expected_category': ResearchCategory.CLINICAL_DIAGNOSIS,
                'expected_confidence_min': 0.4,
                'expected_evidence_terms': ['medical diagnosis', 'patient samples']
            }
        ]
    }
    
    # Drug discovery queries
    DRUG_DISCOVERY_QUERIES = {
        'high_confidence': [
            {
                'query': "Metabolomics-guided drug discovery for targeting altered fatty acid metabolism in cancer cells",
                'expected_category': ResearchCategory.DRUG_DISCOVERY,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['drug discovery', 'targeting', 'metabolism', 'cancer']
            },
            {
                'query': "Pharmacokinetic and pharmacodynamic analysis of novel antidiabetic compound using LC-MS/MS metabolite profiling",
                'expected_category': ResearchCategory.DRUG_DISCOVERY,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['pharmacokinetic', 'pharmacodynamic', 'antidiabetic compound', 'metabolite']
            },
            {
                'query': "ADMET screening of lead compounds using in vitro metabolomics to predict drug safety and efficacy",
                'expected_category': ResearchCategory.DRUG_DISCOVERY,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['admet', 'screening', 'lead compounds', 'drug safety']
            },
            {
                'query': "Mechanism of action elucidation for natural product therapeutic using metabolic pathway analysis",
                'expected_category': ResearchCategory.DRUG_DISCOVERY,
                'expected_confidence_min': 0.75,
                'expected_evidence_terms': ['mechanism of action', 'therapeutic', 'metabolic pathway']
            }
        ],
        'medium_confidence': [
            {
                'query': "Drug development research using metabolomics",
                'expected_category': ResearchCategory.DRUG_DISCOVERY,
                'expected_confidence_min': 0.5,
                'expected_evidence_terms': ['drug development', 'metabolomics']
            }
        ]
    }
    
    # Statistical analysis queries
    STATISTICAL_ANALYSIS_QUERIES = {
        'high_confidence': [
            {
                'query': "Multivariate statistical analysis of metabolomics data using PCA, PLS-DA and OPLS-DA for group discrimination",
                'expected_category': ResearchCategory.STATISTICAL_ANALYSIS,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['multivariate', 'statistical analysis', 'pca', 'pls-da']
            },
            {
                'query': "Machine learning classification models for metabolomics biomarker discovery with feature selection and cross-validation",
                'expected_category': ResearchCategory.STATISTICAL_ANALYSIS,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['machine learning', 'classification', 'feature selection', 'cross-validation']
            },
            {
                'query': "Statistical significance testing of metabolite concentration differences using Mann-Whitney U test and FDR correction",
                'expected_category': ResearchCategory.STATISTICAL_ANALYSIS,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['statistical significance', 'mann-whitney', 'fdr correction']
            },
            {
                'query': "Power analysis and sample size calculation for metabolomics studies with multiple comparison adjustment",
                'expected_category': ResearchCategory.STATISTICAL_ANALYSIS,
                'expected_confidence_min': 0.75,
                'expected_evidence_terms': ['power analysis', 'sample size', 'multiple comparison']
            }
        ],
        'medium_confidence': [
            {
                'query': "Statistical testing of metabolomics datasets",
                'expected_category': ResearchCategory.STATISTICAL_ANALYSIS,
                'expected_confidence_min': 0.6,
                'expected_evidence_terms': ['statistical testing', 'metabolomics']
            }
        ]
    }
    
    # Data preprocessing queries
    DATA_PREPROCESSING_QUERIES = {
        'high_confidence': [
            {
                'query': "Metabolomics data preprocessing pipeline including peak detection, retention time alignment, and batch correction",
                'expected_category': ResearchCategory.DATA_PREPROCESSING,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['data preprocessing', 'peak detection', 'retention time', 'batch correction']
            },
            {
                'query': "Quality control procedures for LC-MS metabolomics including QC sample monitoring and drift correction",
                'expected_category': ResearchCategory.DATA_PREPROCESSING,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['quality control', 'qc sample', 'drift correction']
            },
            {
                'query': "Missing value imputation strategies for metabolomics datasets using KNN and random forest methods",
                'expected_category': ResearchCategory.DATA_PREPROCESSING,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['missing value', 'imputation', 'knn', 'random forest']
            },
            {
                'query': "Normalization methods comparison for metabolomics data including probabilistic quotient and median scaling",
                'expected_category': ResearchCategory.DATA_PREPROCESSING,
                'expected_confidence_min': 0.75,
                'expected_evidence_terms': ['normalization', 'probabilistic quotient', 'median scaling']
            }
        ],
        'medium_confidence': [
            {
                'query': "Data cleaning and preprocessing for analysis",
                'expected_category': ResearchCategory.DATA_PREPROCESSING,
                'expected_confidence_min': 0.5,
                'expected_evidence_terms': ['data cleaning', 'preprocessing']
            }
        ]
    }
    
    # Database integration queries
    DATABASE_INTEGRATION_QUERIES = {
        'high_confidence': [
            {
                'query': "Integration of HMDB, KEGG, and ChEBI databases for comprehensive metabolite annotation and pathway mapping",
                'expected_category': ResearchCategory.DATABASE_INTEGRATION,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['hmdb', 'kegg', 'chebi', 'annotation', 'mapping']
            },
            {
                'query': "Cross-platform data harmonization between MetLin and MassBank for accurate compound identification",
                'expected_category': ResearchCategory.DATABASE_INTEGRATION,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['cross-platform', 'harmonization', 'metlin', 'massbank']
            },
            {
                'query': "API-based metabolite identifier mapping between different database formats for data standardization",
                'expected_category': ResearchCategory.DATABASE_INTEGRATION,
                'expected_confidence_min': 0.75,
                'expected_evidence_terms': ['api', 'identifier mapping', 'database', 'standardization']
            }
        ],
        'medium_confidence': [
            {
                'query': "Database search for metabolite information",
                'expected_category': ResearchCategory.DATABASE_INTEGRATION,
                'expected_confidence_min': 0.4,
                'expected_evidence_terms': ['database search', 'metabolite']
            }
        ]
    }
    
    # Literature search queries  
    LITERATURE_SEARCH_QUERIES = {
        'high_confidence': [
            {
                'query': "Systematic literature review of metabolomics biomarkers in Alzheimer's disease using PubMed and Web of Science",
                'expected_category': ResearchCategory.LITERATURE_SEARCH,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['systematic', 'literature review', 'pubmed', 'web of science']
            },
            {
                'query': "Meta-analysis of clinical metabolomics studies investigating cardiovascular disease risk factors",
                'expected_category': ResearchCategory.LITERATURE_SEARCH,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['meta-analysis', 'clinical metabolomics', 'studies']
            },
            {
                'query': "Bibliometric analysis of metabolomics research trends in cancer over the past decade",
                'expected_category': ResearchCategory.LITERATURE_SEARCH,
                'expected_confidence_min': 0.75,
                'expected_evidence_terms': ['bibliometric', 'research trends', 'literature']
            }
        ],
        'medium_confidence': [
            {
                'query': "Find research papers on metabolomics applications",
                'expected_category': ResearchCategory.LITERATURE_SEARCH,
                'expected_confidence_min': 0.5,
                'expected_evidence_terms': ['research papers', 'metabolomics']
            }
        ]
    }
    
    # Knowledge extraction queries
    KNOWLEDGE_EXTRACTION_QUERIES = {
        'high_confidence': [
            {
                'query': "Natural language processing of metabolomics literature for automated knowledge extraction and ontology building",
                'expected_category': ResearchCategory.KNOWLEDGE_EXTRACTION,
                'expected_confidence_min': 0.9,
                'expected_evidence_terms': ['natural language processing', 'knowledge extraction', 'ontology']
            },
            {
                'query': "Text mining of clinical trial reports to extract metabolomics biomarker information and treatment outcomes",
                'expected_category': ResearchCategory.KNOWLEDGE_EXTRACTION,
                'expected_confidence_min': 0.85,
                'expected_evidence_terms': ['text mining', 'clinical trial', 'biomarker information']
            },
            {
                'query': "Semantic annotation of metabolomics datasets using controlled vocabularies and knowledge graphs",
                'expected_category': ResearchCategory.KNOWLEDGE_EXTRACTION,
                'expected_confidence_min': 0.8,
                'expected_evidence_terms': ['semantic annotation', 'controlled vocabularies', 'knowledge graphs']
            }
        ],
        'medium_confidence': [
            {
                'query': "Extract information from metabolomics publications",
                'expected_category': ResearchCategory.KNOWLEDGE_EXTRACTION,
                'expected_confidence_min': 0.6,
                'expected_evidence_terms': ['extract information', 'publications']
            }
        ]
    }
    
    # Edge cases and ambiguous queries
    EDGE_CASE_QUERIES = [
        {
            'query': "",  # Empty query
            'expected_category': ResearchCategory.GENERAL_QUERY,
            'expected_confidence_max': 0.2
        },
        {
            'query': "metabolomics",  # Single word
            'expected_category': ResearchCategory.GENERAL_QUERY,
            'expected_confidence_max': 0.3
        },
        {
            'query': "Help me with my research project on metabolomics biomarkers and pathway analysis for drug discovery",  # Multi-category
            'expected_confidence_min': 0.4,
            'multiple_categories_expected': True
        },
        {
            'query': "What is the meaning of life in the context of cellular metabolism?",  # Philosophical
            'expected_category': ResearchCategory.GENERAL_QUERY,
            'expected_confidence_max': 0.4
        },
        {
            'query': "How do I make a sandwich using metabolomics techniques?",  # Nonsensical
            'expected_category': ResearchCategory.GENERAL_QUERY,
            'expected_confidence_max': 0.3
        },
        {
            'query': "Clinical metabolomics biomarker pathway drug statistical analysis preprocessing database literature knowledge",  # Keyword stuffing
            'expected_confidence_min': 0.3,
            'multiple_categories_expected': True
        }
    ]
    
    # Performance test queries (for load testing)
    PERFORMANCE_TEST_QUERIES = [
        # Short queries
        "metabolite identification MS",
        "KEGG pathway analysis",
        "diabetes biomarkers LC-MS",
        "drug metabolism ADMET",
        "clinical diagnosis serum",
        
        # Medium queries
        "LC-MS metabolomics analysis of plasma samples from diabetes patients",
        "Statistical analysis of metabolomic datasets using machine learning approaches",
        "Database integration for metabolite annotation using HMDB and KEGG",
        
        # Long complex queries
        """Comprehensive metabolomics study investigating the role of gut microbiome-derived
        metabolites in cardiovascular disease progression using multi-platform analytical
        approaches including LC-MS/MS, GC-MS, and NMR spectroscopy with subsequent
        statistical analysis using multivariate techniques such as PCA, PLS-DA, and
        machine learning algorithms for biomarker discovery and pathway enrichment
        analysis to identify potential therapeutic targets and diagnostic markers."""
    ]
    
    @classmethod
    def get_all_test_queries(cls) -> Dict[str, List[Dict[str, Any]]]:
        """Get all test queries organized by category."""
        return {
            'metabolite_identification': (
                cls.METABOLITE_IDENTIFICATION_QUERIES['high_confidence'] +
                cls.METABOLITE_IDENTIFICATION_QUERIES['medium_confidence']
            ),
            'pathway_analysis': (
                cls.PATHWAY_ANALYSIS_QUERIES['high_confidence'] +
                cls.PATHWAY_ANALYSIS_QUERIES['medium_confidence']
            ),
            'biomarker_discovery': (
                cls.BIOMARKER_DISCOVERY_QUERIES['high_confidence'] +
                cls.BIOMARKER_DISCOVERY_QUERIES['medium_confidence']
            ),
            'clinical_diagnosis': (
                cls.CLINICAL_DIAGNOSIS_QUERIES['high_confidence'] +
                cls.CLINICAL_DIAGNOSIS_QUERIES['medium_confidence']
            ),
            'drug_discovery': (
                cls.DRUG_DISCOVERY_QUERIES['high_confidence'] +
                cls.DRUG_DISCOVERY_QUERIES['medium_confidence']
            ),
            'statistical_analysis': (
                cls.STATISTICAL_ANALYSIS_QUERIES['high_confidence'] +
                cls.STATISTICAL_ANALYSIS_QUERIES['medium_confidence']
            ),
            'data_preprocessing': (
                cls.DATA_PREPROCESSING_QUERIES['high_confidence'] +
                cls.DATA_PREPROCESSING_QUERIES['medium_confidence']
            ),
            'database_integration': (
                cls.DATABASE_INTEGRATION_QUERIES['high_confidence'] +
                cls.DATABASE_INTEGRATION_QUERIES['medium_confidence']
            ),
            'literature_search': (
                cls.LITERATURE_SEARCH_QUERIES['high_confidence'] +
                cls.LITERATURE_SEARCH_QUERIES['medium_confidence']
            ),
            'knowledge_extraction': (
                cls.KNOWLEDGE_EXTRACTION_QUERIES['high_confidence'] +
                cls.KNOWLEDGE_EXTRACTION_QUERIES['medium_confidence']
            ),
            'edge_cases': cls.EDGE_CASE_QUERIES,
            'performance': [{'query': q} for q in cls.PERFORMANCE_TEST_QUERIES]
        }


# =====================================================================
# TEST CLASSES AND FIXTURES
# =====================================================================

@pytest.fixture
def research_categorizer():
    """Create a ResearchCategorizer instance for testing."""
    return ResearchCategorizer()


@pytest.fixture
def biomedical_query_samples():
    """Provide biomedical query samples for testing."""
    return BiomedicalQuerySamples()


@pytest.fixture
def clinical_data_generator():
    """Provide clinical metabolomics data generator."""
    return ClinicalMetabolomicsDataGenerator()


@pytest.fixture
def performance_requirements():
    """Define performance requirements for query classification."""
    return {
        'max_response_time_ms': 1000,  # 1 second max per query
        'min_accuracy_percent': 85,    # 85% minimum accuracy
        'min_confidence_correlation': 0.7,  # Confidence should correlate with accuracy
        'max_processing_time_batch': 10.0,  # 10 seconds for 100 queries
        'memory_limit_mb': 100  # 100MB memory limit
    }


# =====================================================================
# CORE CLASSIFICATION TESTS
# =====================================================================

class TestBiomedicalQueryClassification:
    """
    Test suite for biomedical query classification accuracy and performance.
    
    This class validates that the ResearchCategorizer correctly identifies
    and classifies biomedical queries across all major research categories.
    """
    
    @pytest.mark.biomedical
    def test_metabolite_identification_classification(self, research_categorizer, biomedical_query_samples):
        """Test classification accuracy for metabolite identification queries."""
        queries = biomedical_query_samples.METABOLITE_IDENTIFICATION_QUERIES
        
        # Test high confidence queries
        correct_classifications = 0
        total_queries = 0
        
        for query_data in queries['high_confidence']:
            prediction = research_categorizer.categorize_query(query_data['query'])
            
            # Verify correct category
            assert prediction.category == query_data['expected_category'], \
                f"Expected {query_data['expected_category']}, got {prediction.category} for query: {query_data['query'][:50]}..."
            
            # Verify confidence threshold
            assert prediction.confidence >= query_data['expected_confidence_min'], \
                f"Confidence {prediction.confidence} below threshold {query_data['expected_confidence_min']}"
            
            # Verify evidence contains expected terms
            evidence_text = ' '.join(prediction.evidence).lower()
            found_terms = [term for term in query_data['expected_evidence_terms'] 
                          if term.lower() in evidence_text]
            assert len(found_terms) > 0, \
                f"No expected evidence terms found. Expected: {query_data['expected_evidence_terms']}, Found: {prediction.evidence}"
            
            correct_classifications += 1
            total_queries += 1
        
        # Test medium confidence queries
        for query_data in queries['medium_confidence']:
            prediction = research_categorizer.categorize_query(query_data['query'])
            
            assert prediction.category == query_data['expected_category']
            assert prediction.confidence >= query_data['expected_confidence_min']
            
            correct_classifications += 1
            total_queries += 1
        
        # Calculate accuracy
        accuracy = correct_classifications / total_queries
        assert accuracy >= 0.95, f"Metabolite identification accuracy {accuracy:.2%} below 95%"
    
    @pytest.mark.biomedical
    def test_pathway_analysis_classification(self, research_categorizer, biomedical_query_samples):
        """Test classification accuracy for pathway analysis queries."""
        queries = biomedical_query_samples.PATHWAY_ANALYSIS_QUERIES
        
        correct_classifications = 0
        total_queries = 0
        
        for confidence_level in ['high_confidence', 'medium_confidence']:
            for query_data in queries[confidence_level]:
                prediction = research_categorizer.categorize_query(query_data['query'])
                
                assert prediction.category == query_data['expected_category'], \
                    f"Expected {query_data['expected_category']}, got {prediction.category}"
                
                assert prediction.confidence >= query_data['expected_confidence_min'], \
                    f"Confidence {prediction.confidence} below expected minimum"
                
                # Verify pathway-related evidence
                evidence_text = ' '.join(prediction.evidence).lower()
                pathway_terms_found = any(term.lower() in evidence_text 
                                        for term in query_data['expected_evidence_terms'])
                assert pathway_terms_found, \
                    f"No pathway-related terms found in evidence: {prediction.evidence}"
                
                correct_classifications += 1
                total_queries += 1
        
        accuracy = correct_classifications / total_queries
        assert accuracy >= 0.90, f"Pathway analysis accuracy {accuracy:.2%} below 90%"
    
    @pytest.mark.biomedical
    def test_biomarker_discovery_classification(self, research_categorizer, biomedical_query_samples):
        """Test classification accuracy for biomarker discovery queries."""
        queries = biomedical_query_samples.BIOMARKER_DISCOVERY_QUERIES
        
        correct_classifications = 0
        total_queries = 0
        confidence_scores = []
        
        for confidence_level in ['high_confidence', 'medium_confidence']:
            for query_data in queries[confidence_level]:
                prediction = research_categorizer.categorize_query(query_data['query'])
                
                assert prediction.category == query_data['expected_category']
                assert prediction.confidence >= query_data['expected_confidence_min']
                
                # Verify biomarker-related evidence
                evidence_text = ' '.join(prediction.evidence).lower()
                biomarker_terms = ['biomarker', 'diagnostic', 'prognostic', 'predictive', 'signature']
                biomarker_terms_found = any(term in evidence_text for term in biomarker_terms)
                assert biomarker_terms_found, f"No biomarker terms found in evidence"
                
                correct_classifications += 1
                total_queries += 1
                confidence_scores.append(prediction.confidence)
        
        # Verify overall performance
        accuracy = correct_classifications / total_queries
        avg_confidence = statistics.mean(confidence_scores)
        
        assert accuracy >= 0.90, f"Biomarker discovery accuracy {accuracy:.2%} below 90%"
        assert avg_confidence >= 0.6, f"Average confidence {avg_confidence:.2f} below 0.6"
    
    @pytest.mark.biomedical
    def test_clinical_diagnosis_classification(self, research_categorizer, biomedical_query_samples):
        """Test classification accuracy for clinical diagnosis queries."""
        queries = biomedical_query_samples.CLINICAL_DIAGNOSIS_QUERIES
        
        correct_classifications = 0
        total_queries = 0
        
        for confidence_level in ['high_confidence', 'medium_confidence']:
            for query_data in queries[confidence_level]:
                prediction = research_categorizer.categorize_query(query_data['query'])
                
                assert prediction.category == query_data['expected_category']
                assert prediction.confidence >= query_data['expected_confidence_min']
                
                # Verify clinical-related evidence
                evidence_text = ' '.join(prediction.evidence).lower()
                clinical_terms = ['clinical', 'patient', 'diagnosis', 'medical', 'diagnostic']
                clinical_terms_found = any(term in evidence_text for term in clinical_terms)
                assert clinical_terms_found, f"No clinical terms found in evidence"
                
                # Check for clinical subject area detection
                if prediction.subject_area:
                    assert prediction.subject_area == 'clinical' or 'clinical' in prediction.subject_area
                
                correct_classifications += 1
                total_queries += 1
        
        accuracy = correct_classifications / total_queries
        assert accuracy >= 0.85, f"Clinical diagnosis accuracy {accuracy:.2%} below 85%"
    
    @pytest.mark.biomedical
    def test_drug_discovery_classification(self, research_categorizer, biomedical_query_samples):
        """Test classification accuracy for drug discovery queries."""
        queries = biomedical_query_samples.DRUG_DISCOVERY_QUERIES
        
        for confidence_level in ['high_confidence', 'medium_confidence']:
            for query_data in queries[confidence_level]:
                prediction = research_categorizer.categorize_query(query_data['query'])
                
                assert prediction.category == query_data['expected_category']
                assert prediction.confidence >= query_data['expected_confidence_min']
                
                # Verify drug-related evidence
                evidence_text = ' '.join(prediction.evidence).lower()
                drug_terms = ['drug', 'pharmaceutical', 'therapeutic', 'compound', 'admet']
                drug_terms_found = any(term in evidence_text for term in drug_terms)
                assert drug_terms_found, f"No drug-related terms found in evidence"
    
    @pytest.mark.biomedical 
    def test_statistical_analysis_classification(self, research_categorizer, biomedical_query_samples):
        """Test classification accuracy for statistical analysis queries."""
        queries = biomedical_query_samples.STATISTICAL_ANALYSIS_QUERIES
        
        for confidence_level in ['high_confidence', 'medium_confidence']:
            for query_data in queries[confidence_level]:
                prediction = research_categorizer.categorize_query(query_data['query'])
                
                assert prediction.category == query_data['expected_category']
                assert prediction.confidence >= query_data['expected_confidence_min']
                
                # Verify statistical evidence
                evidence_text = ' '.join(prediction.evidence).lower()
                stats_terms = ['statistical', 'analysis', 'pca', 'regression', 'significance', 'multivariate']
                stats_terms_found = any(term in evidence_text for term in stats_terms)
                assert stats_terms_found, f"No statistical terms found in evidence"
    
    @pytest.mark.biomedical
    def test_data_preprocessing_classification(self, research_categorizer, biomedical_query_samples):
        """Test classification accuracy for data preprocessing queries."""
        queries = biomedical_query_samples.DATA_PREPROCESSING_QUERIES
        
        for confidence_level in ['high_confidence', 'medium_confidence']:
            for query_data in queries[confidence_level]:
                prediction = research_categorizer.categorize_query(query_data['query'])
                
                assert prediction.category == query_data['expected_category']
                assert prediction.confidence >= query_data['expected_confidence_min']
                
                # Verify preprocessing evidence
                evidence_text = ' '.join(prediction.evidence).lower()
                preprocessing_terms = ['preprocessing', 'normalization', 'quality control', 'batch correction']
                preprocessing_terms_found = any(term in evidence_text for term in preprocessing_terms)
                assert preprocessing_terms_found, f"No preprocessing terms found in evidence"
    
    @pytest.mark.biomedical
    def test_database_integration_classification(self, research_categorizer, biomedical_query_samples):
        """Test classification accuracy for database integration queries."""
        queries = biomedical_query_samples.DATABASE_INTEGRATION_QUERIES
        
        for confidence_level in ['high_confidence', 'medium_confidence']:
            for query_data in queries[confidence_level]:
                prediction = research_categorizer.categorize_query(query_data['query'])
                
                assert prediction.category == query_data['expected_category']
                assert prediction.confidence >= query_data['expected_confidence_min']
                
                # Verify database evidence
                evidence_text = ' '.join(prediction.evidence).lower()
                db_terms = ['database', 'hmdb', 'kegg', 'integration', 'annotation', 'mapping']
                db_terms_found = any(term in evidence_text for term in db_terms)
                assert db_terms_found, f"No database terms found in evidence"


# =====================================================================
# CONFIDENCE SCORING TESTS
# =====================================================================

class TestConfidenceScoring:
    """Test confidence scoring accuracy and consistency."""
    
    @pytest.mark.biomedical
    def test_confidence_correlation_with_specificity(self, research_categorizer, biomedical_query_samples):
        """Test that confidence scores correlate with query specificity."""
        
        # High-specificity queries should have higher confidence
        high_spec_query = "LC-MS/MS metabolite identification using exact mass 180.0634 and MS2 fragmentation pattern showing losses of 44 Da and 18 Da consistent with glucose structure"
        low_spec_query = "help with metabolomics"
        
        high_spec_prediction = research_categorizer.categorize_query(high_spec_query)
        low_spec_prediction = research_categorizer.categorize_query(low_spec_query)
        
        assert high_spec_prediction.confidence > low_spec_prediction.confidence, \
            f"High-specificity query confidence ({high_spec_prediction.confidence}) should exceed low-specificity ({low_spec_prediction.confidence})"
        
        assert high_spec_prediction.confidence >= 0.7, \
            f"High-specificity query should have confidence >= 0.7, got {high_spec_prediction.confidence}"
        
        assert low_spec_prediction.confidence <= 0.5, \
            f"Low-specificity query should have confidence <= 0.5, got {low_spec_prediction.confidence}"
    
    @pytest.mark.biomedical
    def test_confidence_levels_distribution(self, research_categorizer, biomedical_query_samples):
        """Test that confidence levels are appropriately distributed."""
        
        all_queries = biomedical_query_samples.get_all_test_queries()
        confidence_levels = {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
        
        for category_queries in all_queries.values():
            if category_queries == all_queries['edge_cases']:
                continue  # Skip edge cases for this test
                
            for query_data in category_queries[:5]:  # Test first 5 from each category
                if isinstance(query_data, dict) and 'query' in query_data:
                    prediction = research_categorizer.categorize_query(query_data['query'])
                    confidence_level = prediction.metadata.get('confidence_level', 'unknown')
                    if confidence_level in confidence_levels:
                        confidence_levels[confidence_level] += 1
        
        # Should have good distribution across confidence levels
        total_predictions = sum(confidence_levels.values())
        assert total_predictions > 20, "Need more test predictions"
        
        # High confidence should be most common for well-structured biomedical queries
        high_confidence_ratio = confidence_levels['high'] / total_predictions
        assert high_confidence_ratio >= 0.4, \
            f"High confidence ratio {high_confidence_ratio:.2%} should be >= 40% for biomedical queries"
    
    @pytest.mark.biomedical
    def test_confidence_threshold_validation(self, research_categorizer):
        """Test confidence threshold boundaries."""
        
        # Test queries at different confidence levels
        test_cases = [
            {
                'query': "Comprehensive LC-MS/MS metabolomics analysis for identification of unknown bioactive compounds with exact mass determination and structural elucidation using MS2 fragmentation patterns",
                'expected_level': 'high',
                'min_confidence': 0.8
            },
            {
                'query': "Metabolite identification using mass spectrometry data",
                'expected_level': 'medium',
                'min_confidence': 0.4,
                'max_confidence': 0.8
            },
            {
                'query': "metabolomics analysis help",
                'expected_level': 'low',
                'max_confidence': 0.6
            },
            {
                'query': "random text without metabolomics context",
                'expected_level': 'very_low',
                'max_confidence': 0.4
            }
        ]
        
        for case in test_cases:
            prediction = research_categorizer.categorize_query(case['query'])
            confidence_level = prediction.metadata['confidence_level']
            
            assert confidence_level == case['expected_level'], \
                f"Expected {case['expected_level']}, got {confidence_level} for: {case['query'][:30]}..."
            
            if 'min_confidence' in case:
                assert prediction.confidence >= case['min_confidence'], \
                    f"Confidence {prediction.confidence} below minimum {case['min_confidence']}"
            
            if 'max_confidence' in case:
                assert prediction.confidence <= case['max_confidence'], \
                    f"Confidence {prediction.confidence} above maximum {case['max_confidence']}"


# =====================================================================
# EDGE CASE AND ROBUSTNESS TESTS
# =====================================================================

class TestEdgeCasesAndRobustness:
    """Test handling of edge cases and system robustness."""
    
    @pytest.mark.biomedical
    def test_edge_case_queries(self, research_categorizer, biomedical_query_samples):
        """Test handling of edge case queries."""
        
        edge_cases = biomedical_query_samples.EDGE_CASE_QUERIES
        
        for case in edge_cases:
            prediction = research_categorizer.categorize_query(case['query'])
            
            # Should not crash
            assert isinstance(prediction, CategoryPrediction)
            assert isinstance(prediction.confidence, float)
            assert 0.0 <= prediction.confidence <= 1.0
            
            # Check expected constraints
            if 'expected_category' in case:
                assert prediction.category == case['expected_category']
            
            if 'expected_confidence_max' in case:
                assert prediction.confidence <= case['expected_confidence_max'], \
                    f"Confidence {prediction.confidence} exceeds maximum {case['expected_confidence_max']} for: {case['query']}"
            
            if 'expected_confidence_min' in case:
                assert prediction.confidence >= case['expected_confidence_min']
            
            if case.get('multiple_categories_expected'):
                # Should detect multiple categories in metadata
                all_scores = prediction.metadata.get('all_scores', {})
                significant_categories = [cat for cat, score in all_scores.items() if score > 0.1]
                assert len(significant_categories) > 1, \
                    f"Expected multiple categories, found: {significant_categories}"
    
    @pytest.mark.biomedical
    def test_query_length_robustness(self, research_categorizer):
        """Test robustness with queries of various lengths."""
        
        # Very short query
        short_query = "LC-MS"
        short_prediction = research_categorizer.categorize_query(short_query)
        assert isinstance(short_prediction, CategoryPrediction)
        
        # Medium query
        medium_query = "LC-MS metabolomics analysis of plasma samples for biomarker discovery in diabetes patients"
        medium_prediction = research_categorizer.categorize_query(medium_query)
        assert isinstance(medium_prediction, CategoryPrediction)
        
        # Very long query (1000+ characters)
        long_query = """
        Comprehensive untargeted metabolomics profiling using high-resolution liquid chromatography
        tandem mass spectrometry (LC-MS/MS) for the identification and quantification of endogenous
        metabolites and xenobiotics in human plasma and urine samples collected from a large cohort
        of patients diagnosed with type 2 diabetes mellitus and age-matched healthy controls to
        investigate alterations in metabolic pathways including glycolysis, gluconeogenesis, lipid
        metabolism, amino acid metabolism, and gut microbiome-derived metabolite profiles for the
        discovery and validation of novel diagnostic biomarkers and prognostic indicators that
        could potentially be implemented in clinical practice for improved patient stratification,
        treatment monitoring, and personalized therapeutic interventions based on individual
        metabolic phenotypes and disease progression patterns with subsequent pathway enrichment
        analysis using KEGG, Reactome, and BioCyc databases to elucidate the underlying molecular
        mechanisms and metabolic dysregulation associated with insulin resistance, beta-cell
        dysfunction, and diabetic complications including nephropathy, retinopathy, and neuropathy.
        """ * 2  # Double it to make it even longer
        
        long_prediction = research_categorizer.categorize_query(long_query)
        assert isinstance(long_prediction, CategoryPrediction)
        assert long_prediction.confidence > 0  # Should find relevant patterns
    
    @pytest.mark.biomedical
    def test_special_characters_and_encoding(self, research_categorizer):
        """Test handling of special characters and encoding issues."""
        
        special_char_queries = [
            "LC-MS/MS análisis metabólicos",  # Non-ASCII characters
            "Glucose concentration (μM) in plasma samples",  # Greek letters
            "β-oxidation pathway analysis using ¹³C-labeled metabolites",  # Superscript
            "Cost-effective approach: $50 per sample → analysis",  # Special symbols
            "Metabolite identification @ 280.1 m/z (±0.01 Da tolerance)",  # Special chars
            "What's the difference between LC-MS & GC-MS?",  # Contractions and symbols
        ]
        
        for query in special_char_queries:
            prediction = research_categorizer.categorize_query(query)
            assert isinstance(prediction, CategoryPrediction)
            assert prediction.confidence >= 0.0
    
    @pytest.mark.biomedical  
    def test_ambiguous_multi_category_queries(self, research_categorizer):
        """Test handling of queries that could belong to multiple categories."""
        
        multi_category_queries = [
            {
                'query': "LC-MS metabolomics biomarker discovery and pathway analysis for drug development in cancer patients",
                'possible_categories': [
                    ResearchCategory.BIOMARKER_DISCOVERY,
                    ResearchCategory.PATHWAY_ANALYSIS,
                    ResearchCategory.DRUG_DISCOVERY,
                    ResearchCategory.CLINICAL_DIAGNOSIS
                ]
            },
            {
                'query': "Statistical analysis of metabolomics data for preprocessing and quality control in clinical studies",
                'possible_categories': [
                    ResearchCategory.STATISTICAL_ANALYSIS,
                    ResearchCategory.DATA_PREPROCESSING,
                    ResearchCategory.CLINICAL_DIAGNOSIS
                ]
            },
            {
                'query': "Database integration and literature mining for metabolite identification and pathway annotation",
                'possible_categories': [
                    ResearchCategory.DATABASE_INTEGRATION,
                    ResearchCategory.LITERATURE_SEARCH,
                    ResearchCategory.METABOLITE_IDENTIFICATION,
                    ResearchCategory.PATHWAY_ANALYSIS
                ]
            }
        ]
        
        for query_data in multi_category_queries:
            prediction = research_categorizer.categorize_query(query_data['query'])
            
            # Primary category should be one of the expected categories
            assert prediction.category in query_data['possible_categories'], \
                f"Category {prediction.category} not in expected categories {query_data['possible_categories']}"
            
            # Should have reasonable confidence (not too low due to ambiguity)
            assert prediction.confidence >= 0.3, \
                f"Confidence {prediction.confidence} too low for multi-category query"
            
            # Metadata should show multiple categories were considered
            all_scores = prediction.metadata.get('all_scores', {})
            categories_with_scores = [cat for cat, score in all_scores.items() if score > 0.1]
            assert len(categories_with_scores) >= 2, \
                f"Expected multiple categories with significant scores, got: {categories_with_scores}"


# =====================================================================
# PERFORMANCE TESTS
# =====================================================================

class TestPerformanceAndScalability:
    """Test performance and scalability requirements."""
    
    @pytest.mark.biomedical
    @pytest.mark.performance
    def test_single_query_response_time(self, research_categorizer, performance_requirements):
        """Test response time for individual queries."""
        
        test_queries = [
            "LC-MS metabolite identification using exact mass and fragmentation patterns",
            "KEGG pathway enrichment analysis for diabetes biomarker discovery",
            "Statistical analysis of metabolomics datasets using PCA and PLS-DA methods",
            "Clinical diagnosis applications of metabolomics in personalized medicine"
        ]
        
        response_times = []
        
        for query in test_queries:
            start_time = time.time()
            prediction = research_categorizer.categorize_query(query)
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            
            # Verify response time requirement
            assert response_time_ms <= performance_requirements['max_response_time_ms'], \
                f"Query response time {response_time_ms:.2f}ms exceeds limit {performance_requirements['max_response_time_ms']}ms"
            
            # Verify valid prediction
            assert isinstance(prediction, CategoryPrediction)
            assert prediction.confidence >= 0.0
        
        # Check average response time
        avg_response_time = statistics.mean(response_times)
        assert avg_response_time <= performance_requirements['max_response_time_ms'] * 0.5, \
            f"Average response time {avg_response_time:.2f}ms should be well below limit"
    
    @pytest.mark.biomedical
    @pytest.mark.performance
    def test_batch_processing_performance(self, research_categorizer, biomedical_query_samples, performance_requirements):
        """Test batch processing performance."""
        
        # Create batch of 50 diverse queries
        all_queries = biomedical_query_samples.get_all_test_queries()
        batch_queries = []
        
        for category, queries in all_queries.items():
            if category in ['edge_cases', 'performance']:
                continue
            # Take first 5 queries from each category
            for query_data in queries[:5]:
                if isinstance(query_data, dict) and 'query' in query_data:
                    batch_queries.append(query_data['query'])
        
        batch_queries = batch_queries[:50]  # Limit to 50 queries
        
        # Process batch and measure time
        start_time = time.time()
        predictions = []
        
        for query in batch_queries:
            prediction = research_categorizer.categorize_query(query)
            predictions.append(prediction)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify batch processing time requirement
        assert total_time <= performance_requirements['max_processing_time_batch'], \
            f"Batch processing time {total_time:.2f}s exceeds limit {performance_requirements['max_processing_time_batch']}s"
        
        # Verify all predictions are valid
        assert len(predictions) == len(batch_queries)
        for prediction in predictions:
            assert isinstance(prediction, CategoryPrediction)
            assert 0.0 <= prediction.confidence <= 1.0
        
        # Calculate throughput
        throughput = len(batch_queries) / total_time
        assert throughput >= 5.0, f"Throughput {throughput:.1f} queries/sec should be >= 5"
    
    @pytest.mark.biomedical
    @pytest.mark.performance
    def test_concurrent_processing(self, research_categorizer):
        """Test concurrent query processing."""
        import concurrent.futures
        import threading
        
        test_queries = [
            "Metabolite identification using LC-MS/MS",
            "KEGG pathway analysis for diabetes",
            "Biomarker discovery in cardiovascular disease",
            "Clinical diagnosis using metabolomics",
            "Drug discovery and ADMET screening"
        ] * 10  # 50 queries total
        
        # Process concurrently
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(research_categorizer.categorize_query, query) 
                      for query in test_queries]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all results
        assert len(results) == len(test_queries)
        for result in results:
            assert isinstance(result, CategoryPrediction)
        
        # Should be faster than sequential processing
        sequential_estimate = len(test_queries) * 0.1  # Assume 100ms per query
        efficiency = sequential_estimate / total_time
        assert efficiency >= 1.5, f"Concurrent processing efficiency {efficiency:.2f} should be >= 1.5x"
    
    @pytest.mark.biomedical
    @pytest.mark.performance
    def test_memory_usage_stability(self, research_categorizer):
        """Test memory usage remains stable during extended operation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many queries to test for memory leaks
        test_query = "LC-MS metabolomics analysis for biomarker discovery in clinical samples"
        
        for i in range(200):  # Process 200 queries
            prediction = research_categorizer.categorize_query(f"{test_query} iteration {i}")
            assert isinstance(prediction, CategoryPrediction)
            
            if i % 50 == 0:  # Check memory every 50 iterations
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory should not increase significantly
                assert memory_increase <= 50, \
                    f"Memory increased by {memory_increase:.1f}MB after {i} queries"
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory
        
        # Total memory increase should be reasonable
        assert total_memory_increase <= 100, \
            f"Total memory increase {total_memory_increase:.1f}MB exceeds 100MB limit"


# =====================================================================
# INTEGRATION AND VALIDATION TESTS
# =====================================================================

class TestIntegrationAndValidation:
    """Test integration with biomedical data and validation against expected results."""
    
    @pytest.mark.biomedical
    def test_integration_with_clinical_data(self, research_categorizer, clinical_data_generator):
        """Test integration with clinical metabolomics data."""
        
        # Generate clinical study data
        study_data = clinical_data_generator.generate_clinical_study_dataset(
            disease='diabetes',
            sample_size=150,
            study_type='case_control'
        )
        
        # Create queries based on study data
        study_based_queries = [
            f"Analysis of {', '.join(study_data.biomarkers_identified)} biomarkers in {study_data.disease_condition}",
            f"Statistical analysis using {', '.join(study_data.statistical_methods[:2])} for {study_data.study_type} study",
            f"{study_data.analytical_platform} analysis of clinical samples for biomarker discovery",
            f"Pathway analysis of metabolites associated with {study_data.disease_condition}"
        ]
        
        for query in study_based_queries:
            prediction = research_categorizer.categorize_query(query)
            
            # Should classify correctly based on query content
            assert isinstance(prediction, CategoryPrediction)
            assert prediction.confidence > 0.3  # Should have reasonable confidence
            
            # Clinical queries should often detect clinical subject area
            if 'clinical' in query.lower() or 'patient' in query.lower():
                assert prediction.subject_area in [None, 'clinical'] or 'clinical' in str(prediction.subject_area)
    
    @pytest.mark.biomedical
    def test_validation_against_metabolite_database(self, research_categorizer, clinical_data_generator):
        """Test validation against metabolite database information."""
        
        metabolite_db = clinical_data_generator.METABOLITE_DATABASE
        
        for metabolite_name, metabolite_data in list(metabolite_db.items())[:5]:
            # Create queries based on metabolite properties
            queries = [
                f"Identification of {metabolite_data.name} with molecular formula {metabolite_data.molecular_formula}",
                f"Analysis of {metabolite_data.name} in {', '.join(metabolite_data.pathways[:2])} pathways",
                f"Clinical significance of {metabolite_data.name} in {list(metabolite_data.disease_associations.keys())[0] if metabolite_data.disease_associations else 'metabolic disorders'}",
                f"{metabolite_data.analytical_platforms[0] if metabolite_data.analytical_platforms else 'LC-MS'} analysis of {metabolite_data.name}"
            ]
            
            for query in queries:
                prediction = research_categorizer.categorize_query(query)
                
                # Should classify appropriately based on query type
                assert isinstance(prediction, CategoryPrediction)
                assert prediction.confidence > 0.2
                
                # Check for technical terms detection
                if metabolite_data.analytical_platforms:
                    analysis_details = prediction.metadata.get('analysis_details', {})
                    technical_terms = analysis_details.get('has_technical_terms', False)
                    if any(platform.lower() in query.lower() for platform in metabolite_data.analytical_platforms):
                        assert technical_terms, f"Should detect technical terms in: {query}"
    
    @pytest.mark.biomedical
    def test_cross_validation_with_disease_panels(self, research_categorizer, clinical_data_generator):
        """Test cross-validation with disease-specific metabolite panels."""
        
        disease_panels = clinical_data_generator.DISEASE_PANELS
        
        for disease, panel_info in disease_panels.items():
            # Create queries based on disease panel information
            panel_queries = [
                f"Biomarker discovery for {disease} using {', '.join(panel_info['primary_markers'])}",
                f"Pathway analysis of {', '.join(panel_info['pathways'])} in {disease} patients",
                f"Clinical diagnosis of {disease} using metabolomics profiling",
                f"Statistical analysis of metabolite changes in {disease}: {', '.join(panel_info['typical_changes'].keys())}"
            ]
            
            for query in panel_queries:
                prediction = research_categorizer.categorize_query(query)
                
                # Should classify with reasonable confidence
                assert prediction.confidence > 0.3, \
                    f"Low confidence {prediction.confidence} for disease panel query: {query[:50]}..."
                
                # Disease-related queries should often be clinical
                if 'patient' in query or 'clinical' in query or 'diagnosis' in query:
                    expected_categories = [
                        ResearchCategory.CLINICAL_DIAGNOSIS,
                        ResearchCategory.BIOMARKER_DISCOVERY
                    ]
                    assert prediction.category in expected_categories, \
                        f"Expected clinical category for: {query[:30]}..."


# =====================================================================
# COMPREHENSIVE INTEGRATION TEST
# =====================================================================

class TestComprehensiveQueryClassificationValidation:
    """
    Comprehensive integration test that validates the entire query classification
    system against realistic biomedical scenarios and requirements.
    """
    
    @pytest.mark.biomedical
    @pytest.mark.integration
    def test_comprehensive_biomedical_query_validation(
        self, 
        research_categorizer, 
        biomedical_query_samples,
        performance_requirements
    ):
        """
        Comprehensive validation test that exercises the complete query classification
        system with realistic biomedical queries and validates against all requirements.
        """
        
        # Get all test queries
        all_queries = biomedical_query_samples.get_all_test_queries()
        
        # Validation metrics
        total_queries = 0
        correct_classifications = 0
        confidence_scores = []
        response_times = []
        category_performance = {}
        
        # Test each category
        for category_name, queries in all_queries.items():
            if category_name in ['edge_cases', 'performance']:
                continue
            
            category_correct = 0
            category_total = 0
            
            for query_data in queries:
                if not isinstance(query_data, dict) or 'query' not in query_data:
                    continue
                
                # Measure response time
                start_time = time.time()
                prediction = research_categorizer.categorize_query(query_data['query'])
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # ms
                response_times.append(response_time)
                
                # Validate prediction
                assert isinstance(prediction, CategoryPrediction)
                assert 0.0 <= prediction.confidence <= 1.0
                assert prediction.category in ResearchCategory
                
                # Check expected results
                if 'expected_category' in query_data:
                    is_correct = prediction.category == query_data['expected_category']
                    if is_correct:
                        correct_classifications += 1
                        category_correct += 1
                    else:
                        # Log misclassification for analysis
                        print(f"Misclassified: {query_data['query'][:50]}... - Expected: {query_data['expected_category']}, Got: {prediction.category}")
                
                confidence_scores.append(prediction.confidence)
                total_queries += 1
                category_total += 1
                
                # Validate confidence requirements
                if 'expected_confidence_min' in query_data:
                    assert prediction.confidence >= query_data['expected_confidence_min'], \
                        f"Confidence {prediction.confidence} below minimum {query_data['expected_confidence_min']}"
                
                # Validate response time
                assert response_time <= performance_requirements['max_response_time_ms'], \
                    f"Response time {response_time:.2f}ms exceeds limit"
            
            # Store category performance
            if category_total > 0:
                category_performance[category_name] = {
                    'accuracy': category_correct / category_total,
                    'total_queries': category_total,
                    'correct': category_correct
                }
        
        # Overall validation
        overall_accuracy = correct_classifications / total_queries
        avg_confidence = statistics.mean(confidence_scores)
        avg_response_time = statistics.mean(response_times)
        
        # Performance requirements validation
        assert overall_accuracy >= (performance_requirements['min_accuracy_percent'] / 100), \
            f"Overall accuracy {overall_accuracy:.2%} below requirement {performance_requirements['min_accuracy_percent']}%"
        
        assert avg_response_time <= performance_requirements['max_response_time_ms'], \
            f"Average response time {avg_response_time:.2f}ms exceeds limit"
        
        # Category-specific validation
        for category_name, perf in category_performance.items():
            assert perf['accuracy'] >= 0.7, \
                f"Category {category_name} accuracy {perf['accuracy']:.2%} below 70%"
        
        # Generate validation report
        validation_report = {
            'total_queries_tested': total_queries,
            'overall_accuracy': overall_accuracy,
            'average_confidence': avg_confidence,
            'average_response_time_ms': avg_response_time,
            'category_performance': category_performance,
            'performance_requirements_met': {
                'accuracy': overall_accuracy >= (performance_requirements['min_accuracy_percent'] / 100),
                'response_time': avg_response_time <= performance_requirements['max_response_time_ms'],
                'confidence_reasonable': avg_confidence >= 0.5
            }
        }
        
        print(f"\n=== Query Classification Validation Report ===")
        print(f"Total Queries Tested: {validation_report['total_queries_tested']}")
        print(f"Overall Accuracy: {validation_report['overall_accuracy']:.2%}")
        print(f"Average Confidence: {validation_report['average_confidence']:.3f}")
        print(f"Average Response Time: {validation_report['average_response_time_ms']:.2f}ms")
        print(f"Performance Requirements Met: {all(validation_report['performance_requirements_met'].values())}")
        
        print(f"\nCategory Performance:")
        for category, perf in validation_report['category_performance'].items():
            print(f"  {category}: {perf['accuracy']:.2%} ({perf['correct']}/{perf['total_queries']})")
        
        # Final assertion - all requirements met
        assert all(validation_report['performance_requirements_met'].values()), \
            "Not all performance requirements were met"


# =====================================================================
# MAIN TEST EXECUTION
# =====================================================================

if __name__ == "__main__":
    # Run tests with comprehensive reporting
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-m", "biomedical",
        "--no-header",
        "--show-capture=no"
    ])