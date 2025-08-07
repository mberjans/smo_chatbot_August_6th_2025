#!/usr/bin/env python3
"""
Test Fixtures for Clinical Metabolomics Relevance Scoring System Tests.

This module provides comprehensive test fixtures, mock data, and utilities
specifically designed for testing the relevance scoring system.

Features:
- Comprehensive test query collections by category
- High-quality test responses with varying characteristics
- Mock components for isolated unit testing
- Performance test data generators
- Biomedical domain-specific test cases
- Edge case scenarios and stress test data

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import random
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json


# =====================================================================
# TEST DATA CLASSES
# =====================================================================

@dataclass
class TestQuery:
    """Represents a test query with metadata."""
    text: str
    expected_type: str
    complexity: str = "medium"  # low, medium, high
    domain_specificity: str = "moderate"  # low, moderate, high
    keywords: List[str] = field(default_factory=list)
    expected_score_range: Tuple[float, float] = (40.0, 100.0)
    notes: str = ""

@dataclass
class TestResponse:
    """Represents a test response with quality characteristics."""
    text: str
    quality_level: str  # excellent, good, fair, poor
    length_category: str  # short, medium, long, very_long
    structure_quality: str  # excellent, good, fair, poor
    biomedical_density: str  # high, medium, low
    technical_accuracy: str  # high, medium, low, questionable
    expected_scores: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    notes: str = ""

@dataclass
class TestScenario:
    """Represents a complete test scenario with query and response."""
    name: str
    query: TestQuery
    response: TestResponse
    expected_overall_score_range: Tuple[float, float] = (0.0, 100.0)
    test_categories: List[str] = field(default_factory=list)
    notes: str = ""


# =====================================================================
# COMPREHENSIVE QUERY FIXTURES
# =====================================================================

class QueryFixtures:
    """Comprehensive collection of test queries by category."""
    
    BASIC_DEFINITION_QUERIES = [
        TestQuery(
            text="What is metabolomics?",
            expected_type="basic_definition",
            complexity="low",
            keywords=["metabolomics", "definition"],
            expected_score_range=(60.0, 100.0)
        ),
        TestQuery(
            text="Define biomarker in the context of clinical research",
            expected_type="basic_definition",
            complexity="medium",
            keywords=["biomarker", "clinical", "research"],
            expected_score_range=(50.0, 95.0)
        ),
        TestQuery(
            text="Explain the basic principles of mass spectrometry",
            expected_type="basic_definition",
            complexity="medium",
            keywords=["mass spectrometry", "principles"],
            expected_score_range=(45.0, 90.0)
        ),
        TestQuery(
            text="What does LC-MS stand for and how does it work?",
            expected_type="basic_definition",
            complexity="medium",
            keywords=["LC-MS", "chromatography", "mass spectrometry"],
            expected_score_range=(50.0, 95.0)
        ),
        TestQuery(
            text="Introduction to metabolomic data analysis",
            expected_type="basic_definition",
            complexity="high",
            keywords=["metabolomics", "data analysis"],
            expected_score_range=(40.0, 85.0)
        )
    ]
    
    CLINICAL_APPLICATION_QUERIES = [
        TestQuery(
            text="How is metabolomics used in clinical diagnosis?",
            expected_type="clinical_application",
            complexity="medium",
            domain_specificity="high",
            keywords=["metabolomics", "clinical", "diagnosis"],
            expected_score_range=(60.0, 100.0)
        ),
        TestQuery(
            text="Clinical applications of biomarker discovery in personalized medicine",
            expected_type="clinical_application", 
            complexity="high",
            domain_specificity="high",
            keywords=["biomarker", "clinical", "personalized medicine"],
            expected_score_range=(55.0, 95.0)
        ),
        TestQuery(
            text="Patient monitoring using metabolomic profiling",
            expected_type="clinical_application",
            complexity="medium",
            keywords=["patient", "monitoring", "metabolomics"],
            expected_score_range=(50.0, 90.0)
        ),
        TestQuery(
            text="Therapeutic drug monitoring with metabolomics approaches",
            expected_type="clinical_application",
            complexity="high",
            keywords=["therapeutic", "drug monitoring", "metabolomics"],
            expected_score_range=(55.0, 95.0)
        ),
        TestQuery(
            text="Clinical implementation challenges for metabolomics",
            expected_type="clinical_application",
            complexity="high",
            keywords=["clinical", "implementation", "metabolomics"],
            expected_score_range=(45.0, 85.0)
        )
    ]
    
    ANALYTICAL_METHOD_QUERIES = [
        TestQuery(
            text="LC-MS protocol for metabolomics analysis",
            expected_type="analytical_method",
            complexity="high",
            domain_specificity="high",
            keywords=["LC-MS", "protocol", "metabolomics"],
            expected_score_range=(65.0, 100.0)
        ),
        TestQuery(
            text="GC-MS sample preparation procedures",
            expected_type="analytical_method",
            complexity="high",
            keywords=["GC-MS", "sample preparation"],
            expected_score_range=(60.0, 95.0)
        ),
        TestQuery(
            text="HILIC chromatography for polar metabolite analysis",
            expected_type="analytical_method",
            complexity="high",
            domain_specificity="high",
            keywords=["HILIC", "chromatography", "polar metabolites"],
            expected_score_range=(60.0, 100.0)
        ),
        TestQuery(
            text="NMR spectroscopy methods in metabolomics",
            expected_type="analytical_method",
            complexity="high",
            keywords=["NMR", "spectroscopy", "metabolomics"],
            expected_score_range=(55.0, 95.0)
        ),
        TestQuery(
            text="Quality control measures in metabolomic analysis",
            expected_type="analytical_method",
            complexity="medium",
            keywords=["quality control", "metabolomics"],
            expected_score_range=(50.0, 90.0)
        )
    ]
    
    RESEARCH_DESIGN_QUERIES = [
        TestQuery(
            text="Study design considerations for metabolomics research",
            expected_type="research_design",
            complexity="high",
            keywords=["study design", "metabolomics", "research"],
            expected_score_range=(55.0, 95.0)
        ),
        TestQuery(
            text="Statistical analysis methods for metabolomic data",
            expected_type="research_design",
            complexity="high",
            keywords=["statistical analysis", "metabolomics"],
            expected_score_range=(60.0, 100.0)
        ),
        TestQuery(
            text="Sample size calculation for biomarker discovery studies",
            expected_type="research_design",
            complexity="high",
            keywords=["sample size", "biomarker", "studies"],
            expected_score_range=(50.0, 90.0)
        ),
        TestQuery(
            text="Validation strategies for metabolomic biomarkers",
            expected_type="research_design",
            complexity="high",
            keywords=["validation", "metabolomics", "biomarkers"],
            expected_score_range=(55.0, 95.0)
        ),
        TestQuery(
            text="Reproducibility challenges in metabolomics studies",
            expected_type="research_design",
            complexity="high",
            keywords=["reproducibility", "metabolomics"],
            expected_score_range=(45.0, 85.0)
        )
    ]
    
    DISEASE_SPECIFIC_QUERIES = [
        TestQuery(
            text="Metabolomics in diabetes research and biomarker discovery",
            expected_type="disease_specific",
            complexity="high",
            domain_specificity="high",
            keywords=["metabolomics", "diabetes", "biomarker"],
            expected_score_range=(60.0, 100.0)
        ),
        TestQuery(
            text="Cancer metabolomics and therapeutic targets",
            expected_type="disease_specific",
            complexity="high",
            keywords=["cancer", "metabolomics", "therapeutic"],
            expected_score_range=(55.0, 95.0)
        ),
        TestQuery(
            text="Cardiovascular disease metabolic signatures",
            expected_type="disease_specific",
            complexity="medium",
            keywords=["cardiovascular", "metabolic", "signatures"],
            expected_score_range=(50.0, 90.0)
        ),
        TestQuery(
            text="Alzheimer's disease biomarkers through metabolomics",
            expected_type="disease_specific",
            complexity="high",
            keywords=["Alzheimer", "biomarkers", "metabolomics"],
            expected_score_range=(55.0, 95.0)
        ),
        TestQuery(
            text="Liver disease metabolic profiling and diagnosis",
            expected_type="disease_specific",
            complexity="medium",
            keywords=["liver disease", "metabolic", "diagnosis"],
            expected_score_range=(50.0, 90.0)
        )
    ]
    
    EDGE_CASE_QUERIES = [
        TestQuery(
            text="",
            expected_type="general",
            complexity="low",
            expected_score_range=(0.0, 20.0),
            notes="Empty query"
        ),
        TestQuery(
            text="?",
            expected_type="general", 
            complexity="low",
            expected_score_range=(0.0, 30.0),
            notes="Single character query"
        ),
        TestQuery(
            text="What is the weather like today?",
            expected_type="general",
            complexity="low",
            expected_score_range=(0.0, 40.0),
            notes="Non-biomedical query"
        ),
        TestQuery(
            text="metabolomics " * 50,
            expected_type="general",
            complexity="low",
            expected_score_range=(20.0, 60.0),
            notes="Repetitive long query"
        ),
        TestQuery(
            text="xyzabc defghi jklmno pqrstu vwxyz",
            expected_type="general",
            complexity="low",
            expected_score_range=(0.0, 30.0),
            notes="Nonsensical query"
        )
    ]
    
    @classmethod
    def get_all_queries(cls) -> List[TestQuery]:
        """Get all test queries."""
        return (
            cls.BASIC_DEFINITION_QUERIES +
            cls.CLINICAL_APPLICATION_QUERIES +
            cls.ANALYTICAL_METHOD_QUERIES + 
            cls.RESEARCH_DESIGN_QUERIES +
            cls.DISEASE_SPECIFIC_QUERIES +
            cls.EDGE_CASE_QUERIES
        )
    
    @classmethod
    def get_queries_by_type(cls, query_type: str) -> List[TestQuery]:
        """Get queries of specific type."""
        type_mapping = {
            'basic_definition': cls.BASIC_DEFINITION_QUERIES,
            'clinical_application': cls.CLINICAL_APPLICATION_QUERIES,
            'analytical_method': cls.ANALYTICAL_METHOD_QUERIES,
            'research_design': cls.RESEARCH_DESIGN_QUERIES,
            'disease_specific': cls.DISEASE_SPECIFIC_QUERIES,
            'edge_cases': cls.EDGE_CASE_QUERIES
        }
        return type_mapping.get(query_type, [])


# =====================================================================
# COMPREHENSIVE RESPONSE FIXTURES
# =====================================================================

class ResponseFixtures:
    """Comprehensive collection of test responses with varying quality."""
    
    EXCELLENT_RESPONSES = [
        TestResponse(
            text="""# Metabolomics in Clinical Applications

## Definition and Overview
Metabolomics is the comprehensive study of small molecules called metabolites in biological systems. This rapidly evolving field focuses on the systematic analysis of the complete set of metabolites present in cells, tissues, or biological fluids under specific physiological or pathological conditions.

## Clinical Applications

### Biomarker Discovery
- **Disease-specific signatures**: Identification of metabolic fingerprints associated with specific diseases
- **Early detection**: Discovery of metabolites that change before clinical symptoms appear
- **Progression monitoring**: Tracking metabolite changes during disease development
- **Treatment response**: Assessment of metabolic changes following therapeutic interventions

### Diagnostic Applications
- **Non-invasive testing**: Analysis of easily accessible samples (blood, urine, saliva)
- **Improved sensitivity**: Detection of subtle metabolic changes with high precision
- **Specificity enhancement**: Discrimination between similar disease conditions
- **Personalized diagnostics**: Tailored testing approaches based on individual metabolic profiles

### Therapeutic Monitoring
- **Drug efficacy assessment**: Evaluation of treatment effectiveness through metabolic changes
- **Toxicity monitoring**: Early detection of adverse drug reactions
- **Dosage optimization**: Adjustment of treatment protocols based on metabolic responses
- **Companion diagnostics**: Integration with therapeutic decision-making

## Analytical Methodologies

### Mass Spectrometry Approaches
- **LC-MS/MS**: Liquid chromatography coupled with tandem mass spectrometry for comprehensive metabolite profiling
- **GC-MS**: Gas chromatography-mass spectrometry for volatile and derivatized metabolites
- **UPLC-QTOF**: Ultra-performance liquid chromatography with quadrupole time-of-flight mass spectrometry

### Nuclear Magnetic Resonance (NMR)
- **Structural elucidation**: Detailed molecular structure determination
- **Quantitative analysis**: Absolute quantification without reference standards
- **Non-destructive analysis**: Sample preservation for additional analyses

### Sample Preparation Considerations
- **Pre-analytical factors**: Sample collection, storage, and processing protocols
- **Extraction methods**: Optimized procedures for different metabolite classes
- **Quality control**: Internal standards and reference samples for analytical validation

## Data Analysis and Bioinformatics

### Statistical Methods
- **Multivariate analysis**: Principal component analysis (PCA) and partial least squares discriminant analysis (PLS-DA)
- **Univariate statistics**: T-tests, ANOVA, and non-parametric alternatives
- **Machine learning**: Random forests, support vector machines, and neural networks

### Pathway Analysis
- **Metabolic pathway mapping**: Integration with KEGG, BioCyc, and Reactome databases
- **Network analysis**: Investigation of metabolite-metabolite and metabolite-protein interactions
- **Functional enrichment**: Identification of dysregulated biological processes

## Challenges and Future Directions

### Technical Challenges
- **Standardization**: Development of harmonized protocols across laboratories
- **Reference materials**: Availability of certified reference standards
- **Data integration**: Combination of multi-platform and multi-laboratory data

### Clinical Translation
- **Regulatory approval**: Navigation of FDA and EMA requirements for clinical implementation
- **Cost-effectiveness**: Economic evaluation of metabolomic testing
- **Clinical utility**: Demonstration of improved patient outcomes

### Emerging Opportunities
- **Precision medicine**: Integration with genomics and proteomics for comprehensive patient profiling
- **Real-time monitoring**: Development of point-of-care metabolomic devices
- **Artificial intelligence**: Application of deep learning for pattern recognition and biomarker discovery

## Conclusion
Metabolomics represents a powerful approach for understanding disease mechanisms, discovering biomarkers, and advancing personalized medicine. Continued technological improvements, standardization efforts, and clinical validation studies will further enhance its impact on healthcare delivery and patient outcomes.""",
            quality_level="excellent",
            length_category="long",
            structure_quality="excellent",
            biomedical_density="high",
            technical_accuracy="high",
            expected_scores={
                "metabolomics_relevance": (85.0, 100.0),
                "clinical_applicability": (80.0, 100.0),
                "query_alignment": (85.0, 100.0),
                "scientific_rigor": (80.0, 95.0),
                "biomedical_context_depth": (85.0, 100.0),
                "response_length_quality": (85.0, 100.0),
                "response_structure_quality": (90.0, 100.0)
            }
        ),
        
        TestResponse(
            text="""LC-MS (Liquid Chromatography-Mass Spectrometry) represents a gold-standard analytical platform for metabolomics research, combining the separation power of liquid chromatography with the identification and quantification capabilities of mass spectrometry.

## Analytical Workflow

### Sample Preparation
1. **Protein precipitation**: Removal of proteins using organic solvents (methanol, acetonitrile)
2. **Liquid-liquid extraction**: Separation of metabolites based on polarity
3. **Solid-phase extraction**: Selective enrichment of specific metabolite classes
4. **Quality control**: Integration of pooled samples and internal standards

### Chromatographic Separation
- **C18 reverse-phase**: Optimal for lipids and moderately polar metabolites
- **HILIC (Hydrophilic Interaction Chromatography)**: Ideal for polar and charged metabolites
- **Ion-pair chromatography**: Effective for highly polar and ionic compounds
- **Gradient optimization**: Systematic development for maximum peak resolution

### Mass Spectrometry Detection
- **Electrospray ionization (ESI)**: Soft ionization technique for intact molecular ions
- **Positive/negative ion modes**: Comprehensive coverage of different metabolite classes
- **High-resolution mass spectrometry**: Accurate mass determination for molecular formula assignment
- **Tandem MS (MS/MS)**: Structural confirmation through fragmentation patterns

## Data Processing and Analysis

### Peak Detection and Alignment
- **XCMS**: Open-source platform for peak detection and retention time alignment
- **MZmine**: Comprehensive tool for mass spectrometry data processing
- **Compound Discoverer**: Thermo Scientific's integrated workflow solution

### Statistical Analysis
- **Normalization**: Correction for systematic variations and batch effects
- **Missing value imputation**: Handling of below-detection-limit metabolites
- **Multivariate statistics**: PCA, PLS-DA, and OPLS-DA for pattern recognition
- **Pathway analysis**: Metabolite set enrichment and network analysis

## Clinical Applications

### Biomarker Discovery
- Identification of disease-specific metabolic signatures
- Validation in independent cohorts
- Assessment of diagnostic accuracy (ROC analysis, sensitivity, specificity)

### Therapeutic Monitoring
- Pharmacokinetic profiling of drug metabolites
- Assessment of treatment efficacy and toxicity
- Personalized dosing strategies

This comprehensive LC-MS approach enables robust, reproducible, and clinically relevant metabolomic analyses across diverse research applications.""",
            quality_level="excellent",
            length_category="long",
            structure_quality="excellent",
            biomedical_density="high", 
            technical_accuracy="high"
        )
    ]
    
    GOOD_RESPONSES = [
        TestResponse(
            text="""Metabolomics is the scientific study of small molecules called metabolites in biological systems. It provides a comprehensive view of the metabolic state of cells, tissues, or organisms.

## Key Applications

### Clinical Diagnosis
- Disease biomarker identification
- Early detection of pathological conditions
- Monitoring treatment responses
- Personalized medicine approaches

### Research Applications
- Understanding disease mechanisms
- Drug development and testing
- Nutritional studies
- Environmental health assessments

## Analytical Methods

**Mass Spectrometry**
- LC-MS: Liquid chromatography-mass spectrometry
- GC-MS: Gas chromatography-mass spectrometry
- High sensitivity and specificity

**NMR Spectroscopy**
- Provides structural information
- Quantitative analysis capabilities
- Non-destructive sample analysis

## Challenges
- Data complexity and standardization
- Need for specialized expertise
- Cost and accessibility considerations
- Regulatory requirements for clinical use

Metabolomics continues to advance with technological improvements and growing clinical applications, making it an increasingly valuable tool for precision medicine.""",
            quality_level="good",
            length_category="medium",
            structure_quality="good",
            biomedical_density="medium",
            technical_accuracy="high"
        ),
        
        TestResponse(
            text="""LC-MS analysis involves several key steps for metabolomics research:

1. **Sample Collection and Preparation**
   - Standardized collection protocols
   - Protein removal using precipitation
   - Metabolite extraction optimization

2. **Chromatographic Separation**
   - Column selection based on metabolite properties
   - Mobile phase optimization
   - Gradient development for best resolution

3. **Mass Spectrometry Detection**
   - Ionization method selection (ESI, APCI)
   - Mass analyzer configuration
   - Data acquisition in positive/negative modes

4. **Data Processing**
   - Peak detection and integration
   - Retention time alignment
   - Statistical analysis and visualization

5. **Results Interpretation**
   - Metabolite identification using databases
   - Pathway analysis and biological interpretation
   - Biomarker validation studies

This systematic approach ensures reliable and reproducible metabolomic results for clinical and research applications.""",
            quality_level="good",
            length_category="medium",
            structure_quality="good",
            biomedical_density="high",
            technical_accuracy="high"
        )
    ]
    
    FAIR_RESPONSES = [
        TestResponse(
            text="""Metabolomics studies small molecules in biological samples. It's used for finding biomarkers and understanding diseases.

Common methods include:
- LC-MS for liquid samples
- GC-MS for volatile compounds
- NMR for structure analysis

Clinical uses:
- Disease diagnosis
- Drug testing
- Treatment monitoring

The field has challenges with data analysis and standardization, but it's growing quickly and becoming more important in medicine.""",
            quality_level="fair",
            length_category="short",
            structure_quality="fair",
            biomedical_density="medium",
            technical_accuracy="medium"
        )
    ]
    
    POOR_RESPONSES = [
        TestResponse(
            text="Metabolomics is good for research. It uses machines to analyze samples and find things.",
            quality_level="poor",
            length_category="short",
            structure_quality="poor",
            biomedical_density="low",
            technical_accuracy="low"
        ),
        
        TestResponse(
            text="LC-MS works well and gives results. Scientists use it for studies.",
            quality_level="poor",
            length_category="short",
            structure_quality="poor",
            biomedical_density="low",
            technical_accuracy="low"
        )
    ]
    
    EDGE_CASE_RESPONSES = [
        TestResponse(
            text="",
            quality_level="poor",
            length_category="short",
            structure_quality="poor",
            biomedical_density="low",
            technical_accuracy="low",
            notes="Empty response"
        ),
        
        TestResponse(
            text="The weather is nice today. Traffic is moving smoothly. Pizza is delicious.",
            quality_level="poor",
            length_category="short",
            structure_quality="poor",
            biomedical_density="low",
            technical_accuracy="low",
            notes="Non-biomedical content"
        ),
        
        TestResponse(
            text="""This response contains contradictory information. Metabolomics is always 100% accurate and never fails to identify every metabolite perfectly. However, it sometimes gives uncertain results and may not be completely reliable. The field is both revolutionary and traditional, offering groundbreaking discoveries while maintaining established conventional methods. It definitely maybe provides possibly reliable results.""",
            quality_level="poor",
            length_category="medium",
            structure_quality="fair",
            biomedical_density="medium",
            technical_accuracy="questionable",
            notes="Contradictory and inconsistent content"
        ),
        
        TestResponse(
            text="Metabolomics research analysis " * 200,
            quality_level="poor",
            length_category="very_long",
            structure_quality="poor",
            biomedical_density="low",
            technical_accuracy="low",
            notes="Repetitive very long response"
        )
    ]
    
    @classmethod
    def get_all_responses(cls) -> List[TestResponse]:
        """Get all test responses."""
        return (
            cls.EXCELLENT_RESPONSES +
            cls.GOOD_RESPONSES +
            cls.FAIR_RESPONSES +
            cls.POOR_RESPONSES +
            cls.EDGE_CASE_RESPONSES
        )
    
    @classmethod
    def get_responses_by_quality(cls, quality_level: str) -> List[TestResponse]:
        """Get responses by quality level."""
        quality_mapping = {
            'excellent': cls.EXCELLENT_RESPONSES,
            'good': cls.GOOD_RESPONSES,
            'fair': cls.FAIR_RESPONSES,
            'poor': cls.POOR_RESPONSES,
            'edge_cases': cls.EDGE_CASE_RESPONSES
        }
        return quality_mapping.get(quality_level, [])


# =====================================================================
# COMPREHENSIVE TEST SCENARIOS
# =====================================================================

class ScenarioFixtures:
    """Comprehensive test scenarios combining queries and responses."""
    
    @classmethod
    def generate_standard_scenarios(cls) -> List[TestScenario]:
        """Generate standard test scenarios."""
        scenarios = []
        
        # High-quality matches
        scenarios.extend([
            TestScenario(
                name="Excellent Metabolomics Overview",
                query=QueryFixtures.BASIC_DEFINITION_QUERIES[0],  # "What is metabolomics?"
                response=ResponseFixtures.EXCELLENT_RESPONSES[0],
                expected_overall_score_range=(85.0, 100.0),
                test_categories=["dimensions", "integration", "quality"],
                notes="Perfect query-response alignment with comprehensive content"
            ),
            
            TestScenario(
                name="Technical LC-MS Query with Expert Response",
                query=QueryFixtures.ANALYTICAL_METHOD_QUERIES[0],  # LC-MS protocol
                response=ResponseFixtures.EXCELLENT_RESPONSES[1],  # LC-MS detailed response
                expected_overall_score_range=(85.0, 100.0),
                test_categories=["dimensions", "weighting", "biomedical"],
                notes="Technical query with matching expert-level response"
            )
        ])
        
        # Medium-quality matches
        scenarios.extend([
            TestScenario(
                name="Clinical Query with Good Response",
                query=QueryFixtures.CLINICAL_APPLICATION_QUERIES[0],
                response=ResponseFixtures.GOOD_RESPONSES[0],
                expected_overall_score_range=(65.0, 85.0),
                test_categories=["classification", "weighting"],
                notes="Clinical query with adequate response"
            )
        ])
        
        # Poor matches
        scenarios.extend([
            TestScenario(
                name="Complex Query with Poor Response",
                query=QueryFixtures.RESEARCH_DESIGN_QUERIES[0],
                response=ResponseFixtures.POOR_RESPONSES[0],
                expected_overall_score_range=(10.0, 40.0),
                test_categories=["dimensions", "quality"],
                notes="Complex query poorly addressed"
            )
        ])
        
        # Edge cases
        scenarios.extend([
            TestScenario(
                name="Empty Query and Response",
                query=QueryFixtures.EDGE_CASE_QUERIES[0],  # Empty query
                response=ResponseFixtures.EDGE_CASE_RESPONSES[0],  # Empty response
                expected_overall_score_range=(0.0, 20.0),
                test_categories=["edge_cases"],
                notes="Both query and response are empty"
            ),
            
            TestScenario(
                name="Biomedical Query with Non-Biomedical Response",
                query=QueryFixtures.DISEASE_SPECIFIC_QUERIES[0],
                response=ResponseFixtures.EDGE_CASE_RESPONSES[1],  # Weather response
                expected_overall_score_range=(0.0, 30.0),
                test_categories=["edge_cases", "biomedical"],
                notes="Complete domain mismatch"
            )
        ])
        
        return scenarios
    
    @classmethod
    def generate_performance_scenarios(cls, count: int = 100) -> List[TestScenario]:
        """Generate scenarios for performance testing."""
        scenarios = []
        queries = QueryFixtures.get_all_queries()
        responses = ResponseFixtures.get_all_responses()
        
        for i in range(count):
            query = random.choice(queries)
            response = random.choice(responses)
            
            scenarios.append(TestScenario(
                name=f"Performance Test {i+1}",
                query=query,
                response=response,
                test_categories=["performance"]
            ))
        
        return scenarios
    
    @classmethod
    def generate_stress_scenarios(cls, count: int = 500) -> List[TestScenario]:
        """Generate scenarios for stress testing."""
        scenarios = []
        
        # Create many variations
        base_queries = QueryFixtures.get_all_queries()
        base_responses = ResponseFixtures.get_all_responses()
        
        for i in range(count):
            # Select random components
            query = random.choice(base_queries)
            response = random.choice(base_responses)
            
            # Add some variation
            if i % 10 == 0:
                # Make some queries longer
                query_text = query.text + " " + "Please provide detailed analysis." * random.randint(1, 5)
                query = TestQuery(
                    text=query_text,
                    expected_type=query.expected_type,
                    complexity="high"
                )
            
            scenarios.append(TestScenario(
                name=f"Stress Test {i+1}",
                query=query,
                response=response,
                test_categories=["stress"]
            ))
        
        return scenarios


# =====================================================================
# PERFORMANCE TEST DATA GENERATORS
# =====================================================================

class PerformanceDataGenerator:
    """Generates data for performance testing."""
    
    @staticmethod
    def generate_concurrent_test_pairs(count: int = 50) -> List[Tuple[str, str]]:
        """Generate query-response pairs for concurrent testing."""
        pairs = []
        
        for i in range(count):
            query = f"Performance test query {i} about metabolomics and clinical applications"
            response = f"Performance test response {i} discussing LC-MS analysis, biomarker discovery, and metabolomic profiling for disease diagnosis and treatment monitoring."
            pairs.append((query, response))
        
        return pairs
    
    @staticmethod
    def generate_variable_length_content(min_words: int = 10, max_words: int = 1000) -> List[Tuple[str, str]]:
        """Generate content with variable lengths."""
        pairs = []
        base_words = ["metabolomics", "biomarker", "LC-MS", "clinical", "analysis", "research", "diagnosis", "treatment", "patient", "study"]
        
        for word_count in [min_words, min_words*2, min_words*5, min_words*10, max_words//2, max_words]:
            query_words = random.choices(base_words, k=min(word_count//4, 50))
            response_words = random.choices(base_words, k=word_count)
            
            query = "What is " + " ".join(query_words) + "?"
            response = " ".join(response_words) + ". This analysis provides insights into metabolic pathways."
            
            pairs.append((query, response))
        
        return pairs
    
    @staticmethod
    def generate_memory_test_data(iterations: int = 1000) -> List[Tuple[str, str]]:
        """Generate data for memory efficiency testing."""
        pairs = []
        
        for i in range(iterations):
            query = f"Memory test {i}: metabolomics research in clinical applications"
            response = f"Memory test response {i}: Clinical metabolomics involves comprehensive analysis of small molecules to identify biomarkers, monitor disease progression, and guide therapeutic interventions using advanced analytical techniques."
            pairs.append((query, response))
        
        return pairs


# =====================================================================
# MOCK COMPONENTS FOR ISOLATED TESTING
# =====================================================================

class MockComponents:
    """Mock components for isolated unit testing."""
    
    @staticmethod
    def create_mock_query_classifier():
        """Create mock query classifier."""
        from unittest.mock import Mock
        
        mock_classifier = Mock()
        mock_classifier.classify_query.return_value = "basic_definition"
        return mock_classifier
    
    @staticmethod
    def create_mock_semantic_engine():
        """Create mock semantic similarity engine.""" 
        from unittest.mock import AsyncMock
        
        mock_engine = AsyncMock()
        mock_engine.calculate_similarity.return_value = 75.0
        return mock_engine
    
    @staticmethod
    def create_mock_domain_validator():
        """Create mock domain expertise validator."""
        from unittest.mock import AsyncMock
        
        mock_validator = AsyncMock()
        mock_validator.validate_domain_expertise.return_value = 80.0
        return mock_validator
    
    @staticmethod
    def create_mock_weighting_manager():
        """Create mock weighting scheme manager."""
        from unittest.mock import Mock
        
        mock_manager = Mock()
        mock_manager.get_weights.return_value = {
            'metabolomics_relevance': 0.3,
            'clinical_applicability': 0.2,
            'query_alignment': 0.2,
            'scientific_rigor': 0.15,
            'biomedical_context_depth': 0.15
        }
        return mock_manager


# =====================================================================
# FIXTURE REGISTRATION
# =====================================================================

@pytest.fixture
def query_fixtures():
    """Provide query fixtures."""
    return QueryFixtures()

@pytest.fixture
def response_fixtures():
    """Provide response fixtures."""
    return ResponseFixtures()

@pytest.fixture
def scenario_fixtures():
    """Provide scenario fixtures."""
    return ScenarioFixtures()

@pytest.fixture
def performance_data_generator():
    """Provide performance data generator."""
    return PerformanceDataGenerator()

@pytest.fixture
def mock_components():
    """Provide mock components."""
    return MockComponents()

@pytest.fixture
def comprehensive_test_queries():
    """Provide comprehensive test queries."""
    return QueryFixtures.get_all_queries()

@pytest.fixture
def comprehensive_test_responses():
    """Provide comprehensive test responses."""
    return ResponseFixtures.get_all_responses()

@pytest.fixture
def standard_test_scenarios():
    """Provide standard test scenarios."""
    return ScenarioFixtures.generate_standard_scenarios()

@pytest.fixture
def performance_test_scenarios():
    """Provide performance test scenarios."""
    return ScenarioFixtures.generate_performance_scenarios(50)

@pytest.fixture
def concurrent_test_pairs():
    """Provide concurrent test pairs."""
    return PerformanceDataGenerator.generate_concurrent_test_pairs()

@pytest.fixture
def variable_length_test_pairs():
    """Provide variable length test pairs."""
    return PerformanceDataGenerator.generate_variable_length_content()


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def save_fixtures_to_json(output_path: Path):
    """Save all fixtures to JSON file for external use."""
    fixtures_data = {
        "queries": {
            "basic_definition": [{"text": q.text, "expected_type": q.expected_type, "complexity": q.complexity} for q in QueryFixtures.BASIC_DEFINITION_QUERIES],
            "clinical_application": [{"text": q.text, "expected_type": q.expected_type, "complexity": q.complexity} for q in QueryFixtures.CLINICAL_APPLICATION_QUERIES],
            "analytical_method": [{"text": q.text, "expected_type": q.expected_type, "complexity": q.complexity} for q in QueryFixtures.ANALYTICAL_METHOD_QUERIES],
            "research_design": [{"text": q.text, "expected_type": q.expected_type, "complexity": q.complexity} for q in QueryFixtures.RESEARCH_DESIGN_QUERIES],
            "disease_specific": [{"text": q.text, "expected_type": q.expected_type, "complexity": q.complexity} for q in QueryFixtures.DISEASE_SPECIFIC_QUERIES],
            "edge_cases": [{"text": q.text, "expected_type": q.expected_type, "complexity": q.complexity} for q in QueryFixtures.EDGE_CASE_QUERIES]
        },
        "responses": {
            "excellent": [{"text": r.text, "quality_level": r.quality_level, "length_category": r.length_category} for r in ResponseFixtures.EXCELLENT_RESPONSES],
            "good": [{"text": r.text, "quality_level": r.quality_level, "length_category": r.length_category} for r in ResponseFixtures.GOOD_RESPONSES],
            "fair": [{"text": r.text, "quality_level": r.quality_level, "length_category": r.length_category} for r in ResponseFixtures.FAIR_RESPONSES],
            "poor": [{"text": r.text, "quality_level": r.quality_level, "length_category": r.length_category} for r in ResponseFixtures.POOR_RESPONSES],
            "edge_cases": [{"text": r.text, "quality_level": r.quality_level, "length_category": r.length_category} for r in ResponseFixtures.EDGE_CASE_RESPONSES]
        },
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(QueryFixtures.get_all_queries()),
            "total_responses": len(ResponseFixtures.get_all_responses()),
            "total_scenarios": len(ScenarioFixtures.generate_standard_scenarios())
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(fixtures_data, f, indent=2)
    
    print(f"Fixtures saved to {output_path}")


if __name__ == "__main__":
    # Generate and save fixtures
    output_path = Path(__file__).parent / "relevance_scorer_fixtures.json"
    save_fixtures_to_json(output_path)