#!/usr/bin/env python3
"""
Comprehensive Sample Biomedical Queries for CMO-LIGHTRAG-012-T01 Testing

This module provides a structured collection of diverse biomedical queries covering
different research categories, complexity levels, and metabolomics scenarios.
Designed for testing the query classification system's ability to handle real-world
biomedical research queries.

Author: Claude Code (Anthropic) & SMO Chatbot Development Team
Created: August 8, 2025
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, NamedTuple, Optional
from dataclasses import dataclass


class ResearchCategory(Enum):
    """Research categories for biomedical query classification."""
    METABOLITE_IDENTIFICATION = "metabolite_identification"
    PATHWAY_ANALYSIS = "pathway_analysis"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    DRUG_DISCOVERY = "drug_discovery"
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"
    DATA_PREPROCESSING = "data_preprocessing"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    LITERATURE_SEARCH = "literature_search"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    DATABASE_INTEGRATION = "database_integration"


class ComplexityLevel(Enum):
    """Complexity levels for biomedical queries."""
    BASIC = "basic"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class QueryTestCase:
    """
    Test case structure for biomedical queries.
    
    Attributes:
        query: The biomedical query text
        primary_category: Primary research category
        secondary_categories: Alternative valid categories (for ambiguous queries)
        complexity: Query complexity level
        expected_confidence: Expected confidence score range (min, max)
        keywords: Key metabolomics terms expected in the query
        description: Brief description of what this query tests
        edge_case: Whether this is an edge case or ambiguous query
    """
    query: str
    primary_category: ResearchCategory
    secondary_categories: Optional[List[ResearchCategory]] = None
    complexity: ComplexityLevel = ComplexityLevel.BASIC
    expected_confidence: tuple = (0.7, 1.0)
    keywords: Optional[List[str]] = None
    description: str = ""
    edge_case: bool = False


# =============================================================================
# METABOLITE IDENTIFICATION QUERIES
# =============================================================================

METABOLITE_IDENTIFICATION_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="What is the molecular formula of glucose?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 1.0),
        keywords=["molecular formula", "glucose"],
        description="Simple metabolite identification - basic structural information"
    ),
    
    QueryTestCase(
        query="How can I identify an unknown metabolite using mass spectrometry?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["identify", "unknown metabolite", "mass spectrometry"],
        description="Basic identification methodology question"
    ),
    
    QueryTestCase(
        query="What database should I use to identify metabolites?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        secondary_categories=[ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.7, 0.9),
        keywords=["database", "identify metabolites"],
        description="Database selection for metabolite identification",
        edge_case=True
    ),
    
    # Medium Level
    QueryTestCase(
        query="I have a peak at m/z 180.0634 with retention time 5.2 minutes. What metabolite could this be?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.85, 1.0),
        keywords=["peak", "m/z", "retention time", "metabolite"],
        description="Specific analytical data for identification"
    ),
    
    QueryTestCase(
        query="How do I use MS/MS fragmentation patterns to identify lipid metabolites?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["MS/MS", "fragmentation patterns", "identify", "lipid metabolites"],
        description="Advanced identification technique for specific metabolite class"
    ),
    
    QueryTestCase(
        query="What are the characteristic NMR signals for amino acid metabolites?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["NMR signals", "amino acid metabolites"],
        description="Spectroscopic identification of metabolite class"
    ),
    
    # Complex Level
    QueryTestCase(
        query="I need to identify unknown metabolites in human plasma using UPLC-QTOF-MS with accurate mass and isotope patterns. What workflow should I follow?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        secondary_categories=[ResearchCategory.DATA_PREPROCESSING],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.8, 0.95),
        keywords=["unknown metabolites", "plasma", "UPLC-QTOF-MS", "accurate mass", "isotope patterns", "workflow"],
        description="Comprehensive identification workflow with advanced instrumentation"
    ),
    
    QueryTestCase(
        query="How can I distinguish between structural isomers of glucose metabolites using ion mobility spectrometry coupled to mass spectrometry?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.8, 1.0),
        keywords=["structural isomers", "glucose metabolites", "ion mobility spectrometry", "mass spectrometry"],
        description="Advanced technique for isomer differentiation"
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm analyzing xenobiotic biotransformation products in liver microsomes using suspect screening with molecular networking. How do I optimize the identification workflow for phase I and II metabolites with low MS/MS spectral similarity to parent compounds?",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        secondary_categories=[ResearchCategory.DATA_PREPROCESSING, ResearchCategory.DRUG_DISCOVERY],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.7, 0.9),
        keywords=["xenobiotic", "biotransformation", "liver microsomes", "suspect screening", "molecular networking", "phase I", "phase II", "metabolites", "MS/MS"],
        description="Expert-level xenobiotic metabolism identification",
        edge_case=True
    ),
]

# =============================================================================
# PATHWAY ANALYSIS QUERIES
# =============================================================================

PATHWAY_ANALYSIS_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="What is the glycolysis pathway?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 1.0),
        keywords=["glycolysis pathway"],
        description="Basic pathway definition question"
    ),
    
    QueryTestCase(
        query="How does glucose metabolism connect to other metabolic pathways?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 1.0),
        keywords=["glucose metabolism", "metabolic pathways"],
        description="Basic pathway interconnections"
    ),
    
    QueryTestCase(
        query="What metabolites are involved in the TCA cycle?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        secondary_categories=[ResearchCategory.METABOLITE_IDENTIFICATION],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.7, 0.9),
        keywords=["metabolites", "TCA cycle"],
        description="Pathway metabolite composition",
        edge_case=True
    ),
    
    # Medium Level
    QueryTestCase(
        query="How do I perform pathway enrichment analysis on my metabolomics dataset?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["pathway enrichment analysis", "metabolomics dataset"],
        description="Statistical pathway analysis methodology",
        edge_case=True
    ),
    
    QueryTestCase(
        query="Which KEGG pathways are significantly altered in diabetes patients based on plasma metabolite levels?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        secondary_categories=[ResearchCategory.CLINICAL_DIAGNOSIS, ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.7, 0.9),
        keywords=["KEGG pathways", "diabetes", "plasma metabolite levels"],
        description="Disease-specific pathway alterations",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How do I use Reactome to understand fatty acid oxidation pathways?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        secondary_categories=[ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["Reactome", "fatty acid oxidation pathways"],
        description="Database-specific pathway analysis",
        edge_case=True
    ),
    
    # Complex Level
    QueryTestCase(
        query="I need to reconstruct metabolic networks from untargeted metabolomics data and identify key regulatory nodes. What computational approaches should I use?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.DATA_PREPROCESSING],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["metabolic networks", "untargeted metabolomics", "regulatory nodes", "computational approaches"],
        description="Network reconstruction and analysis"
    ),
    
    QueryTestCase(
        query="How can I integrate transcriptomics and metabolomics data to identify dysregulated pathways in cancer metabolism?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        secondary_categories=[ResearchCategory.BIOMARKER_DISCOVERY, ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["transcriptomics", "metabolomics", "dysregulated pathways", "cancer metabolism"],
        description="Multi-omics pathway integration"
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm studying flux distributions in central carbon metabolism using dynamic 13C-labeling experiments. How do I model the kinetic parameters and identify rate-limiting steps while accounting for compartmentalization effects in different cell types?",
        primary_category=ResearchCategory.PATHWAY_ANALYSIS,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.7, 0.9),
        keywords=["flux distributions", "central carbon metabolism", "13C-labeling", "kinetic parameters", "rate-limiting steps", "compartmentalization"],
        description="Advanced metabolic flux analysis with isotope labeling"
    ),
]

# =============================================================================
# BIOMARKER DISCOVERY QUERIES
# =============================================================================

BIOMARKER_DISCOVERY_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="What are biomarkers in clinical metabolomics?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["biomarkers", "clinical metabolomics"],
        description="Basic biomarker concept definition"
    ),
    
    QueryTestCase(
        query="How do I find metabolite biomarkers for disease diagnosis?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        secondary_categories=[ResearchCategory.CLINICAL_DIAGNOSIS],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["metabolite biomarkers", "disease diagnosis"],
        description="Basic biomarker discovery methodology",
        edge_case=True
    ),
    
    QueryTestCase(
        query="What metabolites are good biomarkers for cardiovascular disease?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        secondary_categories=[ResearchCategory.CLINICAL_DIAGNOSIS],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["metabolites", "biomarkers", "cardiovascular disease"],
        description="Disease-specific biomarker identification",
        edge_case=True
    ),
    
    # Medium Level
    QueryTestCase(
        query="How do I validate potential biomarkers using ROC curve analysis and cross-validation?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.7, 0.9),
        keywords=["validate", "biomarkers", "ROC curve", "cross-validation"],
        description="Biomarker validation methodology",
        edge_case=True
    ),
    
    QueryTestCase(
        query="I need to identify prognostic biomarkers for cancer progression using longitudinal metabolomics data. What statistical approaches should I use?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["prognostic biomarkers", "cancer progression", "longitudinal", "metabolomics", "statistical approaches"],
        description="Prognostic biomarker discovery with temporal data"
    ),
    
    QueryTestCase(
        query="What is the difference between diagnostic and prognostic biomarkers in metabolomics?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["diagnostic", "prognostic", "biomarkers", "metabolomics"],
        description="Biomarker type differentiation"
    ),
    
    # Complex Level
    QueryTestCase(
        query="I'm developing a multi-metabolite signature for early detection of Alzheimer's disease. How do I handle class imbalance, select optimal feature combinations, and ensure clinical translatability?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.CLINICAL_DIAGNOSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["multi-metabolite signature", "Alzheimer's", "early detection", "class imbalance", "feature combinations", "clinical translatability"],
        description="Comprehensive biomarker signature development"
    ),
    
    QueryTestCase(
        query="How can I integrate metabolomics with genetic markers to identify personalized biomarkers for drug response prediction?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        secondary_categories=[ResearchCategory.DRUG_DISCOVERY, ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["metabolomics", "genetic markers", "personalized biomarkers", "drug response prediction"],
        description="Multi-omics personalized biomarker discovery"
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm developing a dynamic biomarker model that incorporates circadian rhythms, dietary patterns, and metabolite ratios for real-time health monitoring. How do I model temporal variability and establish population-specific reference ranges while maintaining diagnostic performance across diverse demographics?",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.CLINICAL_DIAGNOSIS],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.6, 0.8),
        keywords=["dynamic biomarker", "circadian rhythms", "dietary patterns", "metabolite ratios", "temporal variability", "reference ranges", "demographics"],
        description="Advanced dynamic biomarker modeling with multiple variables",
        edge_case=True
    ),
]

# =============================================================================
# DRUG DISCOVERY QUERIES
# =============================================================================

DRUG_DISCOVERY_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="How can metabolomics be used in drug discovery?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["metabolomics", "drug discovery"],
        description="Basic application of metabolomics in drug development"
    ),
    
    QueryTestCase(
        query="What are the metabolic effects of aspirin?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        secondary_categories=[ResearchCategory.PATHWAY_ANALYSIS],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["metabolic effects", "aspirin"],
        description="Drug metabolic impact assessment",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How do I study drug metabolism using metabolomics?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 1.0),
        keywords=["drug metabolism", "metabolomics"],
        description="Basic drug metabolism study approach"
    ),
    
    # Medium Level
    QueryTestCase(
        query="I need to identify drug targets for diabetes treatment using metabolic pathway analysis. What approach should I take?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        secondary_categories=[ResearchCategory.PATHWAY_ANALYSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.7, 0.9),
        keywords=["drug targets", "diabetes treatment", "metabolic pathway analysis"],
        description="Target identification through pathway analysis",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How can I use metabolomics to assess drug toxicity and side effects?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        secondary_categories=[ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["metabolomics", "drug toxicity", "side effects"],
        description="Toxicity assessment using metabolomics",
        edge_case=True
    ),
    
    QueryTestCase(
        query="What metabolic biomarkers can predict drug efficacy in cancer treatment?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        secondary_categories=[ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.7, 0.9),
        keywords=["metabolic biomarkers", "drug efficacy", "cancer treatment"],
        description="Efficacy biomarkers for drug response",
        edge_case=True
    ),
    
    # Complex Level
    QueryTestCase(
        query="I'm developing a pharmacometabolomics approach to optimize dosing regimens for personalized medicine. How do I integrate PK/PD modeling with metabolic profiling?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["pharmacometabolomics", "dosing regimens", "personalized medicine", "PK/PD modeling", "metabolic profiling"],
        description="Advanced pharmacometabolomics for personalized dosing"
    ),
    
    QueryTestCase(
        query="How can I use systems pharmacology approaches to predict drug-drug interactions based on metabolic pathway perturbations?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        secondary_categories=[ResearchCategory.PATHWAY_ANALYSIS, ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["systems pharmacology", "drug-drug interactions", "metabolic pathway perturbations"],
        description="Systems-level drug interaction prediction"
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm developing a machine learning model to predict novel drug repurposing opportunities by analyzing metabolic signatures of disease states and known drug effects. How do I handle multi-target polypharmacology while accounting for metabolic network robustness and compensatory mechanisms?",
        primary_category=ResearchCategory.DRUG_DISCOVERY,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.PATHWAY_ANALYSIS],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.6, 0.8),
        keywords=["machine learning", "drug repurposing", "metabolic signatures", "multi-target", "polypharmacology", "metabolic network", "compensatory mechanisms"],
        description="AI-driven drug repurposing with complex metabolic modeling",
        edge_case=True
    ),
]

# =============================================================================
# CLINICAL DIAGNOSIS QUERIES
# =============================================================================

CLINICAL_DIAGNOSIS_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="How is metabolomics used in clinical diagnosis?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["metabolomics", "clinical diagnosis"],
        description="Basic clinical application overview"
    ),
    
    QueryTestCase(
        query="What metabolites are measured in blood for diabetes diagnosis?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["metabolites", "blood", "diabetes diagnosis"],
        description="Disease-specific clinical metabolites",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How do I collect and process urine samples for metabolomic analysis?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.DATA_PREPROCESSING],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.7, 0.9),
        keywords=["urine samples", "metabolomic analysis"],
        description="Sample collection and processing",
        edge_case=True
    ),
    
    # Medium Level
    QueryTestCase(
        query="I need to develop a clinical metabolomics assay for kidney disease diagnosis using serum samples. What analytical considerations should I address?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.BIOMARKER_DISCOVERY, ResearchCategory.DATA_PREPROCESSING],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.7, 0.9),
        keywords=["clinical metabolomics assay", "kidney disease", "serum samples", "analytical considerations"],
        description="Clinical assay development methodology"
    ),
    
    QueryTestCase(
        query="How do I validate metabolomics-based diagnostic tests for regulatory approval?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["validate", "metabolomics-based diagnostic tests", "regulatory approval"],
        description="Regulatory validation requirements",
        edge_case=True
    ),
    
    QueryTestCase(
        query="What quality control measures are needed for clinical metabolomics laboratories?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.DATA_PREPROCESSING],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["quality control", "clinical metabolomics laboratories"],
        description="Laboratory quality assurance",
        edge_case=True
    ),
    
    # Complex Level
    QueryTestCase(
        query="I'm implementing a point-of-care metabolomics device for emergency department triage. How do I ensure analytical performance meets clinical requirements while maintaining cost-effectiveness?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.DATA_PREPROCESSING, ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["point-of-care", "metabolomics device", "emergency department", "triage", "analytical performance", "clinical requirements", "cost-effectiveness"],
        description="Point-of-care diagnostic device development"
    ),
    
    QueryTestCase(
        query="How can I integrate metabolomics data with electronic health records to improve diagnostic accuracy while ensuring patient privacy?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.DATABASE_INTEGRATION, ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["metabolomics data", "electronic health records", "diagnostic accuracy", "patient privacy"],
        description="Clinical data integration with privacy considerations"
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm developing a real-time metabolomics monitoring system for ICU patients that can predict organ failure 24-48 hours before clinical manifestation. How do I handle continuous data streams, patient-specific baselines, and integrate with existing clinical decision support systems while maintaining regulatory compliance?",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.6, 0.8),
        keywords=["real-time", "metabolomics monitoring", "ICU", "organ failure", "continuous data", "clinical decision support", "regulatory compliance"],
        description="Advanced real-time clinical monitoring system",
        edge_case=True
    ),
]

# =============================================================================
# DATA PREPROCESSING QUERIES
# =============================================================================

DATA_PREPROCESSING_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="How do I normalize metabolomics data?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["normalize", "metabolomics data"],
        description="Basic data normalization question"
    ),
    
    QueryTestCase(
        query="What is quality control in metabolomics experiments?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["quality control", "metabolomics experiments"],
        description="Quality control concept explanation"
    ),
    
    QueryTestCase(
        query="How do I handle missing values in my metabolomics dataset?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["missing values", "metabolomics dataset"],
        description="Missing data handling methodology",
        edge_case=True
    ),
    
    # Medium Level
    QueryTestCase(
        query="I need to correct for batch effects in my LC-MS metabolomics data. What methods should I use?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["batch effects", "LC-MS", "metabolomics data"],
        description="Batch effect correction methodology",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How do I detect and remove outliers from metabolomics data without losing biological variation?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["outliers", "metabolomics data", "biological variation"],
        description="Outlier detection with biological preservation",
        edge_case=True
    ),
    
    QueryTestCase(
        query="What peak alignment methods work best for GC-MS metabolomics data?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["peak alignment", "GC-MS", "metabolomics data"],
        description="Platform-specific peak alignment"
    ),
    
    # Complex Level
    QueryTestCase(
        query="I'm processing untargeted metabolomics data with significant signal drift and matrix effects. How do I implement a comprehensive quality control strategy that includes QC samples, internal standards, and statistical corrections?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["untargeted metabolomics", "signal drift", "matrix effects", "quality control", "QC samples", "internal standards", "statistical corrections"],
        description="Comprehensive quality control strategy"
    ),
    
    QueryTestCase(
        query="How can I integrate data from multiple analytical platforms (LC-MS, GC-MS, NMR) while accounting for platform-specific biases and scaling differences?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["multiple platforms", "LC-MS", "GC-MS", "NMR", "platform-specific biases", "scaling differences"],
        description="Multi-platform data integration"
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm developing an automated preprocessing pipeline for high-throughput metabolomics that must handle diverse sample types, varying acquisition parameters, and dynamic quality thresholds. How do I implement adaptive algorithms that can self-optimize while maintaining reproducibility across different studies and laboratories?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.6, 0.8),
        keywords=["automated preprocessing", "high-throughput", "diverse samples", "acquisition parameters", "dynamic quality", "adaptive algorithms", "reproducibility"],
        description="Advanced automated preprocessing with adaptive algorithms",
        edge_case=True
    ),
]

# =============================================================================
# STATISTICAL ANALYSIS QUERIES
# =============================================================================

STATISTICAL_ANALYSIS_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="How do I perform PCA analysis on metabolomics data?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["PCA analysis", "metabolomics data"],
        description="Basic multivariate analysis technique"
    ),
    
    QueryTestCase(
        query="What statistical test should I use to compare metabolite levels between two groups?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 1.0),
        keywords=["statistical test", "metabolite levels", "two groups"],
        description="Basic group comparison statistics"
    ),
    
    QueryTestCase(
        query="How do I interpret p-values in metabolomics studies?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["p-values", "metabolomics studies"],
        description="Statistical significance interpretation"
    ),
    
    # Medium Level
    QueryTestCase(
        query="I need to correct for multiple testing in my metabolomics analysis. Should I use Bonferroni or FDR correction?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["multiple testing", "Bonferroni", "FDR correction"],
        description="Multiple testing correction methods"
    ),
    
    QueryTestCase(
        query="How do I perform OPLS-DA analysis and validate the model using permutation testing?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["OPLS-DA", "validate", "permutation testing"],
        description="Advanced supervised multivariate analysis"
    ),
    
    QueryTestCase(
        query="What machine learning algorithms work best for metabolomics classification problems?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        secondary_categories=[ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["machine learning", "metabolomics", "classification"],
        description="ML algorithms for metabolomics classification",
        edge_case=True
    ),
    
    # Complex Level
    QueryTestCase(
        query="I'm analyzing longitudinal metabolomics data with repeated measurements and missing time points. How do I model temporal trends while accounting for individual variability?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["longitudinal", "metabolomics", "repeated measurements", "missing time points", "temporal trends", "individual variability"],
        description="Complex longitudinal data modeling"
    ),
    
    QueryTestCase(
        query="How can I use network analysis to identify metabolite clusters and pathway modules from correlation matrices?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        secondary_categories=[ResearchCategory.PATHWAY_ANALYSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["network analysis", "metabolite clusters", "pathway modules", "correlation matrices"],
        description="Network-based statistical analysis",
        edge_case=True
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm developing a Bayesian hierarchical model to integrate multi-omics data (metabolomics, proteomics, transcriptomics) while accounting for measurement uncertainty, batch effects, and population stratification. How do I specify appropriate priors and ensure model convergence in high-dimensional settings?",
        primary_category=ResearchCategory.STATISTICAL_ANALYSIS,
        secondary_categories=[ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.6, 0.8),
        keywords=["Bayesian hierarchical", "multi-omics", "metabolomics", "proteomics", "transcriptomics", "measurement uncertainty", "batch effects", "population stratification", "priors", "model convergence"],
        description="Advanced Bayesian multi-omics integration",
        edge_case=True
    ),
]

# =============================================================================
# LITERATURE SEARCH QUERIES
# =============================================================================

LITERATURE_SEARCH_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="Find recent papers on metabolomics in cancer research.",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["papers", "metabolomics", "cancer research"],
        description="Basic literature search request"
    ),
    
    QueryTestCase(
        query="What are the latest publications on diabetes metabolomics?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["publications", "diabetes metabolomics"],
        description="Disease-specific literature search"
    ),
    
    QueryTestCase(
        query="I need references on mass spectrometry methods for metabolite identification.",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        secondary_categories=[ResearchCategory.METABOLITE_IDENTIFICATION],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["references", "mass spectrometry", "metabolite identification"],
        description="Method-specific literature search",
        edge_case=True
    ),
    
    # Medium Level
    QueryTestCase(
        query="Find systematic reviews and meta-analyses on metabolomics biomarkers for cardiovascular disease published in the last 3 years.",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        secondary_categories=[ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["systematic reviews", "meta-analyses", "metabolomics biomarkers", "cardiovascular disease"],
        description="Specific publication type and timeframe search",
        edge_case=True
    ),
    
    QueryTestCase(
        query="I need to conduct a comprehensive literature review on NMR-based metabolomics applications in plant biology. What search strategy should I use?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["literature review", "NMR-based metabolomics", "plant biology", "search strategy"],
        description="Structured literature review methodology"
    ),
    
    QueryTestCase(
        query="How do I find clinical trials using metabolomics as primary or secondary endpoints?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        secondary_categories=[ResearchCategory.CLINICAL_DIAGNOSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["clinical trials", "metabolomics", "endpoints"],
        description="Clinical trial literature search",
        edge_case=True
    ),
    
    # Complex Level
    QueryTestCase(
        query="I'm preparing a grant proposal and need to identify research gaps in gut microbiome-metabolome interactions. How do I systematically map the current knowledge landscape?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        secondary_categories=[ResearchCategory.KNOWLEDGE_EXTRACTION],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["grant proposal", "research gaps", "gut microbiome", "metabolome interactions", "knowledge landscape"],
        description="Gap analysis and knowledge mapping",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How can I use bibliometric analysis and citation networks to identify emerging trends in metabolomics methodology development?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        secondary_categories=[ResearchCategory.KNOWLEDGE_EXTRACTION],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["bibliometric analysis", "citation networks", "emerging trends", "metabolomics methodology"],
        description="Advanced bibliometric trend analysis",
        edge_case=True
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm developing an AI-powered literature monitoring system that can automatically identify breakthrough discoveries in metabolomics, assess their clinical translatability, and predict future research directions. How do I implement semantic analysis and knowledge graph construction from biomedical literature?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        secondary_categories=[ResearchCategory.KNOWLEDGE_EXTRACTION, ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.6, 0.8),
        keywords=["AI-powered", "literature monitoring", "breakthrough discoveries", "clinical translatability", "semantic analysis", "knowledge graph"],
        description="AI-driven literature analysis and prediction system",
        edge_case=True
    ),
]

# =============================================================================
# KNOWLEDGE EXTRACTION QUERIES
# =============================================================================

KNOWLEDGE_EXTRACTION_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="How do I extract metabolite information from scientific papers?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        secondary_categories=[ResearchCategory.LITERATURE_SEARCH],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["extract", "metabolite information", "scientific papers"],
        description="Basic information extraction from literature",
        edge_case=True
    ),
    
    QueryTestCase(
        query="What databases contain metabolite pathway information?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        secondary_categories=[ResearchCategory.DATABASE_INTEGRATION, ResearchCategory.PATHWAY_ANALYSIS],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.7, 0.9),
        keywords=["databases", "metabolite pathway information"],
        description="Database identification for pathway knowledge",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How can I mine text for metabolomics experimental conditions?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 1.0),
        keywords=["mine text", "metabolomics", "experimental conditions"],
        description="Text mining for experimental metadata"
    ),
    
    # Medium Level
    QueryTestCase(
        query="I need to automatically extract metabolite-disease associations from PubMed abstracts. What natural language processing approaches should I use?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        secondary_categories=[ResearchCategory.LITERATURE_SEARCH],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.7, 0.9),
        keywords=["extract", "metabolite-disease associations", "PubMed abstracts", "natural language processing"],
        description="Automated biomedical relation extraction",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How can I create a knowledge graph linking metabolites, pathways, and diseases from multiple data sources?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        secondary_categories=[ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.7, 0.9),
        keywords=["knowledge graph", "metabolites", "pathways", "diseases", "data sources"],
        description="Knowledge graph construction from multiple sources",
        edge_case=True
    ),
    
    QueryTestCase(
        query="What ontologies are available for standardizing metabolomics data annotation?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        secondary_categories=[ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["ontologies", "metabolomics data annotation"],
        description="Ontology resources for data standardization",
        edge_case=True
    ),
    
    # Complex Level
    QueryTestCase(
        query="I'm developing a system to automatically extract experimental protocols from metabolomics papers and convert them into machine-readable formats. How do I handle protocol variability and standardization?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        secondary_categories=[ResearchCategory.LITERATURE_SEARCH, ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["extract", "experimental protocols", "metabolomics papers", "machine-readable", "protocol variability", "standardization"],
        description="Automated protocol extraction and standardization"
    ),
    
    QueryTestCase(
        query="How can I use deep learning to identify novel metabolite-protein interactions from heterogeneous biomedical datasets?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["deep learning", "metabolite-protein interactions", "heterogeneous", "biomedical datasets"],
        description="AI-driven interaction discovery from heterogeneous data"
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm building a dynamic knowledge base that continuously learns from new publications, experimental data, and clinical records to predict metabolic phenotypes. How do I implement incremental learning while maintaining knowledge consistency and handling conflicting information from different sources?",
        primary_category=ResearchCategory.KNOWLEDGE_EXTRACTION,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.DATABASE_INTEGRATION],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.6, 0.8),
        keywords=["dynamic knowledge base", "continuous learning", "publications", "experimental data", "clinical records", "metabolic phenotypes", "incremental learning", "knowledge consistency"],
        description="Advanced dynamic knowledge base with continuous learning",
        edge_case=True
    ),
]

# =============================================================================
# DATABASE INTEGRATION QUERIES
# =============================================================================

DATABASE_INTEGRATION_QUERIES = [
    # Basic Level
    QueryTestCase(
        query="What metabolomics databases should I use for my research?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.9, 1.0),
        keywords=["metabolomics databases"],
        description="Basic database selection guidance"
    ),
    
    QueryTestCase(
        query="How do I access HMDB for metabolite information?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        secondary_categories=[ResearchCategory.METABOLITE_IDENTIFICATION],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["HMDB", "metabolite information"],
        description="Specific database access question",
        edge_case=True
    ),
    
    QueryTestCase(
        query="What is the difference between KEGG and Reactome databases?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        secondary_categories=[ResearchCategory.PATHWAY_ANALYSIS],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.8, 0.95),
        keywords=["KEGG", "Reactome", "databases"],
        description="Database comparison and selection",
        edge_case=True
    ),
    
    # Medium Level
    QueryTestCase(
        query="I need to programmatically query multiple metabolomics databases (HMDB, METLIN, ChEBI) for compound information. What APIs are available?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["programmatically query", "HMDB", "METLIN", "ChEBI", "APIs"],
        description="Programmatic database access"
    ),
    
    QueryTestCase(
        query="How can I integrate metabolomics data with genomics databases like Ensembl and NCBI?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 1.0),
        keywords=["integrate", "metabolomics data", "genomics databases", "Ensembl", "NCBI"],
        description="Multi-omics database integration"
    ),
    
    QueryTestCase(
        query="What standardized formats exist for sharing metabolomics data between different databases?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        secondary_categories=[ResearchCategory.KNOWLEDGE_EXTRACTION],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.8, 0.95),
        keywords=["standardized formats", "sharing", "metabolomics data", "databases"],
        description="Data format standardization",
        edge_case=True
    ),
    
    # Complex Level
    QueryTestCase(
        query="I'm developing a federated query system that can search across HMDB, METLIN, LipidMaps, and custom databases simultaneously. How do I handle schema differences and result harmonization?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        secondary_categories=[ResearchCategory.KNOWLEDGE_EXTRACTION],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["federated query", "HMDB", "METLIN", "LipidMaps", "schema differences", "result harmonization"],
        description="Federated database query system development"
    ),
    
    QueryTestCase(
        query="How can I create automated pipelines to synchronize local metabolomics databases with public repositories while maintaining data provenance?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        secondary_categories=[ResearchCategory.DATA_PREPROCESSING],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.7, 0.9),
        keywords=["automated pipelines", "synchronize", "local databases", "public repositories", "data provenance"],
        description="Database synchronization with provenance tracking",
        edge_case=True
    ),
    
    # Expert Level
    QueryTestCase(
        query="I'm building a semantic web platform that automatically discovers and integrates new metabolomics resources, maps between different identifier systems, and maintains temporal versions of evolving databases. How do I implement automated ontology alignment and handle version conflicts?",
        primary_category=ResearchCategory.DATABASE_INTEGRATION,
        secondary_categories=[ResearchCategory.KNOWLEDGE_EXTRACTION],
        complexity=ComplexityLevel.EXPERT,
        expected_confidence=(0.6, 0.8),
        keywords=["semantic web", "automatically discovers", "integrates", "identifier systems", "temporal versions", "ontology alignment", "version conflicts"],
        description="Advanced semantic web platform for database integration",
        edge_case=True
    ),
]

# =============================================================================
# EDGE CASE AND AMBIGUOUS QUERIES
# =============================================================================

EDGE_CASE_QUERIES = [
    # Multi-category ambiguous queries
    QueryTestCase(
        query="I need to analyze metabolomics data from cancer patients to find biomarkers and understand pathway changes for drug development.",
        primary_category=ResearchCategory.BIOMARKER_DISCOVERY,
        secondary_categories=[ResearchCategory.PATHWAY_ANALYSIS, ResearchCategory.DRUG_DISCOVERY, ResearchCategory.CLINICAL_DIAGNOSIS],
        complexity=ComplexityLevel.COMPLEX,
        expected_confidence=(0.6, 0.8),
        keywords=["metabolomics data", "cancer", "biomarkers", "pathway changes", "drug development"],
        description="Multi-objective research spanning multiple categories",
        edge_case=True
    ),
    
    QueryTestCase(
        query="How do I normalize and statistically analyze metabolomics data to identify disease biomarkers?",
        primary_category=ResearchCategory.DATA_PREPROCESSING,
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.6, 0.8),
        keywords=["normalize", "statistically analyze", "metabolomics data", "disease biomarkers"],
        description="Workflow spanning preprocessing, statistics, and biomarker discovery",
        edge_case=True
    ),
    
    # General/vague queries
    QueryTestCase(
        query="What is metabolomics?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,
        secondary_categories=[ResearchCategory.KNOWLEDGE_EXTRACTION],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.6, 0.8),
        keywords=["metabolomics"],
        description="Very general definitional query",
        edge_case=True
    ),
    
    QueryTestCase(
        query="Tell me about mass spectrometry in metabolomics.",
        primary_category=ResearchCategory.METABOLITE_IDENTIFICATION,
        secondary_categories=[ResearchCategory.DATA_PREPROCESSING, ResearchCategory.LITERATURE_SEARCH],
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.6, 0.8),
        keywords=["mass spectrometry", "metabolomics"],
        description="General technique overview spanning multiple categories",
        edge_case=True
    ),
    
    # Borderline queries
    QueryTestCase(
        query="I want to study the metabolic effects of exercise interventions in diabetic patients using NMR spectroscopy.",
        primary_category=ResearchCategory.CLINICAL_DIAGNOSIS,
        secondary_categories=[ResearchCategory.PATHWAY_ANALYSIS, ResearchCategory.BIOMARKER_DISCOVERY],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.6, 0.8),
        keywords=["metabolic effects", "exercise interventions", "diabetic patients", "NMR spectroscopy"],
        description="Clinical intervention study with multiple analytical aspects",
        edge_case=True
    ),
    
    # Non-metabolomics queries that might be submitted by mistake
    QueryTestCase(
        query="How do I perform RNA sequencing analysis?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,  # Fallback category
        secondary_categories=[ResearchCategory.STATISTICAL_ANALYSIS],
        complexity=ComplexityLevel.MEDIUM,
        expected_confidence=(0.3, 0.6),
        keywords=["RNA sequencing"],
        description="Non-metabolomics query that should have low confidence",
        edge_case=True
    ),
    
    QueryTestCase(
        query="What is the weather forecast for tomorrow?",
        primary_category=ResearchCategory.LITERATURE_SEARCH,  # Fallback category
        complexity=ComplexityLevel.BASIC,
        expected_confidence=(0.1, 0.3),
        keywords=["weather forecast"],
        description="Completely unrelated query for testing system robustness",
        edge_case=True
    ),
]

# =============================================================================
# COMPREHENSIVE TEST SUITE COMPILATION
# =============================================================================

def get_all_test_queries() -> Dict[str, List[QueryTestCase]]:
    """
    Get all test queries organized by research category.
    
    Returns:
        Dictionary mapping category names to lists of QueryTestCase objects
    """
    return {
        "metabolite_identification": METABOLITE_IDENTIFICATION_QUERIES,
        "pathway_analysis": PATHWAY_ANALYSIS_QUERIES,
        "biomarker_discovery": BIOMARKER_DISCOVERY_QUERIES,
        "drug_discovery": DRUG_DISCOVERY_QUERIES,
        "clinical_diagnosis": CLINICAL_DIAGNOSIS_QUERIES,
        "data_preprocessing": DATA_PREPROCESSING_QUERIES,
        "statistical_analysis": STATISTICAL_ANALYSIS_QUERIES,
        "literature_search": LITERATURE_SEARCH_QUERIES,
        "knowledge_extraction": KNOWLEDGE_EXTRACTION_QUERIES,
        "database_integration": DATABASE_INTEGRATION_QUERIES,
        "edge_cases": EDGE_CASE_QUERIES,
    }


def get_queries_by_complexity(complexity: ComplexityLevel) -> List[QueryTestCase]:
    """
    Get all queries of a specific complexity level.
    
    Args:
        complexity: The complexity level to filter by
        
    Returns:
        List of QueryTestCase objects matching the complexity level
    """
    all_queries = get_all_test_queries()
    matching_queries = []
    
    for category_queries in all_queries.values():
        for query in category_queries:
            if query.complexity == complexity:
                matching_queries.append(query)
    
    return matching_queries


def get_edge_case_queries() -> List[QueryTestCase]:
    """
    Get all queries marked as edge cases.
    
    Returns:
        List of QueryTestCase objects that are edge cases
    """
    all_queries = get_all_test_queries()
    edge_cases = []
    
    for category_queries in all_queries.values():
        for query in category_queries:
            if query.edge_case:
                edge_cases.append(query)
    
    return edge_cases


def get_query_statistics() -> Dict[str, int]:
    """
    Get statistics about the query test suite.
    
    Returns:
        Dictionary with counts for different query characteristics
    """
    all_queries = get_all_test_queries()
    
    total_queries = sum(len(queries) for queries in all_queries.values())
    edge_cases = len(get_edge_case_queries())
    
    complexity_counts = {
        "basic": len(get_queries_by_complexity(ComplexityLevel.BASIC)),
        "medium": len(get_queries_by_complexity(ComplexityLevel.MEDIUM)),
        "complex": len(get_queries_by_complexity(ComplexityLevel.COMPLEX)),
        "expert": len(get_queries_by_complexity(ComplexityLevel.EXPERT)),
    }
    
    category_counts = {
        category: len(queries) for category, queries in all_queries.items()
    }
    
    return {
        "total_queries": total_queries,
        "edge_cases": edge_cases,
        "complexity_distribution": complexity_counts,
        "category_distribution": category_counts,
    }


if __name__ == "__main__":
    """
    Demonstrate the test fixture capabilities.
    """
    print("=== Biomedical Query Test Fixtures ===\n")
    
    # Print statistics
    stats = get_query_statistics()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Edge cases: {stats['edge_cases']}")
    print("\nComplexity distribution:")
    for complexity, count in stats['complexity_distribution'].items():
        print(f"  {complexity}: {count}")
    
    print("\nCategory distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"  {category}: {count}")
    
    # Show sample queries from each category
    print("\n=== Sample Queries by Category ===\n")
    all_queries = get_all_test_queries()
    
    for category, queries in all_queries.items():
        print(f"\n{category.upper()}:")
        if queries:
            sample_query = queries[0]
            print(f"  Query: {sample_query.query}")
            print(f"  Complexity: {sample_query.complexity.value}")
            print(f"  Expected confidence: {sample_query.expected_confidence}")
            if sample_query.keywords:
                print(f"  Keywords: {sample_query.keywords[:3]}...")