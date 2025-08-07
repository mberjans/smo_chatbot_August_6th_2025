#!/usr/bin/env python3
"""
Query Test Fixtures for Clinical Metabolomics Oracle.

This module provides comprehensive test fixtures for generating and testing
clinical metabolomics queries across different research categories and use cases.
It includes realistic query scenarios, expected responses, and validation patterns
specifically tailored for biomedical research.

Components:
- ClinicalQueryGenerator: Creates realistic clinical metabolomics queries
- ResearchCategoryQueryBuilder: Builds queries for specific research categories
- QueryResponseValidator: Validates query responses for biomedical accuracy
- QueryComplexityScaler: Generates queries of varying complexity levels
- MetabolomicsQueryPatterns: Provides domain-specific query templates
- ClinicalScenarioBuilder: Creates comprehensive clinical query scenarios

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import random
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import re

# Import research categories from the main module
from lightrag_integration.research_categorizer import ResearchCategory


@dataclass
class QueryTestCase:
    """Represents a complete query test case with expected outcomes."""
    query_id: str
    query_text: str
    research_category: ResearchCategory
    complexity_level: str  # 'simple', 'medium', 'complex', 'expert'
    query_type: str  # 'question', 'search', 'analysis', 'comparison', 'explanation'
    domain_area: str  # 'clinical', 'research', 'diagnostic', 'therapeutic'
    expected_entities: List[str] = field(default_factory=list)
    expected_relationships: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    response_template: str = ""
    confidence_threshold: float = 0.7
    processing_timeout: float = 10.0
    cost_estimate: float = 0.05
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def test_summary(self) -> str:
        """Generate test case summary."""
        return f"Query: {self.query_text[:100]}... | Category: {self.research_category.value} | Complexity: {self.complexity_level}"


@dataclass
class QueryScenario:
    """Represents a comprehensive query testing scenario."""
    scenario_name: str
    scenario_description: str
    query_sequence: List[QueryTestCase]
    expected_workflow: List[str]
    success_criteria: Dict[str, Any]
    failure_conditions: List[str]
    resource_requirements: Dict[str, Any]
    estimated_duration: float
    
    @property
    def total_queries(self) -> int:
        """Get total number of queries in scenario."""
        return len(self.query_sequence)


class ClinicalQueryGenerator:
    """
    Generates realistic clinical metabolomics queries across different complexity levels
    and research categories.
    """
    
    # Clinical metabolomics query templates organized by research category
    QUERY_TEMPLATES = {
        ResearchCategory.METABOLITE_IDENTIFICATION: {
            'simple': [
                "What is {metabolite}?",
                "Identify compound with molecular formula {formula}",
                "Find metabolite with HMDB ID {hmdb_id}",
                "What metabolite has mass {mass}?"
            ],
            'medium': [
                "Identify unknown metabolite with m/z {mass} and retention time {rt} minutes using {technique}",
                "What are the fragmentation patterns for {metabolite} in {ms_mode} mode?",
                "Compare spectral data for {metabolite} across different analytical platforms",
                "Structural elucidation of compound showing {spectral_features}"
            ],
            'complex': [
                "Comprehensive identification of unknown metabolite using multi-platform approach: LC-MS/MS, GC-MS, and NMR data integration for compound with molecular ion {mass}, showing characteristic {fragments} fragmentation pattern",
                "Advanced metabolite identification workflow incorporating isotope patterns, retention time prediction, and spectral library matching for {sample_type} samples",
                "Machine learning-assisted compound identification using spectral features, chemical properties, and pathway context for novel biomarker discovery in {disease}"
            ],
            'expert': [
                "Develop comprehensive metabolite identification strategy integrating high-resolution mass spectrometry, ion mobility, and computational prediction for untargeted metabolomics analysis of {disease} patient samples, incorporating false discovery rate control and statistical validation",
                "Design multi-dimensional metabolite identification approach using orthogonal analytical techniques, spectral databases, and pathway network analysis for discovering novel {disease} biomarkers with clinical validation requirements"
            ]
        },
        
        ResearchCategory.PATHWAY_ANALYSIS: {
            'simple': [
                "What is {pathway} pathway?",
                "List metabolites in {pathway}",
                "How is {metabolite} involved in {pathway}?",
                "What pathways involve {metabolite}?"
            ],
            'medium': [
                "Analyze pathway enrichment for metabolites {metabolite_list} in {disease}",
                "Compare pathway activity between {condition1} and {condition2}",
                "Identify dysregulated pathways in {disease} using metabolomics data",
                "Map metabolomic changes to KEGG pathways for {study_type}"
            ],
            'complex': [
                "Comprehensive pathway analysis integrating metabolomics and proteomics data to identify dysregulated networks in {disease}, including pathway enrichment, flux analysis, and regulatory network reconstruction",
                "Systems-level pathway analysis incorporating metabolite-protein interactions, pathway crosstalk, and temporal dynamics for understanding {disease} progression mechanisms",
                "Multi-omics pathway integration using metabolomics, transcriptomics, and clinical data to identify therapeutic targets in {disease}"
            ],
            'expert': [
                "Advanced pathway network analysis incorporating metabolic flux modeling, constraint-based analysis, and machine learning approaches to identify novel therapeutic targets and biomarkers for {disease} with validation in independent cohorts",
                "Design comprehensive systems biology approach integrating metabolomics, proteomics, and genomics data with pathway databases and regulatory networks to understand {disease} pathophysiology and identify precision medicine opportunities"
            ]
        },
        
        ResearchCategory.BIOMARKER_DISCOVERY: {
            'simple': [
                "What are biomarkers for {disease}?",
                "Find metabolite markers for {condition}",
                "List diagnostic markers for {disease}",
                "What metabolites are elevated in {disease}?"
            ],
            'medium': [
                "Identify metabolomic biomarkers for early detection of {disease} in {population}",
                "Validate biomarker panel for {disease} diagnosis using {analytical_method}",
                "Compare biomarker performance between {method1} and {method2} for {disease}",
                "Develop prognostic biomarkers for {disease} outcome prediction"
            ],
            'complex': [
                "Comprehensive biomarker discovery study for {disease} incorporating untargeted metabolomics, statistical validation, and clinical correlation analysis with {sample_size} participants across {num_centers} clinical centers",
                "Multi-stage biomarker validation including discovery, verification, and validation phases for {disease} diagnostic panel with regulatory pathway considerations",
                "Longitudinal biomarker analysis for {disease} progression monitoring incorporating temporal metabolomic changes and clinical endpoints"
            ],
            'expert': [
                "Design comprehensive biomarker discovery and validation pipeline for {disease} incorporating multi-omics integration, machine learning algorithms, clinical validation, regulatory requirements, and health economic assessment for clinical implementation",
                "Develop precision medicine biomarker strategy for {disease} including population stratification, personalized risk assessment, treatment response prediction, and companion diagnostic development with regulatory approval pathway"
            ]
        },
        
        ResearchCategory.CLINICAL_DIAGNOSIS: {
            'simple': [
                "Diagnose {disease} using metabolomics",
                "Clinical tests for {condition}",
                "Laboratory values for {disease}",
                "Metabolomic profile of {disease}"
            ],
            'medium': [
                "Clinical metabolomics approach for diagnosing {disease} in {population}",
                "Integrate metabolomic and clinical data for {disease} diagnosis",
                "Point-of-care metabolomic testing for {condition}",
                "Metabolomic-guided clinical decision making for {disease}"
            ],
            'complex': [
                "Comprehensive clinical metabolomics platform for diagnosing {disease} incorporating sample collection protocols, analytical workflows, data processing pipelines, and clinical interpretation guidelines",
                "Clinical implementation of metabolomic diagnostics for {disease} including workflow integration, quality control, regulatory compliance, and healthcare provider training",
                "Personalized medicine approach using metabolomics for {disease} diagnosis, treatment selection, and monitoring with electronic health record integration"
            ],
            'expert': [
                "Design clinical metabolomics laboratory for routine {disease} diagnostics including instrument selection, method validation, quality management systems, regulatory compliance, and cost-effectiveness analysis",
                "Develop comprehensive clinical decision support system integrating metabolomics, clinical data, and artificial intelligence for {disease} diagnosis, prognosis, and treatment optimization with real-world evidence generation"
            ]
        },
        
        ResearchCategory.DRUG_DISCOVERY: {
            'simple': [
                "Drug targets for {disease}",
                "Metabolites affected by {drug}",
                "Drug metabolism of {compound}",
                "Therapeutic targets in {pathway}"
            ],
            'medium': [
                "Identify drug targets using metabolomics for {disease} treatment",
                "Pharmacometabolomics analysis of {drug} in {population}",
                "Drug mechanism of action study using metabolomics for {compound}",
                "Metabolomic biomarkers for drug efficacy in {disease}"
            ],
            'complex': [
                "Comprehensive drug discovery program using metabolomics for {disease} including target identification, lead optimization, mechanism of action studies, and biomarker development",
                "Pharmacometabolomics-guided drug development for {disease} incorporating personalized dosing, efficacy prediction, and safety monitoring",
                "Systems pharmacology approach integrating metabolomics, pharmacokinetics, and pharmacodynamics for {disease} drug development"
            ],
            'expert': [
                "Design integrated drug discovery platform combining metabolomics, systems biology, and artificial intelligence for {disease} therapeutic development with translational pathway from discovery to clinical trials",
                "Develop precision medicine drug development strategy using metabolomics-guided patient stratification, companion diagnostics, and biomarker-driven clinical trial design for {disease} therapeutics"
            ]
        },
        
        ResearchCategory.STATISTICAL_ANALYSIS: {
            'simple': [
                "Statistical analysis of metabolomics data",
                "Compare metabolite levels between groups",
                "Correlation analysis for {metabolites}",
                "Statistical significance of {biomarker}"
            ],
            'medium': [
                "Multivariate analysis of metabolomics data for {disease} study",
                "Power analysis for metabolomics biomarker study with {effect_size}",
                "Statistical validation of metabolomic biomarkers for {disease}",
                "Machine learning analysis of metabolomics data for {application}"
            ],
            'complex': [
                "Comprehensive statistical analysis pipeline for metabolomics study including normalization, missing value imputation, multivariate analysis, and multiple testing correction",
                "Advanced machine learning approaches for metabolomics data analysis including feature selection, model validation, and interpretation for {disease} research",
                "Longitudinal statistical analysis of metabolomics data incorporating repeated measures, mixed-effects modeling, and trajectory analysis"
            ],
            'expert': [
                "Design sophisticated statistical framework for metabolomics mega-analysis incorporating batch effect correction, meta-analysis methods, and heterogeneity assessment across multiple studies and platforms",
                "Develop advanced computational pipeline for metabolomics data integration including multi-omics statistical methods, network analysis, and causal inference for {disease} research"
            ]
        },
        
        ResearchCategory.LITERATURE_SEARCH: {
            'simple': [
                "Papers about {metabolite} and {disease}",
                "Literature on {topic} metabolomics",
                "Research articles on {biomarker}",
                "Studies of {pathway} in {disease}"
            ],
            'medium': [
                "Systematic review of metabolomics in {disease} diagnosis",
                "Meta-analysis of {biomarker} studies in {disease}",
                "Literature review of {analytical_method} applications in metabolomics",
                "Research trends in {disease} metabolomics over past {years} years"
            ],
            'complex': [
                "Comprehensive systematic review and meta-analysis of metabolomics biomarkers for {disease} including study quality assessment, heterogeneity analysis, and clinical translation potential",
                "Bibliometric analysis of {disease} metabolomics research including citation networks, collaboration patterns, and knowledge evolution",
                "Evidence synthesis for metabolomics applications in {disease} incorporating multiple study designs and outcome measures"
            ],
            'expert': [
                "Design comprehensive knowledge synthesis platform combining systematic literature review, expert curation, and artificial intelligence for metabolomics knowledge extraction and integration",
                "Develop advanced literature mining pipeline for metabolomics research incorporating natural language processing, knowledge graph construction, and automated evidence assessment"
            ]
        }
    }
    
    # Domain-specific variables and contexts
    DOMAIN_VARIABLES = {
        'metabolites': ['glucose', 'lactate', 'cholesterol', 'creatinine', 'urea', 'bilirubin', 'TMAO', 'acetate'],
        'diseases': ['diabetes', 'cardiovascular disease', 'cancer', 'kidney disease', 'liver disease', 'Alzheimer disease'],
        'pathways': ['glycolysis', 'TCA cycle', 'cholesterol biosynthesis', 'amino acid metabolism', 'fatty acid oxidation'],
        'analytical_methods': ['LC-MS/MS', 'GC-MS', 'NMR spectroscopy', 'CE-MS', 'HILIC-MS'],
        'sample_types': ['plasma', 'serum', 'urine', 'tissue', 'saliva', 'CSF'],
        'populations': ['pediatric patients', 'elderly population', 'diabetic patients', 'healthy controls'],
        'formulas': ['C6H12O6', 'C3H6O3', 'C27H46O', 'C4H7N3O', 'CH4N2O'],
        'masses': ['180.0634', '90.0317', '386.6535', '113.0589', '60.0553'],
        'hmdb_ids': ['HMDB0000122', 'HMDB0000190', 'HMDB0000067', 'HMDB0000562', 'HMDB0000294']
    }
    
    @classmethod
    def generate_query_test_case(cls,
                               research_category: ResearchCategory,
                               complexity_level: str = 'medium',
                               query_type: str = 'question') -> QueryTestCase:
        """Generate a comprehensive query test case."""
        
        # Get template for the research category and complexity
        if research_category not in cls.QUERY_TEMPLATES:
            research_category = ResearchCategory.METABOLITE_IDENTIFICATION
        
        category_templates = cls.QUERY_TEMPLATES[research_category]
        
        if complexity_level not in category_templates:
            complexity_level = 'medium'
        
        templates = category_templates[complexity_level]
        selected_template = random.choice(templates)
        
        # Fill template with domain-specific variables
        query_text = cls._fill_template(selected_template)
        
        # Generate query ID
        query_id = f"Q_{research_category.value[:8]}_{complexity_level}_{random.randint(1000, 9999)}"
        
        # Determine domain area
        domain_area = cls._determine_domain_area(query_text)
        
        # Generate expected entities and relationships
        expected_entities = cls._extract_expected_entities(query_text, research_category)
        expected_relationships = cls._extract_expected_relationships(research_category, expected_entities)
        expected_keywords = cls._extract_expected_keywords(query_text, research_category)
        
        # Generate validation criteria
        validation_criteria = cls._generate_validation_criteria(research_category, complexity_level)
        
        # Generate response template
        response_template = cls._generate_response_template(research_category, complexity_level)
        
        # Set confidence threshold based on complexity
        confidence_thresholds = {
            'simple': 0.9,
            'medium': 0.7,
            'complex': 0.6,
            'expert': 0.5
        }
        confidence_threshold = confidence_thresholds.get(complexity_level, 0.7)
        
        # Set processing timeout based on complexity
        timeout_values = {
            'simple': 5.0,
            'medium': 10.0,
            'complex': 20.0,
            'expert': 30.0
        }
        processing_timeout = timeout_values.get(complexity_level, 10.0)
        
        # Estimate cost based on complexity
        cost_estimates = {
            'simple': 0.02,
            'medium': 0.05,
            'complex': 0.10,
            'expert': 0.20
        }
        cost_estimate = cost_estimates.get(complexity_level, 0.05)
        
        # Generate metadata
        metadata = {
            'template_used': selected_template,
            'variables_filled': cls._get_filled_variables(selected_template, query_text),
            'domain_terms_count': len(cls._extract_domain_terms(query_text)),
            'technical_complexity': complexity_level,
            'expected_response_length': cls._estimate_response_length(complexity_level)
        }
        
        return QueryTestCase(
            query_id=query_id,
            query_text=query_text,
            research_category=research_category,
            complexity_level=complexity_level,
            query_type=query_type,
            domain_area=domain_area,
            expected_entities=expected_entities,
            expected_relationships=expected_relationships,
            expected_keywords=expected_keywords,
            validation_criteria=validation_criteria,
            response_template=response_template,
            confidence_threshold=confidence_threshold,
            processing_timeout=processing_timeout,
            cost_estimate=cost_estimate,
            metadata=metadata
        )
    
    @classmethod
    def _fill_template(cls, template: str) -> str:
        """Fill template with realistic domain variables."""
        filled_template = template
        
        # Extract placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        for placeholder in placeholders:
            if placeholder in cls.DOMAIN_VARIABLES:
                replacement = random.choice(cls.DOMAIN_VARIABLES[placeholder])
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder.endswith('_list'):
                # Handle list placeholders
                base_key = placeholder.replace('_list', '')
                if base_key in cls.DOMAIN_VARIABLES:
                    replacements = random.sample(cls.DOMAIN_VARIABLES[base_key], 
                                               min(3, len(cls.DOMAIN_VARIABLES[base_key])))
                    replacement = ', '.join(replacements)
                    filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder in ['condition1', 'condition2']:
                replacement = random.choice(cls.DOMAIN_VARIABLES['diseases'])
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder in ['method1', 'method2']:
                replacement = random.choice(cls.DOMAIN_VARIABLES['analytical_methods'])
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'sample_size':
                replacement = str(random.randint(100, 1000))
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'num_centers':
                replacement = str(random.randint(2, 10))
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'effect_size':
                replacement = f"{random.uniform(0.3, 1.5):.1f}"
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'years':
                replacement = str(random.randint(5, 15))
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'rt':
                replacement = f"{random.uniform(1.0, 15.0):.1f}"
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'technique':
                replacement = random.choice(cls.DOMAIN_VARIABLES['analytical_methods'])
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'ms_mode':
                replacement = random.choice(['positive', 'negative'])
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'spectral_features':
                features = ['neutral loss', 'characteristic ions', 'isotope patterns']
                replacement = random.choice(features)
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
            elif placeholder == 'fragments':
                fragments = ['m/z 163', 'm/z 147', 'm/z 129']
                replacement = random.choice(fragments)
                filled_template = filled_template.replace(f'{{{placeholder}}}', replacement)
        
        return filled_template
    
    @classmethod
    def _determine_domain_area(cls, query_text: str) -> str:
        """Determine domain area based on query content."""
        query_lower = query_text.lower()
        
        if any(term in query_lower for term in ['clinical', 'patient', 'diagnosis', 'diagnostic']):
            return 'clinical'
        elif any(term in query_lower for term in ['drug', 'therapeutic', 'treatment', 'therapy']):
            return 'therapeutic'
        elif any(term in query_lower for term in ['research', 'study', 'analysis', 'discovery']):
            return 'research'
        else:
            return 'diagnostic'
    
    @classmethod
    def _extract_expected_entities(cls, query_text: str, category: ResearchCategory) -> List[str]:
        """Extract expected entities from query text."""
        entities = []
        query_lower = query_text.lower()
        
        # Extract metabolites
        for metabolite in cls.DOMAIN_VARIABLES['metabolites']:
            if metabolite.lower() in query_lower:
                entities.append(f"METABOLITE:{metabolite}")
        
        # Extract diseases
        for disease in cls.DOMAIN_VARIABLES['diseases']:
            if disease.lower() in query_lower:
                entities.append(f"DISEASE:{disease}")
        
        # Extract pathways
        for pathway in cls.DOMAIN_VARIABLES['pathways']:
            if pathway.lower() in query_lower:
                entities.append(f"PATHWAY:{pathway}")
        
        # Add category-specific entities
        if category == ResearchCategory.DRUG_DISCOVERY:
            entities.extend(["DRUG:compound", "TARGET:protein"])
        elif category == ResearchCategory.STATISTICAL_ANALYSIS:
            entities.extend(["METHOD:statistics", "ANALYSIS:multivariate"])
        
        return entities[:10]  # Limit to top 10
    
    @classmethod
    def _extract_expected_relationships(cls, category: ResearchCategory, entities: List[str]) -> List[str]:
        """Generate expected relationships based on category and entities."""
        relationships = []
        
        if len(entities) >= 2:
            if category == ResearchCategory.BIOMARKER_DISCOVERY:
                relationships.append("biomarker_for")
                relationships.append("associated_with")
            elif category == ResearchCategory.PATHWAY_ANALYSIS:
                relationships.append("participates_in")
                relationships.append("regulates")
            elif category == ResearchCategory.DRUG_DISCOVERY:
                relationships.append("targets")
                relationships.append("affects")
            elif category == ResearchCategory.METABOLITE_IDENTIFICATION:
                relationships.append("has_mass")
                relationships.append("has_formula")
        
        return relationships
    
    @classmethod
    def _extract_expected_keywords(cls, query_text: str, category: ResearchCategory) -> List[str]:
        """Extract expected keywords from query."""
        keywords = []
        
        # Category-specific keywords
        category_keywords = {
            ResearchCategory.METABOLITE_IDENTIFICATION: ['identification', 'compound', 'structure', 'mass'],
            ResearchCategory.PATHWAY_ANALYSIS: ['pathway', 'network', 'enrichment', 'regulation'],
            ResearchCategory.BIOMARKER_DISCOVERY: ['biomarker', 'diagnostic', 'prognostic', 'signature'],
            ResearchCategory.CLINICAL_DIAGNOSIS: ['clinical', 'diagnosis', 'patient', 'laboratory'],
            ResearchCategory.DRUG_DISCOVERY: ['drug', 'therapeutic', 'target', 'mechanism'],
            ResearchCategory.STATISTICAL_ANALYSIS: ['statistical', 'analysis', 'significance', 'correlation']
        }
        
        keywords.extend(category_keywords.get(category, []))
        
        # Extract technical terms
        technical_terms = ['metabolomics', 'LC-MS', 'GC-MS', 'NMR', 'biomarker', 'pathway']
        for term in technical_terms:
            if term.lower() in query_text.lower():
                keywords.append(term)
        
        return list(set(keywords))[:8]  # Remove duplicates and limit
    
    @classmethod
    def _generate_validation_criteria(cls, category: ResearchCategory, complexity: str) -> Dict[str, Any]:
        """Generate validation criteria for the query."""
        base_criteria = {
            'min_response_length': 100,
            'max_response_length': 2000,
            'must_contain_keywords': True,
            'must_mention_entities': True,
            'require_references': False,
            'require_quantitative_data': False
        }
        
        # Adjust based on complexity
        if complexity in ['complex', 'expert']:
            base_criteria.update({
                'min_response_length': 300,
                'max_response_length': 5000,
                'require_references': True,
                'require_quantitative_data': True,
                'require_methodology': True
            })
        
        # Adjust based on category
        if category == ResearchCategory.STATISTICAL_ANALYSIS:
            base_criteria.update({
                'require_quantitative_data': True,
                'require_statistical_methods': True
            })
        elif category == ResearchCategory.CLINICAL_DIAGNOSIS:
            base_criteria.update({
                'require_clinical_context': True,
                'require_reference_ranges': True
            })
        
        return base_criteria
    
    @classmethod
    def _generate_response_template(cls, category: ResearchCategory, complexity: str) -> str:
        """Generate response template for the category and complexity."""
        
        templates = {
            ResearchCategory.METABOLITE_IDENTIFICATION: {
                'simple': "The metabolite {name} is {description}. Its molecular formula is {formula} and molecular weight is {weight}.",
                'complex': "Comprehensive identification of {name}: Chemical structure: {structure}. Analytical characteristics: {analytical_data}. Biological significance: {biological_role}. Clinical relevance: {clinical_importance}."
            },
            ResearchCategory.BIOMARKER_DISCOVERY: {
                'simple': "Key biomarkers for {disease} include {biomarkers}. These metabolites show {changes} in disease conditions.",
                'complex': "Biomarker discovery analysis reveals: Primary biomarkers: {primary_markers} with sensitivity {sensitivity}% and specificity {specificity}%. Pathway involvement: {pathways}. Clinical validation: {validation_data}."
            },
            ResearchCategory.PATHWAY_ANALYSIS: {
                'simple': "The {pathway} pathway involves {metabolites} and is regulated by {regulators}.",
                'complex': "Pathway analysis results: Enriched pathways: {enriched_pathways}. Key metabolites: {key_metabolites}. Regulatory networks: {networks}. Functional implications: {implications}."
            }
        }
        
        category_templates = templates.get(category, templates[ResearchCategory.METABOLITE_IDENTIFICATION])
        return category_templates.get(complexity, category_templates.get('simple', ''))
    
    @classmethod
    def _get_filled_variables(cls, template: str, filled_text: str) -> Dict[str, str]:
        """Get mapping of variables that were filled in template."""
        placeholders = re.findall(r'\{(\w+)\}', template)
        return {placeholder: 'filled' for placeholder in placeholders}
    
    @classmethod
    def _extract_domain_terms(cls, query_text: str) -> List[str]:
        """Extract domain-specific terms from query."""
        domain_terms = []
        query_lower = query_text.lower()
        
        all_terms = []
        for term_list in cls.DOMAIN_VARIABLES.values():
            all_terms.extend(term_list)
        
        for term in all_terms:
            if term.lower() in query_lower:
                domain_terms.append(term)
        
        return domain_terms
    
    @classmethod
    def _estimate_response_length(cls, complexity: str) -> int:
        """Estimate expected response length based on complexity."""
        length_estimates = {
            'simple': 200,
            'medium': 500,
            'complex': 1000,
            'expert': 2000
        }
        return length_estimates.get(complexity, 500)


class QueryScenarioBuilder:
    """
    Builds comprehensive query testing scenarios for different research workflows.
    """
    
    SCENARIO_TEMPLATES = {
        'biomarker_discovery_workflow': {
            'description': 'Complete biomarker discovery workflow from initial query to validation',
            'query_sequence': [
                ('What are potential biomarkers for diabetes?', ResearchCategory.BIOMARKER_DISCOVERY, 'simple'),
                ('Identify metabolomic biomarkers for diabetes diagnosis using LC-MS/MS', ResearchCategory.BIOMARKER_DISCOVERY, 'medium'),
                ('Validate glucose and lactate as biomarkers for diabetes in clinical cohort', ResearchCategory.BIOMARKER_DISCOVERY, 'complex'),
                ('Statistical analysis of biomarker performance for diabetes diagnosis', ResearchCategory.STATISTICAL_ANALYSIS, 'medium'),
                ('Literature review of diabetes biomarker studies', ResearchCategory.LITERATURE_SEARCH, 'medium')
            ],
            'expected_workflow': ['discovery', 'validation', 'statistical_analysis', 'literature_review'],
            'success_criteria': {
                'all_queries_processed': True,
                'biomarkers_identified': True,
                'validation_completed': True,
                'statistical_significance': True
            }
        },
        
        'clinical_diagnosis_workflow': {
            'description': 'Clinical diagnostic workflow using metabolomics',
            'query_sequence': [
                ('Clinical metabolomic profile for kidney disease', ResearchCategory.CLINICAL_DIAGNOSIS, 'simple'),
                ('Diagnostic metabolomics panel for kidney disease using creatinine and urea', ResearchCategory.CLINICAL_DIAGNOSIS, 'medium'),
                ('Integrate metabolomic and clinical data for kidney disease diagnosis', ResearchCategory.CLINICAL_DIAGNOSIS, 'complex'),
                ('Statistical validation of kidney disease diagnostic panel', ResearchCategory.STATISTICAL_ANALYSIS, 'medium')
            ],
            'expected_workflow': ['clinical_assessment', 'diagnostic_panel', 'integration', 'validation'],
            'success_criteria': {
                'diagnostic_accuracy': 0.85,
                'clinical_applicability': True,
                'cost_effectiveness': True
            }
        },
        
        'drug_discovery_workflow': {
            'description': 'Drug discovery workflow using metabolomics',
            'query_sequence': [
                ('Drug targets for cardiovascular disease', ResearchCategory.DRUG_DISCOVERY, 'simple'),
                ('Pharmacometabolomics analysis of statins in cardiovascular disease', ResearchCategory.DRUG_DISCOVERY, 'medium'),
                ('Metabolomic biomarkers for statin efficacy and safety', ResearchCategory.DRUG_DISCOVERY, 'complex'),
                ('Pathway analysis of statin mechanism of action', ResearchCategory.PATHWAY_ANALYSIS, 'medium'),
                ('Literature review of statin pharmacometabolomics studies', ResearchCategory.LITERATURE_SEARCH, 'medium')
            ],
            'expected_workflow': ['target_identification', 'mechanism_study', 'biomarker_development', 'pathway_analysis'],
            'success_criteria': {
                'targets_identified': True,
                'mechanism_elucidated': True,
                'biomarkers_validated': True
            }
        }
    }
    
    @classmethod
    def build_scenario(cls, scenario_name: str) -> QueryScenario:
        """Build comprehensive query scenario."""
        
        if scenario_name not in cls.SCENARIO_TEMPLATES:
            scenario_name = 'biomarker_discovery_workflow'
        
        template = cls.SCENARIO_TEMPLATES[scenario_name]
        
        # Generate query test cases
        query_cases = []
        for i, (query_text, category, complexity) in enumerate(template['query_sequence']):
            query_case = QueryTestCase(
                query_id=f"{scenario_name}_Q{i+1:02d}",
                query_text=query_text,
                research_category=category,
                complexity_level=complexity,
                query_type='question',
                domain_area='research',
                expected_entities=ClinicalQueryGenerator._extract_expected_entities(query_text, category),
                expected_relationships=ClinicalQueryGenerator._extract_expected_relationships(category, []),
                expected_keywords=ClinicalQueryGenerator._extract_expected_keywords(query_text, category),
                validation_criteria=ClinicalQueryGenerator._generate_validation_criteria(category, complexity),
                confidence_threshold=0.7,
                processing_timeout=15.0,
                cost_estimate=0.08,
                metadata={'scenario': scenario_name, 'sequence_position': i+1}
            )
            query_cases.append(query_case)
        
        # Calculate resource requirements
        total_cost = sum(case.cost_estimate for case in query_cases)
        total_timeout = sum(case.processing_timeout for case in query_cases)
        
        resource_requirements = {
            'total_cost_estimate': total_cost,
            'total_timeout': total_timeout,
            'memory_requirement_mb': len(query_cases) * 100,
            'concurrent_queries': min(3, len(query_cases))
        }
        
        # Add failure conditions
        failure_conditions = [
            'Any query fails to process',
            'Response quality below threshold',
            'Cost exceeds budget limit',
            'Processing timeout exceeded'
        ]
        
        return QueryScenario(
            scenario_name=scenario_name,
            scenario_description=template['description'],
            query_sequence=query_cases,
            expected_workflow=template['expected_workflow'],
            success_criteria=template['success_criteria'],
            failure_conditions=failure_conditions,
            resource_requirements=resource_requirements,
            estimated_duration=total_timeout * 1.2  # Add 20% buffer
        )


# Pytest fixtures for query testing
@pytest.fixture
def clinical_query_generator():
    """Provide clinical query generator."""
    return ClinicalQueryGenerator()

@pytest.fixture
def query_scenario_builder():
    """Provide query scenario builder."""
    return QueryScenarioBuilder()

@pytest.fixture
def sample_query_test_cases():
    """Provide sample query test cases across categories."""
    test_cases = []
    
    categories = [
        ResearchCategory.METABOLITE_IDENTIFICATION,
        ResearchCategory.BIOMARKER_DISCOVERY,
        ResearchCategory.PATHWAY_ANALYSIS,
        ResearchCategory.CLINICAL_DIAGNOSIS,
        ResearchCategory.STATISTICAL_ANALYSIS
    ]
    
    complexities = ['simple', 'medium', 'complex']
    
    for category in categories:
        for complexity in complexities:
            test_case = ClinicalQueryGenerator.generate_query_test_case(
                research_category=category,
                complexity_level=complexity,
                query_type='question'
            )
            test_cases.append(test_case)
    
    return test_cases

@pytest.fixture
def biomarker_discovery_scenario():
    """Provide biomarker discovery query scenario."""
    return QueryScenarioBuilder.build_scenario('biomarker_discovery_workflow')

@pytest.fixture
def clinical_diagnosis_scenario():
    """Provide clinical diagnosis query scenario."""
    return QueryScenarioBuilder.build_scenario('clinical_diagnosis_workflow')

@pytest.fixture
def drug_discovery_scenario():
    """Provide drug discovery query scenario."""
    return QueryScenarioBuilder.build_scenario('drug_discovery_workflow')

@pytest.fixture
def multi_complexity_query_set():
    """Provide queries of varying complexity for the same category."""
    category = ResearchCategory.BIOMARKER_DISCOVERY
    complexities = ['simple', 'medium', 'complex', 'expert']
    
    query_set = []
    for complexity in complexities:
        query_case = ClinicalQueryGenerator.generate_query_test_case(
            research_category=category,
            complexity_level=complexity
        )
        query_set.append(query_case)
    
    return query_set

@pytest.fixture
def metabolomics_query_collection():
    """Provide comprehensive collection of metabolomics queries."""
    collection = {
        'by_category': defaultdict(list),
        'by_complexity': defaultdict(list),
        'by_domain': defaultdict(list)
    }
    
    # Generate queries for each category
    for category in ResearchCategory:
        for complexity in ['simple', 'medium', 'complex']:
            query_case = ClinicalQueryGenerator.generate_query_test_case(
                research_category=category,
                complexity_level=complexity
            )
            
            collection['by_category'][category.value].append(query_case)
            collection['by_complexity'][complexity].append(query_case)
            collection['by_domain'][query_case.domain_area].append(query_case)
    
    return dict(collection)

@pytest.fixture
def query_validation_test_data():
    """Provide data for testing query validation."""
    return {
        'valid_queries': [
            "What are the metabolomic biomarkers for diabetes?",
            "Identify compounds using LC-MS/MS for cancer diagnosis",
            "Pathway analysis of glucose metabolism in liver disease"
        ],
        'invalid_queries': [
            "",  # Empty query
            "x" * 10000,  # Too long
            "What is the meaning of life?",  # Non-biomedical
            "Show me all data"  # Too vague
        ],
        'edge_cases': [
            "What are biomarkers?",  # Very general
            "LC-MS analysis glucose diabetes pathway metabolomics",  # No proper grammar
            "How do you analyze metabolomic data using statistical methods with machine learning?"  # Very complex
        ]
    }

@pytest.fixture
def research_workflow_scenarios():
    """Provide multiple research workflow scenarios."""
    scenarios = []
    
    scenario_names = ['biomarker_discovery_workflow', 'clinical_diagnosis_workflow', 'drug_discovery_workflow']
    
    for scenario_name in scenario_names:
        scenario = QueryScenarioBuilder.build_scenario(scenario_name)
        scenarios.append(scenario)
    
    return scenarios

@pytest.fixture
def query_performance_test_data():
    """Provide data for query performance testing."""
    return {
        'lightweight_queries': [
            ClinicalQueryGenerator.generate_query_test_case(
                ResearchCategory.METABOLITE_IDENTIFICATION, 'simple'
            ) for _ in range(10)
        ],
        'medium_queries': [
            ClinicalQueryGenerator.generate_query_test_case(
                ResearchCategory.BIOMARKER_DISCOVERY, 'medium'
            ) for _ in range(10)
        ],
        'heavy_queries': [
            ClinicalQueryGenerator.generate_query_test_case(
                ResearchCategory.PATHWAY_ANALYSIS, 'complex'
            ) for _ in range(5)
        ],
        'concurrent_test_queries': [
            ClinicalQueryGenerator.generate_query_test_case(
                random.choice(list(ResearchCategory)), 
                random.choice(['simple', 'medium'])
            ) for _ in range(20)
        ]
    }