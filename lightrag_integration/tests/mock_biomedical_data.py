#!/usr/bin/env python3
"""
Mock Biomedical Data Generators for Clinical Metabolomics Oracle Testing.

This module provides sophisticated mock data generators for creating realistic
biomedical research papers, clinical studies, and scientific literature content
specifically tailored for metabolomics research scenarios.

Components:
- BiomedicalPaperGenerator: Creates realistic research paper abstracts and content
- ClinicalStudyMockGenerator: Generates comprehensive clinical study datasets
- LiteratureSearchMockData: Provides mock literature search results
- ResearchAbstractGenerator: Creates domain-specific research abstracts
- MultimicsIntegrationMockData: Generates multi-omics research scenarios
- ResearchMethodologyGenerator: Creates realistic research methodology descriptions

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import random
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np


@dataclass
class ResearchPaper:
    """Represents a biomedical research paper with comprehensive metadata."""
    title: str
    authors: List[str]
    affiliations: List[str]
    journal: str
    publication_year: int
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: str = ""
    pmid: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    research_area: str = ""
    study_type: str = ""
    sample_size: int = 0
    analytical_methods: List[str] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    biomarkers_identified: List[str] = field(default_factory=list)
    pathways_analyzed: List[str] = field(default_factory=list)
    statistical_methods: List[str] = field(default_factory=list)
    funding_sources: List[str] = field(default_factory=list)
    ethical_approval: bool = True
    data_availability: str = ""
    supplementary_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def citation(self) -> str:
        """Generate citation string."""
        authors_str = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        return f"{authors_str} ({self.publication_year}). {self.title}. {self.journal}."
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata dictionary."""
        return {
            'title': self.title,
            'authors': self.authors,
            'affiliations': self.affiliations,
            'journal': self.journal,
            'publication_year': self.publication_year,
            'doi': self.doi,
            'pmid': self.pmid,
            'keywords': self.keywords,
            'mesh_terms': self.mesh_terms,
            'research_area': self.research_area,
            'study_type': self.study_type,
            'sample_size': self.sample_size,
            'analytical_methods': self.analytical_methods,
            'biomarkers_identified': self.biomarkers_identified,
            'pathways_analyzed': self.pathways_analyzed,
            'statistical_methods': self.statistical_methods
        }


@dataclass 
class ClinicalTrialData:
    """Represents clinical trial data with realistic characteristics."""
    nct_id: str
    title: str
    status: str  # 'recruiting', 'completed', 'terminated', 'suspended'
    phase: str  # 'Phase I', 'Phase II', 'Phase III', 'Phase IV'
    study_type: str
    condition: str
    intervention: str
    primary_endpoints: List[str]
    secondary_endpoints: List[str]
    enrollment: int
    start_date: str
    sponsor: str
    completion_date: Optional[str] = None
    collaborators: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    biomarkers_measured: List[str] = field(default_factory=list)
    analytical_platforms: List[str] = field(default_factory=list)
    results_summary: Optional[Dict[str, Any]] = None


class BiomedicalPaperGenerator:
    """
    Generates realistic biomedical research papers with domain-specific content
    tailored for metabolomics, proteomics, and clinical research.
    """
    
    # Comprehensive author database with realistic names and affiliations
    AUTHOR_DATABASE = {
        'metabolomics_experts': [
            'Dr. Sarah Chen', 'Dr. Michael Rodriguez', 'Dr. Emily Johnson', 'Dr. David Kumar',
            'Dr. Lisa Zhang', 'Dr. Robert Anderson', 'Dr. Maria Gonzalez', 'Dr. James Wilson',
            'Prof. Anna Kowalski', 'Prof. Thomas Mueller', 'Dr. Jennifer Lee', 'Dr. Ahmed Hassan'
        ],
        'clinical_researchers': [
            'Dr. Patricia Davis', 'Dr. Christopher Brown', 'Dr. Rachel Green', 'Dr. Kevin Park',
            'Dr. Michelle Taylor', 'Dr. Brian Foster', 'Dr. Susan Liu', 'Dr. Mark Thompson',
            'Prof. Catherine White', 'Prof. Daniel Miller', 'Dr. Amy Roberts', 'Dr. Paul Kim'
        ],
        'bioinformatics_experts': [
            'Dr. Julia Martinez', 'Dr. Steven Chang', 'Dr. Rebecca Walsh', 'Dr. Gary Chen',
            'Dr. Nicole Adams', 'Dr. Ryan O\'Connor', 'Dr. Samantha Lee', 'Dr. Eric Nakamura',
            'Prof. Helen Rodriguez', 'Prof. Jonathan Smith', 'Dr. Diana Patel', 'Dr. Alex Wang'
        ]
    }
    
    AFFILIATION_DATABASE = [
        'Harvard Medical School, Boston, MA, USA',
        'Stanford University School of Medicine, Stanford, CA, USA',
        'Mayo Clinic, Rochester, MN, USA',
        'Johns Hopkins University, Baltimore, MD, USA',
        'University of California San Francisco, San Francisco, CA, USA',
        'Massachusetts General Hospital, Boston, MA, USA',
        'MD Anderson Cancer Center, Houston, TX, USA',
        'Cleveland Clinic, Cleveland, OH, USA',
        'University of Oxford, Oxford, UK',
        'University of Cambridge, Cambridge, UK',
        'Karolinska Institute, Stockholm, Sweden',
        'University of Toronto, Toronto, ON, Canada',
        'National Institutes of Health, Bethesda, MD, USA',
        'Centers for Disease Control and Prevention, Atlanta, GA, USA',
        'European Medicines Agency, Amsterdam, Netherlands'
    ]
    
    JOURNAL_DATABASE = {
        'high_impact': [
            'Nature Medicine', 'The Lancet', 'New England Journal of Medicine',
            'Nature', 'Science', 'Cell', 'Nature Biotechnology'
        ],
        'specialized': [
            'Metabolomics', 'Journal of Proteome Research', 'Analytical Chemistry',
            'Clinical Chemistry', 'Biomarkers in Medicine', 'Molecular Systems Biology',
            'Journal of Clinical Investigation', 'Clinical Biochemistry'
        ],
        'clinical': [
            'Journal of Clinical Oncology', 'Diabetes Care', 'Circulation',
            'Journal of the American College of Cardiology', 'Kidney International',
            'Hepatology', 'Clinical Infectious Diseases'
        ]
    }
    
    # Disease-specific research templates
    DISEASE_RESEARCH_TEMPLATES = {
        'diabetes': {
            'title_patterns': [
                'Metabolomic Profiling Reveals Novel Biomarkers for {diabetes_type} Diabetes',
                'Plasma Metabolomics in {diabetes_type} Diabetes: Pathway Analysis and Clinical Implications',
                'Integration of Metabolomics and Clinical Data for {diabetes_type} Diabetes Prediction',
                'Urinary Metabolomic Signatures in {diabetes_type} Diabetes Patients'
            ],
            'abstracts': {
                'background': [
                    'Diabetes mellitus is a complex metabolic disorder affecting millions worldwide.',
                    'Early detection and monitoring of diabetes progression remains challenging.',
                    'Metabolomic approaches offer new insights into diabetes pathophysiology.',
                    'Personalized diabetes management requires better biomarker strategies.'
                ],
                'objectives': [
                    'To identify novel metabolic biomarkers for diabetes diagnosis and monitoring.',
                    'To characterize metabolomic profiles associated with diabetes progression.',
                    'To develop a metabolomics-based predictive model for diabetes risk.',
                    'To investigate metabolic pathway alterations in diabetes patients.'
                ],
                'methods': [
                    'We analyzed plasma samples from {} diabetes patients and {} controls using LC-MS/MS.',
                    'Urine samples were collected from {} participants and analyzed by GC-MS.',
                    'A comprehensive metabolomic analysis was performed using multiple analytical platforms.',
                    'Multivariate statistical analysis was applied to identify discriminating metabolites.'
                ],
                'results': [
                    'We identified {} significantly altered metabolites with p<0.05.',
                    'Pathway analysis revealed dysregulation of glucose and amino acid metabolism.',
                    'The metabolomic signature showed {} sensitivity and {} specificity.',
                    'Key biomarkers included elevated glucose, lactate, and branched-chain amino acids.'
                ],
                'conclusions': [
                    'This metabolomic approach provides new insights into diabetes pathophysiology.',
                    'The identified biomarkers may improve diabetes diagnosis and monitoring.',
                    'Integration with clinical data enhances predictive accuracy.',
                    'These findings support personalized diabetes management strategies.'
                ]
            },
            'keywords': ['diabetes', 'metabolomics', 'biomarkers', 'glucose metabolism', 'insulin resistance'],
            'mesh_terms': ['Diabetes Mellitus', 'Metabolomics', 'Biomarkers', 'Glucose', 'Insulin'],
            'biomarkers': ['glucose', 'HbA1c', 'lactate', 'alanine', 'valine', 'leucine', 'isoleucine'],
            'pathways': ['Glycolysis', 'Gluconeogenesis', 'Amino acid metabolism', 'Fatty acid metabolism'],
            'methods': ['LC-MS/MS', 'GC-MS', 'Enzymatic assays', 'Immunoassays']
        },
        'cardiovascular': {
            'title_patterns': [
                'Lipidomic Analysis of Cardiovascular Disease: Novel Risk Markers',
                'Metabolomic Signatures of Coronary Artery Disease in Large-Scale Cohort Study',
                'Plasma Metabolomics for Cardiovascular Risk Prediction and Stratification',
                'Integration of Metabolomics and Proteomics in Heart Failure Patients'
            ],
            'abstracts': {
                'background': [
                    'Cardiovascular disease remains the leading cause of mortality worldwide.',
                    'Current risk stratification methods have limitations in accuracy.',
                    'Metabolomic biomarkers offer potential for improved cardiovascular risk assessment.',
                    'Understanding metabolic changes in cardiovascular disease is crucial.'
                ],
                'objectives': [
                    'To identify metabolomic biomarkers for cardiovascular disease prediction.',
                    'To characterize metabolic alterations associated with coronary artery disease.',
                    'To develop a metabolomics-based cardiovascular risk score.',
                    'To investigate lipid metabolism changes in heart failure patients.'
                ]
            },
            'keywords': ['cardiovascular disease', 'metabolomics', 'lipids', 'coronary artery disease', 'risk prediction'],
            'mesh_terms': ['Cardiovascular Diseases', 'Metabolomics', 'Lipids', 'Coronary Disease', 'Risk Assessment'],
            'biomarkers': ['cholesterol', 'TMAO', 'lactate', 'BNP', 'troponin', 'CRP'],
            'pathways': ['Cholesterol metabolism', 'Fatty acid oxidation', 'Inflammation', 'Energy metabolism'],
            'methods': ['LC-MS/MS', 'NMR spectroscopy', 'Enzymatic assays', 'Immunoassays']
        },
        'cancer': {
            'title_patterns': [
                'Metabolomic Profiling of {} Cancer: Tumor vs. Normal Tissue Analysis',
                'Serum Metabolomics for Early Detection of {} Cancer',
                'Metabolic Reprogramming in {} Cancer: Therapeutic Target Identification',
                'Integration of Metabolomics and Genomics in {} Cancer Progression'
            ],
            'abstracts': {
                'background': [
                    'Cancer cells exhibit distinct metabolic reprogramming patterns.',
                    'Early cancer detection remains a major clinical challenge.',
                    'Metabolomic approaches can reveal novel cancer biomarkers.',
                    'Understanding tumor metabolism is crucial for therapeutic development.'
                ],
                'objectives': [
                    'To characterize metabolic alterations in cancer tissue vs. normal tissue.',
                    'To identify serum metabolomic biomarkers for early cancer detection.',
                    'To investigate metabolic pathways involved in cancer progression.',
                    'To develop metabolomics-based cancer diagnostic tools.'
                ]
            },
            'keywords': ['cancer', 'metabolomics', 'tumor metabolism', 'biomarkers', 'early detection'],
            'mesh_terms': ['Neoplasms', 'Metabolomics', 'Tumor Biomarkers', 'Cell Transformation', 'Metabolism'],
            'biomarkers': ['lactate', 'glutamine', 'glucose', 'succinate', 'alpha-ketoglutarate'],
            'pathways': ['Glycolysis', 'Glutamine metabolism', 'TCA cycle', 'Warburg effect'],
            'methods': ['LC-MS/MS', 'GC-MS', 'NMR spectroscopy', 'Metabolic flux analysis']
        }
    }
    
    @classmethod
    def generate_research_paper(cls, 
                              research_area: str = 'metabolomics',
                              disease: str = 'diabetes',
                              complexity: str = 'medium') -> ResearchPaper:
        """Generate a comprehensive research paper with realistic content."""
        
        # Select disease template
        disease_template = cls.DISEASE_RESEARCH_TEMPLATES.get(disease, 
                                                            cls.DISEASE_RESEARCH_TEMPLATES['diabetes'])
        
        # Generate title
        title_pattern = random.choice(disease_template['title_patterns'])
        if disease == 'diabetes':
            diabetes_type = random.choice(['Type 2', 'Type 1', 'Gestational'])
            title = title_pattern.format(diabetes_type=diabetes_type)
        elif disease == 'cancer':
            cancer_type = random.choice(['Breast', 'Lung', 'Colorectal', 'Prostate', 'Pancreatic'])
            title = title_pattern.format(cancer_type)
        else:
            title = title_pattern
        
        # Generate authors and affiliations
        if research_area == 'clinical':
            author_pool = cls.AUTHOR_DATABASE['clinical_researchers']
        elif research_area == 'bioinformatics':
            author_pool = cls.AUTHOR_DATABASE['bioinformatics_experts']
        else:
            author_pool = cls.AUTHOR_DATABASE['metabolomics_experts']
        
        num_authors = random.randint(4, 10)
        authors = random.sample(author_pool, min(num_authors, len(author_pool)))
        affiliations = random.sample(cls.AFFILIATION_DATABASE, random.randint(2, 5))
        
        # Select journal
        if random.random() < 0.2:  # 20% chance for high-impact journal
            journal = random.choice(cls.JOURNAL_DATABASE['high_impact'])
        elif disease in ['diabetes', 'cardiovascular', 'cancer']:
            journal = random.choice(cls.JOURNAL_DATABASE['clinical'])
        else:
            journal = random.choice(cls.JOURNAL_DATABASE['specialized'])
        
        # Generate publication details
        year = random.randint(2019, 2024)
        volume = f"{random.randint(10, 50)}"
        issue = f"{random.randint(1, 12)}"
        pages = f"{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        doi = f"10.1000/{disease}.{year}.{random.randint(1000, 9999):04d}"
        pmid = f"{random.randint(30000000, 35000000)}"
        
        # Generate abstract
        abstract = cls._generate_abstract(disease_template, complexity)
        
        # Generate study characteristics
        sample_size = cls._generate_sample_size(complexity)
        
        # Generate research-specific data
        keywords = disease_template['keywords'] + [research_area, 'clinical study']
        mesh_terms = disease_template['mesh_terms']
        biomarkers = random.sample(disease_template['biomarkers'], 
                                 random.randint(3, len(disease_template['biomarkers'])))
        pathways = random.sample(disease_template['pathways'],
                               random.randint(2, len(disease_template['pathways'])))
        methods = random.sample(disease_template['methods'],
                              random.randint(2, len(disease_template['methods'])))
        
        # Generate statistical methods
        statistical_methods = random.sample([
            'Student t-test', 'Mann-Whitney U test', 'ANOVA', 'Kruskal-Wallis test',
            'Pearson correlation', 'Spearman correlation', 'Multiple linear regression',
            'Logistic regression', 'PCA', 'PLS-DA', 'Random forest', 'SVM'
        ], random.randint(3, 6))
        
        # Generate funding sources
        funding_sources = random.sample([
            'National Institutes of Health (NIH)',
            'National Science Foundation (NSF)',
            'American Heart Association',
            'American Diabetes Association',
            'European Research Council',
            'Wellcome Trust',
            'Canadian Institutes of Health Research'
        ], random.randint(1, 3))
        
        # Generate key findings
        key_findings = cls._generate_key_findings(disease, biomarkers, pathways)
        
        # Generate supplementary data
        supplementary_data = {
            'metabolite_data': f"Supplementary Table S1: Complete metabolite concentration data ({sample_size} samples)",
            'pathway_analysis': "Supplementary Figure S1: Pathway enrichment analysis results",
            'statistical_analysis': "Supplementary Methods: Detailed statistical analysis procedures",
            'biomarker_validation': "Supplementary Table S2: Biomarker validation results"
        }
        
        return ResearchPaper(
            title=title,
            authors=authors,
            affiliations=affiliations,
            journal=journal,
            publication_year=year,
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
            pmid=pmid,
            abstract=abstract,
            keywords=keywords,
            mesh_terms=mesh_terms,
            research_area=research_area,
            study_type=random.choice(['case_control', 'cohort', 'cross_sectional', 'intervention']),
            sample_size=sample_size,
            analytical_methods=methods,
            key_findings=key_findings,
            biomarkers_identified=biomarkers,
            pathways_analyzed=pathways,
            statistical_methods=statistical_methods,
            funding_sources=funding_sources,
            ethical_approval=True,
            data_availability="Data available upon request" if random.random() < 0.7 else "Data publicly available",
            supplementary_data=supplementary_data
        )
    
    @classmethod
    def _generate_abstract(cls, disease_template: Dict[str, Any], complexity: str) -> str:
        """Generate realistic abstract based on disease template."""
        
        abstract_template = disease_template.get('abstracts', {})
        
        # Select content based on complexity
        if complexity == 'simple':
            sections = 3
        elif complexity == 'complex':
            sections = 5
        else:  # medium
            sections = 4
        
        abstract_parts = []
        
        # Background
        if 'background' in abstract_template:
            background = random.choice(abstract_template['background'])
            abstract_parts.append(f"Background: {background}")
        
        # Objective
        if 'objectives' in abstract_template:
            objective = random.choice(abstract_template['objectives'])
            abstract_parts.append(f"Objective: {objective}")
        
        # Methods
        if 'methods' in abstract_template and sections >= 3:
            method = random.choice(abstract_template['methods'])
            sample_size = random.randint(100, 500)
            control_size = random.randint(50, 200)
            method_text = method.format(sample_size, control_size)
            abstract_parts.append(f"Methods: {method_text}")
        
        # Results
        if 'results' in abstract_template and sections >= 4:
            result = random.choice(abstract_template['results'])
            num_metabolites = random.randint(15, 80)
            sensitivity = random.randint(75, 95)
            specificity = random.randint(75, 95)
            result_text = result.format(num_metabolites, sensitivity, specificity)
            abstract_parts.append(f"Results: {result_text}")
        
        # Conclusions
        if 'conclusions' in abstract_template and sections >= 5:
            conclusion = random.choice(abstract_template['conclusions'])
            abstract_parts.append(f"Conclusions: {conclusion}")
        
        return " ".join(abstract_parts)
    
    @classmethod
    def _generate_sample_size(cls, complexity: str) -> int:
        """Generate realistic sample size based on study complexity."""
        if complexity == 'simple':
            return random.randint(50, 150)
        elif complexity == 'complex':
            return random.randint(300, 1000)
        else:  # medium
            return random.randint(150, 300)
    
    @classmethod
    def _generate_key_findings(cls, disease: str, biomarkers: List[str], pathways: List[str]) -> List[str]:
        """Generate realistic key findings based on disease and biomarkers."""
        findings = []
        
        # Primary biomarker finding
        if biomarkers:
            primary_biomarker = biomarkers[0]
            change_direction = random.choice(['significantly elevated', 'significantly reduced', 'altered'])
            findings.append(f"{primary_biomarker.title()} was {change_direction} in {disease} patients vs. controls (p<0.001)")
        
        # Pathway finding
        if pathways:
            primary_pathway = pathways[0]
            findings.append(f"Pathway analysis revealed significant dysregulation of {primary_pathway.lower()}")
        
        # Statistical performance finding
        if len(biomarkers) > 1:
            auc = random.uniform(0.75, 0.95)
            findings.append(f"Combined biomarker panel achieved AUC of {auc:.3f} for {disease} classification")
        
        # Clinical relevance finding
        findings.append(f"Metabolomic changes correlated with {disease} severity and clinical outcomes")
        
        return findings[:3]  # Return top 3 findings


class ClinicalTrialMockGenerator:
    """
    Generates realistic clinical trial data for testing biomedical integration scenarios.
    """
    
    TRIAL_PHASES = ['Phase I', 'Phase II', 'Phase III', 'Phase IV']
    TRIAL_STATUS = ['recruiting', 'active', 'completed', 'terminated', 'suspended']
    
    INTERVENTION_TYPES = {
        'metabolomics': [
            'Metabolomic biomarker validation study',
            'Targeted metabolomics for treatment monitoring',
            'Metabolomic-guided therapy selection',
            'Pharmacometabolomics study'
        ],
        'drug': [
            'Novel antidiabetic compound',
            'Cardiovascular risk reduction therapy',
            'Cancer immunotherapy',
            'Personalized medicine approach'
        ],
        'lifestyle': [
            'Dietary intervention with metabolomic monitoring',
            'Exercise program with biomarker tracking',
            'Weight management with metabolomic assessment',
            'Stress reduction intervention'
        ]
    }
    
    @classmethod
    def generate_clinical_trial(cls, 
                              disease: str = 'diabetes',
                              intervention_type: str = 'metabolomics') -> ClinicalTrialData:
        """Generate realistic clinical trial data."""
        
        # Generate NCT ID
        nct_id = f"NCT{random.randint(10000000, 99999999):08d}"
        
        # Generate title
        interventions = cls.INTERVENTION_TYPES.get(intervention_type, cls.INTERVENTION_TYPES['metabolomics'])
        intervention = random.choice(interventions)
        title = f"{intervention} in {disease.replace('_', ' ').title()} Patients"
        
        # Generate trial characteristics
        phase = random.choice(cls.TRIAL_PHASES)
        status = random.choice(cls.TRIAL_STATUS)
        study_type = random.choice(['Interventional', 'Observational'])
        
        # Generate enrollment and dates
        enrollment = random.randint(50, 500)
        start_date = datetime.now() - timedelta(days=random.randint(30, 1095))
        
        completion_date = None
        if status == 'completed':
            completion_date = start_date + timedelta(days=random.randint(90, 730))
        
        # Generate endpoints
        primary_endpoints = [
            f"Change in {disease}-specific metabolomic biomarkers from baseline to week 12",
            f"Efficacy of {intervention.lower()} measured by clinical outcomes",
            "Safety and tolerability assessment"
        ][:random.randint(1, 3)]
        
        secondary_endpoints = [
            "Quality of life assessment",
            "Healthcare utilization analysis",
            "Cost-effectiveness evaluation",
            "Long-term safety follow-up"
        ][:random.randint(1, 4)]
        
        # Generate sponsor and locations
        sponsors = [
            "National Institutes of Health",
            "Mayo Clinic",
            "Harvard Medical School",
            "Stanford University",
            "Pfizer Inc.",
            "Novartis Pharmaceuticals",
            "Johnson & Johnson"
        ]
        sponsor = random.choice(sponsors)
        
        locations = random.sample([
            "Boston, MA", "New York, NY", "Los Angeles, CA", "Chicago, IL",
            "Houston, TX", "Philadelphia, PA", "Phoenix, AZ", "San Francisco, CA"
        ], random.randint(1, 5))
        
        # Generate inclusion/exclusion criteria
        inclusion_criteria = [
            f"Confirmed diagnosis of {disease}",
            "Age 18-75 years",
            "Stable medication regimen for ≥3 months",
            "Ability to provide informed consent"
        ]
        
        exclusion_criteria = [
            "Pregnancy or nursing",
            "Severe renal or hepatic impairment",
            "Active substance abuse",
            "Participation in another clinical trial within 30 days"
        ]
        
        # Generate biomarkers and analytical platforms
        biomarkers_measured = random.sample([
            'glucose', 'HbA1c', 'cholesterol', 'triglycerides', 'CRP',
            'insulin', 'lactate', 'creatinine', 'BUN', 'ALT', 'AST'
        ], random.randint(3, 8))
        
        analytical_platforms = random.sample([
            'LC-MS/MS', 'GC-MS', 'NMR spectroscopy', 'Immunoassays', 
            'Clinical chemistry analyzers', 'Point-of-care testing'
        ], random.randint(2, 4))
        
        return ClinicalTrialData(
            nct_id=nct_id,
            title=title,
            status=status,
            phase=phase,
            study_type=study_type,
            condition=disease,
            intervention=intervention,
            primary_endpoints=primary_endpoints,
            secondary_endpoints=secondary_endpoints,
            enrollment=enrollment,
            start_date=start_date.strftime('%Y-%m-%d'),
            completion_date=completion_date.strftime('%Y-%m-%d') if completion_date else None,
            sponsor=sponsor,
            collaborators=random.sample(["University Medical Center", "Research Institute"], 
                                       random.randint(0, 2)),
            locations=locations,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            biomarkers_measured=biomarkers_measured,
            analytical_platforms=analytical_platforms
        )


class LiteratureSearchMockData:
    """
    Generates mock literature search results for testing search and retrieval functionality.
    """
    
    @classmethod
    def generate_pubmed_search_results(cls, 
                                     query: str,
                                     num_results: int = 20) -> Dict[str, Any]:
        """Generate mock PubMed search results."""
        
        # Analyze query to determine relevant papers
        query_lower = query.lower()
        
        # Determine primary research area
        if any(term in query_lower for term in ['metabolomics', 'metabolite', 'biomarker']):
            research_area = 'metabolomics'
        elif any(term in query_lower for term in ['proteomics', 'protein']):
            research_area = 'proteomics'
        elif any(term in query_lower for term in ['genomics', 'gene', 'dna']):
            research_area = 'genomics'
        else:
            research_area = 'metabolomics'  # Default
        
        # Determine disease focus
        disease = 'diabetes'  # Default
        for disease_term in ['diabetes', 'cardiovascular', 'cancer', 'kidney', 'liver']:
            if disease_term in query_lower:
                disease = disease_term
                break
        
        # Generate papers
        papers = []
        for i in range(num_results):
            paper = BiomedicalPaperGenerator.generate_research_paper(
                research_area=research_area,
                disease=disease,
                complexity=random.choice(['simple', 'medium', 'complex'])
            )
            papers.append(paper)
        
        # Generate search metadata
        search_results = {
            'query': query,
            'total_results': num_results,
            'search_date': datetime.now().strftime('%Y-%m-%d'),
            'database': 'PubMed',
            'papers': papers,
            'search_statistics': {
                'by_year': cls._generate_year_distribution(papers),
                'by_journal': cls._generate_journal_distribution(papers),
                'by_research_area': cls._count_by_attribute(papers, 'research_area'),
                'by_study_type': cls._count_by_attribute(papers, 'study_type')
            },
            'related_terms': cls._generate_related_terms(query_lower),
            'mesh_terms': cls._extract_common_mesh_terms(papers)
        }
        
        return search_results
    
    @classmethod
    def _generate_year_distribution(cls, papers: List[ResearchPaper]) -> Dict[int, int]:
        """Generate year distribution of papers."""
        year_counts = {}
        for paper in papers:
            year = paper.publication_year
            year_counts[year] = year_counts.get(year, 0) + 1
        return dict(sorted(year_counts.items()))
    
    @classmethod
    def _generate_journal_distribution(cls, papers: List[ResearchPaper]) -> Dict[str, int]:
        """Generate journal distribution of papers."""
        journal_counts = {}
        for paper in papers:
            journal = paper.journal
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
        return dict(sorted(journal_counts.items(), key=lambda x: x[1], reverse=True))
    
    @classmethod
    def _count_by_attribute(cls, papers: List[ResearchPaper], attribute: str) -> Dict[str, int]:
        """Count papers by specified attribute."""
        counts = {}
        for paper in papers:
            value = getattr(paper, attribute, 'Unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    @classmethod
    def _generate_related_terms(cls, query: str) -> List[str]:
        """Generate related search terms."""
        base_terms = {
            'metabolomics': ['biomarkers', 'mass spectrometry', 'LC-MS', 'pathway analysis'],
            'diabetes': ['insulin resistance', 'glucose metabolism', 'HbA1c', 'type 2 diabetes'],
            'cardiovascular': ['heart disease', 'cholesterol', 'lipids', 'atherosclerosis'],
            'cancer': ['tumor metabolism', 'oncology', 'biomarkers', 'chemotherapy']
        }
        
        related = []
        for term, related_list in base_terms.items():
            if term in query:
                related.extend(related_list[:2])
        
        return related[:5]
    
    @classmethod
    def _extract_common_mesh_terms(cls, papers: List[ResearchPaper]) -> List[str]:
        """Extract most common MeSH terms from papers."""
        mesh_counts = {}
        for paper in papers:
            for mesh_term in paper.mesh_terms:
                mesh_counts[mesh_term] = mesh_counts.get(mesh_term, 0) + 1
        
        return [term for term, count in sorted(mesh_counts.items(), 
                                             key=lambda x: x[1], reverse=True)][:10]


# Pytest fixtures for integration
@pytest.fixture
def biomedical_paper_generator():
    """Provide biomedical paper generator."""
    return BiomedicalPaperGenerator()

@pytest.fixture
def sample_research_papers():
    """Provide collection of sample research papers."""
    papers = []
    diseases = ['diabetes', 'cardiovascular', 'cancer']
    areas = ['metabolomics', 'clinical', 'bioinformatics']
    
    for disease in diseases:
        for area in areas:
            paper = BiomedicalPaperGenerator.generate_research_paper(
                research_area=area,
                disease=disease,
                complexity='medium'
            )
            papers.append(paper)
    
    return papers

@pytest.fixture
def clinical_trial_generator():
    """Provide clinical trial generator."""
    return ClinicalTrialMockGenerator()

@pytest.fixture
def sample_clinical_trials():
    """Provide sample clinical trials."""
    trials = []
    diseases = ['diabetes', 'cardiovascular_disease', 'cancer']
    intervention_types = ['metabolomics', 'drug', 'lifestyle']
    
    for disease in diseases:
        for intervention in intervention_types:
            trial = ClinicalTrialMockGenerator.generate_clinical_trial(
                disease=disease,
                intervention_type=intervention
            )
            trials.append(trial)
    
    return trials

@pytest.fixture
def literature_search_mock():
    """Provide literature search mock data generator."""
    return LiteratureSearchMockData()

@pytest.fixture
def pubmed_search_results():
    """Provide sample PubMed search results."""
    queries = [
        "metabolomics diabetes biomarkers",
        "cardiovascular disease lipidomics",
        "cancer metabolic pathways",
        "kidney disease metabolomics"
    ]
    
    results = {}
    for query in queries:
        results[query] = LiteratureSearchMockData.generate_pubmed_search_results(
            query=query,
            num_results=15
        )
    
    return results

@pytest.fixture
def multi_omics_integration_data():
    """Provide multi-omics integration test data."""
    
    # Generate metabolomics data
    metabolomics_paper = BiomedicalPaperGenerator.generate_research_paper(
        research_area='metabolomics',
        disease='diabetes',
        complexity='complex'
    )
    
    # Generate proteomics data  
    proteomics_paper = BiomedicalPaperGenerator.generate_research_paper(
        research_area='proteomics',
        disease='diabetes',
        complexity='complex'
    )
    
    # Generate integrated analysis
    integration_data = {
        'study_design': 'multi_omics_integration',
        'omics_layers': ['metabolomics', 'proteomics', 'clinical'],
        'sample_size': 200,
        'integration_methods': ['network analysis', 'pathway enrichment', 'machine learning'],
        'key_findings': [
            'Cross-omics correlations identified novel disease mechanisms',
            'Integrated biomarker panel improved diagnostic accuracy',
            'Systems-level analysis revealed therapeutic targets'
        ],
        'metabolomics_data': metabolomics_paper,
        'proteomics_data': proteomics_paper,
        'integration_results': {
            'shared_pathways': ['Glucose metabolism', 'Insulin signaling', 'Inflammation'],
            'cross_correlations': 0.75,
            'combined_auc': 0.89
        }
    }
    
    return integration_data