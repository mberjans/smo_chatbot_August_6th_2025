#!/usr/bin/env python3
"""
Comprehensive Biomedical Test Fixtures for Clinical Metabolomics Oracle.

This module provides specialized test fixtures, mock data generators, and utilities 
specifically designed for biomedical content testing in the Clinical Metabolomics Oracle
LightRAG integration. It extends the base fixtures with biomedical-specific content
including clinical metabolomics research data, disease-specific biomarkers, and
realistic research scenarios.

Components:
- ClinicalMetabolomicsDataGenerator: Generates realistic metabolomics research data
- BiomedicalKnowledgeGenerator: Creates knowledge graph entities and relationships
- ResearchScenarioBuilder: Builds comprehensive research test scenarios
- MetabolomicsPlatformSimulator: Simulates different analytical platforms
- BiomarkerValidationFixtures: Provides disease biomarker test data
- ClinicalStudyGenerator: Creates realistic clinical study datasets

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import random
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Import research categorizer for category-specific fixtures
from lightrag_integration.research_categorizer import ResearchCategory


@dataclass
class MetaboliteData:
    """Represents a metabolite with clinical data."""
    name: str
    hmdb_id: str
    kegg_id: Optional[str] = None
    chebi_id: Optional[str] = None
    molecular_formula: str = ""
    molecular_weight: float = 0.0
    concentration_range: Tuple[float, float] = (0.0, 1.0)
    concentration_units: str = "µM"
    pathways: List[str] = field(default_factory=list)
    disease_associations: Dict[str, str] = field(default_factory=dict)
    analytical_platforms: List[str] = field(default_factory=list)
    clinical_significance: str = ""
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metabolite metadata as dictionary."""
        return {
            'name': self.name,
            'hmdb_id': self.hmdb_id,
            'kegg_id': self.kegg_id,
            'chebi_id': self.chebi_id,
            'molecular_formula': self.molecular_formula,
            'molecular_weight': self.molecular_weight,
            'pathways': self.pathways,
            'disease_associations': self.disease_associations,
            'analytical_platforms': self.analytical_platforms,
            'clinical_significance': self.clinical_significance
        }


@dataclass
class ClinicalStudyData:
    """Represents clinical study metadata and results."""
    study_id: str
    title: str
    study_type: str  # 'case_control', 'cohort', 'cross_sectional', 'intervention'
    disease_condition: str
    sample_size: Dict[str, int]  # {'patients': 100, 'controls': 50}
    demographics: Dict[str, Any]
    analytical_platform: str
    key_findings: List[str]
    biomarkers_identified: List[str]
    statistical_methods: List[str]
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    publication_year: int
    journal: str
    doi: str
    
    @property
    def summary(self) -> str:
        """Generate study summary text."""
        total_participants = sum(self.sample_size.values())
        return f"""
        Clinical Study: {self.title}
        Study Type: {self.study_type}
        Condition: {self.disease_condition}
        Participants: {total_participants} ({self.sample_size})
        Platform: {self.analytical_platform}
        Key Findings: {'; '.join(self.key_findings[:3])}
        Biomarkers: {len(self.biomarkers_identified)} identified
        """


class ClinicalMetabolomicsDataGenerator:
    """
    Generates comprehensive clinical metabolomics test data including realistic
    metabolite profiles, clinical study datasets, and research scenarios.
    """
    
    # Comprehensive metabolomics database
    METABOLITE_DATABASE = {
        'glucose': MetaboliteData(
            name='Glucose',
            hmdb_id='HMDB0000122',
            kegg_id='C00031',
            chebi_id='CHEBI:17234',
            molecular_formula='C6H12O6',
            molecular_weight=180.16,
            concentration_range=(3.9, 6.1),
            concentration_units='mM',
            pathways=['Glycolysis', 'Gluconeogenesis', 'Pentose phosphate pathway'],
            disease_associations={
                'diabetes': 'elevated',
                'hypoglycemia': 'decreased',
                'metabolic_syndrome': 'elevated'
            },
            analytical_platforms=['LC-MS/MS', 'GC-MS', 'Enzymatic assay'],
            clinical_significance='Primary biomarker for glucose metabolism disorders'
        ),
        'lactate': MetaboliteData(
            name='Lactate',
            hmdb_id='HMDB0000190',
            kegg_id='C00186',
            chebi_id='CHEBI:16651',
            molecular_formula='C3H6O3',
            molecular_weight=90.08,
            concentration_range=(0.5, 2.2),
            concentration_units='mM',
            pathways=['Glycolysis', 'Cori cycle'],
            disease_associations={
                'sepsis': 'elevated',
                'heart_failure': 'elevated',
                'cancer': 'elevated'
            },
            analytical_platforms=['LC-MS/MS', 'Enzymatic assay'],
            clinical_significance='Marker of anaerobic metabolism and tissue hypoxia'
        ),
        'creatinine': MetaboliteData(
            name='Creatinine',
            hmdb_id='HMDB0000562',
            kegg_id='C00791',
            chebi_id='CHEBI:16737',
            molecular_formula='C4H7N3O',
            molecular_weight=113.12,
            concentration_range=(0.6, 1.2),
            concentration_units='mg/dL',
            pathways=['Creatine metabolism'],
            disease_associations={
                'kidney_disease': 'elevated',
                'muscle_disorders': 'altered'
            },
            analytical_platforms=['LC-MS/MS', 'HPLC', 'Colorimetric assay'],
            clinical_significance='Gold standard marker for kidney function'
        ),
        'cholesterol': MetaboliteData(
            name='Cholesterol',
            hmdb_id='HMDB0000067',
            kegg_id='C00187',
            chebi_id='CHEBI:16113',
            molecular_formula='C27H46O',
            molecular_weight=386.65,
            concentration_range=(150, 200),
            concentration_units='mg/dL',
            pathways=['Cholesterol biosynthesis', 'Bile acid synthesis'],
            disease_associations={
                'cardiovascular_disease': 'elevated',
                'atherosclerosis': 'elevated'
            },
            analytical_platforms=['LC-MS/MS', 'Enzymatic assay'],
            clinical_significance='Key lipid biomarker for cardiovascular risk'
        ),
        'urea': MetaboliteData(
            name='Urea',
            hmdb_id='HMDB0000294',
            kegg_id='C00086',
            chebi_id='CHEBI:16199',
            molecular_formula='CH4N2O',
            molecular_weight=60.06,
            concentration_range=(2.5, 6.5),
            concentration_units='mM',
            pathways=['Urea cycle'],
            disease_associations={
                'kidney_disease': 'elevated',
                'liver_disease': 'decreased'
            },
            analytical_platforms=['LC-MS/MS', 'Enzymatic assay'],
            clinical_significance='Marker of nitrogen metabolism and kidney function'
        ),
        'bilirubin': MetaboliteData(
            name='Bilirubin',
            hmdb_id='HMDB0000054',
            kegg_id='C00486',
            chebi_id='CHEBI:16990',
            molecular_formula='C33H36N4O6',
            molecular_weight=584.66,
            concentration_range=(0.3, 1.2),
            concentration_units='mg/dL',
            pathways=['Heme degradation'],
            disease_associations={
                'liver_disease': 'elevated',
                'hemolysis': 'elevated',
                'gilbert_syndrome': 'elevated'
            },
            analytical_platforms=['LC-MS/MS', 'HPLC', 'Spectrophotometry'],
            clinical_significance='Primary marker of liver function and hemolysis'
        ),
        'tmao': MetaboliteData(
            name='Trimethylamine N-oxide',
            hmdb_id='HMDB0000925',
            kegg_id='C01104',
            chebi_id='CHEBI:15724',
            molecular_formula='C3H9NO',
            molecular_weight=75.11,
            concentration_range=(1.0, 10.0),
            concentration_units='μM',
            pathways=['Choline metabolism', 'Carnitine metabolism'],
            disease_associations={
                'cardiovascular_disease': 'elevated',
                'kidney_disease': 'elevated',
                'atherosclerosis': 'elevated'
            },
            analytical_platforms=['LC-MS/MS'],
            clinical_significance='Gut microbiome-derived cardiovascular risk marker'
        ),
        'acetate': MetaboliteData(
            name='Acetate',
            hmdb_id='HMDB0000042',
            kegg_id='C00033',
            chebi_id='CHEBI:30089',
            molecular_formula='C2H4O2',
            molecular_weight=60.05,
            concentration_range=(10, 100),
            concentration_units='μM',
            pathways=['Fatty acid synthesis', 'Acetyl-CoA metabolism'],
            disease_associations={
                'metabolic_syndrome': 'altered',
                'diabetes': 'altered'
            },
            analytical_platforms=['LC-MS/MS', 'GC-MS', 'NMR'],
            clinical_significance='Short-chain fatty acid and energy metabolism marker'
        ),
        'alanine': MetaboliteData(
            name='Alanine',
            hmdb_id='HMDB0000161',
            kegg_id='C00041',
            chebi_id='CHEBI:16977',
            molecular_formula='C3H7NO2',
            molecular_weight=89.09,
            concentration_range=(200, 500),
            concentration_units='μM',
            pathways=['Amino acid metabolism', 'Glucose-alanine cycle'],
            disease_associations={
                'diabetes': 'elevated',
                'insulin_resistance': 'elevated'
            },
            analytical_platforms=['LC-MS/MS', 'GC-MS', 'Amino acid analyzer'],
            clinical_significance='Branched-chain amino acid and glucose metabolism marker'
        ),
        'glutamine': MetaboliteData(
            name='Glutamine',
            hmdb_id='HMDB0000641',
            kegg_id='C00064',
            chebi_id='CHEBI:18050',
            molecular_formula='C5H10N2O3',
            molecular_weight=146.14,
            concentration_range=(400, 800),
            concentration_units='μM',
            pathways=['Amino acid metabolism', 'Glutamate metabolism'],
            disease_associations={
                'cancer': 'depleted',
                'critical_illness': 'depleted'
            },
            analytical_platforms=['LC-MS/MS', 'GC-MS', 'Amino acid analyzer'],
            clinical_significance='Most abundant free amino acid, immune function marker'
        )
    }
    
    # Disease-specific metabolite panels
    DISEASE_PANELS = {
        'diabetes': {
            'primary_markers': ['glucose', 'lactate', 'alanine'],
            'secondary_markers': ['acetate', 'urea'],
            'pathways': ['Glycolysis', 'Gluconeogenesis', 'Amino acid metabolism'],
            'typical_changes': {
                'glucose': 'elevated',
                'lactate': 'elevated',
                'alanine': 'elevated'
            }
        },
        'cardiovascular_disease': {
            'primary_markers': ['cholesterol', 'tmao', 'lactate'],
            'secondary_markers': ['glucose', 'creatinine'],
            'pathways': ['Cholesterol metabolism', 'Choline metabolism', 'Energy metabolism'],
            'typical_changes': {
                'cholesterol': 'elevated',
                'tmao': 'elevated',
                'lactate': 'elevated'
            }
        },
        'kidney_disease': {
            'primary_markers': ['creatinine', 'urea', 'tmao'],
            'secondary_markers': ['glucose', 'bilirubin'],
            'pathways': ['Urea cycle', 'Creatine metabolism'],
            'typical_changes': {
                'creatinine': 'elevated',
                'urea': 'elevated',
                'tmao': 'elevated'
            }
        },
        'liver_disease': {
            'primary_markers': ['bilirubin', 'lactate', 'urea'],
            'secondary_markers': ['glucose', 'alanine'],
            'pathways': ['Heme metabolism', 'Urea cycle', 'Gluconeogenesis'],
            'typical_changes': {
                'bilirubin': 'elevated',
                'lactate': 'elevated',
                'urea': 'decreased'
            }
        },
        'cancer': {
            'primary_markers': ['lactate', 'glutamine', 'glucose'],
            'secondary_markers': ['alanine', 'acetate'],
            'pathways': ['Glycolysis', 'Glutamine metabolism', 'Warburg effect'],
            'typical_changes': {
                'lactate': 'elevated',
                'glutamine': 'depleted',
                'glucose': 'elevated'
            }
        }
    }
    
    # Analytical platform specifications
    ANALYTICAL_PLATFORMS = {
        'LC-MS/MS': {
            'full_name': 'Liquid Chromatography Tandem Mass Spectrometry',
            'typical_compounds': ['glucose', 'lactate', 'amino_acids', 'lipids'],
            'sensitivity': 'high',
            'throughput': 'medium',
            'cost': 'high',
            'advantages': ['High specificity', 'Quantitative', 'Wide dynamic range'],
            'limitations': ['Expensive', 'Requires expertise', 'Ion suppression']
        },
        'GC-MS': {
            'full_name': 'Gas Chromatography Mass Spectrometry',
            'typical_compounds': ['volatile_organic_acids', 'amino_acids_derivatized', 'fatty_acids'],
            'sensitivity': 'high',
            'throughput': 'medium',
            'cost': 'medium',
            'advantages': ['Good separation', 'Reproducible', 'Established libraries'],
            'limitations': ['Requires derivatization', 'Limited to volatile compounds']
        },
        'NMR': {
            'full_name': 'Nuclear Magnetic Resonance Spectroscopy',
            'typical_compounds': ['glucose', 'lactate', 'acetate', 'amino_acids'],
            'sensitivity': 'low',
            'throughput': 'high',
            'cost': 'medium',
            'advantages': ['Non-destructive', 'Quantitative', 'No sample prep'],
            'limitations': ['Low sensitivity', 'Limited resolution', 'Water interference']
        },
        'HPLC': {
            'full_name': 'High Performance Liquid Chromatography',
            'typical_compounds': ['amino_acids', 'organic_acids', 'vitamins'],
            'sensitivity': 'medium',
            'throughput': 'high',
            'cost': 'low',
            'advantages': ['Cost effective', 'Reliable', 'Easy operation'],
            'limitations': ['Limited specificity', 'Matrix effects', 'Lower sensitivity']
        }
    }
    
    @classmethod
    def generate_clinical_study_dataset(cls, 
                                      disease: str = 'diabetes',
                                      sample_size: int = 100,
                                      study_type: str = 'case_control') -> ClinicalStudyData:
        """Generate realistic clinical study dataset."""
        
        # Get disease panel information
        panel_info = cls.DISEASE_PANELS.get(disease, cls.DISEASE_PANELS['diabetes'])
        
        # Generate study metadata
        study_id = f"CMO-{disease.upper()}-{random.randint(1000, 9999)}"
        
        # Generate sample size distribution
        if study_type == 'case_control':
            patients = sample_size // 2
            controls = sample_size - patients
            sample_size_dict = {'patients': patients, 'controls': controls}
        elif study_type == 'cohort':
            sample_size_dict = {'participants': sample_size}
        else:
            sample_size_dict = {'participants': sample_size}
        
        # Generate demographics
        demographics = {
            'age_mean': random.uniform(45, 70),
            'age_std': random.uniform(8, 15),
            'gender_distribution': {
                'male': random.uniform(40, 60),
                'female': 100 - random.uniform(40, 60)
            },
            'bmi_mean': random.uniform(24, 32),
            'ethnicity': {
                'caucasian': random.uniform(60, 80),
                'african_american': random.uniform(10, 20),
                'hispanic': random.uniform(5, 15),
                'asian': random.uniform(5, 10),
                'other': random.uniform(1, 5)
            }
        }
        
        # Generate analytical platform
        platform = random.choice(list(cls.ANALYTICAL_PLATFORMS.keys()))
        
        # Generate key findings
        key_findings = [
            f"Significant alterations in {', '.join(panel_info['primary_markers'][:3])} levels",
            f"Dysregulation of {panel_info['pathways'][0]} pathway",
            f"Strong correlation between {panel_info['primary_markers'][0]} and disease severity"
        ]
        
        # Generate biomarkers
        biomarkers = panel_info['primary_markers'] + random.sample(
            panel_info['secondary_markers'], 
            min(2, len(panel_info['secondary_markers']))
        )
        
        # Generate statistical methods
        statistical_methods = random.sample([
            'Student t-test', 'Mann-Whitney U test', 'ANOVA', 'Kruskal-Wallis test',
            'Pearson correlation', 'Spearman correlation', 'Linear regression',
            'Logistic regression', 'Multiple comparison correction (FDR)',
            'Principal component analysis', 'Partial least squares discriminant analysis'
        ], random.randint(3, 6))
        
        # Generate p-values and effect sizes
        p_values = {}
        effect_sizes = {}
        
        for marker in biomarkers:
            p_values[marker] = random.uniform(0.001, 0.049)  # Significant values
            effect_sizes[marker] = random.uniform(0.3, 2.5)  # Cohen's d or fold change
        
        # Generate publication details
        year = random.randint(2018, 2024)
        journals = [
            'Nature Medicine', 'The Lancet', 'New England Journal of Medicine',
            'Journal of Clinical Investigation', 'Clinical Chemistry',
            'Metabolomics', 'Analytical Chemistry', 'Journal of Proteome Research',
            'Biomarkers in Medicine', 'Clinical Biochemistry'
        ]
        journal = random.choice(journals)
        doi = f"10.1000/test.{year}.{random.randint(1000, 9999)}"
        
        return ClinicalStudyData(
            study_id=study_id,
            title=f"Metabolomic Analysis of {disease.replace('_', ' ').title()} Using {platform}",
            study_type=study_type,
            disease_condition=disease,
            sample_size=sample_size_dict,
            demographics=demographics,
            analytical_platform=platform,
            key_findings=key_findings,
            biomarkers_identified=biomarkers,
            statistical_methods=statistical_methods,
            p_values=p_values,
            effect_sizes=effect_sizes,
            publication_year=year,
            journal=journal,
            doi=doi
        )
    
    @classmethod
    def generate_metabolite_concentration_data(cls, 
                                             metabolites: List[str],
                                             n_samples: int = 100,
                                             disease_state: str = 'healthy',
                                             add_noise: bool = True) -> Dict[str, Any]:
        """Generate realistic metabolite concentration data."""
        
        concentration_data = {
            'sample_metadata': {
                'n_samples': n_samples,
                'disease_state': disease_state,
                'timestamp': time.time(),
                'platform': random.choice(list(cls.ANALYTICAL_PLATFORMS.keys()))
            },
            'metabolite_data': {},
            'quality_metrics': {}
        }
        
        for metabolite_name in metabolites:
            if metabolite_name not in cls.METABOLITE_DATABASE:
                continue
                
            metabolite = cls.METABOLITE_DATABASE[metabolite_name]
            base_range = metabolite.concentration_range
            
            # Adjust concentrations based on disease state
            disease_effect = cls._get_disease_effect(metabolite_name, disease_state)
            adjusted_range = cls._apply_disease_effect(base_range, disease_effect)
            
            # Generate concentrations with biological variation
            concentrations = []
            for _ in range(n_samples):
                # Log-normal distribution for biological realism
                mean_conc = np.mean(adjusted_range)
                std_conc = (adjusted_range[1] - adjusted_range[0]) / 4
                
                conc = np.random.lognormal(
                    np.log(mean_conc), 
                    std_conc / mean_conc
                )
                
                # Add analytical noise if requested
                if add_noise:
                    noise_factor = random.uniform(0.95, 1.05)  # ±5% analytical variation
                    conc *= noise_factor
                
                concentrations.append(max(0, conc))  # Ensure non-negative
            
            # Calculate quality metrics
            cv = np.std(concentrations) / np.mean(concentrations) * 100
            detection_rate = sum(1 for c in concentrations if c > base_range[0] * 0.1) / n_samples
            
            concentration_data['metabolite_data'][metabolite_name] = {
                'concentrations': concentrations,
                'units': metabolite.concentration_units,
                'mean': np.mean(concentrations),
                'median': np.median(concentrations),
                'std': np.std(concentrations),
                'cv_percent': cv,
                'min': np.min(concentrations),
                'max': np.max(concentrations),
                'reference_range': base_range,
                'disease_effect': disease_effect
            }
            
            concentration_data['quality_metrics'][metabolite_name] = {
                'cv_percent': cv,
                'detection_rate': detection_rate,
                'outliers': len([c for c in concentrations if abs(c - np.mean(concentrations)) > 3 * np.std(concentrations)]),
                'missing_values': 0,  # For simplicity, no missing values in generated data
                'quality_flag': 'pass' if cv < 30 and detection_rate > 0.8 else 'warning'
            }
        
        return concentration_data
    
    @classmethod
    def _get_disease_effect(cls, metabolite_name: str, disease_state: str) -> str:
        """Get disease effect on metabolite concentration."""
        if disease_state == 'healthy':
            return 'normal'
        
        for disease, panel in cls.DISEASE_PANELS.items():
            if disease_state in disease or disease in disease_state:
                return panel['typical_changes'].get(metabolite_name, 'normal')
        
        return 'normal'
    
    @classmethod
    def _apply_disease_effect(cls, base_range: Tuple[float, float], effect: str) -> Tuple[float, float]:
        """Apply disease effect to concentration range."""
        low, high = base_range
        
        if effect == 'elevated':
            return (high * 0.8, high * 2.5)
        elif effect == 'decreased':
            return (low * 0.1, high * 0.5)
        elif effect == 'depleted':
            return (low * 0.05, high * 0.3)
        else:  # normal
            return base_range
    
    @classmethod
    def generate_pathway_analysis_data(cls, disease: str = 'diabetes') -> Dict[str, Any]:
        """Generate pathway analysis results data."""
        
        # Metabolic pathways database
        pathways_db = {
            'Glycolysis / Gluconeogenesis': {
                'kegg_id': 'hsa00010',
                'metabolites': ['glucose', 'lactate', 'alanine'],
                'enzymes': ['HK1', 'PKM', 'LDHA', 'G6PC'],
                'description': 'Central glucose metabolism pathway'
            },
            'Citrate cycle (TCA cycle)': {
                'kegg_id': 'hsa00020',
                'metabolites': ['acetate', 'lactate'],
                'enzymes': ['CS', 'ACO2', 'IDH1', 'OGDH'],
                'description': 'Central energy metabolism pathway'
            },
            'Amino acid metabolism': {
                'kegg_id': 'hsa01230',
                'metabolites': ['alanine', 'glutamine'],
                'enzymes': ['ALT1', 'AST1', 'GLUL', 'GLS'],
                'description': 'Amino acid synthesis and degradation'
            },
            'Cholesterol metabolism': {
                'kegg_id': 'hsa00100',
                'metabolites': ['cholesterol'],
                'enzymes': ['HMGCR', 'CYP7A1', 'LDLR'],
                'description': 'Cholesterol synthesis and regulation'
            },
            'Urea cycle': {
                'kegg_id': 'hsa00220',
                'metabolites': ['urea'],
                'enzymes': ['CPS1', 'OTC', 'ASS1', 'ARG1'],
                'description': 'Nitrogen disposal pathway'
            }
        }
        
        # Get relevant pathways for disease
        disease_panel = cls.DISEASE_PANELS.get(disease, cls.DISEASE_PANELS['diabetes'])
        relevant_pathways = disease_panel['pathways']
        
        pathway_results = {
            'analysis_metadata': {
                'disease': disease,
                'analysis_date': time.strftime('%Y-%m-%d'),
                'method': 'Over-representation analysis',
                'database': 'KEGG',
                'significance_threshold': 0.05
            },
            'enriched_pathways': [],
            'pathway_details': {},
            'network_statistics': {}
        }
        
        # Generate enrichment results
        for pathway_name in relevant_pathways:
            if pathway_name in pathways_db:
                pathway_info = pathways_db[pathway_name]
                
                # Generate realistic enrichment statistics
                p_value = random.uniform(0.001, 0.049)  # Significant
                fold_enrichment = random.uniform(1.5, 4.0)
                genes_in_pathway = random.randint(15, 80)
                genes_in_data = random.randint(5, min(20, genes_in_pathway))
                
                enrichment_result = {
                    'pathway_name': pathway_name,
                    'kegg_id': pathway_info['kegg_id'],
                    'description': pathway_info['description'],
                    'p_value': p_value,
                    'adjusted_p_value': p_value * 1.2,  # Simple correction
                    'fold_enrichment': fold_enrichment,
                    'genes_in_pathway': genes_in_pathway,
                    'genes_in_data': genes_in_data,
                    'genes_list': pathway_info['enzymes'],
                    'metabolites_involved': pathway_info['metabolites']
                }
                
                pathway_results['enriched_pathways'].append(enrichment_result)
                pathway_results['pathway_details'][pathway_name] = pathway_info
        
        # Sort by p-value
        pathway_results['enriched_pathways'].sort(key=lambda x: x['p_value'])
        
        # Generate network statistics
        pathway_results['network_statistics'] = {
            'total_pathways_tested': len(pathways_db),
            'significant_pathways': len(pathway_results['enriched_pathways']),
            'total_genes': sum(p['genes_in_pathway'] for p in pathway_results['enriched_pathways']),
            'overlap_coefficient': random.uniform(0.2, 0.6),
            'network_density': random.uniform(0.1, 0.3)
        }
        
        return pathway_results
    
    @classmethod
    def generate_biomarker_validation_data(cls, 
                                         biomarkers: List[str],
                                         validation_type: str = 'roc_analysis') -> Dict[str, Any]:
        """Generate biomarker validation study data."""
        
        validation_data = {
            'validation_metadata': {
                'validation_type': validation_type,
                'biomarkers': biomarkers,
                'validation_date': time.strftime('%Y-%m-%d'),
                'cohort_size': random.randint(200, 800)
            },
            'biomarker_performance': {},
            'combined_panel_performance': {},
            'clinical_utility': {}
        }
        
        # Generate performance metrics for each biomarker
        for biomarker in biomarkers:
            if biomarker not in cls.METABOLITE_DATABASE:
                continue
            
            # Generate ROC analysis results
            auc = random.uniform(0.65, 0.95)  # Realistic AUC range
            sensitivity = random.uniform(0.70, 0.95)
            specificity = random.uniform(0.70, 0.95)
            
            # Calculate derived metrics
            ppv = sensitivity * 0.3 / (sensitivity * 0.3 + (1 - specificity) * 0.7)  # Assume 30% prevalence
            npv = specificity * 0.7 / (specificity * 0.7 + (1 - sensitivity) * 0.3)
            
            validation_data['biomarker_performance'][biomarker] = {
                'auc': auc,
                'auc_ci_lower': auc - random.uniform(0.05, 0.15),
                'auc_ci_upper': auc + random.uniform(0.05, 0.15),
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'optimal_cutoff': random.uniform(0.1, 10.0),
                'p_value': random.uniform(0.001, 0.01)
            }
        
        # Generate combined panel performance
        if len(biomarkers) > 1:
            combined_auc = min(0.98, max(validation_data['biomarker_performance'].values(), 
                                      key=lambda x: x['auc'])['auc'] + random.uniform(0.02, 0.08))
            
            validation_data['combined_panel_performance'] = {
                'panel_auc': combined_auc,
                'improvement_over_best_single': combined_auc - max(
                    v['auc'] for v in validation_data['biomarker_performance'].values()
                ),
                'sensitivity': random.uniform(0.85, 0.98),
                'specificity': random.uniform(0.85, 0.98),
                'model_type': random.choice(['Logistic regression', 'Random forest', 'SVM']),
                'cross_validation_auc': combined_auc - random.uniform(0.01, 0.03)
            }
        
        # Generate clinical utility metrics
        validation_data['clinical_utility'] = {
            'net_benefit': random.uniform(0.1, 0.4),
            'clinical_impact': random.choice(['High', 'Moderate', 'Low']),
            'cost_effectiveness': random.uniform(0.5, 2.0),  # Cost per QALY saved
            'implementation_feasibility': random.choice(['High', 'Moderate', 'Low']),
            'regulatory_pathway': random.choice(['FDA 510(k)', 'CLIA', 'LDT', 'Research use only'])
        }
        
        return validation_data


# Additional fixtures for pytest integration
@pytest.fixture
def clinical_metabolomics_data():
    """Provide clinical metabolomics data generator."""
    return ClinicalMetabolomicsDataGenerator()

@pytest.fixture
def metabolite_database():
    """Provide comprehensive metabolite database."""
    return ClinicalMetabolomicsDataGenerator.METABOLITE_DATABASE

@pytest.fixture
def disease_panels():
    """Provide disease-specific metabolite panels."""
    return ClinicalMetabolomicsDataGenerator.DISEASE_PANELS

@pytest.fixture
def analytical_platforms():
    """Provide analytical platform specifications."""
    return ClinicalMetabolomicsDataGenerator.ANALYTICAL_PLATFORMS

@pytest.fixture
def sample_clinical_study():
    """Provide sample clinical study dataset."""
    return ClinicalMetabolomicsDataGenerator.generate_clinical_study_dataset(
        disease='diabetes',
        sample_size=150,
        study_type='case_control'
    )

@pytest.fixture
def diabetes_concentration_data():
    """Provide diabetes-specific metabolite concentration data."""
    metabolites = ['glucose', 'lactate', 'alanine', 'acetate']
    return ClinicalMetabolomicsDataGenerator.generate_metabolite_concentration_data(
        metabolites=metabolites,
        n_samples=100,
        disease_state='diabetes',
        add_noise=True
    )

@pytest.fixture
def pathway_analysis_results():
    """Provide pathway analysis results."""
    return ClinicalMetabolomicsDataGenerator.generate_pathway_analysis_data('diabetes')

@pytest.fixture
def biomarker_validation_results():
    """Provide biomarker validation results."""
    return ClinicalMetabolomicsDataGenerator.generate_biomarker_validation_data(
        biomarkers=['glucose', 'lactate', 'alanine'],
        validation_type='roc_analysis'
    )

@pytest.fixture
def multi_disease_study_collection():
    """Provide collection of studies across multiple diseases."""
    diseases = ['diabetes', 'cardiovascular_disease', 'kidney_disease', 'liver_disease', 'cancer']
    studies = []
    
    for disease in diseases:
        study = ClinicalMetabolomicsDataGenerator.generate_clinical_study_dataset(
            disease=disease,
            sample_size=random.randint(80, 300),
            study_type=random.choice(['case_control', 'cohort', 'cross_sectional'])
        )
        studies.append(study)
    
    return studies

@pytest.fixture
def research_category_test_data():
    """Provide test data mapped to research categories."""
    category_data = {}
    
    # Map each research category to relevant test data
    for category in ResearchCategory:
        if category == ResearchCategory.BIOMARKER_DISCOVERY:
            category_data[category] = {
                'clinical_studies': [
                    ClinicalMetabolomicsDataGenerator.generate_clinical_study_dataset(
                        disease='diabetes',
                        study_type='case_control'
                    )
                ],
                'validation_data': ClinicalMetabolomicsDataGenerator.generate_biomarker_validation_data(
                    ['glucose', 'lactate']
                )
            }
        elif category == ResearchCategory.PATHWAY_ANALYSIS:
            category_data[category] = {
                'pathway_results': ClinicalMetabolomicsDataGenerator.generate_pathway_analysis_data('diabetes')
            }
        elif category == ResearchCategory.CLINICAL_DIAGNOSIS:
            category_data[category] = {
                'concentration_data': ClinicalMetabolomicsDataGenerator.generate_metabolite_concentration_data(
                    metabolites=['glucose', 'creatinine', 'bilirubin'],
                    disease_state='kidney_disease'
                )
            }
    
    return category_data