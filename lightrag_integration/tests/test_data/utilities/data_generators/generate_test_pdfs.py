#!/usr/bin/env python3
"""
Test PDF Data Generator

Generates sample PDF-like content files for testing the LightRAG integration.
Creates realistic biomedical research documents with controlled content.

Usage:
    python generate_test_pdfs.py [options]
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import argparse


class TestPDFGenerator:
    """Generates test PDF content files for biomedical research testing"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Biomedical vocabulary for realistic content generation
        self.metabolites = [
            "glucose", "pyruvate", "lactate", "citrate", "malate", "succinate",
            "acetyl-CoA", "palmitate", "cholesterol", "creatinine", "urea",
            "glutamate", "alanine", "glycine", "taurine", "carnitine"
        ]
        
        self.conditions = [
            "Type 2 diabetes", "cardiovascular disease", "metabolic syndrome",
            "obesity", "hypertension", "dyslipidemia", "insulin resistance",
            "atherosclerosis", "fatty liver disease", "kidney disease"
        ]
        
        self.methods = [
            "LC-MS/MS", "GC-MS", "NMR spectroscopy", "UHPLC-QTOF",
            "targeted metabolomics", "untargeted metabolomics", 
            "lipidomics", "proteomics", "mass spectrometry"
        ]
        
        self.journals = [
            "Journal of Clinical Metabolomics",
            "Metabolomics Research",
            "Clinical Chemistry and Laboratory Medicine",
            "Biomedical Chromatography",
            "Nature Metabolism"
        ]
        
    def generate_research_paper(self, paper_id: str, complexity: str = "medium") -> str:
        """Generate a research paper with specified complexity"""
        
        # Select random components
        metabolites = random.sample(self.metabolites, random.randint(3, 8))
        condition = random.choice(self.conditions)
        method = random.choice(self.methods)
        journal = random.choice(self.journals)
        
        # Generate content based on complexity
        if complexity == "minimal":
            return self._generate_minimal_paper(paper_id, metabolites[0], condition, method, journal)
        elif complexity == "complex":
            return self._generate_complex_paper(paper_id, metabolites, condition, method, journal)
        else:
            return self._generate_medium_paper(paper_id, metabolites[:5], condition, method, journal)
            
    def _generate_minimal_paper(self, paper_id: str, metabolite: str, condition: str, method: str, journal: str) -> str:
        """Generate minimal research paper"""
        return f"""RESEARCH PAPER - {paper_id}

Title: {metabolite.title()} Analysis in {condition} Using {method}
Authors: Dr. Test Author, Research Team
Journal: {journal}
Year: {random.randint(2020, 2024)}

ABSTRACT
This study investigates {metabolite} levels in patients with {condition}. Using {method} analysis, we found significant differences compared to healthy controls.

INTRODUCTION
{condition} is a major health concern. {metabolite.title()} may serve as an important biomarker.

METHODS
Study Design: Case-control study
Participants: {random.randint(50, 200)} patients, {random.randint(30, 100)} controls
Analysis: {method}

RESULTS
{metabolite.title()} levels were {'elevated' if random.random() > 0.5 else 'reduced'} in patients (p < 0.05).

CONCLUSIONS
{metabolite.title()} shows promise as a biomarker for {condition}.

KEYWORDS: {metabolite}, {condition}, biomarkers, {method.lower()}
"""

    def _generate_medium_paper(self, paper_id: str, metabolites: List[str], condition: str, method: str, journal: str) -> str:
        """Generate medium complexity research paper"""
        n_patients = random.randint(100, 300)
        n_controls = random.randint(50, 150)
        
        results_section = "\n".join([
            f"- {met.title()}: {'Significantly elevated' if random.random() > 0.6 else 'Significantly reduced'} "
            f"({random.uniform(1.2, 3.5):.1f}-fold change, p < 0.0{random.randint(1, 5)})"
            for met in metabolites
        ])
        
        return f"""BIOMEDICAL RESEARCH DOCUMENT - {paper_id}

Title: Comprehensive {method} Analysis of Metabolic Alterations in {condition}
Authors: Dr. Principal Investigator, Dr. Co-Investigator, Research Team
Institution: Clinical Metabolomics Research Center
Journal: {journal}
Year: {random.randint(2020, 2024)}

ABSTRACT
Background: {condition} represents a significant healthcare burden with complex metabolic alterations.
Objective: To characterize metabolic profiles using {method} in {condition} patients.
Methods: {n_patients} patients and {n_controls} healthy controls were analyzed.
Results: {len(metabolites)} metabolites showed significant alterations.
Conclusions: Metabolic profiling reveals distinct signatures in {condition}.

KEYWORDS
{', '.join(metabolites[:3])}, {condition.lower()}, {method.lower()}, clinical metabolomics

INTRODUCTION
{condition} affects millions worldwide and involves complex metabolic dysregulation. 
Previous studies have suggested alterations in energy metabolism, lipid metabolism, 
and amino acid metabolism. This study aims to provide comprehensive metabolic 
characterization using state-of-the-art {method} technology.

METHODS
Study Design: Cross-sectional case-control study
Participants: 
- Cases: {n_patients} patients with confirmed {condition}
- Controls: {n_controls} age and sex-matched healthy individuals
Sample Collection: Fasting plasma samples
Analytical Platform: {method}
Statistical Analysis: t-tests, multivariate analysis, pathway enrichment

RESULTS
Metabolite Alterations:
{results_section}

Pathway Analysis:
- Energy metabolism: significantly disrupted
- Lipid metabolism: altered in {random.randint(60, 90)}% of patients
- Amino acid metabolism: moderately affected

Classification Performance:
- AUC: {random.uniform(0.75, 0.95):.3f}
- Sensitivity: {random.uniform(0.70, 0.95):.3f}
- Specificity: {random.uniform(0.75, 0.90):.3f}

DISCUSSION
The metabolic alterations observed in this study are consistent with the pathophysiology
of {condition}. The involvement of {metabolites[0]} and {metabolites[1]} suggests
disruption of central metabolic pathways. These findings may have implications
for diagnosis, monitoring, and treatment of {condition}.

CONCLUSIONS
This comprehensive {method} analysis reveals significant metabolic alterations
in {condition} patients. The identified metabolites may serve as potential
biomarkers for clinical applications.

REFERENCES
1. Reference 1 - Previous metabolomics study
2. Reference 2 - Clinical guidelines for {condition}
3. Reference 3 - {method} methodology
"""

    def _generate_complex_paper(self, paper_id: str, metabolites: List[str], condition: str, method: str, journal: str) -> str:
        """Generate complex research paper with detailed content"""
        # This would include more detailed sections, statistical tables, etc.
        # For brevity, returning a placeholder that indicates complexity
        base_paper = self._generate_medium_paper(paper_id, metabolites, condition, method, journal)
        
        additional_content = f"""

DETAILED STATISTICAL ANALYSIS
Multiple testing correction was applied using Benjamini-Hochberg FDR.
Pathway enrichment analysis was performed using KEGG and BioCyc databases.
Machine learning models (Random Forest, SVM) were trained and validated.

SUPPLEMENTARY DATA
- Table S1: Complete metabolite list with statistical significance
- Figure S1: PCA plot showing group separation
- Figure S2: Heatmap of significantly altered metabolites
- Table S2: Pathway enrichment results

EXTENDED METHODS
Sample Preparation:
1. Plasma samples were thawed on ice
2. Protein precipitation using methanol (1:3 ratio)
3. Centrifugation at 14,000 rpm for 10 minutes
4. Supernatant collection and LC-MS/MS analysis

Quality Control:
- Pooled QC samples analyzed every 10 injections
- Blank samples to monitor contamination
- Internal standards for normalization

Data Processing:
- Peak detection using vendor software
- Manual review of integration
- Normalization to internal standards
- Log transformation and scaling

EXTENDED DISCUSSION
The metabolic alterations identified in this study provide new insights into
the pathophysiology of {condition}. The disruption of {metabolites[0]} metabolism
may be linked to {random.choice(['oxidative stress', 'inflammation', 'insulin resistance'])}.

Future studies should investigate:
1. Temporal changes in metabolite levels
2. Response to therapeutic interventions
3. Integration with genomic data
4. Validation in independent cohorts
"""
        return base_paper + additional_content
        
    def generate_clinical_trial_protocol(self, trial_id: str) -> str:
        """Generate clinical trial protocol document"""
        condition = random.choice(self.conditions)
        method = random.choice(self.methods)
        
        return f"""CLINICAL TRIAL PROTOCOL - {trial_id}

Title: Phase {random.choice(['I', 'II', 'III'])} Clinical Trial of {method}-Guided Therapy in {condition}
Protocol Number: CMO-{trial_id}
Principal Investigator: Dr. Clinical Researcher
Sponsor: Clinical Metabolomics Research Institute

STUDY SYNOPSIS
Objective: To evaluate the efficacy of {method}-guided therapy selection
Design: Randomized, double-blind, placebo-controlled trial
Population: Adult patients with {condition}
Sample Size: {random.randint(100, 500)} participants
Primary Endpoint: Clinical improvement at 12 weeks
Duration: {random.randint(12, 52)} weeks

BACKGROUND AND RATIONALE
{condition} management could benefit from precision medicine approaches.
{method} profiling may guide optimal therapy selection.

STUDY OBJECTIVES
Primary:
- Assess clinical efficacy of {method}-guided therapy

Secondary:
- Evaluate safety and tolerability
- Analyze metabolic biomarker changes
- Determine cost-effectiveness

STUDY DESIGN AND PROCEDURES
Screening Period: 2-4 weeks
Randomization: 1:1 to intervention vs. standard care
Follow-up: Weekly visits for first month, then monthly

Inclusion Criteria:
- Age 18-75 years
- Diagnosed {condition}
- Stable on current medications
- Able to provide informed consent

Exclusion Criteria:
- Severe comorbidities
- Recent participation in other trials
- Pregnancy or lactation

STATISTICAL ANALYSIS PLAN
Primary analysis: Intention-to-treat using t-test
Sample size calculation: 80% power, alpha=0.05
Interim analysis: Planned at 50% enrollment

REGULATORY CONSIDERATIONS
IRB approval required before initiation
FDA IND application submitted
Good Clinical Practice compliance
"""

    def generate_corrupted_document(self, doc_id: str) -> str:
        """Generate document with intentional corruption for error testing"""
        corruption_types = [
            "incomplete_sections",
            "encoding_issues", 
            "malformed_structure",
            "mixed_corruption"
        ]
        
        corruption_type = random.choice(corruption_types)
        
        if corruption_type == "incomplete_sections":
            return f"""CORRUPTED DOCUMENT - {doc_id}

Title: Metabolomics Study of [TRUNCATED
Authors: Dr. Test [MISSING_DATA
Institution: [INCOMPLETE

ABSTRACT
This study investigates metabolomic profiles in diabetes patients. However, the data collection was [INTERRUPTED_CONTENT...

METHODS
Sample Collection: Plasma samples from 100 patien[CUT_OFF
Analytical Platform: LC-MS/MS using [MISSING_INSTRUMENT_INFO
Data Processing: [SECTION_MISSING]

RESULTS
[DATA_CORRUPTED]
- Glucose-6-phosphate: [VALUE_MISSING]
- Pyruvate levels: [ERROR_IN_MEASUREMENT]

CONCLUSIONS
[INCOMPLETE_SECTION]
"""

        elif corruption_type == "encoding_issues":
            return f"""ENCODING ISSUES DOCUMENT - {doc_id}

Title: Clínical Metabólomics Análysis with Special Chars ÄÖÜßàáâã
Authors: Dr. Tëst Àuthör, Rëséärch Tëäm
Journal: Jöurnäl öf Clínicäl Mëtäbölömícs

ABSTRACT
This study investigates metabolomic profiles using spëcíäl chäräctërs and encoding issues.
Binary data corruption: ����þÿ������
Invalid UTF-8 sequences: ���������

METHODS
Sample collection with éncodíng íssúes: àáâãäåæçèéêëìíîï
Statistical analysis: ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß

RESULTS
Mixed encoding results: ���metabolite levels��� showed significant ������.
"""

        else:  # malformed_structure or mixed_corruption
            return f"""MALFORMED STRUCTURE DOCUMENT - {doc_id}

Title: Test Document with Structural Issues
<<INVALID_XML_TAG>>
{{UNCLOSED_JSON: "malformed_data", "missing_quote: value
[BROKEN_MARKDOWN](#invalid-reference-to-nowhere

ABSTRACT
This section has no proper ending

METHODS
- Incomplete bullet point
- Another bullet point with [MISSING_CLOSING_BRACKET
- Normal bullet point

{{"json": "mixed_with_text", "error": true}}

RESULTS
Tabular data with issues:
| Column 1 | Missing columns
| Data 1   | 
| Data 2   | More data | Too many columns |

CONCLUSIONS
Multiple issues:
1. Incomplete numbering
3. Skipped number 2
5. Another skip

[SECTION_END_MISSING]
"""

    def generate_test_dataset(self, n_documents: int = 10, output_formats: List[str] = None) -> Dict[str, int]:
        """Generate a complete test dataset"""
        if output_formats is None:
            output_formats = ['txt']  # Simplified for testing, in practice could include PDF generation
            
        stats = {
            'research_papers': 0,
            'clinical_trials': 0,
            'corrupted_docs': 0,
            'total_documents': 0
        }
        
        for i in range(n_documents):
            doc_id = f"TEST_{datetime.now().strftime('%Y%m%d')}_{i+1:03d}"
            
            # Determine document type
            doc_type = random.choices(
                ['research_paper', 'clinical_trial', 'corrupted'],
                weights=[0.6, 0.3, 0.1]
            )[0]
            
            # Generate content
            if doc_type == 'research_paper':
                complexity = random.choice(['minimal', 'medium', 'complex'])
                content = self.generate_research_paper(doc_id, complexity)
                stats['research_papers'] += 1
            elif doc_type == 'clinical_trial':
                content = self.generate_clinical_trial_protocol(doc_id)
                stats['clinical_trials'] += 1
            else:
                content = self.generate_corrupted_document(doc_id)
                stats['corrupted_docs'] += 1
                
            # Save to file
            for fmt in output_formats:
                output_file = self.output_dir / f"{doc_id}.{fmt}"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            stats['total_documents'] += 1
            
        return stats
        
def main():
    parser = argparse.ArgumentParser(description='Generate test PDF content for LightRAG integration')
    parser.add_argument('--output-dir', default='./generated_test_docs', help='Output directory')
    parser.add_argument('--count', type=int, default=10, help='Number of documents to generate')
    parser.add_argument('--formats', nargs='+', default=['txt'], help='Output formats')
    
    args = parser.parse_args()
    
    generator = TestPDFGenerator(args.output_dir)
    stats = generator.generate_test_dataset(args.count, args.formats)
    
    print("Test Document Generation Complete!")
    print(f"Research Papers: {stats['research_papers']}")
    print(f"Clinical Trials: {stats['clinical_trials']}")
    print(f"Corrupted Documents: {stats['corrupted_docs']}")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Output Directory: {args.output_dir}")
    
    return 0

if __name__ == '__main__':
    main()