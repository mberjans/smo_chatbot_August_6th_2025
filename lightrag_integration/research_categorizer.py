"""
Research Categorization System for Clinical Metabolomics Oracle LightRAG Integration

This module provides intelligent categorization of research queries and operations
for metabolomics-specific cost tracking and analysis.

Classes:
    - QueryAnalyzer: Analyzes query content to determine research categories
    - ResearchCategorizer: Main categorization system with pattern matching
    - CategoryMetrics: Metrics tracking for categorization accuracy
    
The categorization system supports:
    - Automated research category detection based on query content
    - Pattern-based classification with metabolomics-specific knowledge
    - Confidence scoring for categorization decisions
    - Learning and adaptation from user feedback
    - Integration with cost tracking for research-specific analytics
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

from .cost_persistence import ResearchCategory


@dataclass
class CategoryPrediction:
    """
    Represents a research category prediction with confidence scoring.
    """
    
    category: ResearchCategory
    confidence: float
    evidence: List[str]  # List of keywords/patterns that led to this prediction
    subject_area: Optional[str] = None
    query_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category.value,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'subject_area': self.subject_area,
            'query_type': self.query_type,
            'metadata': self.metadata or {}
        }


@dataclass 
class CategoryMetrics:
    """
    Metrics for tracking categorization performance and accuracy.
    """
    
    total_predictions: int = 0
    correct_predictions: int = 0
    category_counts: Dict[str, int] = None
    confidence_distribution: List[float] = None
    average_confidence: float = 0.0
    
    def __post_init__(self):
        if self.category_counts is None:
            self.category_counts = defaultdict(int)
        if self.confidence_distribution is None:
            self.confidence_distribution = []
    
    def update(self, prediction: CategoryPrediction, is_correct: Optional[bool] = None):
        """Update metrics with a new prediction."""
        self.total_predictions += 1
        self.category_counts[prediction.category.value] += 1
        self.confidence_distribution.append(prediction.confidence)
        
        if is_correct is not None and is_correct:
            self.correct_predictions += 1
        
        # Recalculate average confidence
        if self.confidence_distribution:
            self.average_confidence = sum(self.confidence_distribution) / len(self.confidence_distribution)
    
    @property
    def accuracy(self) -> float:
        """Calculate prediction accuracy if feedback is available."""
        return (self.correct_predictions / self.total_predictions) if self.total_predictions > 0 else 0.0


class QueryAnalyzer:
    """
    Analyzes query content to extract features for research categorization.
    
    This class performs text analysis on user queries to identify keywords,
    patterns, and context that can be used for automatic categorization.
    """
    
    def __init__(self):
        """Initialize the query analyzer with metabolomics-specific patterns."""
        
        # Metabolomics-specific keyword patterns
        self.category_patterns = {
            ResearchCategory.METABOLITE_IDENTIFICATION: {
                'keywords': [
                    'metabolite', 'compound', 'molecule', 'identify', 'identification',
                    'mass spectrum', 'ms/ms', 'fragmentation', 'molecular formula',
                    'exact mass', 'isotope pattern', 'retention time', 'chemical shift',
                    'nmr', 'spectroscopy', 'structure', 'chemical structure', 'identification',
                    'unknown compound', 'database search', 'library match'
                ],
                'patterns': [
                    r'\b(?:identify|identification)\s+(?:of\s+)?(?:metabolite|compound|molecule)',
                    r'\bmass\s+spectromet\w+',
                    r'\bms/ms\b|\btandem\s+ms\b',
                    r'\bmolecular\s+(?:formula|weight|mass)',
                    r'\bstructural?\s+(?:elucidation|determination|analysis)',
                    r'\bcompound\s+(?:identification|characterization)',
                    r'\bunknown\s+(?:peak|signal|compound|metabolite)'
                ]
            },
            
            ResearchCategory.PATHWAY_ANALYSIS: {
                'keywords': [
                    'pathway', 'metabolic pathway', 'biochemical pathway', 'network',
                    'metabolic network', 'kegg', 'reactome', 'pathway analysis',
                    'enrichment', 'pathway enrichment', 'metabolic map', 'flux analysis',
                    'systems biology', 'metabolic modeling', 'network analysis',
                    'regulation', 'metabolic regulation', 'enzyme', 'reaction',
                    'metabolism', 'metabolic process', 'biological process'
                ],
                'patterns': [
                    r'\bpathway\s+(?:analysis|enrichment|mapping)',
                    r'\bmetabolic\s+(?:pathway|network|map|flux|modeling)',
                    r'\bbiochemical\s+pathway',
                    r'\bkegg\b|\breactome\b|\bwikipathways\b',
                    r'\benrichment\s+analysis',
                    r'\bsystems\s+biology',
                    r'\bnetwork\s+(?:analysis|topology|reconstruction)',
                    r'\bflux\s+(?:analysis|balance|distribution)'
                ]
            },
            
            ResearchCategory.BIOMARKER_DISCOVERY: {
                'keywords': [
                    'biomarker', 'marker', 'diagnostic', 'prognostic', 'predictive',
                    'signature', 'metabolic signature', 'disease marker', 'clinical marker',
                    'screening', 'diagnosis', 'prognosis', 'therapeutic target',
                    'drug target', 'disease mechanism', 'pathophysiology',
                    'personalized medicine', 'precision medicine', 'risk assessment',
                    'early detection', 'therapeutic monitoring'
                ],
                'patterns': [
                    r'\bbiomarker\s+(?:discovery|identification|validation)',
                    r'\bdiagnostic\s+(?:marker|biomarker|metabolite)',
                    r'\bprognostic\s+(?:marker|signature)',
                    r'\bmetabolic\s+signature',
                    r'\bdisease\s+(?:marker|biomarker|mechanism)',
                    r'\btherapeutic\s+(?:target|monitoring)',
                    r'\bpersonalized\s+medicine',
                    r'\bprecision\s+medicine',
                    r'\bearly\s+detection'
                ]
            },
            
            ResearchCategory.DRUG_DISCOVERY: {
                'keywords': [
                    'drug', 'pharmaceutical', 'therapeutic', 'medicine', 'treatment',
                    'drug discovery', 'drug development', 'drug target', 'lead compound',
                    'pharmaceutical compound', 'active ingredient', 'drug metabolism',
                    'pharmacokinetics', 'pharmacodynamics', 'admet', 'toxicity',
                    'side effect', 'drug interaction', 'mechanism of action',
                    'therapeutic effect', 'drug efficacy', 'clinical trial'
                ],
                'patterns': [
                    r'\bdrug\s+(?:discovery|development|design|screening)',
                    r'\bpharmaceutical\s+(?:compound|development)',
                    r'\btherapeutic\s+(?:compound|agent|target)',
                    r'\bdrug\s+(?:metabolism|target|interaction)',
                    r'\bpharmacokinetic\w*|\bpharmacodynamic\w*',
                    r'\badmet\b|\btoxicity\b',
                    r'\bmechanism\s+of\s+action',
                    r'\btherapeutic\s+effect',
                    r'\bclinical\s+trial'
                ]
            },
            
            ResearchCategory.CLINICAL_DIAGNOSIS: {
                'keywords': [
                    'clinical', 'patient', 'diagnosis', 'diagnostic', 'medical',
                    'healthcare', 'clinical chemistry', 'laboratory medicine',
                    'clinical metabolomics', 'medical diagnosis', 'disease diagnosis',
                    'clinical marker', 'laboratory test', 'clinical sample',
                    'patient sample', 'serum', 'plasma', 'urine', 'blood',
                    'tissue', 'biopsy', 'clinical study', 'clinical research'
                ],
                'patterns': [
                    r'\bclinical\s+(?:diagnosis|application|study|research)',
                    r'\bmedical\s+(?:diagnosis|application)',
                    r'\bpatient\s+(?:sample|data|diagnosis)',
                    r'\bclinical\s+(?:chemistry|metabolomics|marker)',
                    r'\blaboratory\s+(?:medicine|test|analysis)',
                    r'\bdisease\s+diagnosis',
                    r'\bclinical\s+sample',
                    r'\b(?:serum|plasma|urine|blood)\s+(?:analysis|metabolomics)'
                ]
            },
            
            ResearchCategory.DATA_PREPROCESSING: {
                'keywords': [
                    'preprocessing', 'data preprocessing', 'data processing', 'normalization',
                    'quality control', 'qc', 'data cleaning', 'outlier detection',
                    'missing data', 'imputation', 'batch correction', 'drift correction',
                    'peak detection', 'peak alignment', 'retention time correction',
                    'mass calibration', 'signal processing', 'noise reduction',
                    'baseline correction', 'smoothing', 'filtering'
                ],
                'patterns': [
                    r'\bdata\s+(?:preprocessing|processing|cleaning|preparation)',
                    r'\bnormalization\b|\bnormalize\b',
                    r'\bquality\s+control\b|\bqc\b',
                    r'\boutlier\s+detection',
                    r'\bmissing\s+(?:data|value)',
                    r'\bimputation\b|\bimpute\b',
                    r'\bbatch\s+(?:correction|effect)',
                    r'\bdrift\s+correction',
                    r'\bpeak\s+(?:detection|alignment|picking)',
                    r'\bretention\s+time\s+(?:correction|alignment)',
                    r'\bmass\s+calibration',
                    r'\bbaseline\s+correction',
                    r'\bnoise\s+reduction'
                ]
            },
            
            ResearchCategory.STATISTICAL_ANALYSIS: {
                'keywords': [
                    'statistics', 'statistical analysis', 'multivariate analysis',
                    'pca', 'principal component analysis', 'pls-da', 'opls-da',
                    'clustering', 'classification', 'machine learning', 'regression',
                    'correlation', 'significance test', 'hypothesis testing',
                    'anova', 't-test', 'wilcoxon', 'mann-whitney', 'chi-square',
                    'multiple comparison', 'false discovery rate', 'fdr',
                    'p-value', 'statistical significance', 'confidence interval'
                ],
                'patterns': [
                    r'\bstatistical\s+(?:analysis|test|method)',
                    r'\bmultivariate\s+(?:analysis|statistics)',
                    r'\bpca\b|\bprincipal\s+component\s+analysis',
                    r'\bpls-da\b|\bopls-da\b',
                    r'\bclustering\b|\bclassification\b',
                    r'\bmachine\s+learning',
                    r'\bregression\s+(?:analysis|model)',
                    r'\bcorrelation\s+(?:analysis|matrix)',
                    r'\bhypothesis\s+test\w*',
                    r'\banova\b|\bt-test\b|\bwilcoxon\b',
                    r'\bmultiple\s+comparison',
                    r'\bfalse\s+discovery\s+rate|\bfdr\b',
                    r'\bp-value\b|\bstatistical\s+significance'
                ]
            },
            
            ResearchCategory.LITERATURE_SEARCH: {
                'keywords': [
                    'literature', 'publication', 'paper', 'article', 'journal',
                    'literature review', 'systematic review', 'meta-analysis',
                    'pubmed', 'research article', 'scientific literature',
                    'bibliography', 'citation', 'reference', 'study',
                    'research', 'findings', 'results', 'conclusion',
                    'abstract', 'full text', 'peer review'
                ],
                'patterns': [
                    r'\bliterature\s+(?:search|review|survey)',
                    r'\bsystematic\s+review',
                    r'\bmeta-analysis\b|\bmeta\s+analysis',
                    r'\bpubmed\b|\bmedline\b',
                    r'\bresearch\s+(?:article|paper|publication)',
                    r'\bscientific\s+literature',
                    r'\bcitation\s+(?:analysis|search)',
                    r'\bbibliograph\w+',
                    r'\bpeer\s+review\w*',
                    r'\babstract\s+(?:search|analysis)'
                ]
            },
            
            ResearchCategory.KNOWLEDGE_EXTRACTION: {
                'keywords': [
                    'knowledge extraction', 'text mining', 'data mining',
                    'information extraction', 'knowledge discovery',
                    'natural language processing', 'nlp', 'semantic analysis',
                    'ontology', 'knowledge base', 'annotation', 'curation',
                    'database integration', 'data integration', 'knowledge graph',
                    'relationship extraction', 'entity recognition', 'parsing'
                ],
                'patterns': [
                    r'\bknowledge\s+(?:extraction|discovery|mining)',
                    r'\btext\s+mining|\bdata\s+mining',
                    r'\binformation\s+extraction',
                    r'\bnatural\s+language\s+processing|\bnlp\b',
                    r'\bsemantic\s+(?:analysis|search|annotation)',
                    r'\bontology\b|\bknowledge\s+base',
                    r'\bannotation\b|\bcuration\b',
                    r'\bdata\s+integration',
                    r'\bknowledge\s+graph',
                    r'\bentity\s+(?:recognition|extraction)',
                    r'\brelationship\s+extraction'
                ]
            },
            
            ResearchCategory.DATABASE_INTEGRATION: {
                'keywords': [
                    'database', 'integration', 'data integration', 'database query',
                    'hmdb', 'kegg', 'chebi', 'pubchem', 'metlin', 'massbank',
                    'database search', 'cross-reference', 'mapping', 'identifier',
                    'accession', 'annotation', 'metadata', 'standardization',
                    'data harmonization', 'format conversion', 'api', 'web service'
                ],
                'patterns': [
                    r'\bdatabase\s+(?:integration|query|search|mapping)',
                    r'\bdata\s+integration',
                    r'\bhmdb\b|\bkegg\b|\bchebi\b|\bpubchem\b',
                    r'\bmetlin\b|\bmassbank\b',
                    r'\bcross-reference\b|\bcross\s+reference',
                    r'\bmapping\b|\bidentifier\s+mapping',
                    r'\baccession\s+(?:number|id)',
                    r'\bmetadata\b|\bannotation\b',
                    r'\bstandardization\b|\bharmonization',
                    r'\bformat\s+conversion',
                    r'\bapi\b|\bweb\s+service'
                ]
            }
        }
        
        # Query type patterns
        self.query_type_patterns = {
            'question': [r'what\s+is', r'how\s+(?:does|can|to)', r'why\s+(?:does|is)', r'when\s+(?:does|is)', r'where\s+(?:does|is)'],
            'search': [r'find', r'search', r'look\s+for', r'identify', r'locate'],
            'analysis': [r'analyze', r'calculate', r'determine', r'evaluate', r'assess'],
            'comparison': [r'compare', r'difference', r'similarity', r'versus', r'vs\.?'],
            'explanation': [r'explain', r'describe', r'tell\s+me', r'what\s+are'],
            'procedure': [r'how\s+to', r'steps', r'procedure', r'protocol', r'method']
        }
        
        # Subject area patterns for metabolomics subdomains
        self.subject_area_patterns = {
            'lipidomics': [r'lipid\w*', r'fatty\s+acid', r'phospholipid', r'sphingolipid', r'sterol', r'triglyceride'],
            'proteomics': [r'protein\w*', r'peptide\w*', r'amino\s+acid', r'enzyme\w*', r'proteomic\w*'],
            'genomics': [r'gene\w*', r'dna', r'rna', r'genetic\w*', r'genomic\w*', r'transcriptom\w*'],
            'clinical': [r'clinical', r'patient\w*', r'disease\w*', r'disorder\w*', r'syndrome\w*', r'medical'],
            'plant': [r'plant\w*', r'botanical', r'phytochemical', r'natural\s+product', r'herbal'],
            'microbial': [r'microb\w+', r'bacterial?', r'fungal?', r'yeast', r'fermentation'],
            'environmental': [r'environmental', r'ecological', r'soil', r'water', r'atmospheric'],
            'food': [r'food', r'nutrition\w*', r'dietary', r'nutrient\w*', r'agricultural']
        }
    
    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """
        Analyze a query to extract features for categorization.
        
        Args:
            query_text: The user query text to analyze
            
        Returns:
            Dict containing analysis results including matched patterns,
            keywords, query type, and subject area
        """
        query_lower = query_text.lower()
        
        analysis = {
            'original_query': query_text,
            'matched_keywords': defaultdict(list),
            'matched_patterns': defaultdict(list),
            'query_type': self._detect_query_type(query_lower),
            'subject_area': self._detect_subject_area(query_lower),
            'query_length': len(query_text),
            'word_count': len(query_text.split()),
            'has_technical_terms': self._has_technical_terms(query_lower)
        }
        
        # Match keywords and patterns for each category
        for category, patterns in self.category_patterns.items():
            # Match keywords
            for keyword in patterns['keywords']:
                if keyword.lower() in query_lower:
                    analysis['matched_keywords'][category].append(keyword)
            
            # Match regex patterns
            for pattern in patterns['patterns']:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    analysis['matched_patterns'][category].extend(matches)
        
        return analysis
    
    def _detect_query_type(self, query_lower: str) -> Optional[str]:
        """Detect the type of query based on linguistic patterns."""
        for query_type, patterns in self.query_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return query_type
        return None
    
    def _detect_subject_area(self, query_lower: str) -> Optional[str]:
        """Detect the subject area based on domain-specific terms."""
        subject_scores = defaultdict(int)
        
        for subject, patterns in self.subject_area_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                subject_scores[subject] += matches
        
        if subject_scores:
            return max(subject_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _has_technical_terms(self, query_lower: str) -> bool:
        """Check if query contains technical metabolomics terms."""
        technical_terms = [
            'metabolomics', 'mass spectrometry', 'lc-ms', 'gc-ms', 'nmr',
            'chromatography', 'spectroscopy', 'biomarker', 'metabolite',
            'pathway', 'kegg', 'hmdb', 'pubchem', 'chebi'
        ]
        
        return any(term in query_lower for term in technical_terms)


class ResearchCategorizer:
    """
    Main research categorization system for metabolomics queries.
    
    This class combines pattern matching with confidence scoring to 
    automatically categorize research queries into appropriate categories
    for cost tracking and analysis.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the research categorizer."""
        self.logger = logger or logging.getLogger(__name__)
        self.query_analyzer = QueryAnalyzer()
        self.metrics = CategoryMetrics()
        
        # Confidence thresholds for categorization decisions
        self.confidence_thresholds = {
            'high': 0.8,      # High confidence prediction
            'medium': 0.6,    # Medium confidence prediction
            'low': 0.4        # Low confidence prediction
        }
        
        # Weighting factors for different types of evidence
        self.evidence_weights = {
            'exact_keyword_match': 1.0,
            'pattern_match': 0.8,
            'partial_keyword_match': 0.6,
            'context_bonus': 0.3,
            'technical_terms_bonus': 0.2
        }
        
        self.logger.info("Research categorizer initialized")
    
    def categorize_query(self, 
                        query_text: str,
                        context: Optional[Dict[str, Any]] = None) -> CategoryPrediction:
        """
        Categorize a research query and return prediction with confidence.
        
        Args:
            query_text: The user query text to categorize
            context: Optional context information (session data, previous queries, etc.)
            
        Returns:
            CategoryPrediction with category, confidence, and supporting evidence
        """
        # Analyze the query
        analysis = self.query_analyzer.analyze_query(query_text)
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(analysis, context)
        
        # Find the best category
        if not category_scores:
            # Default to general query if no patterns match
            prediction = CategoryPrediction(
                category=ResearchCategory.GENERAL_QUERY,
                confidence=0.1,
                evidence=['no_specific_patterns_found'],
                query_type=analysis.get('query_type'),
                subject_area=analysis.get('subject_area')
            )
        else:
            # Get the highest scoring category
            best_category, best_score = max(category_scores.items(), key=lambda x: x[1]['total_score'])
            
            prediction = CategoryPrediction(
                category=best_category,
                confidence=min(best_score['total_score'], 1.0),  # Cap at 1.0
                evidence=best_score['evidence'],
                query_type=analysis.get('query_type'),
                subject_area=analysis.get('subject_area'),
                metadata={
                    'all_scores': {cat.value: score['total_score'] for cat, score in category_scores.items()},
                    'analysis_details': analysis,
                    'confidence_level': self._get_confidence_level(min(best_score['total_score'], 1.0))
                }
            )
        
        # Update metrics
        self.metrics.update(prediction)
        
        self.logger.debug(f"Categorized query as {prediction.category.value} "
                         f"with confidence {prediction.confidence:.3f}")
        
        return prediction
    
    def _calculate_category_scores(self, 
                                 analysis: Dict[str, Any], 
                                 context: Optional[Dict[str, Any]]) -> Dict[ResearchCategory, Dict[str, Any]]:
        """Calculate scores for each category based on analysis results."""
        scores = {}
        
        for category in ResearchCategory:
            score_data = {
                'total_score': 0.0,
                'evidence': [],
                'keyword_score': 0.0,
                'pattern_score': 0.0,
                'context_score': 0.0,
                'bonus_score': 0.0
            }
            
            # Keyword matching score
            matched_keywords = analysis['matched_keywords'].get(category, [])
            if matched_keywords:
                # Score based on number and quality of keyword matches
                keyword_score = len(matched_keywords) * self.evidence_weights['exact_keyword_match']
                score_data['keyword_score'] = keyword_score
                score_data['evidence'].extend([f"keyword:{kw}" for kw in matched_keywords[:3]])  # Limit evidence
            
            # Pattern matching score
            matched_patterns = analysis['matched_patterns'].get(category, [])
            if matched_patterns:
                pattern_score = len(matched_patterns) * self.evidence_weights['pattern_match']
                score_data['pattern_score'] = pattern_score
                score_data['evidence'].extend([f"pattern:{p}" for p in matched_patterns[:2]])  # Limit evidence
            
            # Context bonuses
            if context:
                context_score = self._calculate_context_score(category, context)
                score_data['context_score'] = context_score
                if context_score > 0:
                    score_data['evidence'].append("context_match")
            
            # Technical terms bonus
            if analysis.get('has_technical_terms'):
                # Give bonus to research-oriented categories
                research_categories = [
                    ResearchCategory.METABOLITE_IDENTIFICATION,
                    ResearchCategory.PATHWAY_ANALYSIS,
                    ResearchCategory.BIOMARKER_DISCOVERY,
                    ResearchCategory.DRUG_DISCOVERY,
                    ResearchCategory.CLINICAL_DIAGNOSIS,
                    ResearchCategory.STATISTICAL_ANALYSIS
                ]
                if category in research_categories:
                    score_data['bonus_score'] += self.evidence_weights['technical_terms_bonus']
                    score_data['evidence'].append("technical_terms")
            
            # Subject area alignment bonus
            if analysis.get('subject_area'):
                subject_bonus = self._calculate_subject_alignment_bonus(category, analysis['subject_area'])
                score_data['bonus_score'] += subject_bonus
                if subject_bonus > 0:
                    score_data['evidence'].append(f"subject_area:{analysis['subject_area']}")
            
            # Calculate total score
            total_score = (score_data['keyword_score'] + 
                          score_data['pattern_score'] + 
                          score_data['context_score'] + 
                          score_data['bonus_score'])
            
            # Apply normalization based on query complexity
            normalization_factor = self._get_normalization_factor(analysis)
            score_data['total_score'] = total_score * normalization_factor
            
            # Only include categories with meaningful scores
            if score_data['total_score'] > 0.1:
                scores[category] = score_data
        
        return scores
    
    def _calculate_context_score(self, 
                                category: ResearchCategory, 
                                context: Dict[str, Any]) -> float:
        """Calculate additional score based on context information."""
        score = 0.0
        
        # Previous queries in session
        if 'previous_categories' in context:
            prev_categories = context['previous_categories']
            if category.value in prev_categories:
                # Bonus for category consistency within session
                score += self.evidence_weights['context_bonus']
        
        # User profile or preferences
        if 'user_research_areas' in context:
            user_areas = context['user_research_areas']
            if category.value in user_areas:
                score += self.evidence_weights['context_bonus'] * 0.5
        
        # Project context
        if 'project_type' in context:
            project_type = context['project_type']
            category_project_mapping = {
                'clinical_study': [ResearchCategory.CLINICAL_DIAGNOSIS, ResearchCategory.BIOMARKER_DISCOVERY],
                'drug_development': [ResearchCategory.DRUG_DISCOVERY, ResearchCategory.PATHWAY_ANALYSIS],
                'basic_research': [ResearchCategory.METABOLITE_IDENTIFICATION, ResearchCategory.PATHWAY_ANALYSIS],
                'data_analysis': [ResearchCategory.STATISTICAL_ANALYSIS, ResearchCategory.DATA_PREPROCESSING]
            }
            
            if category in category_project_mapping.get(project_type, []):
                score += self.evidence_weights['context_bonus']
        
        return score
    
    def _calculate_subject_alignment_bonus(self, 
                                         category: ResearchCategory, 
                                         subject_area: str) -> float:
        """Calculate bonus for subject area alignment with category."""
        
        # Define which subject areas align with which categories
        alignments = {
            ResearchCategory.CLINICAL_DIAGNOSIS: ['clinical', 'medical'],
            ResearchCategory.BIOMARKER_DISCOVERY: ['clinical', 'medical'],
            ResearchCategory.DRUG_DISCOVERY: ['clinical', 'medical'],
            ResearchCategory.METABOLITE_IDENTIFICATION: ['lipidomics', 'proteomics', 'plant', 'microbial', 'food'],
            ResearchCategory.PATHWAY_ANALYSIS: ['proteomics', 'genomics', 'microbial', 'plant'],
            ResearchCategory.DATABASE_INTEGRATION: ['lipidomics', 'proteomics', 'genomics'],
            ResearchCategory.STATISTICAL_ANALYSIS: ['clinical', 'lipidomics', 'proteomics']
        }
        
        category_subjects = alignments.get(category, [])
        if subject_area in category_subjects:
            return self.evidence_weights['technical_terms_bonus']
        
        return 0.0
    
    def _get_normalization_factor(self, analysis: Dict[str, Any]) -> float:
        """Calculate normalization factor based on query characteristics."""
        # Longer, more detailed queries get slight boost
        word_count = analysis.get('word_count', 0)
        if word_count > 10:
            return 1.1
        elif word_count > 5:
            return 1.0
        else:
            return 0.9
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category based on score."""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    def update_from_feedback(self, 
                           query_text: str,
                           predicted_category: ResearchCategory,
                           actual_category: ResearchCategory,
                           confidence: float) -> None:
        """
        Update the categorizer based on user feedback.
        
        Args:
            query_text: Original query text
            predicted_category: Category predicted by the system
            actual_category: Correct category provided by user
            confidence: Confidence of the original prediction
        """
        is_correct = (predicted_category == actual_category)
        
        # Create a dummy prediction for metrics update
        feedback_prediction = CategoryPrediction(
            category=predicted_category,
            confidence=confidence,
            evidence=[]
        )
        
        self.metrics.update(feedback_prediction, is_correct)
        
        # Log feedback for potential model improvement
        self.logger.info(f"Feedback received - Query: {query_text[:100]}... "
                        f"Predicted: {predicted_category.value}, "
                        f"Actual: {actual_category.value}, "
                        f"Correct: {is_correct}")
        
        # Here you could implement learning logic to adjust patterns or weights
        # For now, we just log the feedback
    
    def get_category_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about categorization performance.
        
        Returns:
            Dict containing performance metrics and category distribution
        """
        return {
            'total_predictions': self.metrics.total_predictions,
            'accuracy': self.metrics.accuracy,
            'average_confidence': self.metrics.average_confidence,
            'category_distribution': dict(self.metrics.category_counts),
            'confidence_distribution': {
                'high': len([c for c in self.metrics.confidence_distribution if c >= self.confidence_thresholds['high']]),
                'medium': len([c for c in self.metrics.confidence_distribution 
                              if self.confidence_thresholds['medium'] <= c < self.confidence_thresholds['high']]),
                'low': len([c for c in self.metrics.confidence_distribution 
                           if self.confidence_thresholds['low'] <= c < self.confidence_thresholds['medium']]),
                'very_low': len([c for c in self.metrics.confidence_distribution if c < self.confidence_thresholds['low']])
            },
            'thresholds': self.confidence_thresholds
        }
    
    def reset_metrics(self) -> None:
        """Reset categorization metrics."""
        self.metrics = CategoryMetrics()
        self.logger.info("Categorization metrics reset")