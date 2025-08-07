# Clinical Metabolomics Response Relevance Scoring Algorithm Design

**Document Version**: 1.0.0  
**Created**: August 7, 2025  
**Target Implementation**: CMO-LIGHTRAG-009-T02  
**Author**: Claude Code (Anthropic)

---

## Executive Summary

This document presents a comprehensive design for a response relevance scoring algorithm specifically tailored for clinical metabolomics responses. The algorithm builds upon the existing ResponseQualityAssessor infrastructure and provides specialized scoring dimensions, weighting schemes, and computational approaches optimized for biomedical query-response evaluation.

**Key Features:**
- Multi-dimensional relevance scoring with clinical metabolomics specialization
- Query-type adaptive weighting schemes
- Real-time computational efficiency
- Integration with existing quality assessment infrastructure
- Semantic similarity and domain expertise validation

---

## 1. System Architecture Overview

### 1.1 Infrastructure Integration

The relevance scoring system integrates with existing components:

```
ResponseQualityAssessor (Existing)
├── assess_response_quality() 
├── _assess_relevance() [TO BE ENHANCED]
├── biomedical_keywords
└── quality_weights

ClinicalMetabolomicsRelevanceScorer (New)
├── RelevanceScorer
├── QueryTypeClassifier  
├── SemanticSimilarityEngine
├── DomainExpertiseValidator
└── WeightingSchemeManager
```

### 1.2 Design Principles

- **Modularity**: Independent scoring components for easy testing and maintenance
- **Extensibility**: Support for new scoring dimensions and query types
- **Performance**: Sub-200ms scoring for real-time applications
- **Accuracy**: >85% correlation with expert human relevance assessments
- **Transparency**: Explainable scoring with detailed breakdowns

---

## 2. Relevance Scoring Dimensions

### 2.1 Core Relevance Dimensions

#### 2.1.1 Metabolomics Relevance (metabolomics_relevance)
**Weight**: 25-35% (depending on query type)  
**Description**: Measures alignment with metabolomics concepts, methodologies, and terminology.

**Sub-components**:
- **Analytical Method Relevance** (30%): LC-MS, GC-MS, NMR, UPLC coverage
- **Metabolite Specificity** (25%): Named metabolites, pathways, concentrations
- **Research Context** (20%): Study design, sample types, biomarker discovery
- **Technical Accuracy** (25%): Correct use of metabolomics terminology

**Scoring Approach**:
```python
def calculate_metabolomics_relevance(query, response):
    analytical_score = assess_analytical_methods(response)
    metabolite_score = assess_metabolite_coverage(query, response)
    research_score = assess_research_context(response)
    technical_score = assess_technical_accuracy(response)
    
    return weighted_average([
        (analytical_score, 0.30),
        (metabolite_score, 0.25),
        (research_score, 0.20),
        (technical_score, 0.25)
    ])
```

#### 2.1.2 Clinical Applicability (clinical_applicability)
**Weight**: 20-30% (depending on query type)  
**Description**: Evaluates relevance to clinical practice, patient care, and medical applications.

**Sub-components**:
- **Disease Relevance** (35%): Disease conditions, pathophysiology
- **Diagnostic Utility** (25%): Biomarkers, diagnostic applications
- **Therapeutic Relevance** (25%): Treatment monitoring, drug effects
- **Clinical Workflow** (15%): Practical implementation, clinical guidelines

#### 2.1.3 Query Alignment (query_alignment)
**Weight**: 20-25%  
**Description**: Direct semantic and lexical alignment between query and response.

**Sub-components**:
- **Semantic Similarity** (40%): Vector-based semantic matching
- **Keyword Overlap** (25%): Weighted term frequency matching
- **Intent Matching** (20%): Query intent vs. response focus
- **Context Preservation** (15%): Maintaining query context throughout response

#### 2.1.4 Scientific Rigor (scientific_rigor)
**Weight**: 15-20%  
**Description**: Assessment of scientific accuracy and methodological soundness.

**Sub-components**:
- **Evidence Quality** (30%): Citation quality, study types referenced
- **Statistical Appropriateness** (25%): Correct statistical concepts
- **Methodological Soundness** (25%): Appropriate research methods
- **Uncertainty Acknowledgment** (20%): Appropriate caveats and limitations

#### 2.1.5 Biomedical Context Depth (biomedical_context_depth)
**Weight**: 10-15%  
**Description**: Depth and appropriateness of biomedical contextualization.

**Sub-components**:
- **Biological Pathway Integration** (30%): Pathway context and connections
- **Physiological Relevance** (25%): Biological systems context
- **Multi-omics Integration** (25%): Connection to other omics fields
- **Translational Context** (20%): Bench-to-bedside relevance

---

## 3. Query Type Classification and Weighting Schemes

### 3.1 Query Type Taxonomy

#### 3.1.1 Basic Definition Queries
**Examples**: "What is metabolomics?", "Define LC-MS"  
**Characteristics**: Seeking fundamental understanding

**Weighting Scheme**:
- metabolomics_relevance: 35%
- query_alignment: 25%
- scientific_rigor: 20%
- clinical_applicability: 15%
- biomedical_context_depth: 5%

#### 3.1.2 Clinical Application Queries  
**Examples**: "Metabolomics biomarkers for diabetes", "Clinical applications of metabolomics"  
**Characteristics**: Focus on medical applications

**Weighting Scheme**:
- clinical_applicability: 30%
- metabolomics_relevance: 25%
- query_alignment: 20%
- scientific_rigor: 15%
- biomedical_context_depth: 10%

#### 3.1.3 Analytical Method Queries
**Examples**: "LC-MS vs GC-MS for metabolomics", "Sample preparation for metabolomics"  
**Characteristics**: Technical methodology focus

**Weighting Scheme**:
- metabolomics_relevance: 40%
- query_alignment: 25%
- scientific_rigor: 20%
- biomedical_context_depth: 10%
- clinical_applicability: 5%

#### 3.1.4 Research Design Queries
**Examples**: "Study design for metabolomics research", "Statistical analysis of metabolomics data"  
**Characteristics**: Research methodology and design

**Weighting Scheme**:
- scientific_rigor: 30%
- metabolomics_relevance: 25%
- query_alignment: 20%
- biomedical_context_depth: 15%
- clinical_applicability: 10%

#### 3.1.5 Disease-Specific Queries
**Examples**: "Metabolomics in Alzheimer's disease", "Cancer metabolomics biomarkers"  
**Characteristics**: Disease-focused applications

**Weighting Scheme**:
- clinical_applicability: 30%
- biomedical_context_depth: 25%
- metabolomics_relevance: 20%
- query_alignment: 15%
- scientific_rigor: 10%

### 3.2 Query Classification Algorithm

```python
class QueryTypeClassifier:
    def __init__(self):
        self.classification_keywords = {
            'basic_definition': ['what is', 'define', 'definition', 'explain', 'basics'],
            'clinical_application': ['clinical', 'patient', 'diagnosis', 'treatment', 'medical'],
            'analytical_method': ['LC-MS', 'GC-MS', 'NMR', 'method', 'analysis', 'protocol'],
            'research_design': ['study design', 'statistics', 'analysis', 'methodology'],
            'disease_specific': ['disease', 'cancer', 'diabetes', 'alzheimer', 'cardiovascular']
        }
    
    def classify_query(self, query: str) -> str:
        scores = {}
        query_lower = query.lower()
        
        for query_type, keywords in self.classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[query_type] = score
            
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
```

---

## 4. Semantic Similarity and Keyword Matching

### 4.1 Semantic Similarity Engine

#### 4.1.1 Vector-Based Similarity
**Approach**: Use biomedical embeddings (BioBERT, ClinicalBERT, or domain-specific models)

```python
class SemanticSimilarityEngine:
    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def calculate_similarity(self, query: str, response: str) -> float:
        query_embedding = self._get_embedding(query)
        response_embedding = self._get_embedding(response)
        
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1), 
            response_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
```

#### 4.1.2 Domain-Specific Similarity Adjustments
- **Metabolite Names**: Exact matching with fuzzy matching for common variants
- **Analytical Methods**: Hierarchical matching (LC-MS/MS matches LC-MS queries)
- **Disease Terms**: Medical ontology-based matching (UMLS, MeSH)

### 4.2 Enhanced Keyword Matching

#### 4.2.1 Weighted Term Frequency
```python
class WeightedKeywordMatcher:
    def __init__(self):
        self.term_weights = {
            'metabolomics_core': 3.0,
            'analytical_methods': 2.5,
            'clinical_terms': 2.0,
            'general_biomedical': 1.5,
            'common_words': 0.1
        }
    
    def calculate_weighted_overlap(self, query: str, response: str) -> float:
        query_terms = self._extract_weighted_terms(query)
        response_terms = self._extract_weighted_terms(response)
        
        overlap_score = 0
        for term, weight in query_terms.items():
            if term in response_terms:
                overlap_score += weight * min(query_terms[term], response_terms[term])
        
        return overlap_score / max(sum(query_terms.values()), 1)
```

#### 4.2.2 Biomedical Keyword Categories

**Extended from existing infrastructure**:
```python
BIOMEDICAL_KEYWORDS = {
    'metabolomics_core': {
        'primary': ['metabolomics', 'metabolite', 'metabolism', 'biomarker'],
        'analytical': ['LC-MS', 'GC-MS', 'NMR', 'mass spectrometry', 'chromatography'],
        'quantitative': ['concentration', 'abundance', 'peak area', 'intensity'],
        'pathways': ['metabolic pathway', 'biochemical pathway', 'KEGG', 'metabolism']
    },
    'clinical_terms': {
        'conditions': ['diabetes', 'cancer', 'cardiovascular', 'alzheimer', 'obesity'],
        'applications': ['diagnosis', 'prognosis', 'treatment', 'monitoring', 'screening'],
        'populations': ['patient', 'control', 'cohort', 'clinical trial', 'study']
    },
    'research_methods': {
        'study_design': ['case-control', 'cohort', 'cross-sectional', 'longitudinal'],
        'statistics': ['p-value', 'fold change', 'ROC', 'AUC', 'statistical significance'],
        'validation': ['reproducibility', 'validation', 'quality control', 'standardization']
    }
}
```

---

## 5. Domain Expertise Validation

### 5.1 Expert Knowledge Integration

#### 5.1.1 Metabolomics Domain Rules
```python
class DomainExpertiseValidator:
    def __init__(self):
        self.expertise_rules = {
            'analytical_method_compatibility': {
                'polar_metabolites': ['HILIC', 'C18 negative mode'],
                'lipids': ['C18 positive mode', 'LIPID column'],
                'volatile_compounds': ['GC-MS', 'headspace']
            },
            'statistical_appropriateness': {
                'univariate': ['t-test', 'ANOVA', 'fold change'],
                'multivariate': ['PCA', 'PLS-DA', 'OPLS-DA'],
                'pathway_analysis': ['GSEA', 'pathway enrichment', 'MetaboAnalyst']
            },
            'clinical_validity': {
                'biomarker_criteria': ['sensitivity', 'specificity', 'reproducibility'],
                'study_requirements': ['sample size', 'validation cohort', 'clinical relevance']
            }
        }
    
    def validate_domain_expertise(self, response: str) -> float:
        expertise_score = 0
        total_checks = 0
        
        for category, rules in self.expertise_rules.items():
            category_score = self._check_category_expertise(response, rules)
            expertise_score += category_score
            total_checks += 1
        
        return expertise_score / max(total_checks, 1)
```

### 5.2 Factual Consistency Checking

#### 5.2.1 Common Knowledge Validation
- **Method Limitations**: Correct acknowledgment of analytical limitations
- **Quantitative Ranges**: Reasonable metabolite concentration ranges
- **Statistical Thresholds**: Appropriate statistical significance thresholds
- **Clinical Guidelines**: Alignment with established clinical practices

---

## 6. Computational Implementation

### 6.1 Main Relevance Scorer Class

```python
class ClinicalMetabolomicsRelevanceScorer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.query_classifier = QueryTypeClassifier()
        self.semantic_engine = SemanticSimilarityEngine()
        self.keyword_matcher = WeightedKeywordMatcher()
        self.domain_validator = DomainExpertiseValidator()
        self.weighting_manager = WeightingSchemeManager()
        
    async def calculate_relevance_score(self, 
                                      query: str, 
                                      response: str, 
                                      metadata: Dict[str, Any] = None) -> RelevanceScore:
        """
        Calculate comprehensive relevance score for clinical metabolomics response.
        
        Args:
            query: Original user query
            response: System response to evaluate
            metadata: Optional metadata about the query/response context
            
        Returns:
            RelevanceScore: Comprehensive scoring results
        """
        # Step 1: Classify query type
        query_type = self.query_classifier.classify_query(query)
        
        # Step 2: Get appropriate weighting scheme
        weights = self.weighting_manager.get_weights(query_type)
        
        # Step 3: Calculate dimension scores in parallel
        dimension_scores = await self._calculate_all_dimensions(query, response, metadata)
        
        # Step 4: Calculate weighted overall score
        overall_score = self._calculate_weighted_score(dimension_scores, weights)
        
        # Step 5: Generate explanation and confidence metrics
        explanation = self._generate_explanation(dimension_scores, weights, query_type)
        
        return RelevanceScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            query_type=query_type,
            weights_used=weights,
            explanation=explanation,
            confidence_score=self._calculate_confidence(dimension_scores),
            processing_time_ms=self._get_processing_time()
        )
    
    async def _calculate_all_dimensions(self, query: str, response: str, metadata: Dict) -> Dict[str, float]:
        """Calculate all relevance dimensions efficiently."""
        tasks = [
            self._calculate_metabolomics_relevance(query, response),
            self._calculate_clinical_applicability(query, response),
            self._calculate_query_alignment(query, response),
            self._calculate_scientific_rigor(response),
            self._calculate_biomedical_context_depth(response)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            'metabolomics_relevance': results[0],
            'clinical_applicability': results[1],
            'query_alignment': results[2],
            'scientific_rigor': results[3],
            'biomedical_context_depth': results[4]
        }
```

### 6.2 Performance Optimization

#### 6.2.1 Caching Strategy
```python
class RelevanceScoreCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get_cached_score(self, query_hash: str, response_hash: str) -> Optional[RelevanceScore]:
        cache_key = f"{query_hash}:{response_hash}"
        if cache_key in self.cache:
            score, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                return score
            else:
                del self.cache[cache_key]
        return None
```

#### 6.2.2 Parallel Processing
- Dimension calculations run concurrently
- Semantic similarity computed asynchronously
- Keyword matching optimized with compiled regex patterns
- Domain validation rules pre-compiled and indexed

---

## 7. Scoring Ranges and Interpretation

### 7.1 Score Ranges

| Score Range | Interpretation | Action |
|-------------|----------------|---------|
| 90-100 | Excellent Relevance | Response directly addresses query with high clinical metabolomics relevance |
| 80-89 | Good Relevance | Response is relevant but may lack some specificity or depth |
| 70-79 | Acceptable Relevance | Response addresses query but with notable gaps or tangential content |
| 60-69 | Marginal Relevance | Response has some relevance but significant issues with alignment |
| 50-59 | Poor Relevance | Response barely addresses the query, major relevance issues |
| 0-49 | Irrelevant | Response does not address the query or contains incorrect focus |

### 7.2 Dimension-Specific Thresholds

#### 7.2.1 Quality Gates
- **Metabolomics Relevance**: Minimum 60 for clinical metabolomics queries
- **Clinical Applicability**: Minimum 50 for clinical application queries  
- **Query Alignment**: Minimum 65 for all query types
- **Scientific Rigor**: Minimum 55 for research methodology queries

#### 7.2.2 Confidence Scoring
```python
def calculate_confidence_score(dimension_scores: Dict[str, float], 
                             score_variance: float) -> float:
    """
    Calculate confidence in the relevance score based on:
    - Consistency across dimensions
    - Score variance
    - Query clarity
    - Response length and structure
    """
    consistency_score = 100 - (score_variance * 10)  # Lower variance = higher confidence
    dimension_agreement = calculate_dimension_agreement(dimension_scores)
    
    confidence = (consistency_score * 0.6) + (dimension_agreement * 0.4)
    return max(0, min(100, confidence))
```

---

## 8. Integration Guidelines

### 8.1 Integration with ResponseQualityAssessor

```python
# Enhanced integration in existing ResponseQualityAssessor
class EnhancedResponseQualityAssessor(ResponseQualityAssessor):
    def __init__(self):
        super().__init__()
        self.relevance_scorer = ClinicalMetabolomicsRelevanceScorer()
    
    async def assess_response_quality(self, query, response, source_documents, expected_concepts):
        # Get base quality metrics
        base_metrics = await super().assess_response_quality(
            query, response, source_documents, expected_concepts
        )
        
        # Get enhanced relevance scoring
        relevance_details = await self.relevance_scorer.calculate_relevance_score(
            query, response, {'source_documents': source_documents}
        )
        
        # Update relevance score with enhanced calculation
        base_metrics.relevance_score = relevance_details.overall_score
        base_metrics.assessment_details['relevance_breakdown'] = relevance_details.dimension_scores
        base_metrics.assessment_details['query_type'] = relevance_details.query_type
        base_metrics.assessment_details['relevance_confidence'] = relevance_details.confidence_score
        
        return base_metrics
```

### 8.2 API Integration Points

```python
# Integration with ClinicalMetabolomicsRAG
class ClinicalMetabolomicsRAG:
    def __init__(self):
        self.quality_assessor = EnhancedResponseQualityAssessor()
    
    async def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
        # ... existing query processing ...
        
        # Enhanced quality assessment
        if self.config.enable_relevance_scoring:
            quality_metrics = await self.quality_assessor.assess_response_quality(
                query=query_text,
                response=response,
                source_documents=source_docs,
                expected_concepts=self._extract_expected_concepts(query_text)
            )
            
            result['quality_metrics'] = quality_metrics
            result['relevance_details'] = quality_metrics.assessment_details.get('relevance_breakdown')
        
        return result
```

---

## 9. Validation and Testing Strategy

### 9.1 Validation Methodology

#### 9.1.1 Expert Evaluation Dataset
- **Size**: 200+ query-response pairs
- **Coverage**: All query types and score ranges
- **Experts**: 3+ clinical metabolomics researchers
- **Agreement**: Minimum 80% inter-rater agreement

#### 9.1.2 Automated Testing
```python
class RelevanceScoringTestSuite:
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.scorer = ClinicalMetabolomicsRelevanceScorer()
    
    async def test_score_consistency(self):
        """Test that identical queries get consistent scores."""
        for query, response in self.test_cases:
            scores = []
            for _ in range(5):  # Run 5 times
                score = await self.scorer.calculate_relevance_score(query, response)
                scores.append(score.overall_score)
            
            variance = statistics.variance(scores)
            assert variance < 1.0, f"Score variance too high: {variance}"
    
    async def test_score_boundaries(self):
        """Test that scores stay within expected boundaries."""
        # Test cases with known expected score ranges
        boundary_tests = [
            ("What is metabolomics?", "Metabolomics is the study of metabolites", (70, 90)),
            ("Tell me about cars", "Cars are vehicles with four wheels", (0, 30)),
            ("LC-MS for metabolomics", "LC-MS is liquid chromatography mass spectrometry", (80, 95))
        ]
        
        for query, response, (min_score, max_score) in boundary_tests:
            score = await self.scorer.calculate_relevance_score(query, response)
            assert min_score <= score.overall_score <= max_score
```

### 9.2 Performance Benchmarks

#### 9.2.1 Computational Performance
- **Target Latency**: <200ms per scoring operation
- **Memory Usage**: <50MB for scorer initialization
- **Concurrent Capacity**: 100+ simultaneous scoring operations

#### 9.2.2 Accuracy Benchmarks
- **Expert Agreement**: >85% correlation with expert ratings
- **Cross-Query Type Consistency**: Variance <15% across query types
- **Temporal Stability**: <5% score drift over repeated evaluations

---

## 10. Future Enhancement Opportunities

### 10.1 Advanced Features

#### 10.1.1 Multi-Modal Relevance
- **Image Relevance**: Scoring responses that include metabolic pathway diagrams
- **Table Relevance**: Evaluating tabular data relevance and accuracy
- **Citation Network Analysis**: Assessing citation relevance and quality

#### 10.1.2 Contextual Adaptation
- **User Expertise Level**: Adjusting scoring based on user background
- **Query History**: Learning from previous user interactions
- **Temporal Context**: Considering recent developments in the field

#### 10.1.3 Real-Time Learning
- **Feedback Integration**: Learning from user relevance feedback
- **Expert Annotations**: Incorporating expert corrections and annotations
- **Performance Adaptation**: Self-tuning based on accuracy metrics

### 10.2 Research Integration

#### 10.2.1 Knowledge Graph Integration
- **Metabolite Networks**: Using metabolite relationship graphs for relevance
- **Literature Graphs**: Citation networks for relevance validation
- **Pathway Databases**: Integration with KEGG, Reactome, WikiPathways

#### 10.2.2 Advanced NLP Techniques
- **Attention Mechanisms**: Understanding which parts of responses are most relevant
- **Discourse Analysis**: Evaluating response structure and flow
- **Causal Reasoning**: Assessing causal claims and logical consistency

---

## 11. Implementation Roadmap

### 11.1 Phase 1: Core Implementation (Weeks 1-2)
1. Implement basic RelevanceScorer class structure
2. Develop QueryTypeClassifier with basic keyword matching
3. Create dimension scoring methods (metabolomics_relevance, clinical_applicability)
4. Implement WeightingSchemeManager
5. Basic integration with ResponseQualityAssessor

### 11.2 Phase 2: Enhanced Features (Weeks 3-4)
1. Implement SemanticSimilarityEngine with BioBERT
2. Advanced keyword matching with domain-specific weights
3. DomainExpertiseValidator implementation
4. Performance optimization and caching
5. Comprehensive testing suite

### 11.3 Phase 3: Validation and Refinement (Week 5-6)
1. Expert evaluation dataset creation
2. Validation studies and accuracy benchmarking
3. Performance optimization
4. Documentation and API finalization
5. Integration testing with full system

---

## 12. Configuration and Deployment

### 12.1 Configuration Options

```python
RELEVANCE_SCORING_CONFIG = {
    'enabled': True,
    'model_settings': {
        'semantic_model': 'dmis-lab/biobert-v1.1',
        'embedding_cache_size': 1000,
        'similarity_threshold': 0.5
    },
    'scoring_settings': {
        'enable_caching': True,
        'cache_ttl_seconds': 3600,
        'parallel_processing': True,
        'max_concurrent_scorings': 50
    },
    'quality_thresholds': {
        'minimum_relevance': 60.0,
        'confidence_threshold': 70.0,
        'flag_low_scores': True
    },
    'query_classification': {
        'enable_auto_classification': True,
        'classification_confidence_threshold': 0.7,
        'fallback_query_type': 'general'
    }
}
```

### 12.2 Monitoring and Logging

```python
class RelevanceScoringMonitor:
    def __init__(self):
        self.metrics = {
            'total_scorings': 0,
            'average_score': 0.0,
            'score_distribution': defaultdict(int),
            'processing_times': [],
            'error_count': 0
        }
    
    def log_scoring_event(self, score: RelevanceScore, processing_time: float):
        self.metrics['total_scorings'] += 1
        self.metrics['processing_times'].append(processing_time)
        
        score_bucket = int(score.overall_score // 10) * 10
        self.metrics['score_distribution'][score_bucket] += 1
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        return {
            'period': datetime.now().isoformat(),
            'total_scorings': self.metrics['total_scorings'],
            'average_processing_time': statistics.mean(self.metrics['processing_times']),
            'score_distribution': dict(self.metrics['score_distribution']),
            'error_rate': self.metrics['error_count'] / max(self.metrics['total_scorings'], 1)
        }
```

---

## 13. Conclusion

This design provides a comprehensive, scientifically-grounded approach to response relevance scoring for clinical metabolomics applications. The algorithm balances accuracy, performance, and explainability while providing the flexibility needed for diverse query types and use cases.

**Key Advantages:**
1. **Domain-Specific Optimization**: Tailored for clinical metabolomics terminology and concepts
2. **Multi-Dimensional Scoring**: Comprehensive evaluation across five key relevance dimensions  
3. **Adaptive Weighting**: Query-type specific optimization for maximum accuracy
4. **Real-Time Performance**: Sub-200ms scoring for responsive user experiences
5. **Integration-Ready**: Seamless integration with existing quality assessment infrastructure
6. **Explainable Results**: Detailed scoring breakdowns for transparency and debugging

The implementation of this design will significantly enhance the Clinical Metabolomics Oracle's ability to assess and ensure response relevance, contributing to improved user experience and scientific accuracy in biomedical query-response systems.

---

**Next Steps for CMO-LIGHTRAG-009-T02:**
1. Review and approve this design document
2. Begin Phase 1 implementation focusing on core scorer architecture
3. Develop comprehensive test cases for validation
4. Integrate with existing ResponseQualityAssessor infrastructure
5. Conduct expert evaluation studies for validation and refinement