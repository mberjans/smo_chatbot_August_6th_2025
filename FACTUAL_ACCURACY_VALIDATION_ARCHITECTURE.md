# Factual Accuracy Validation Architecture Design

## Clinical Metabolomics Oracle - LightRAG Integration

**Document Version:** 1.0.0  
**Author:** Claude Code (Anthropic)  
**Date:** August 7, 2025  
**Related Task:** Factual Accuracy Validation System Design

---

## Executive Summary

This document presents the comprehensive architecture design for implementing factual accuracy validation against source documents in the Clinical Metabolomics Oracle LightRAG integration project. The design seamlessly integrates with the existing `ClinicalMetabolomicsRelevanceScorer` and `ResponseQualityAssessor` infrastructure while providing sophisticated claim extraction, document verification, and real-time validation capabilities.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Class Architecture Design](#class-architecture-design)
3. [Claim Extraction Strategy](#claim-extraction-strategy)
4. [Document Verification Process](#document-verification-process)
5. [Scoring and Reporting System](#scoring-and-reporting-system)
6. [Performance Optimization](#performance-optimization)
7. [Integration Points](#integration-points)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Error Handling and Recovery](#error-handling-and-recovery)
10. [Testing Strategy](#testing-strategy)

---

## Architecture Overview

### System Components Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Factual Accuracy Validation System          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Claim Extractor │  │ Document Matcher │  │ Accuracy      │  │
│  │                 │  │                  │  │ Scorer        │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
│           │                      │                     │        │
│           v                      v                     v        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │           Source Document Index & Cache                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Integration Layer                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │         ClinicalMetabolomicsRelevanceScorer                 │ │
│  │                   (Enhanced)                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              ResponseQualityAssessor                        │ │
│  │                   (Enhanced)                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Query → LightRAG → Response
                     ↓
    ┌────────────────────────────────────────┐
    │        Factual Claim Extraction        │
    └────────────────────────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │      Source Document Retrieval         │
    └────────────────────────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │      Claim Verification Process        │
    └────────────────────────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │       Accuracy Scoring & Reporting     │
    └────────────────────────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │    Integration with Quality Pipeline   │
    └────────────────────────────────────────┘
```

---

## Class Architecture Design

### Core Classes and Relationships

```python
# =====================================================================
# FACTUAL ACCURACY VALIDATION CORE CLASSES
# =====================================================================

@dataclass
class FactualClaim:
    """Represents a factual claim extracted from response text."""
    claim_id: str
    claim_text: str
    claim_type: str  # 'numeric', 'qualitative', 'causal', 'temporal'
    confidence: float  # 0-1, extraction confidence
    context: str  # Surrounding text context
    position: Tuple[int, int]  # Start and end position in text
    biomedical_domain: str  # 'metabolomics', 'clinical', 'analytical'
    
    # Extracted components for numeric claims
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    range_bounds: Optional[Tuple[float, float]] = None
    
    # Extracted components for qualitative claims
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    
    # Validation status
    validation_status: str = "pending"  # 'pending', 'verified', 'contradicted', 'unclear'
    validation_confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)

@dataclass  
class DocumentMatch:
    """Represents a match between claim and source document content."""
    document_id: str
    document_title: str
    matched_text: str
    match_score: float  # 0-1, similarity score
    match_type: str  # 'exact', 'semantic', 'numeric', 'contextual'
    page_number: Optional[int] = None
    section: Optional[str] = None
    context_window: str = ""  # Surrounding text for context

@dataclass
class FactualAccuracyResult:
    """Comprehensive factual accuracy validation results."""
    overall_accuracy_score: float  # 0-100
    total_claims: int
    verified_claims: int
    contradicted_claims: int
    unverifiable_claims: int
    
    claim_validations: List[Tuple[FactualClaim, List[DocumentMatch]]] = field(default_factory=list)
    accuracy_breakdown: Dict[str, float] = field(default_factory=dict)  # By claim type
    confidence_assessment: float = 0.0
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def accuracy_grade(self) -> str:
        """Convert accuracy score to grade."""
        if self.overall_accuracy_score >= 90:
            return "Highly Accurate"
        elif self.overall_accuracy_score >= 80:
            return "Accurate"
        elif self.overall_accuracy_score >= 70:
            return "Mostly Accurate"
        elif self.overall_accuracy_score >= 60:
            return "Partially Accurate"
        else:
            return "Low Accuracy"
```

### Main Validator Architecture

```python
class FactualAccuracyValidator:
    """
    Core factual accuracy validation engine for clinical metabolomics responses.
    
    This class orchestrates the entire factual accuracy validation pipeline:
    - Extracts factual claims from LightRAG responses
    - Matches claims against source documents in the knowledge base
    - Validates claims using semantic similarity and exact matching
    - Provides detailed scoring and confidence assessments
    """
    
    def __init__(self, 
                 document_index: 'SourceDocumentIndex',
                 semantic_matcher: 'SemanticSimilarityMatcher',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the factual accuracy validator.
        
        Args:
            document_index: Source document indexing and retrieval system
            semantic_matcher: Semantic similarity matching engine
            config: Configuration parameters for validation behavior
        """
        self.document_index = document_index
        self.semantic_matcher = semantic_matcher  
        self.config = config or self._get_default_config()
        
        # Initialize claim extraction components
        self.claim_extractor = BiomedicalClaimExtractor(self.config)
        self.document_matcher = DocumentContentMatcher(
            self.document_index, self.semantic_matcher, self.config
        )
        self.accuracy_scorer = AccuracyScorer(self.config)
        
        # Performance optimization components
        self.claim_cache = ClaimValidationCache(self.config)
        self.batch_processor = BatchValidationProcessor(self.config)
    
    async def validate_factual_accuracy(self,
                                      response: str,
                                      query_context: Optional[str] = None,
                                      source_document_ids: Optional[List[str]] = None) -> FactualAccuracyResult:
        """
        Main entry point for factual accuracy validation.
        
        Args:
            response: LightRAG response text to validate
            query_context: Original query for context
            source_document_ids: Specific documents to validate against
            
        Returns:
            FactualAccuracyResult: Comprehensive validation results
        """
        
        # Step 1: Extract factual claims from response
        claims = await self.claim_extractor.extract_claims(
            response, query_context
        )
        
        if not claims:
            return self._create_empty_result("No factual claims detected")
        
        # Step 2: Batch validate claims against source documents
        validation_results = await self.batch_processor.validate_claims_batch(
            claims, source_document_ids
        )
        
        # Step 3: Calculate comprehensive accuracy scores
        accuracy_result = self.accuracy_scorer.calculate_accuracy_scores(
            claims, validation_results
        )
        
        # Step 4: Cache results for future use
        await self.claim_cache.cache_validation_results(
            response, accuracy_result
        )
        
        return accuracy_result
```

---

## Claim Extraction Strategy

### Biomedical Claim Categories

```python
class BiomedicalClaimExtractor:
    """
    Specialized claim extraction for biomedical and metabolomics content.
    
    Extracts different types of factual claims with domain-specific patterns
    and natural language processing techniques.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Claim extraction patterns by type
        self.claim_patterns = {
            'numeric_claims': [
                # Concentrations and measurements
                r'(\d+(?:\.\d+)?)\s*(µM|nM|mM|mg/dL|ng/mL|pmol/L)',
                # Percentages and ratios  
                r'(\d+(?:\.\d+)?)\s*(%|percent|fold|times)',
                # Statistical values
                r'p\s*[<>=]\s*(\d+(?:\.\d+)?)',
                r'r\s*=\s*(\d+(?:\.\d+)?)',
                # Sample sizes
                r'n\s*=\s*(\d+)',
            ],
            
            'qualitative_claims': [
                # Causal relationships
                r'(\w+(?:\s+\w+)*)\s+(increases?|decreases?|affects?|causes?|leads\s+to)\s+(\w+(?:\s+\w+)*)',
                # Comparative statements
                r'(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(higher|lower|greater|less|more|fewer)\s+(?:than\s+)?(\w+(?:\s+\w+)*)',
                # Diagnostic relationships
                r'(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(?:a\s+)?(?:biomarker|indicator|marker)\s+(?:for|of)\s+(\w+(?:\s+\w+)*)',
            ],
            
            'temporal_claims': [
                # Time-based relationships
                r'(?:after|before|during|within)\s+(\d+(?:\.\d+)?)\s*(minutes?|hours?|days?|weeks?|months?)',
                # Sequential relationships
                r'(\w+(?:\s+\w+)*)\s+(?:precedes?|follows?|occurs?\s+before|occurs?\s+after)\s+(\w+(?:\s+\w+)*)',
            ],
            
            'methodological_claims': [
                # Analytical methods
                r'(?:using|by|via|through)\s+(LC-MS|GC-MS|NMR|UPLC|HILIC)',
                # Sample preparation
                r'samples?\s+(?:were|are)\s+(extracted|prepared|analyzed|processed)',
                # Study design
                r'(?:randomized|controlled|double-blind|placebo-controlled)\s+(?:trial|study)',
            ]
        }
        
        # Biomedical entity recognition patterns
        self.entity_patterns = {
            'metabolites': r'\b(?:[A-Z][a-z]+-)*[A-Z][a-z]+(?:-\d+)*\b',
            'diseases': r'\b(?:type\s+\d+\s+)?(?:diabetes|cancer|alzheimer|cardiovascular|obesity)\b',
            'pathways': r'\b(?:glycolysis|krebs\s+cycle|fatty\s+acid\s+oxidation|gluconeogenesis)\b',
            'proteins': r'\b[A-Z]{2,}[0-9]*\b',  # Common protein naming pattern
        }
    
    async def extract_claims(self, 
                           response: str, 
                           query_context: Optional[str] = None) -> List[FactualClaim]:
        """
        Extract factual claims from biomedical response text.
        
        Args:
            response: Response text to analyze
            query_context: Original query for contextual understanding
            
        Returns:
            List of extracted FactualClaim objects
        """
        claims = []
        
        # Extract different types of claims
        numeric_claims = self._extract_numeric_claims(response)
        qualitative_claims = self._extract_qualitative_claims(response)
        temporal_claims = self._extract_temporal_claims(response)
        methodological_claims = self._extract_methodological_claims(response)
        
        claims.extend(numeric_claims)
        claims.extend(qualitative_claims)
        claims.extend(temporal_claims)
        claims.extend(methodological_claims)
        
        # Post-process claims for domain classification
        for claim in claims:
            claim.biomedical_domain = self._classify_biomedical_domain(
                claim.claim_text, query_context
            )
        
        return claims
    
    def _extract_numeric_claims(self, text: str) -> List[FactualClaim]:
        """Extract numeric claims with values, units, and statistical measures."""
        claims = []
        
        for pattern in self.claim_patterns['numeric_claims']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                claim_text = match.group(0)
                start, end = match.span()
                
                # Extract numeric value and unit
                groups = match.groups()
                numeric_value = float(groups[0]) if groups[0] else None
                unit = groups[1] if len(groups) > 1 else None
                
                # Get surrounding context
                context = self._get_context_window(text, start, end, window_size=100)
                
                claim = FactualClaim(
                    claim_id=f"numeric_{len(claims)}_{start}",
                    claim_text=claim_text,
                    claim_type="numeric",
                    confidence=self._calculate_extraction_confidence(claim_text, context),
                    context=context,
                    position=(start, end),
                    biomedical_domain="",  # Will be set later
                    numeric_value=numeric_value,
                    unit=unit
                )
                
                claims.append(claim)
        
        return claims
    
    def _extract_qualitative_claims(self, text: str) -> List[FactualClaim]:
        """Extract qualitative claims about relationships and properties."""
        claims = []
        
        for pattern in self.claim_patterns['qualitative_claims']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                claim_text = match.group(0)
                start, end = match.span()
                groups = match.groups()
                
                # Extract subject-predicate-object structure
                subject = groups[0] if groups else None
                predicate = groups[1] if len(groups) > 1 else None
                obj = groups[2] if len(groups) > 2 else None
                
                context = self._get_context_window(text, start, end, window_size=150)
                
                claim = FactualClaim(
                    claim_id=f"qualitative_{len(claims)}_{start}",
                    claim_text=claim_text,
                    claim_type="qualitative",
                    confidence=self._calculate_extraction_confidence(claim_text, context),
                    context=context,
                    position=(start, end),
                    biomedical_domain="",  # Will be set later
                    subject=subject,
                    predicate=predicate,
                    object=obj
                )
                
                claims.append(claim)
        
        return claims
```

---

## Document Verification Process

### Source Document Indexing

```python
class SourceDocumentIndex:
    """
    Advanced indexing system for source documents with multiple access patterns.
    
    Provides fast retrieval by content similarity, entity mentions, 
    and structured metadata queries.
    """
    
    def __init__(self, 
                 document_store: 'DocumentStore',
                 embedding_model: 'EmbeddingModel',
                 config: Dict[str, Any]):
        self.document_store = document_store
        self.embedding_model = embedding_model
        self.config = config
        
        # Multi-level indexing structures
        self.content_index = {}  # Full-text search index
        self.entity_index = {}   # Entity mention index
        self.semantic_index = {} # Embedding-based semantic index
        self.metadata_index = {} # Structured metadata index
        
        # Caching for performance
        self.document_cache = LRUCache(maxsize=self.config.get('document_cache_size', 1000))
        self.embedding_cache = LRUCache(maxsize=self.config.get('embedding_cache_size', 5000))
    
    async def build_index(self, documents: List['Document']) -> None:
        """Build comprehensive index from source documents."""
        
        tasks = []
        for doc in documents:
            # Process documents in batches for memory efficiency
            task = asyncio.create_task(self._index_document(doc))
            tasks.append(task)
            
            # Process in batches to avoid memory overflow
            if len(tasks) >= self.config.get('batch_size', 50):
                await asyncio.gather(*tasks)
                tasks = []
        
        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks)
    
    async def find_relevant_documents(self,
                                    claim: FactualClaim,
                                    max_documents: int = 10) -> List['Document']:
        """Find documents most relevant to validating a specific claim."""
        
        # Multi-strategy document retrieval
        strategies = [
            self._find_by_semantic_similarity(claim),
            self._find_by_entity_mentions(claim),
            self._find_by_content_keywords(claim),
            self._find_by_domain_metadata(claim)
        ]
        
        # Execute strategies concurrently
        strategy_results = await asyncio.gather(*strategies, return_exceptions=True)
        
        # Combine and rank results
        combined_results = self._combine_retrieval_results(
            strategy_results, claim, max_documents
        )
        
        return combined_results
    
    async def _find_by_semantic_similarity(self, claim: FactualClaim) -> List[Tuple['Document', float]]:
        """Find documents using semantic similarity to claim text."""
        
        # Generate embedding for claim
        claim_embedding = await self.embedding_model.encode(claim.claim_text)
        
        # Find similar document chunks
        similar_chunks = await self.semantic_index.find_similar(
            claim_embedding, 
            top_k=self.config.get('semantic_retrieval_k', 20)
        )
        
        # Group by document and aggregate scores
        document_scores = {}
        for chunk_id, similarity_score in similar_chunks:
            doc_id = self._get_document_id_from_chunk(chunk_id)
            if doc_id not in document_scores:
                document_scores[doc_id] = []
            document_scores[doc_id].append(similarity_score)
        
        # Calculate aggregate scores and return sorted results
        results = []
        for doc_id, scores in document_scores.items():
            document = await self.document_store.get_document(doc_id)
            if document:
                # Use max score as aggregate (could also use mean or weighted sum)
                aggregate_score = max(scores)
                results.append((document, aggregate_score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
```

### Document Content Matching

```python
class DocumentContentMatcher:
    """
    Advanced content matching between claims and source documents.
    
    Supports multiple matching strategies including exact matching,
    semantic similarity, and contextual understanding.
    """
    
    def __init__(self, 
                 document_index: SourceDocumentIndex,
                 semantic_matcher: 'SemanticSimilarityMatcher',
                 config: Dict[str, Any]):
        self.document_index = document_index
        self.semantic_matcher = semantic_matcher
        self.config = config
        
        # Matching thresholds
        self.exact_match_threshold = config.get('exact_match_threshold', 0.95)
        self.semantic_match_threshold = config.get('semantic_match_threshold', 0.75)
        self.contextual_match_threshold = config.get('contextual_match_threshold', 0.70)
    
    async def find_claim_matches(self, 
                               claim: FactualClaim,
                               documents: List['Document']) -> List[DocumentMatch]:
        """
        Find all matches for a claim across the provided documents.
        
        Args:
            claim: Factual claim to validate
            documents: Source documents to search
            
        Returns:
            List of DocumentMatch objects ranked by confidence
        """
        matches = []
        
        # Process documents concurrently for performance
        matching_tasks = [
            self._match_claim_in_document(claim, doc) 
            for doc in documents
        ]
        
        document_matches = await asyncio.gather(*matching_tasks, return_exceptions=True)
        
        # Collect all valid matches
        for doc_matches in document_matches:
            if isinstance(doc_matches, Exception):
                continue
            matches.extend(doc_matches)
        
        # Sort by match score (highest first)
        matches.sort(key=lambda m: m.match_score, reverse=True)
        
        return matches
    
    async def _match_claim_in_document(self, 
                                     claim: FactualClaim, 
                                     document: 'Document') -> List[DocumentMatch]:
        """Match a claim within a single document using multiple strategies."""
        matches = []
        
        # Strategy 1: Exact/Near-exact matching for numeric claims
        if claim.claim_type == "numeric":
            exact_matches = self._find_exact_numeric_matches(claim, document)
            matches.extend(exact_matches)
        
        # Strategy 2: Semantic similarity matching
        semantic_matches = await self._find_semantic_matches(claim, document)
        matches.extend(semantic_matches)
        
        # Strategy 3: Entity-based contextual matching
        contextual_matches = self._find_contextual_matches(claim, document)
        matches.extend(contextual_matches)
        
        # Strategy 4: Pattern-based matching for methodological claims
        if claim.claim_type == "methodological":
            pattern_matches = self._find_pattern_matches(claim, document)
            matches.extend(pattern_matches)
        
        return matches
    
    def _find_exact_numeric_matches(self, 
                                   claim: FactualClaim, 
                                   document: 'Document') -> List[DocumentMatch]:
        """Find exact or near-exact matches for numeric claims."""
        matches = []
        
        if not claim.numeric_value:
            return matches
        
        # Create search patterns for the numeric value
        value_patterns = [
            rf'{re.escape(str(claim.numeric_value))}\s*{re.escape(claim.unit or "")}',
            rf'{claim.numeric_value:.1f}\s*{re.escape(claim.unit or "")}' if claim.unit else f'{claim.numeric_value:.1f}',
        ]
        
        # Allow for small variations in numeric values (±5%)
        tolerance = claim.numeric_value * 0.05
        lower_bound = claim.numeric_value - tolerance
        upper_bound = claim.numeric_value + tolerance
        
        for chunk in document.text_chunks:
            # Look for exact matches
            for pattern in value_patterns:
                for match in re.finditer(pattern, chunk.text, re.IGNORECASE):
                    match_score = self.exact_match_threshold
                    context = self._get_chunk_context(chunk, match.span())
                    
                    doc_match = DocumentMatch(
                        document_id=document.document_id,
                        document_title=document.title,
                        matched_text=match.group(0),
                        match_score=match_score,
                        match_type="exact",
                        page_number=chunk.page_number,
                        section=chunk.section,
                        context_window=context
                    )
                    matches.append(doc_match)
            
            # Look for values within tolerance range
            numeric_pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(claim.unit or "")}'
            for match in re.finditer(numeric_pattern, chunk.text, re.IGNORECASE):
                try:
                    found_value = float(match.group(1))
                    if lower_bound <= found_value <= upper_bound:
                        # Calculate match score based on proximity
                        proximity = 1.0 - abs(found_value - claim.numeric_value) / claim.numeric_value
                        match_score = self.exact_match_threshold * proximity
                        
                        context = self._get_chunk_context(chunk, match.span())
                        
                        doc_match = DocumentMatch(
                            document_id=document.document_id,
                            document_title=document.title,
                            matched_text=match.group(0),
                            match_score=match_score,
                            match_type="numeric",
                            page_number=chunk.page_number,
                            section=chunk.section,
                            context_window=context
                        )
                        matches.append(doc_match)
                except ValueError:
                    continue
        
        return matches
    
    async def _find_semantic_matches(self, 
                                   claim: FactualClaim, 
                                   document: 'Document') -> List[DocumentMatch]:
        """Find matches using semantic similarity."""
        matches = []
        
        # Generate embedding for claim text
        claim_embedding = await self.semantic_matcher.encode(claim.claim_text)
        
        # Compare against document chunks
        for chunk in document.text_chunks:
            chunk_embedding = await self.semantic_matcher.encode(chunk.text)
            
            similarity = await self.semantic_matcher.calculate_similarity(
                claim_embedding, chunk_embedding
            )
            
            if similarity >= self.semantic_match_threshold:
                # Find the most relevant sentence within the chunk
                best_sentence, sentence_similarity = await self._find_best_sentence_match(
                    claim.claim_text, chunk.text
                )
                
                doc_match = DocumentMatch(
                    document_id=document.document_id,
                    document_title=document.title,
                    matched_text=best_sentence,
                    match_score=sentence_similarity,
                    match_type="semantic",
                    page_number=chunk.page_number,
                    section=chunk.section,
                    context_window=chunk.text
                )
                matches.append(doc_match)
        
        return matches
```

---

## Scoring and Reporting System

### Multi-dimensional Accuracy Scoring

```python
class AccuracyScorer:
    """
    Comprehensive accuracy scoring system for factual claims.
    
    Provides multi-dimensional scoring with confidence levels
    and detailed breakdown by claim types and validation methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Scoring weights by claim type
        self.claim_type_weights = {
            'numeric': 0.35,      # High weight for quantitative claims
            'qualitative': 0.25,  # Moderate weight for relationships
            'temporal': 0.15,     # Lower weight for time-based claims  
            'methodological': 0.25 # Moderate weight for methods
        }
        
        # Validation method weights
        self.validation_weights = {
            'exact': 1.0,         # Full confidence for exact matches
            'semantic': 0.8,      # High confidence for semantic matches
            'contextual': 0.6,    # Moderate confidence for contextual matches
            'numeric': 0.9        # High confidence for numeric matches
        }
    
    def calculate_accuracy_scores(self, 
                                claims: List[FactualClaim],
                                validation_results: List[Tuple[FactualClaim, List[DocumentMatch]]]) -> FactualAccuracyResult:
        """
        Calculate comprehensive accuracy scores from validation results.
        
        Args:
            claims: Original extracted claims
            validation_results: Results of claim validation against documents
            
        Returns:
            FactualAccuracyResult: Comprehensive accuracy assessment
        """
        
        if not claims:
            return FactualAccuracyResult(
                overall_accuracy_score=0.0,
                total_claims=0,
                verified_claims=0,
                contradicted_claims=0,
                unverifiable_claims=0
            )
        
        # Initialize counters and accumulators
        verified_claims = 0
        contradicted_claims = 0
        unverifiable_claims = 0
        claim_scores = []
        claim_validations = []
        accuracy_by_type = {}
        
        # Process each claim and its validation results
        for claim, matches in validation_results:
            claim_accuracy = self._calculate_claim_accuracy(claim, matches)
            claim_scores.append(claim_accuracy)
            claim_validations.append((claim, matches))
            
            # Categorize claim validation status
            if claim_accuracy >= self.config.get('verification_threshold', 0.8):
                verified_claims += 1
                claim.validation_status = "verified"
            elif claim_accuracy <= self.config.get('contradiction_threshold', 0.3):
                contradicted_claims += 1
                claim.validation_status = "contradicted"
            else:
                unverifiable_claims += 1
                claim.validation_status = "unclear"
            
            claim.validation_confidence = claim_accuracy
            
            # Track accuracy by claim type
            if claim.claim_type not in accuracy_by_type:
                accuracy_by_type[claim.claim_type] = []
            accuracy_by_type[claim.claim_type].append(claim_accuracy)
        
        # Calculate overall accuracy score
        if claim_scores:
            # Weighted average by claim type importance
            weighted_scores = []
            for claim, score in zip(claims, claim_scores):
                weight = self.claim_type_weights.get(claim.claim_type, 0.25)
                weighted_scores.append(score * weight)
            
            overall_accuracy = sum(weighted_scores) / len(weighted_scores) * 100
        else:
            overall_accuracy = 0.0
        
        # Calculate accuracy breakdown by type
        type_accuracy = {}
        for claim_type, scores in accuracy_by_type.items():
            type_accuracy[claim_type] = sum(scores) / len(scores) * 100 if scores else 0.0
        
        # Calculate confidence assessment
        confidence = self._calculate_confidence_assessment(claims, validation_results)
        
        # Prepare validation details
        validation_details = {
            'total_matches_found': sum(len(matches) for _, matches in validation_results),
            'avg_matches_per_claim': sum(len(matches) for _, matches in validation_results) / len(claims) if claims else 0,
            'validation_methods_used': list(set(
                match.match_type for _, matches in validation_results for match in matches
            )),
            'high_confidence_claims': sum(1 for score in claim_scores if score >= 0.9),
            'low_confidence_claims': sum(1 for score in claim_scores if score <= 0.5),
        }
        
        return FactualAccuracyResult(
            overall_accuracy_score=overall_accuracy,
            total_claims=len(claims),
            verified_claims=verified_claims,
            contradicted_claims=contradicted_claims,
            unverifiable_claims=unverifiable_claims,
            claim_validations=claim_validations,
            accuracy_breakdown=type_accuracy,
            confidence_assessment=confidence,
            validation_details=validation_details
        )
    
    def _calculate_claim_accuracy(self, 
                                claim: FactualClaim, 
                                matches: List[DocumentMatch]) -> float:
        """Calculate accuracy score for a single claim based on its matches."""
        
        if not matches:
            return 0.0
        
        # Different strategies based on claim type
        if claim.claim_type == "numeric":
            return self._calculate_numeric_claim_accuracy(claim, matches)
        elif claim.claim_type == "qualitative":
            return self._calculate_qualitative_claim_accuracy(claim, matches)
        else:
            return self._calculate_general_claim_accuracy(claim, matches)
    
    def _calculate_numeric_claim_accuracy(self, 
                                        claim: FactualClaim, 
                                        matches: List[DocumentMatch]) -> float:
        """Calculate accuracy for numeric claims with special handling for exact values."""
        
        # For numeric claims, exact matches get highest scores
        exact_matches = [m for m in matches if m.match_type == "exact"]
        numeric_matches = [m for m in matches if m.match_type == "numeric"]
        
        if exact_matches:
            # Use highest exact match score
            best_exact = max(exact_matches, key=lambda m: m.match_score)
            return best_exact.match_score
        
        elif numeric_matches:
            # Use highest numeric match score
            best_numeric = max(numeric_matches, key=lambda m: m.match_score)
            return best_numeric.match_score * self.validation_weights["numeric"]
        
        else:
            # Fall back to semantic matches
            return self._calculate_general_claim_accuracy(claim, matches)
    
    def _calculate_qualitative_claim_accuracy(self, 
                                            claim: FactualClaim, 
                                            matches: List[DocumentMatch]) -> float:
        """Calculate accuracy for qualitative relationship claims."""
        
        # For qualitative claims, we need to check for both supporting and contradicting evidence
        supporting_matches = []
        contradicting_matches = []
        
        for match in matches:
            # This would involve more sophisticated NLP to determine if the match
            # supports or contradicts the claim. For now, we'll use match score as proxy.
            if match.match_score >= self.config.get('support_threshold', 0.7):
                supporting_matches.append(match)
            elif self._detect_contradiction(claim, match):
                contradicting_matches.append(match)
        
        if contradicting_matches:
            # Strong contradicting evidence reduces accuracy significantly
            contradiction_penalty = len(contradicting_matches) * 0.3
            base_accuracy = max(0.1, 1.0 - contradiction_penalty)
        elif supporting_matches:
            # Multiple supporting matches increase confidence
            best_support = max(supporting_matches, key=lambda m: m.match_score)
            support_boost = min(0.2, (len(supporting_matches) - 1) * 0.05)
            base_accuracy = best_support.match_score + support_boost
        else:
            base_accuracy = 0.0
        
        return min(1.0, base_accuracy)
    
    def _detect_contradiction(self, claim: FactualClaim, match: DocumentMatch) -> bool:
        """Detect if a document match contradicts the claim."""
        
        # Simple contradiction detection based on negation patterns
        contradiction_patterns = [
            r'\b(?:not|no|never|neither|none)\b',
            r'\b(?:opposite|contrary|different|unlike)\b',
            r'\b(?:decreased?|reduced?|lower|less)\b' if 'increase' in claim.claim_text.lower() else None,
            r'\b(?:increased?|higher|greater|more)\b' if 'decrease' in claim.claim_text.lower() else None,
        ]
        
        contradiction_patterns = [p for p in contradiction_patterns if p is not None]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, match.context_window, re.IGNORECASE):
                return True
        
        return False
```

---

## Performance Optimization

### Caching Strategy

```python
class ClaimValidationCache:
    """
    Multi-level caching system for claim validation results.
    
    Provides caching at multiple levels:
    - Response-level caching for complete validation results
    - Claim-level caching for individual claim validations
    - Document match caching for reusable document matches
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Response-level cache (complete validation results)
        self.response_cache = TTLCache(
            maxsize=config.get('response_cache_size', 1000),
            ttl=config.get('response_cache_ttl', 3600)  # 1 hour
        )
        
        # Claim-level cache (individual claim validations)
        self.claim_cache = TTLCache(
            maxsize=config.get('claim_cache_size', 5000),
            ttl=config.get('claim_cache_ttl', 7200)  # 2 hours
        )
        
        # Document match cache (reusable document matches)
        self.match_cache = TTLCache(
            maxsize=config.get('match_cache_size', 10000),
            ttl=config.get('match_cache_ttl', 14400)  # 4 hours
        )
        
        # Cache hit/miss statistics
        self.cache_stats = {
            'response_hits': 0,
            'response_misses': 0,
            'claim_hits': 0,
            'claim_misses': 0,
            'match_hits': 0,
            'match_misses': 0
        }
    
    def _generate_response_cache_key(self, response: str) -> str:
        """Generate cache key for complete response validation."""
        import hashlib
        return hashlib.md5(response.encode()).hexdigest()
    
    def _generate_claim_cache_key(self, claim: FactualClaim) -> str:
        """Generate cache key for individual claim validation."""
        import hashlib
        key_data = f"{claim.claim_text}_{claim.claim_type}_{claim.biomedical_domain}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_cached_validation(self, response: str) -> Optional[FactualAccuracyResult]:
        """Retrieve cached validation result for complete response."""
        cache_key = self._generate_response_cache_key(response)
        
        if cache_key in self.response_cache:
            self.cache_stats['response_hits'] += 1
            return self.response_cache[cache_key]
        
        self.cache_stats['response_misses'] += 1
        return None
    
    async def cache_validation_results(self, 
                                     response: str, 
                                     result: FactualAccuracyResult) -> None:
        """Cache complete validation results."""
        cache_key = self._generate_response_cache_key(response)
        self.response_cache[cache_key] = result
        
        # Also cache individual claim validations
        for claim, matches in result.claim_validations:
            await self.cache_claim_validation(claim, matches)
    
    async def get_cached_claim_validation(self, 
                                        claim: FactualClaim) -> Optional[List[DocumentMatch]]:
        """Retrieve cached validation for individual claim."""
        cache_key = self._generate_claim_cache_key(claim)
        
        if cache_key in self.claim_cache:
            self.cache_stats['claim_hits'] += 1
            return self.claim_cache[cache_key]
        
        self.cache_stats['claim_misses'] += 1
        return None
    
    async def cache_claim_validation(self, 
                                   claim: FactualClaim, 
                                   matches: List[DocumentMatch]) -> None:
        """Cache individual claim validation results."""
        cache_key = self._generate_claim_cache_key(claim)
        self.claim_cache[cache_key] = matches
```

### Parallel Processing Architecture

```python
class BatchValidationProcessor:
    """
    High-performance batch processing system for claim validation.
    
    Provides parallel processing capabilities with intelligent batching,
    resource management, and error recovery.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Parallel processing configuration
        self.max_workers = config.get('max_workers', min(32, os.cpu_count() + 4))
        self.batch_size = config.get('batch_size', 10)
        self.max_concurrent_docs = config.get('max_concurrent_docs', 5)
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Performance tracking
        self.processing_stats = {
            'total_claims_processed': 0,
            'avg_processing_time_ms': 0.0,
            'parallel_efficiency': 0.0,
            'cache_hit_rate': 0.0
        }
    
    async def validate_claims_batch(self, 
                                  claims: List[FactualClaim],
                                  source_document_ids: Optional[List[str]] = None) -> List[Tuple[FactualClaim, List[DocumentMatch]]]:
        """
        Process a batch of claims for validation with parallel processing.
        
        Args:
            claims: List of claims to validate
            source_document_ids: Specific documents to validate against
            
        Returns:
            List of tuples containing claims and their validation matches
        """
        
        start_time = time.time()
        
        # Organize claims into processing batches
        claim_batches = self._create_claim_batches(claims)
        
        # Process batches with controlled concurrency
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_docs)
        
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await self._process_claim_batch(batch, source_document_ids)
        
        # Execute batches concurrently
        batch_tasks = [
            process_batch_with_semaphore(batch) 
            for batch in claim_batches
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results and handle exceptions
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing error: {batch_result}")
                continue
            results.extend(batch_result)
        
        # Update performance statistics
        processing_time = (time.time() - start_time) * 1000
        self._update_processing_stats(len(claims), processing_time)
        
        return results
    
    def _create_claim_batches(self, claims: List[FactualClaim]) -> List[List[FactualClaim]]:
        """Create optimally-sized batches of claims for processing."""
        
        # Group claims by type and complexity for more efficient processing
        claim_groups = {}
        for claim in claims:
            group_key = f"{claim.claim_type}_{claim.biomedical_domain}"
            if group_key not in claim_groups:
                claim_groups[group_key] = []
            claim_groups[group_key].append(claim)
        
        # Create batches ensuring good distribution of claim types
        batches = []
        current_batch = []
        
        for group_claims in claim_groups.values():
            for claim in group_claims:
                current_batch.append(claim)
                
                if len(current_batch) >= self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
        
        # Add remaining claims
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _process_claim_batch(self, 
                                 batch: List[FactualClaim],
                                 source_document_ids: Optional[List[str]]) -> List[Tuple[FactualClaim, List[DocumentMatch]]]:
        """Process a single batch of claims."""
        
        results = []
        
        # Process claims in the batch concurrently
        claim_tasks = [
            self._validate_single_claim(claim, source_document_ids)
            for claim in batch
        ]
        
        claim_results = await asyncio.gather(*claim_tasks, return_exceptions=True)
        
        # Collect results
        for claim, result in zip(batch, claim_results):
            if isinstance(result, Exception):
                logger.error(f"Claim validation error: {result}")
                results.append((claim, []))  # Empty matches for failed claims
            else:
                results.append((claim, result))
        
        return results
```

---

## Integration Points

### Enhanced ClinicalMetabolomicsRelevanceScorer Integration

```python
# Addition to ClinicalMetabolomicsRelevanceScorer class

class ClinicalMetabolomicsRelevanceScorer:
    """Enhanced with factual accuracy validation capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ... existing initialization code ...
        
        # Initialize factual accuracy validation components
        if config and config.get('enable_factual_validation', False):
            self.factual_validator = FactualAccuracyValidator(
                document_index=config.get('document_index'),
                semantic_matcher=config.get('semantic_matcher'),
                config=config.get('factual_validation_config', {})
            )
            self.factual_validation_enabled = True
        else:
            self.factual_validator = None
            self.factual_validation_enabled = False
    
    async def calculate_relevance_score(self,
                                     query: str,
                                     response: str,
                                     metadata: Optional[Dict[str, Any]] = None) -> RelevanceScore:
        """Enhanced to include factual accuracy validation."""
        
        # ... existing relevance scoring code ...
        
        # Add factual accuracy validation if enabled
        if self.factual_validation_enabled and self.factual_validator:
            try:
                factual_accuracy_result = await self.factual_validator.validate_factual_accuracy(
                    response=response,
                    query_context=query,
                    source_document_ids=metadata.get('source_document_ids') if metadata else None
                )
                
                # Add factual accuracy to dimension scores
                dimension_scores['factual_accuracy'] = factual_accuracy_result.overall_accuracy_score
                
                # Update metadata with detailed factual analysis
                result.metadata.update({
                    'factual_accuracy_details': {
                        'total_claims': factual_accuracy_result.total_claims,
                        'verified_claims': factual_accuracy_result.verified_claims,
                        'accuracy_grade': factual_accuracy_result.accuracy_grade,
                        'confidence_assessment': factual_accuracy_result.confidence_assessment
                    }
                })
                
            except Exception as e:
                logger.warning(f"Factual accuracy validation failed: {e}")
                dimension_scores['factual_accuracy'] = 0.0
        
        # ... rest of existing scoring code ...
        
        return result
```

### Enhanced ResponseQualityAssessor Integration

```python
class ResponseQualityAssessor:
    """Enhanced with factual accuracy assessment."""
    
    def __init__(self):
        # ... existing initialization code ...
        
        # Add factual accuracy to quality weights
        self.quality_weights['factual_accuracy'] = 0.15
        
        # Adjust existing weights to maintain total of 1.0
        self.quality_weights.update({
            'relevance': 0.20,  # Reduced from 0.25
            'accuracy': 0.15,   # Reduced from 0.20
            'completeness': 0.20,
            'clarity': 0.15,
            'biomedical_terminology': 0.10,
            'source_citation': 0.05  # Reduced from 0.10
        })
    
    async def assess_response_quality(self, 
                                    query: str,
                                    response: str,
                                    source_documents: List[str],
                                    expected_concepts: List[str]) -> ResponseQualityMetrics:
        """Enhanced with factual accuracy assessment."""
        
        # ... existing assessment code ...
        
        # Get factual accuracy score from the existing placeholder method
        # This now uses the full FactualAccuracyValidator if available
        factual_accuracy = await self._assess_factual_accuracy_enhanced(
            response, source_documents, query
        )
        
        # Update overall score calculation
        overall_score = (
            relevance * self.quality_weights['relevance'] +
            accuracy * self.quality_weights['accuracy'] +
            completeness * self.quality_weights['completeness'] +
            clarity * self.quality_weights['clarity'] +
            biomedical_terminology * self.quality_weights['biomedical_terminology'] +
            source_citation * self.quality_weights['source_citation'] +
            factual_accuracy * self.quality_weights['factual_accuracy']
        )
        
        # ... rest of existing code ...
        
        return ResponseQualityMetrics(
            # ... existing fields ...
            factual_accuracy_score=factual_accuracy,
            # ... rest of fields ...
        )
    
    async def _assess_factual_accuracy_enhanced(self, 
                                              response: str, 
                                              source_documents: List[str],
                                              query_context: str) -> float:
        """Enhanced factual accuracy assessment using full validation system."""
        
        # If full factual validator is available, use it
        if hasattr(self, 'factual_validator') and self.factual_validator:
            try:
                accuracy_result = await self.factual_validator.validate_factual_accuracy(
                    response=response,
                    query_context=query_context,
                    source_document_ids=None  # Let validator find relevant docs
                )
                return accuracy_result.overall_accuracy_score
            except Exception as e:
                logger.warning(f"Full factual validation failed, using fallback: {e}")
        
        # Fallback to existing simplified method
        return self._assess_factual_accuracy(response, source_documents)
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)

1. **Foundation Classes**
   - Implement `FactualClaim`, `DocumentMatch`, `FactualAccuracyResult` dataclasses
   - Create `FactualAccuracyValidator` base structure
   - Set up basic claim extraction framework

2. **Document Infrastructure** 
   - Implement `SourceDocumentIndex` for document indexing
   - Create document retrieval and caching mechanisms
   - Set up basic semantic matching infrastructure

### Phase 2: Claim Extraction (Weeks 3-4)

1. **BiomedicalClaimExtractor Implementation**
   - Implement numeric claim extraction with unit parsing
   - Create qualitative claim extraction with relationship parsing
   - Add temporal and methodological claim extraction
   - Implement biomedical domain classification

2. **Testing and Validation**
   - Create comprehensive test suite for claim extraction
   - Validate extraction accuracy on biomedical texts
   - Optimize extraction patterns and confidence scoring

### Phase 3: Document Matching (Weeks 5-6)

1. **DocumentContentMatcher Implementation**
   - Implement exact matching for numeric claims
   - Create semantic similarity matching pipeline
   - Add contextual matching capabilities
   - Implement contradiction detection

2. **Performance Optimization**
   - Add parallel processing for document matching
   - Implement caching strategies
   - Optimize for real-time performance requirements

### Phase 4: Scoring and Integration (Weeks 7-8)

1. **AccuracyScorer Implementation**
   - Implement multi-dimensional accuracy scoring
   - Create confidence assessment algorithms
   - Add detailed reporting and breakdown features

2. **Quality Pipeline Integration**
   - Integrate with `ClinicalMetabolomicsRelevanceScorer`
   - Enhance `ResponseQualityAssessor` with factual accuracy
   - Update existing test suites and validation

### Phase 5: Testing and Deployment (Weeks 9-10)

1. **Comprehensive Testing**
   - End-to-end integration testing
   - Performance benchmarking and optimization
   - Error handling and recovery testing

2. **Documentation and Deployment**
   - Complete API documentation
   - Performance tuning for production deployment
   - Monitor and validate real-world performance

---

## Error Handling and Recovery

### Comprehensive Error Handling Strategy

```python
class FactualValidationError(Exception):
    """Base exception for factual validation errors."""
    pass

class ClaimExtractionError(FactualValidationError):
    """Exception for claim extraction failures."""
    pass

class DocumentRetrievalError(FactualValidationError):
    """Exception for document retrieval failures."""
    pass

class ValidationTimeoutError(FactualValidationError):
    """Exception for validation timeout conditions."""
    pass

class FactualAccuracyValidator:
    """Enhanced with comprehensive error handling."""
    
    async def validate_factual_accuracy(self, 
                                      response: str,
                                      query_context: Optional[str] = None,
                                      source_document_ids: Optional[List[str]] = None) -> FactualAccuracyResult:
        """Main validation method with comprehensive error handling."""
        
        try:
            # Validation with timeout protection
            return await asyncio.wait_for(
                self._perform_validation(response, query_context, source_document_ids),
                timeout=self.config.get('validation_timeout', 30.0)
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Factual accuracy validation timed out after {self.config.get('validation_timeout', 30)}s")
            return self._create_timeout_result(response)
            
        except ClaimExtractionError as e:
            logger.error(f"Claim extraction failed: {e}")
            return self._create_extraction_error_result(response, str(e))
            
        except DocumentRetrievalError as e:
            logger.error(f"Document retrieval failed: {e}")
            return self._create_retrieval_error_result(response, str(e))
            
        except Exception as e:
            logger.error(f"Unexpected error in factual accuracy validation: {e}")
            return self._create_generic_error_result(response, str(e))
    
    def _create_timeout_result(self, response: str) -> FactualAccuracyResult:
        """Create result for timeout scenarios."""
        return FactualAccuracyResult(
            overall_accuracy_score=0.0,
            total_claims=0,
            verified_claims=0,
            contradicted_claims=0,
            unverifiable_claims=0,
            validation_details={
                'error_type': 'timeout',
                'error_message': 'Validation process timed out',
                'fallback_applied': True
            }
        )
    
    async def _perform_validation_with_retry(self, 
                                           response: str,
                                           query_context: Optional[str],
                                           source_document_ids: Optional[List[str]],
                                           max_retries: int = 3) -> FactualAccuracyResult:
        """Perform validation with retry logic for transient failures."""
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await self._perform_validation(response, query_context, source_document_ids)
                
            except (DocumentRetrievalError, ConnectionError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Validation attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"All {max_retries} validation attempts failed")
                    break
            
            except Exception as e:
                # Non-retryable errors
                raise e
        
        # If we get here, all retries failed
        raise last_exception or FactualValidationError("Validation failed after all retries")
```

---

## Testing Strategy

### Comprehensive Test Suite Architecture

```python
# =====================================================================
# FACTUAL ACCURACY VALIDATION TEST SUITE
# =====================================================================

class TestFactualAccuracyValidation:
    """Comprehensive test suite for factual accuracy validation system."""
    
    @pytest.fixture
    def sample_biomedical_claims(self):
        """Sample claims for testing."""
        return [
            FactualClaim(
                claim_id="test_numeric_1",
                claim_text="glucose concentration was 5.5 mM",
                claim_type="numeric",
                confidence=0.9,
                context="In diabetic patients, glucose concentration was 5.5 mM compared to 4.2 mM in controls",
                position=(25, 50),
                biomedical_domain="metabolomics",
                numeric_value=5.5,
                unit="mM"
            ),
            FactualClaim(
                claim_id="test_qualitative_1", 
                claim_text="metabolomics increases diagnostic accuracy",
                claim_type="qualitative",
                confidence=0.8,
                context="Studies show that metabolomics increases diagnostic accuracy for early disease detection",
                position=(15, 55),
                biomedical_domain="clinical",
                subject="metabolomics",
                predicate="increases",
                object="diagnostic accuracy"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_claim_extraction_accuracy(self, factual_validator):
        """Test accuracy of claim extraction from biomedical text."""
        
        response_text = """
        Clinical metabolomics analysis revealed glucose-6-phosphate concentrations 
        of 2.3 µM in diabetic patients, significantly higher than the 1.8 µM observed 
        in healthy controls (p < 0.05). LC-MS analysis was used for quantification.
        This increase correlates with disease severity and suggests that glucose-6-phosphate 
        serves as a potential biomarker for diabetes progression.
        """
        
        claims = await factual_validator.claim_extractor.extract_claims(
            response_text, "What are glucose metabolite levels in diabetes?"
        )
        
        # Verify claim extraction
        assert len(claims) >= 2, "Should extract at least numeric and qualitative claims"
        
        # Check numeric claim
        numeric_claims = [c for c in claims if c.claim_type == "numeric"]
        assert len(numeric_claims) >= 2, "Should extract concentration values"
        
        glucose_claim = next((c for c in numeric_claims if "2.3" in c.claim_text), None)
        assert glucose_claim is not None, "Should extract 2.3 µM glucose concentration"
        assert glucose_claim.numeric_value == 2.3
        assert glucose_claim.unit == "µM"
        
        # Check qualitative claim
        qualitative_claims = [c for c in claims if c.claim_type == "qualitative"]
        assert len(qualitative_claims) >= 1, "Should extract biomarker relationship"
    
    @pytest.mark.asyncio
    async def test_document_matching_precision(self, factual_validator, sample_documents):
        """Test precision of document matching against source texts."""
        
        claim = FactualClaim(
            claim_id="test_match",
            claim_text="glucose levels 5.2 mM",
            claim_type="numeric",
            confidence=0.9,
            context="",
            position=(0, 20),
            biomedical_domain="metabolomics",
            numeric_value=5.2,
            unit="mM"
        )
        
        matches = await factual_validator.document_matcher.find_claim_matches(
            claim, sample_documents
        )
        
        # Verify matching quality
        assert len(matches) > 0, "Should find document matches"
        
        # Check for exact numeric matches
        exact_matches = [m for m in matches if m.match_type == "exact"]
        if exact_matches:
            best_match = exact_matches[0]
            assert best_match.match_score >= 0.9, "Exact matches should have high scores"
            assert "5.2" in best_match.matched_text or "5.2" in best_match.context_window
    
    @pytest.mark.asyncio
    async def test_accuracy_scoring_consistency(self, factual_validator, sample_biomedical_claims):
        """Test consistency and reliability of accuracy scoring."""
        
        # Create mock validation results
        validation_results = []
        for claim in sample_biomedical_claims:
            # Mock high-confidence matches
            matches = [
                DocumentMatch(
                    document_id="test_doc_1",
                    document_title="Test Document",
                    matched_text=claim.claim_text,
                    match_score=0.95,
                    match_type="exact" if claim.claim_type == "numeric" else "semantic"
                )
            ]
            validation_results.append((claim, matches))
        
        # Calculate accuracy scores multiple times
        scores = []
        for _ in range(5):
            result = factual_validator.accuracy_scorer.calculate_accuracy_scores(
                sample_biomedical_claims, validation_results
            )
            scores.append(result.overall_accuracy_score)
        
        # Verify consistency (should be deterministic)
        assert all(abs(score - scores[0]) < 0.01 for score in scores), \
            f"Accuracy scoring should be consistent: {scores}"
        
        # Verify reasonable score range
        assert all(80 <= score <= 100 for score in scores), \
            f"High-confidence matches should yield high accuracy scores: {scores}"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_validation_performance_benchmarks(self, factual_validator):
        """Test performance meets real-time requirements."""
        
        # Test response with multiple claims
        complex_response = """
        Metabolomics analysis using LC-MS revealed significant differences in glucose 
        metabolism between diabetic (n=45, mean age 58.3±12.1 years) and control 
        subjects (n=42, mean age 56.7±10.9 years). Glucose-6-phosphate concentrations 
        were 2.87±0.45 µM vs 1.92±0.32 µM respectively (p<0.001). Fructose-6-phosphate 
        levels showed similar elevation (3.21±0.52 µM vs 2.14±0.38 µM, p<0.001). 
        These findings suggest enhanced glycolytic activity in diabetic patients.
        Statistical analysis used Student's t-test with Bonferroni correction.
        """
        
        start_time = time.time()
        
        result = await factual_validator.validate_factual_accuracy(
            response=complex_response,
            query_context="What are the metabolic differences in diabetes?"
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Performance assertions
        assert processing_time < 5000, f"Validation should complete within 5 seconds: {processing_time:.2f}ms"
        assert result.total_claims > 0, "Should extract claims from complex response"
        assert result.overall_accuracy_score >= 0, "Should produce valid accuracy score"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, factual_validator):
        """Test error handling and graceful degradation."""
        
        # Test with malformed response
        malformed_response = ""
        result = await factual_validator.validate_factual_accuracy(malformed_response)
        
        assert isinstance(result, FactualAccuracyResult), "Should return valid result even for empty input"
        assert result.total_claims == 0, "Should handle empty input gracefully"
        
        # Test with very long response (potential memory issues)
        very_long_response = "Clinical metabolomics analysis shows. " * 10000
        result = await factual_validator.validate_factual_accuracy(very_long_response)
        
        assert isinstance(result, FactualAccuracyResult), "Should handle large inputs without crashing"
        
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, clinical_rag_system, factual_validator):
        """Test end-to-end integration with existing quality assessment pipeline."""
        
        # Generate response using LightRAG
        query = "What is the role of glucose-6-phosphate in diabetes?"
        response = await clinical_rag_system.query(query)
        
        # Validate factual accuracy
        accuracy_result = await factual_validator.validate_factual_accuracy(
            response=response,
            query_context=query
        )
        
        # Verify integration
        assert isinstance(accuracy_result, FactualAccuracyResult)
        assert accuracy_result.overall_accuracy_score >= 0
        
        # Test integration with relevance scorer
        relevance_scorer = ClinicalMetabolomicsRelevanceScorer(config={
            'enable_factual_validation': True,
            'factual_validator': factual_validator
        })
        
        relevance_result = await relevance_scorer.calculate_relevance_score(
            query=query,
            response=response
        )
        
        # Verify factual accuracy is included in results
        assert 'factual_accuracy' in relevance_result.dimension_scores
        assert 'factual_accuracy_details' in relevance_result.metadata
```

---

## Conclusion

This comprehensive architecture design provides a robust foundation for implementing factual accuracy validation in the Clinical Metabolomics Oracle LightRAG integration. The design emphasizes:

1. **Seamless Integration**: Works naturally with existing `ClinicalMetabolomicsRelevanceScorer` and `ResponseQualityAssessor` infrastructure
2. **Sophisticated Validation**: Multi-strategy claim extraction and verification against source documents  
3. **Real-time Performance**: Optimized for production use with caching, parallel processing, and efficient algorithms
4. **Comprehensive Error Handling**: Robust error recovery and graceful degradation
5. **Extensible Design**: Modular architecture allows for future enhancements and domain-specific customizations

The implementation can be completed in phases over 10 weeks, with each phase building upon previous work and maintaining system stability throughout the development process.

---

**Next Steps:**
1. Review and approve this architectural design
2. Begin Phase 1 implementation with core infrastructure classes
3. Set up development environment with necessary dependencies
4. Create initial test framework and validation datasets
5. Begin iterative development and testing cycle

This design document serves as the definitive guide for implementing factual accuracy validation and should be referenced throughout the development process to ensure consistency and completeness.