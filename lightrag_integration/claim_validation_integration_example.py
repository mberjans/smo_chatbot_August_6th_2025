#!/usr/bin/env python3
"""
Claim Validation Integration Example for Clinical Metabolomics Oracle.

This script demonstrates the complete integration of the BiomedicalClaimExtractor
with the existing document indexing and quality assessment systems for comprehensive
factual accuracy validation.

Integration Components:
    - BiomedicalClaimExtractor: Extract factual claims from LightRAG responses
    - SourceDocumentIndex: Index and retrieve source document content
    - Quality Assessment Pipeline: Validate claims against source documents
    - Relevance Scoring System: Assess claim relevance and accuracy

Workflow:
    1. Extract claims from LightRAG response
    2. Classify and prioritize claims for verification
    3. Search indexed source documents for supporting evidence
    4. Validate claims against source content
    5. Generate factual accuracy assessment
    6. Provide recommendations for response improvement

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Factual Claim Extraction and Validation Integration
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# Import claim extraction system
try:
    from claim_extractor import (
        BiomedicalClaimExtractor,
        ExtractedClaim,
        prepare_claims_for_quality_assessment
    )
    CLAIM_EXTRACTOR_AVAILABLE = True
except ImportError:
    CLAIM_EXTRACTOR_AVAILABLE = False

# Import document indexing system
try:
    from document_indexer import SourceDocumentIndex
    DOCUMENT_INDEXER_AVAILABLE = True
except ImportError:
    DOCUMENT_INDEXER_AVAILABLE = False

# Import relevance scoring system
try:
    from relevance_scorer import ClinicalMetabolomicsRelevanceScorer
    RELEVANCE_SCORER_AVAILABLE = True
except ImportError:
    RELEVANCE_SCORER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClaimValidationResult:
    """Results of claim validation against source documents."""
    
    claim_id: str
    claim_text: str
    validation_status: str  # 'supported', 'contradicted', 'unclear', 'unsupported'
    confidence_score: float
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    verification_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FactualAccuracyAssessment:
    """Overall factual accuracy assessment for a response."""
    
    response_id: str
    total_claims: int
    validated_claims: int
    accuracy_score: float
    reliability_grade: str
    claim_validations: List[ClaimValidationResult] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    assessment_metadata: Dict[str, Any] = field(default_factory=dict)


class IntegratedClaimValidator:
    """
    Integrated system for extracting and validating factual claims.
    
    Combines claim extraction, document indexing, and quality assessment
    to provide comprehensive factual accuracy validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integrated validator."""
        
        self.config = config or {}
        
        # Initialize components if available
        self.claim_extractor = None
        self.document_index = None
        self.relevance_scorer = None
        
        if CLAIM_EXTRACTOR_AVAILABLE:
            self.claim_extractor = BiomedicalClaimExtractor(config.get('claim_extractor') if config else None)
        
        if DOCUMENT_INDEXER_AVAILABLE:
            self.document_index = SourceDocumentIndex(config.get('document_index') if config else None)
        
        if RELEVANCE_SCORER_AVAILABLE:
            self.relevance_scorer = ClinicalMetabolomicsRelevanceScorer(config.get('relevance_scorer') if config else None)
        
        logger.info(f"Integrated validator initialized with components: "
                   f"claim_extractor={self.claim_extractor is not None}, "
                   f"document_index={self.document_index is not None}, "
                   f"relevance_scorer={self.relevance_scorer is not None}")
    
    async def validate_response_accuracy(
        self,
        response_text: str,
        query: Optional[str] = None,
        source_documents: Optional[List[str]] = None,
        response_id: Optional[str] = None
    ) -> FactualAccuracyAssessment:
        """
        Perform comprehensive factual accuracy validation of a response.
        
        Args:
            response_text: The LightRAG response to validate
            query: Original query for context
            source_documents: List of source document identifiers
            response_id: Unique identifier for the response
            
        Returns:
            Comprehensive factual accuracy assessment
        """
        
        response_id = response_id or f"response_{int(time.time())}"
        
        logger.info(f"Starting factual accuracy validation for {response_id}")
        
        # Step 1: Extract claims from response
        if not self.claim_extractor:
            raise ValueError("Claim extractor not available")
        
        claims = await self.claim_extractor.extract_claims(response_text, query)
        logger.info(f"Extracted {len(claims)} claims from response")
        
        if not claims:
            return FactualAccuracyAssessment(
                response_id=response_id,
                total_claims=0,
                validated_claims=0,
                accuracy_score=0.0,
                reliability_grade="No Claims",
                quality_flags=["no_factual_claims_found"],
                improvement_suggestions=["Add more specific factual content"]
            )
        
        # Step 2: Prioritize claims for validation
        high_priority_claims = await self.claim_extractor.filter_high_confidence_claims(
            claims, min_confidence=50.0
        )
        
        logger.info(f"Identified {len(high_priority_claims)} high-priority claims for validation")
        
        # Step 3: Validate claims against source documents
        validation_results = []
        
        for claim in high_priority_claims:
            validation_result = await self._validate_single_claim(
                claim, source_documents
            )
            validation_results.append(validation_result)
        
        # Step 4: Calculate overall accuracy assessment
        assessment = await self._calculate_accuracy_assessment(
            response_id, claims, validation_results
        )
        
        logger.info(f"Completed validation: {assessment.reliability_grade} "
                   f"({assessment.accuracy_score:.1f}%)")
        
        return assessment
    
    async def _validate_single_claim(
        self,
        claim: ExtractedClaim,
        source_documents: Optional[List[str]] = None
    ) -> ClaimValidationResult:
        """Validate a single claim against source documents."""
        
        logger.info(f"Validating claim: {claim.claim_id}")
        
        # Initialize validation result
        validation_result = ClaimValidationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            validation_status="unsupported",
            confidence_score=0.0
        )
        
        try:
            # Search for supporting evidence
            if self.document_index and source_documents:
                supporting_evidence = await self._search_supporting_evidence(
                    claim, source_documents
                )
                validation_result.supporting_evidence = supporting_evidence
            
            # Assess validation status
            validation_result = await self._assess_validation_status(
                validation_result, claim
            )
            
            # Generate recommendations
            validation_result.recommendations = await self._generate_claim_recommendations(
                validation_result, claim
            )
            
        except Exception as e:
            logger.error(f"Error validating claim {claim.claim_id}: {str(e)}")
            validation_result.validation_status = "error"
            validation_result.verification_details = {"error": str(e)}
        
        return validation_result
    
    async def _search_supporting_evidence(
        self,
        claim: ExtractedClaim,
        source_documents: List[str]
    ) -> List[Dict[str, Any]]:
        """Search for supporting evidence in source documents."""
        
        supporting_evidence = []
        
        # Generate search terms from claim
        search_terms = []
        search_terms.extend(claim.keywords[:5])  # Top keywords
        
        # Add numeric values as search terms
        if claim.numeric_values:
            search_terms.extend([str(v) for v in claim.numeric_values])
        
        # Add units as search terms
        if claim.units:
            search_terms.extend(claim.units)
        
        # Mock evidence search (would use actual document index)
        if self.document_index:
            try:
                # This would be the actual search implementation
                search_results = await self._mock_document_search(
                    search_terms, source_documents
                )
                supporting_evidence.extend(search_results)
            except Exception as e:
                logger.warning(f"Document search failed: {str(e)}")
        
        return supporting_evidence
    
    async def _mock_document_search(
        self,
        search_terms: List[str],
        source_documents: List[str]
    ) -> List[Dict[str, Any]]:
        """Mock document search for demonstration."""
        
        # This is a mock implementation for demonstration
        mock_evidence = []
        
        for term in search_terms[:3]:  # Limit to first 3 terms
            mock_evidence.append({
                'document_id': f"doc_{hash(term) % 1000}",
                'content_snippet': f"Context containing {term}...",
                'relevance_score': 0.7 + (hash(term) % 30) / 100,
                'match_type': 'keyword_match',
                'source_section': 'results'
            })
        
        return mock_evidence
    
    async def _assess_validation_status(
        self,
        validation_result: ClaimValidationResult,
        claim: ExtractedClaim
    ) -> ClaimValidationResult:
        """Assess the validation status based on available evidence."""
        
        # Simple validation logic based on evidence and claim confidence
        supporting_count = len(validation_result.supporting_evidence)
        contradicting_count = len(validation_result.contradicting_evidence)
        claim_confidence = claim.confidence.overall_confidence
        
        if supporting_count > 0 and contradicting_count == 0:
            validation_result.validation_status = "supported"
            validation_result.confidence_score = min(95.0, claim_confidence + 20)
            
        elif contradicting_count > supporting_count:
            validation_result.validation_status = "contradicted"
            validation_result.confidence_score = 20.0
            
        elif supporting_count > 0 and contradicting_count > 0:
            validation_result.validation_status = "unclear"
            validation_result.confidence_score = 50.0
            
        else:
            validation_result.validation_status = "unsupported"
            validation_result.confidence_score = max(10.0, claim_confidence - 30)
        
        # Store validation details
        validation_result.verification_details = {
            'supporting_evidence_count': supporting_count,
            'contradicting_evidence_count': contradicting_count,
            'original_confidence': claim_confidence,
            'assessment_method': 'evidence_count_heuristic'
        }
        
        return validation_result
    
    async def _generate_claim_recommendations(
        self,
        validation_result: ClaimValidationResult,
        claim: ExtractedClaim
    ) -> List[str]:
        """Generate recommendations for claim improvement."""
        
        recommendations = []
        
        if validation_result.validation_status == "supported":
            recommendations.append("Claim is well-supported by source documents")
            if len(validation_result.supporting_evidence) > 2:
                recommendations.append("Consider citing specific sources")
        
        elif validation_result.validation_status == "contradicted":
            recommendations.append("Claim contradicts source documents - verify accuracy")
            recommendations.append("Review original sources and correct if necessary")
        
        elif validation_result.validation_status == "unclear":
            recommendations.append("Mixed evidence - clarify the claim or add qualifiers")
            recommendations.append("Consider mentioning conflicting findings")
        
        else:  # unsupported
            recommendations.append("No supporting evidence found in source documents")
            if claim.confidence.overall_confidence < 60:
                recommendations.append("Consider removing or qualifying this claim")
            else:
                recommendations.append("Add citations or supporting evidence")
        
        # Type-specific recommendations
        if claim.claim_type == "numeric":
            if not claim.units:
                recommendations.append("Add units to numeric values")
            if claim.confidence.verification_confidence < 70:
                recommendations.append("Verify numeric accuracy against sources")
        
        elif claim.claim_type == "qualitative":
            if "correlates" in claim.claim_text.lower():
                recommendations.append("Clarify whether correlation implies causation")
        
        return recommendations
    
    async def _calculate_accuracy_assessment(
        self,
        response_id: str,
        all_claims: List[ExtractedClaim],
        validation_results: List[ClaimValidationResult]
    ) -> FactualAccuracyAssessment:
        """Calculate overall accuracy assessment."""
        
        total_claims = len(all_claims)
        validated_claims = len(validation_results)
        
        if validated_claims == 0:
            accuracy_score = 0.0
            reliability_grade = "Unvalidated"
        else:
            # Calculate accuracy based on validation results
            supported_count = len([r for r in validation_results if r.validation_status == "supported"])
            contradicted_count = len([r for r in validation_results if r.validation_status == "contradicted"])
            
            # Weighted accuracy score
            accuracy_score = (
                (supported_count * 100) +
                (len([r for r in validation_results if r.validation_status == "unclear"]) * 50) +
                (len([r for r in validation_results if r.validation_status == "unsupported"]) * 25) -
                (contradicted_count * 100)
            ) / (validated_claims * 100) * 100
            
            accuracy_score = max(0.0, min(100.0, accuracy_score))
        
        # Determine reliability grade
        if accuracy_score >= 90:
            reliability_grade = "Excellent"
        elif accuracy_score >= 80:
            reliability_grade = "Good"
        elif accuracy_score >= 70:
            reliability_grade = "Acceptable"
        elif accuracy_score >= 50:
            reliability_grade = "Poor"
        else:
            reliability_grade = "Unreliable"
        
        # Generate quality flags
        quality_flags = []
        contradicted_claims = [r for r in validation_results if r.validation_status == "contradicted"]
        if contradicted_claims:
            quality_flags.append("contradicted_claims_found")
        
        unsupported_claims = [r for r in validation_results if r.validation_status == "unsupported"]
        if len(unsupported_claims) > validated_claims * 0.5:
            quality_flags.append("high_unsupported_claims_ratio")
        
        # Generate improvement suggestions
        improvement_suggestions = []
        
        if contradicted_claims:
            improvement_suggestions.append("Review and correct contradicted claims")
        
        if unsupported_claims:
            improvement_suggestions.append("Add supporting evidence or citations")
        
        if accuracy_score < 80:
            improvement_suggestions.append("Improve factual accuracy through source verification")
        
        # Create assessment
        assessment = FactualAccuracyAssessment(
            response_id=response_id,
            total_claims=total_claims,
            validated_claims=validated_claims,
            accuracy_score=accuracy_score,
            reliability_grade=reliability_grade,
            claim_validations=validation_results,
            quality_flags=quality_flags,
            improvement_suggestions=improvement_suggestions,
            assessment_metadata={
                'validation_timestamp': datetime.now().isoformat(),
                'supported_claims': supported_count,
                'contradicted_claims': len(contradicted_claims),
                'unsupported_claims': len(unsupported_claims),
                'unclear_claims': len([r for r in validation_results if r.validation_status == "unclear"])
            }
        )
        
        return assessment
    
    def generate_validation_report(
        self,
        assessment: FactualAccuracyAssessment,
        output_format: str = "json"
    ) -> str:
        """Generate a comprehensive validation report."""
        
        if output_format == "json":
            # Convert to JSON-serializable format
            report_data = {
                'response_id': assessment.response_id,
                'accuracy_summary': {
                    'total_claims': assessment.total_claims,
                    'validated_claims': assessment.validated_claims,
                    'accuracy_score': assessment.accuracy_score,
                    'reliability_grade': assessment.reliability_grade
                },
                'claim_validations': [
                    {
                        'claim_id': cv.claim_id,
                        'claim_text': cv.claim_text[:100] + ("..." if len(cv.claim_text) > 100 else ""),
                        'validation_status': cv.validation_status,
                        'confidence_score': cv.confidence_score,
                        'supporting_evidence_count': len(cv.supporting_evidence),
                        'recommendations': cv.recommendations
                    }
                    for cv in assessment.claim_validations
                ],
                'quality_flags': assessment.quality_flags,
                'improvement_suggestions': assessment.improvement_suggestions,
                'metadata': assessment.assessment_metadata
            }
            
            return json.dumps(report_data, indent=2)
        
        elif output_format == "text":
            # Generate human-readable text report
            report_lines = [
                "FACTUAL ACCURACY VALIDATION REPORT",
                "=" * 50,
                f"Response ID: {assessment.response_id}",
                f"Overall Grade: {assessment.reliability_grade} ({assessment.accuracy_score:.1f}%)",
                f"Claims Analyzed: {assessment.validated_claims}/{assessment.total_claims}",
                "",
                "CLAIM VALIDATION RESULTS:",
                "-" * 30
            ]
            
            for i, cv in enumerate(assessment.claim_validations, 1):
                status_symbol = {
                    'supported': '✓',
                    'contradicted': '✗',
                    'unclear': '?',
                    'unsupported': '○'
                }.get(cv.validation_status, '?')
                
                report_lines.extend([
                    f"{i}. {status_symbol} [{cv.validation_status.upper()}] "
                    f"Confidence: {cv.confidence_score:.1f}%",
                    f"   {cv.claim_text[:80]}{'...' if len(cv.claim_text) > 80 else ''}",
                    f"   Evidence: {len(cv.supporting_evidence)} supporting",
                    ""
                ])
            
            if assessment.quality_flags:
                report_lines.extend([
                    "QUALITY FLAGS:",
                    "-" * 15
                ])
                for flag in assessment.quality_flags:
                    report_lines.append(f"• {flag.replace('_', ' ').title()}")
                report_lines.append("")
            
            if assessment.improvement_suggestions:
                report_lines.extend([
                    "IMPROVEMENT SUGGESTIONS:",
                    "-" * 25
                ])
                for suggestion in assessment.improvement_suggestions:
                    report_lines.append(f"• {suggestion}")
            
            return "\n".join(report_lines)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


async def demonstrate_integrated_validation():
    """Demonstrate the integrated claim validation system."""
    
    print("=" * 80)
    print("INTEGRATED CLAIM VALIDATION DEMONSTRATION")
    print("=" * 80)
    
    # Sample LightRAG response for validation
    sample_response = """
    A comprehensive metabolomics analysis using LC-MS/MS identified 342 metabolites 
    in plasma samples from diabetic patients. Glucose concentrations were significantly 
    elevated at 9.8 ± 2.1 mmol/L compared to healthy controls (5.2 ± 0.8 mmol/L, p < 0.001).
    
    The study revealed a strong correlation between insulin resistance and branched-chain 
    amino acid levels (r = 0.73, p < 0.001). Oxidative stress markers increased by 
    2.3-fold in diabetic patients, while antioxidant metabolites decreased by 45%.
    
    These findings suggest that metabolic dysregulation in diabetes involves complex 
    perturbations across multiple biochemical pathways. The analysis was performed 
    using validated analytical methods with detection limits ranging from 0.5 to 50 ng/mL.
    """
    
    sample_query = "What metabolomic changes are observed in diabetes patients?"
    sample_documents = ["doc_001", "doc_002", "doc_003"]
    
    # Initialize validator
    validator = IntegratedClaimValidator()
    
    if not validator.claim_extractor:
        print("⚠ Claim extractor not available - using mock validation")
        return
    
    print("Starting integrated validation workflow...")
    print()
    
    # Perform validation
    assessment = await validator.validate_response_accuracy(
        response_text=sample_response,
        query=sample_query,
        source_documents=sample_documents,
        response_id="demo_response_001"
    )
    
    # Display results
    print("VALIDATION RESULTS:")
    print("-" * 40)
    print(f"Overall Grade: {assessment.reliability_grade}")
    print(f"Accuracy Score: {assessment.accuracy_score:.1f}%")
    print(f"Claims Validated: {assessment.validated_claims}/{assessment.total_claims}")
    print()
    
    # Show individual claim validations
    if assessment.claim_validations:
        print("CLAIM VALIDATION DETAILS:")
        print("-" * 30)
        
        for i, cv in enumerate(assessment.claim_validations[:5], 1):  # Show first 5
            status_emoji = {
                'supported': '✅',
                'contradicted': '❌',
                'unclear': '⚠️',
                'unsupported': '❓'
            }.get(cv.validation_status, '❓')
            
            print(f"{i}. {status_emoji} {cv.validation_status.upper()} "
                  f"(Confidence: {cv.confidence_score:.1f}%)")
            print(f"   {cv.claim_text[:80]}{'...' if len(cv.claim_text) > 80 else ''}")
            print(f"   Evidence: {len(cv.supporting_evidence)} supporting")
            if cv.recommendations:
                print(f"   Recommendation: {cv.recommendations[0]}")
            print()
    
    # Show quality flags and suggestions
    if assessment.quality_flags:
        print("QUALITY FLAGS:")
        for flag in assessment.quality_flags:
            print(f"  🚩 {flag.replace('_', ' ').title()}")
        print()
    
    if assessment.improvement_suggestions:
        print("IMPROVEMENT SUGGESTIONS:")
        for suggestion in assessment.improvement_suggestions:
            print(f"  💡 {suggestion}")
        print()
    
    # Generate and save detailed report
    json_report = validator.generate_validation_report(assessment, "json")
    text_report = validator.generate_validation_report(assessment, "text")
    
    # Save reports
    with open("validation_report_demo.json", "w") as f:
        f.write(json_report)
    
    with open("validation_report_demo.txt", "w") as f:
        f.write(text_report)
    
    print("📊 Detailed reports saved:")
    print("   • validation_report_demo.json")
    print("   • validation_report_demo.txt")
    print()
    
    print("=" * 80)
    print("INTEGRATION DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("The integrated claim validation system successfully:")
    print("  ✅ Extracted factual claims from LightRAG response")
    print("  ✅ Classified and prioritized claims for validation")
    print("  ✅ Searched for supporting evidence (mock implementation)")
    print("  ✅ Assessed validation status and confidence")
    print("  ✅ Generated comprehensive accuracy assessment")
    print("  ✅ Provided actionable improvement recommendations")
    print("  ✅ Created detailed validation reports")


async def main():
    """Main function for demonstration."""
    
    print("Integrated Claim Validation System")
    print("Clinical Metabolomics Oracle - LightRAG Integration")
    print()
    
    if not CLAIM_EXTRACTOR_AVAILABLE:
        print("❌ Claim extractor not available")
        return
    
    try:
        await demonstrate_integrated_validation()
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    """Run the integrated validation demonstration."""
    asyncio.run(main())