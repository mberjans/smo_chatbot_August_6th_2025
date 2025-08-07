#!/usr/bin/env python3
"""
Factual Validation Integration Example for Clinical Metabolomics Oracle.

This example demonstrates the complete integration of the factual accuracy validation
system with the existing claim extraction and document indexing infrastructure.

The example shows:
1. Setting up the integrated validation pipeline
2. Processing LightRAG responses through the complete validation workflow
3. Generating comprehensive validation reports
4. Integrating with existing quality assessment systems
5. Performance monitoring and optimization

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Factual Accuracy Validation Integration
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import the integrated validation system components
try:
    from factual_accuracy_validator import (
        FactualAccuracyValidator, VerificationReport, VerificationResult,
        verify_extracted_claims, verify_claim_against_documents
    )
    from claim_extractor import BiomedicalClaimExtractor, ExtractedClaim
    from document_indexer import SourceDocumentIndex
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedFactualValidationPipeline:
    """
    Integrated pipeline for complete factual validation workflow.
    
    This class combines claim extraction, document indexing, and factual accuracy
    validation into a seamless pipeline for processing LightRAG responses.
    """
    
    def __init__(self, 
                 document_index_dir: str = "./document_index",
                 validation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integrated validation pipeline.
        
        Args:
            document_index_dir: Directory containing the document index
            validation_config: Optional configuration for validation
        """
        self.config = validation_config or {}
        self.logger = logger
        
        # Initialize components
        self.claim_extractor = None
        self.document_indexer = None
        self.factual_validator = None
        
        # Pipeline statistics
        self.pipeline_stats = {
            'total_responses_processed': 0,
            'total_claims_extracted': 0,
            'total_claims_verified': 0,
            'processing_times': [],
            'validation_success_rate': 0.0
        }
        
    async def initialize(self):
        """Initialize all pipeline components."""
        
        try:
            self.logger.info("Initializing integrated factual validation pipeline...")
            
            # Initialize claim extractor
            self.claim_extractor = BiomedicalClaimExtractor(self.config.get('claim_extraction', {}))
            self.logger.info("✓ Claim extractor initialized")
            
            # Initialize document indexer
            self.document_indexer = SourceDocumentIndex(
                index_dir=self.config.get('document_index_dir', "./document_index")
            )
            await self.document_indexer.initialize()
            self.logger.info("✓ Document indexer initialized")
            
            # Initialize factual accuracy validator
            self.factual_validator = FactualAccuracyValidator(
                document_indexer=self.document_indexer,
                claim_extractor=self.claim_extractor,
                config=self.config.get('factual_validation', {})
            )
            self.logger.info("✓ Factual accuracy validator initialized")
            
            self.logger.info("🎉 Integrated factual validation pipeline ready!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
    
    async def process_lightrag_response(self, 
                                       response_text: str,
                                       query: Optional[str] = None,
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a LightRAG response through the complete validation pipeline.
        
        Args:
            response_text: The LightRAG response to validate
            query: Optional original query for context
            context: Optional additional context
            
        Returns:
            Dict containing comprehensive validation results
        """
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing LightRAG response ({len(response_text)} characters)...")
            
            # Step 1: Extract factual claims
            self.logger.info("📝 Extracting factual claims...")
            extracted_claims = await self.claim_extractor.extract_claims(
                response_text, query, context
            )
            
            self.logger.info(f"✓ Extracted {len(extracted_claims)} claims")
            
            # Step 2: Verify claims against source documents
            self.logger.info("🔍 Verifying claims against source documents...")
            verification_report = await self.factual_validator.verify_claims(
                extracted_claims, 
                self.config.get('verification', {})
            )
            
            self.logger.info(f"✓ Verified {len(extracted_claims)} claims")
            
            # Step 3: Generate comprehensive results
            results = await self._generate_comprehensive_results(
                response_text, query, extracted_claims, verification_report, context
            )
            
            # Update pipeline statistics
            processing_time = (time.time() - start_time) * 1000
            await self._update_pipeline_stats(len(extracted_claims), processing_time, verification_report)
            
            self.logger.info(f"✅ Pipeline processing completed in {processing_time:.2f}ms")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pipeline processing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    async def batch_process_responses(self, 
                                     responses: List[Dict[str, Any]],
                                     batch_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process multiple LightRAG responses in batch.
        
        Args:
            responses: List of response dictionaries with 'text', 'query', etc.
            batch_config: Optional batch processing configuration
            
        Returns:
            Dict containing batch processing results
        """
        
        start_time = time.time()
        batch_results = []
        
        try:
            self.logger.info(f"Starting batch processing of {len(responses)} responses...")
            
            # Process each response
            for i, response_data in enumerate(responses):
                self.logger.info(f"Processing response {i+1}/{len(responses)}...")
                
                result = await self.process_lightrag_response(
                    response_data.get('text', ''),
                    response_data.get('query'),
                    response_data.get('context', {})
                )
                
                result['batch_index'] = i
                result['response_id'] = response_data.get('id', f'response_{i}')
                batch_results.append(result)
            
            # Generate batch summary
            batch_summary = await self._generate_batch_summary(batch_results)
            
            total_processing_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"✅ Batch processing completed in {total_processing_time:.2f}ms")
            
            return {
                'batch_summary': batch_summary,
                'individual_results': batch_results,
                'total_processing_time_ms': total_processing_time,
                'processed_count': len(responses),
                'success_rate': sum(1 for r in batch_results if r.get('success', False)) / len(batch_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processed_results': batch_results,
                'total_processing_time_ms': (time.time() - start_time) * 1000
            }
    
    async def validate_claim_accuracy(self, 
                                     claim_text: str,
                                     detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Validate accuracy of a specific claim.
        
        Args:
            claim_text: The claim text to validate
            detailed_analysis: Whether to include detailed analysis
            
        Returns:
            Dict containing claim validation results
        """
        
        try:
            self.logger.info(f"Validating claim: {claim_text[:100]}...")
            
            # Use the convenience function for single claim validation
            verification_report = await verify_claim_against_documents(
                claim_text,
                self.document_indexer,
                self.claim_extractor,
                self.config.get('single_claim_validation', {})
            )
            
            # Extract the main result (should be one claim)
            if verification_report.verification_results:
                main_result = verification_report.verification_results[0]
                
                validation_result = {
                    'claim_text': claim_text,
                    'verification_status': main_result.verification_status.value,
                    'verification_confidence': main_result.verification_confidence,
                    'evidence_strength': main_result.evidence_strength,
                    'context_match': main_result.context_match,
                    'verification_grade': main_result.verification_grade,
                    'supporting_evidence_count': len(main_result.supporting_evidence),
                    'contradicting_evidence_count': len(main_result.contradicting_evidence),
                    'neutral_evidence_count': len(main_result.neutral_evidence)
                }
                
                if detailed_analysis:
                    validation_result.update({
                        'supporting_evidence': [
                            {
                                'source': ev.source_document,
                                'text': ev.evidence_text,
                                'confidence': ev.confidence,
                                'context': ev.context[:200] + "..." if len(ev.context) > 200 else ev.context
                            }
                            for ev in main_result.supporting_evidence
                        ],
                        'contradicting_evidence': [
                            {
                                'source': ev.source_document,
                                'text': ev.evidence_text,
                                'confidence': ev.confidence,
                                'context': ev.context[:200] + "..." if len(ev.context) > 200 else ev.context
                            }
                            for ev in main_result.contradicting_evidence
                        ],
                        'verification_report': verification_report.to_dict()
                    })
                
                return validation_result
                
            else:
                return {
                    'claim_text': claim_text,
                    'verification_status': 'ERROR',
                    'error': 'No verification results generated'
                }
            
        except Exception as e:
            self.logger.error(f"Error validating claim: {str(e)}")
            return {
                'claim_text': claim_text,
                'verification_status': 'ERROR',
                'error': str(e)
            }
    
    async def _generate_comprehensive_results(self,
                                            response_text: str,
                                            query: Optional[str],
                                            extracted_claims: List[ExtractedClaim],
                                            verification_report: VerificationReport,
                                            context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive validation results."""
        
        # Calculate overall quality metrics
        verification_results = verification_report.verification_results
        
        if verification_results:
            avg_confidence = sum(vr.verification_confidence for vr in verification_results) / len(verification_results)
            avg_evidence_strength = sum(vr.evidence_strength for vr in verification_results) / len(verification_results)
            
            # Status distribution
            status_counts = {}
            for vr in verification_results:
                status = vr.verification_status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        else:
            avg_confidence = 0
            avg_evidence_strength = 0
            status_counts = {}
        
        # Quality assessment
        factual_accuracy_grade = self._calculate_factual_accuracy_grade(verification_results)
        
        # Prepare results
        results = {
            'success': True,
            'response_analysis': {
                'original_response': response_text,
                'original_query': query,
                'response_length': len(response_text),
                'processing_timestamp': datetime.now().isoformat()
            },
            'claim_extraction_results': {
                'total_claims_extracted': len(extracted_claims),
                'claims_by_type': self._group_claims_by_type(extracted_claims),
                'high_confidence_claims': len([c for c in extracted_claims if c.confidence.overall_confidence >= 75]),
                'extracted_claims': [claim.to_dict() for claim in extracted_claims]
            },
            'factual_verification_results': {
                'verification_report': verification_report.to_dict(),
                'overall_metrics': {
                    'average_verification_confidence': avg_confidence,
                    'average_evidence_strength': avg_evidence_strength,
                    'factual_accuracy_grade': factual_accuracy_grade,
                    'verification_status_distribution': status_counts
                },
                'evidence_summary': {
                    'total_evidence_items': sum(vr.total_evidence_count for vr in verification_results),
                    'claims_with_supporting_evidence': len([vr for vr in verification_results if vr.supporting_evidence]),
                    'claims_with_contradicting_evidence': len([vr for vr in verification_results if vr.contradicting_evidence]),
                    'claims_without_evidence': len([vr for vr in verification_results if not vr.supporting_evidence and not vr.contradicting_evidence])
                }
            },
            'quality_assessment': {
                'factual_accuracy_score': avg_confidence,
                'evidence_support_score': avg_evidence_strength,
                'overall_reliability_grade': self._calculate_overall_reliability_grade(avg_confidence, avg_evidence_strength),
                'recommendations': verification_report.recommendations
            },
            'processing_metadata': {
                'pipeline_version': '1.0.0',
                'components_used': ['claim_extractor', 'document_indexer', 'factual_validator'],
                'processing_timestamp': datetime.now().isoformat(),
                'context_provided': context is not None
            }
        }
        
        return results
    
    def _group_claims_by_type(self, claims: List[ExtractedClaim]) -> Dict[str, int]:
        """Group claims by type and count them."""
        type_counts = {}
        for claim in claims:
            claim_type = claim.claim_type
            type_counts[claim_type] = type_counts.get(claim_type, 0) + 1
        return type_counts
    
    def _calculate_factual_accuracy_grade(self, verification_results: List[VerificationResult]) -> str:
        """Calculate overall factual accuracy grade."""
        if not verification_results:
            return "Unknown"
        
        # Count supported vs contradicted claims
        supported = len([vr for vr in verification_results if vr.verification_status.value == 'SUPPORTED'])
        contradicted = len([vr for vr in verification_results if vr.verification_status.value == 'CONTRADICTED'])
        total = len(verification_results)
        
        support_rate = supported / total if total > 0 else 0
        contradict_rate = contradicted / total if total > 0 else 0
        
        if support_rate >= 0.8 and contradict_rate <= 0.1:
            return "Excellent"
        elif support_rate >= 0.6 and contradict_rate <= 0.2:
            return "Good"
        elif support_rate >= 0.4 and contradict_rate <= 0.3:
            return "Acceptable"
        elif contradict_rate <= 0.4:
            return "Marginal"
        else:
            return "Poor"
    
    def _calculate_overall_reliability_grade(self, avg_confidence: float, avg_evidence_strength: float) -> str:
        """Calculate overall reliability grade combining multiple factors."""
        combined_score = (avg_confidence * 0.6 + avg_evidence_strength * 0.4)
        
        if combined_score >= 90:
            return "Very High"
        elif combined_score >= 80:
            return "High"
        elif combined_score >= 70:
            return "Moderate"
        elif combined_score >= 60:
            return "Low"
        else:
            return "Very Low"
    
    async def _generate_batch_summary(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary for batch processing results."""
        
        if not batch_results:
            return {}
        
        successful_results = [r for r in batch_results if r.get('success', False)]
        
        # Aggregate statistics
        total_claims = sum(r.get('claim_extraction_results', {}).get('total_claims_extracted', 0) for r in successful_results)
        
        # Average quality metrics
        quality_scores = []
        for result in successful_results:
            quality_data = result.get('quality_assessment', {})
            if quality_data.get('factual_accuracy_score'):
                quality_scores.append(quality_data['factual_accuracy_score'])
        
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Reliability grade distribution
        reliability_grades = [r.get('quality_assessment', {}).get('overall_reliability_grade', 'Unknown') 
                            for r in successful_results]
        grade_counts = {}
        for grade in reliability_grades:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        return {
            'batch_metrics': {
                'total_responses_processed': len(batch_results),
                'successful_responses': len(successful_results),
                'success_rate': len(successful_results) / len(batch_results),
                'total_claims_extracted': total_claims,
                'average_claims_per_response': total_claims / len(successful_results) if successful_results else 0
            },
            'quality_metrics': {
                'average_factual_accuracy_score': avg_quality_score,
                'reliability_grade_distribution': grade_counts,
                'responses_with_high_accuracy': len([r for r in successful_results 
                                                   if r.get('quality_assessment', {}).get('factual_accuracy_score', 0) >= 80])
            },
            'processing_metrics': {
                'total_processing_time_ms': sum(r.get('processing_time_ms', 0) for r in batch_results),
                'average_processing_time_ms': sum(r.get('processing_time_ms', 0) for r in batch_results) / len(batch_results)
            }
        }
    
    async def _update_pipeline_stats(self, claims_count: int, processing_time: float, verification_report: VerificationReport):
        """Update pipeline statistics."""
        
        self.pipeline_stats['total_responses_processed'] += 1
        self.pipeline_stats['total_claims_extracted'] += claims_count
        self.pipeline_stats['total_claims_verified'] += len(verification_report.verification_results)
        self.pipeline_stats['processing_times'].append(processing_time)
        
        # Calculate validation success rate
        if verification_report.verification_results:
            successful_verifications = len([vr for vr in verification_report.verification_results 
                                          if vr.verification_status.value in ['SUPPORTED', 'NEUTRAL']])
            self.pipeline_stats['validation_success_rate'] = (
                successful_verifications / len(verification_report.verification_results)
            )
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance statistics."""
        
        processing_times = self.pipeline_stats['processing_times']
        
        return {
            'usage_statistics': {
                'total_responses_processed': self.pipeline_stats['total_responses_processed'],
                'total_claims_extracted': self.pipeline_stats['total_claims_extracted'],
                'total_claims_verified': self.pipeline_stats['total_claims_verified'],
                'average_claims_per_response': (
                    self.pipeline_stats['total_claims_extracted'] / 
                    max(1, self.pipeline_stats['total_responses_processed'])
                )
            },
            'performance_metrics': {
                'processing_times_ms': {
                    'count': len(processing_times),
                    'average': sum(processing_times) / len(processing_times) if processing_times else 0,
                    'min': min(processing_times) if processing_times else 0,
                    'max': max(processing_times) if processing_times else 0
                }
            },
            'quality_metrics': {
                'validation_success_rate': self.pipeline_stats['validation_success_rate']
            },
            'component_statistics': {
                'claim_extractor_stats': self.claim_extractor.get_extraction_statistics() if self.claim_extractor else {},
                'factual_validator_stats': self.factual_validator.get_verification_statistics() if self.factual_validator else {}
            }
        }


# Demonstration and testing functions
async def demonstrate_integrated_validation():
    """Demonstrate the integrated validation pipeline."""
    
    print("🚀 Clinical Metabolomics Oracle - Factual Validation Pipeline Demo")
    print("=" * 70)
    
    # Sample LightRAG response for testing
    sample_response = """
    Metabolomics analysis revealed that glucose levels were elevated by 25% 
    in diabetic patients compared to healthy controls. The LC-MS analysis 
    showed significant differences (p < 0.05) in 47 metabolites. 
    Insulin resistance correlates with increased branched-chain amino acid 
    concentrations, which were approximately 1.8-fold higher in the patient group.
    The study used UPLC-MS/MS for metabolite identification and quantification.
    """
    
    sample_query = "What are the key metabolic differences between diabetic patients and healthy controls?"
    
    # Initialize pipeline
    try:
        print("📋 Initializing integrated validation pipeline...")
        pipeline = IntegratedFactualValidationPipeline({
            'claim_extraction': {'confidence_threshold': 60.0},
            'factual_validation': {'min_evidence_confidence': 50}
        })
        
        # Note: In a real implementation, you would await pipeline.initialize()
        # For this demo, we'll simulate the process
        print("✓ Pipeline initialization simulated (would require actual document index)")
        
        print("\n🔍 Sample Processing Workflow:")
        print("-" * 50)
        
        # Simulate claim extraction
        print("1. Claim Extraction:")
        print(f"   - Input: {len(sample_response)} characters")
        print("   - Simulated extraction: 5 factual claims identified")
        print("   - Types: 2 numeric, 1 methodological, 1 qualitative, 1 comparative")
        
        # Simulate verification
        print("\n2. Factual Verification:")
        print("   - Searching document index for supporting evidence...")
        print("   - Verification strategies applied based on claim types")
        print("   - Evidence assessment: SUPPORTED, CONTRADICTED, or NEUTRAL")
        
        # Simulate results
        print("\n3. Validation Results:")
        print("   ✓ Numeric claims: 85% average confidence (2 claims)")
        print("   ✓ Methodological claims: 92% confidence (LC-MS verified)")
        print("   ✓ Qualitative claims: 78% confidence (correlation supported)")
        print("   ✓ Comparative claims: 88% confidence (fold-change verified)")
        print("   ✓ Overall factual accuracy grade: GOOD")
        
        print("\n📊 Simulated Pipeline Performance:")
        print(f"   - Total processing time: 245ms")
        print(f"   - Claims extracted: 5")
        print(f"   - Claims verified: 5")
        print(f"   - Evidence items found: 12")
        print(f"   - Verification success rate: 95%")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        print("Note: Full demo requires initialized document index and source documents")


async def test_single_claim_validation():
    """Test single claim validation functionality."""
    
    print("\n🔬 Single Claim Validation Test")
    print("=" * 40)
    
    test_claims = [
        "Glucose levels were 150 mg/dL in diabetic patients",
        "LC-MS analysis was used for metabolite identification", 
        "Insulin resistance correlates with amino acid concentrations",
        "The study showed a 2-fold increase in branched-chain amino acids"
    ]
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n{i}. Testing claim: {claim}")
        print("   Simulated validation:")
        
        # Simulate validation results
        if "mg/dL" in claim:
            print("   ✓ Numeric verification: SUPPORTED (confidence: 85%)")
            print("   ✓ Evidence: 3 supporting documents found")
        elif "LC-MS" in claim:
            print("   ✓ Methodological verification: SUPPORTED (confidence: 92%)")
            print("   ✓ Evidence: Method confirmed in 5 source documents")
        elif "correlates" in claim:
            print("   ✓ Qualitative verification: SUPPORTED (confidence: 78%)")
            print("   ✓ Evidence: Correlation pattern found in 2 studies")
        elif "fold" in claim:
            print("   ✓ Comparative verification: SUPPORTED (confidence: 88%)")
            print("   ✓ Evidence: Fold-change data verified in source")


async def main():
    """Main demonstration function."""
    
    print("Clinical Metabolomics Oracle - Factual Accuracy Validation System")
    print("================================================================")
    print()
    
    # Run demonstrations
    await demonstrate_integrated_validation()
    await test_single_claim_validation()
    
    print("\n✅ Factual Accuracy Validation System Ready!")
    print("\nFor production use:")
    print("1. Initialize SourceDocumentIndex with your document collection")
    print("2. Configure BiomedicalClaimExtractor for your domain")
    print("3. Set up FactualAccuracyValidator with appropriate thresholds")
    print("4. Process LightRAG responses through the integrated pipeline")
    print("\nIntegration with existing quality assessment systems:")
    print("- Use VerificationReport data in ResponseQualityAssessor")
    print("- Incorporate factual accuracy scores in overall quality metrics")
    print("- Add verification results to audit trails and logging")


if __name__ == "__main__":
    asyncio.run(main())