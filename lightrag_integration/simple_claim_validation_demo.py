#!/usr/bin/env python3
"""
Simple Claim Validation Demonstration for Clinical Metabolomics Oracle.

This script demonstrates the factual claim extraction system and its integration
with quality assessment workflows, focusing on the core claim extraction capabilities.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

# Import claim extraction system
from claim_extractor import (
    BiomedicalClaimExtractor,
    ExtractedClaim,
    prepare_claims_for_quality_assessment
)


async def demonstrate_claim_validation_workflow():
    """Demonstrate the complete claim validation workflow."""
    
    print("=" * 80)
    print("BIOMEDICAL CLAIM EXTRACTION AND VALIDATION WORKFLOW")
    print("=" * 80)
    print()
    
    # Sample biomedical responses for validation
    test_responses = {
        'high_quality_response': """
        A targeted metabolomics analysis using LC-MS/MS identified significant alterations 
        in 47 metabolites in plasma samples from 150 diabetic patients compared to 120 
        healthy controls. Glucose concentrations were markedly elevated (9.2 ± 1.4 mmol/L 
        vs 5.8 ± 0.7 mmol/L, p < 0.001, effect size = 2.8).
        
        Branched-chain amino acids (leucine, isoleucine, valine) showed a 1.8-fold increase 
        in diabetic patients, correlating strongly with insulin resistance measured by 
        HOMA-IR (r = 0.73, p < 0.001). The analysis employed validated analytical methods 
        with precision < 15% CV and accuracy within 95-105%.
        """,
        
        'medium_quality_response': """
        Studies suggest that metabolomics may reveal biomarkers associated with diabetes. 
        Some research indicates that glucose levels might be elevated in diabetic patients 
        compared to controls. Certain amino acids could potentially correlate with insulin 
        resistance, although the evidence varies between studies.
        
        The analytical methods used in these studies typically involve mass spectrometry 
        approaches. Statistical significance is usually assessed using appropriate tests, 
        with p-values often reported as significant when less than 0.05.
        """,
        
        'poor_quality_response': """
        Diabetes affects metabolism in many ways. Patients generally have different 
        metabolite patterns than healthy people. Research shows various changes in 
        blood chemistry. Some studies find correlations with different factors.
        
        Many analytical techniques are available for these studies. Results vary 
        depending on the study design and population examined.
        """
    }
    
    # Initialize extractor
    extractor = BiomedicalClaimExtractor()
    
    # Process each response
    all_results = {}
    
    for response_name, response_text in test_responses.items():
        print(f"Analyzing: {response_name.replace('_', ' ').title()}")
        print("-" * 60)
        
        # Extract claims
        claims = await extractor.extract_claims(response_text)
        
        print(f"✓ Extracted {len(claims)} claims")
        
        # Classify claims by type
        classified = await extractor.classify_claims_by_type(claims)
        print(f"  Claim types: {list(classified.keys())}")
        
        # Filter high-confidence claims
        high_conf_claims = await extractor.filter_high_confidence_claims(claims, 60.0)
        print(f"  High-confidence claims: {len(high_conf_claims)}")
        
        # Prepare for quality assessment
        quality_data = await prepare_claims_for_quality_assessment(claims, 50.0)
        print(f"  Claims for quality assessment: {quality_data['claim_count']}")
        
        # Show top claims
        if claims:
            top_claims = sorted(claims, key=lambda c: c.confidence.overall_confidence, reverse=True)[:3]
            print("  Top claims:")
            for i, claim in enumerate(top_claims, 1):
                preview = claim.claim_text[:60] + ("..." if len(claim.claim_text) > 60 else "")
                print(f"    {i}. [{claim.claim_type.upper()}] {preview}")
                print(f"       Confidence: {claim.confidence.overall_confidence:.1f}% | Priority: {claim.priority_score:.1f}")
        
        # Store results
        all_results[response_name] = {
            'total_claims': len(claims),
            'high_confidence_claims': len(high_conf_claims),
            'claim_types': list(classified.keys()),
            'avg_confidence': sum(c.confidence.overall_confidence for c in claims) / len(claims) if claims else 0,
            'quality_assessment_ready': quality_data['claim_count']
        }
        
        print()
    
    # Overall analysis
    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    for response_name, results in all_results.items():
        quality_score = (
            results['high_confidence_claims'] * 20 +
            results['avg_confidence'] +
            len(results['claim_types']) * 5
        )
        
        quality_grade = "Excellent" if quality_score >= 90 else \
                       "Good" if quality_score >= 70 else \
                       "Fair" if quality_score >= 50 else "Poor"
        
        print(f"{response_name.replace('_', ' ').title()}:")
        print(f"  Quality Score: {quality_score:.1f} ({quality_grade})")
        print(f"  Total Claims: {results['total_claims']}")
        print(f"  High-Confidence Claims: {results['high_confidence_claims']}")
        print(f"  Average Confidence: {results['avg_confidence']:.1f}%")
        print(f"  Claim Type Diversity: {len(results['claim_types'])}")
        print()
    
    # Demonstrate verification preparation
    print("=" * 80)
    print("VERIFICATION PREPARATION WORKFLOW")
    print("=" * 80)
    
    # Get all claims from high-quality response
    high_quality_claims = await extractor.extract_claims(test_responses['high_quality_response'])
    
    # Prepare for verification
    verification_data = await extractor.prepare_claims_for_verification(high_quality_claims)
    
    print(f"Verification Candidates: {len(verification_data['verification_candidates'])}")
    print(f"Claims by Type: {len(verification_data['claims_by_type'])} types")
    print(f"High Priority Claims: {len(verification_data['high_priority_claims'])}")
    
    # Show verification candidates
    candidates = verification_data['verification_candidates'][:5]  # Top 5
    
    if candidates:
        print("\nTop Verification Candidates:")
        print("-" * 40)
        
        for i, candidate in enumerate(candidates, 1):
            print(f"{i}. [{candidate['claim_type'].upper()}] Priority: {candidate['priority_score']:.1f}")
            print(f"   {candidate['claim_text'][:70]}...")
            print(f"   Verification Targets: {', '.join(candidate['verification_targets'][:3])}")
            print(f"   Keywords: {', '.join(candidate['search_keywords'][:5])}")
            print()
    
    # Generate summary report
    print("=" * 80)
    print("INTEGRATION SUMMARY")
    print("=" * 80)
    
    total_claims_all = sum(r['total_claims'] for r in all_results.values())
    total_high_conf = sum(r['high_confidence_claims'] for r in all_results.values())
    
    print("System Capabilities Demonstrated:")
    print("  ✅ Multi-type claim extraction (numeric, qualitative, methodological, temporal, comparative)")
    print("  ✅ Confidence scoring and reliability assessment")
    print("  ✅ Biomedical domain specialization")
    print("  ✅ Claim classification and prioritization")
    print("  ✅ Quality assessment preparation")
    print("  ✅ Verification workflow preparation")
    print("  ✅ Performance optimization and tracking")
    print()
    
    print("Processing Statistics:")
    print(f"  • Total Claims Processed: {total_claims_all}")
    print(f"  • High-Confidence Claims: {total_high_conf} ({(total_high_conf/max(1,total_claims_all))*100:.1f}%)")
    print(f"  • Response Quality Range: Poor to Excellent")
    print(f"  • Verification Candidates: {len(verification_data['verification_candidates'])}")
    
    stats = extractor.get_extraction_statistics()
    print(f"  • Average Processing Time: {stats['processing_times']['average_ms']:.1f}ms")
    print(f"  • Average Claims per Extraction: {stats['average_claims_per_extraction']:.1f}")
    print()
    
    # Save detailed results
    detailed_results = {
        'demonstration_timestamp': datetime.now().isoformat(),
        'system_version': '1.0.0',
        'response_analyses': all_results,
        'verification_preparation': {
            'total_candidates': len(verification_data['verification_candidates']),
            'candidate_types': list(verification_data['claims_by_type'].keys()),
            'extraction_metadata': verification_data['extraction_metadata']
        },
        'performance_statistics': stats
    }
    
    with open('claim_validation_workflow_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("📊 Detailed results saved to: claim_validation_workflow_results.json")
    print()
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("The BiomedicalClaimExtractor is ready for integration with the LightRAG quality validation infrastructure.")


async def main():
    """Main demonstration function."""
    
    print("Clinical Metabolomics Oracle")
    print("Biomedical Claim Extraction and Validation System")
    print("=" * 80)
    print()
    
    try:
        await demonstrate_claim_validation_workflow()
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    """Run the claim validation workflow demonstration."""
    asyncio.run(main())