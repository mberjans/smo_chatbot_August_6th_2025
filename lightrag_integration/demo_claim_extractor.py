#!/usr/bin/env python3
"""
Biomedical Claim Extractor Demonstration and Integration Example.

This script demonstrates the capabilities of the BiomedicalClaimExtractor
and shows how it integrates with the existing quality assessment pipeline
in the Clinical Metabolomics Oracle LightRAG integration system.

Features Demonstrated:
    - Comprehensive claim extraction from biomedical responses
    - Multi-type claim classification and analysis
    - Confidence scoring and filtering
    - Integration with quality assessment systems
    - Verification preparation workflow
    - Performance monitoring and statistics

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Factual Claim Extraction Implementation
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import the claim extractor
try:
    from claim_extractor import (
        BiomedicalClaimExtractor,
        ExtractedClaim,
        extract_claims_from_response,
        prepare_claims_for_quality_assessment
    )
    CLAIM_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    CLAIM_EXTRACTOR_AVAILABLE = False
    print(f"Claim extractor not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimExtractorDemo:
    """Demonstration class for the BiomedicalClaimExtractor."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.extractor = BiomedicalClaimExtractor()
        self.demo_responses = self._load_demo_responses()
        self.results = {}
    
    def _load_demo_responses(self) -> Dict[str, str]:
        """Load demonstration response samples."""
        
        return {
            'comprehensive_metabolomics_study': """
            A comprehensive untargeted metabolomics analysis was performed using LC-MS/MS 
            on plasma samples from 247 type 2 diabetes patients and 198 age-matched healthy 
            controls. The study identified 1,247 metabolites with high confidence (CV < 15%). 
            
            Glucose concentrations were significantly elevated in diabetic patients 
            (9.8 ± 2.1 mmol/L) compared to controls (5.2 ± 0.8 mmol/L, p < 0.001). 
            Insulin resistance, measured by HOMA-IR, correlated strongly with branched-chain 
            amino acid levels (r = 0.73, p < 0.001). 
            
            The analysis revealed a 2.3-fold increase in oxidative stress markers and 
            a 45% reduction in antioxidant metabolites in the diabetic cohort. These 
            findings suggest that metabolic dysregulation in diabetes involves complex 
            perturbations across multiple biochemical pathways.
            
            Statistical analysis was performed using multivariate PCA and PLS-DA methods, 
            with false discovery rate correction (FDR < 0.05). The study was conducted 
            over 18 months following a randomized controlled trial design.
            """,
            
            'clinical_biomarker_validation': """
            Targeted metabolomics analysis using LC-MS/MS validated potential biomarkers 
            for early-stage pancreatic cancer. Serum samples from 156 cancer patients 
            and 142 healthy controls were analyzed using a validated analytical method 
            with detection limits ranging from 0.5 to 50 ng/mL.
            
            Three metabolites showed exceptional diagnostic performance: CA 19-9 levels 
            were elevated 8.7-fold (p < 0.001), while glutamine concentrations decreased 
            by 35% (p = 0.003), and lactate levels increased by approximately 2.1-fold 
            in cancer patients versus controls.
            
            The combined biomarker panel achieved 89% sensitivity and 94% specificity 
            for cancer detection, with an AUC of 0.96 (95% CI: 0.93-0.99). These results 
            demonstrate the potential clinical utility of metabolomic profiling for 
            early cancer diagnosis.
            """,
            
            'pharmacokinetic_study': """
            A Phase I pharmacokinetic study evaluated the metabolic fate of compound XYZ-123 
            in 24 healthy volunteers. Blood samples were collected at 0, 0.5, 1, 2, 4, 8, 12, 
            and 24 hours post-administration. LC-MS/MS analysis identified the parent compound 
            and five major metabolites.
            
            Peak plasma concentration (Cmax) was reached at 2.5 ± 0.8 hours, with 
            maximum levels of 127 ± 34 ng/mL. The elimination half-life was calculated 
            as 6.8 ± 1.2 hours, and total clearance was 2.4 ± 0.6 L/hr/kg. 
            
            Metabolite M1 accounted for 65% of total drug-related exposure, while M2 
            represented 23% of the circulating compounds. The study protocol was approved 
            by the institutional review board and conducted according to GCP guidelines 
            over a 12-week period.
            """,
            
            'method_development_validation': """
            A novel UPLC-MS/MS method was developed and validated for simultaneous 
            quantification of 85 endogenous metabolites in human plasma. The method 
            employed a HILIC column (2.1 × 100 mm, 1.7 μm particles) with gradient 
            elution using acetonitrile and 10 mM ammonium formate buffer.
            
            Method validation demonstrated excellent linearity (r² > 0.995) across 
            the concentration range of 0.1-1000 ng/mL for all analytes. Precision 
            values ranged from 2.1% to 8.9% CV, while accuracy was between 95-108% 
            for all quality control samples.
            
            Lower limits of quantification (LLOQ) ranged from 0.05 to 2.5 ng/mL, 
            with matrix effects below 15% for 98% of analyzed compounds. The total 
            run time was optimized to 12 minutes per sample, enabling high-throughput 
            analysis of large clinical studies.
            """,
            
            'uncertain_preliminary_findings': """
            Preliminary data suggests that certain metabolic profiles might be associated 
            with disease progression in some patients. It appears that biomarker levels 
            could potentially fluctuate depending on various factors that may include 
            diet, exercise, and possibly genetic variations.
            
            Some studies have reported that metabolite ratios might correlate with 
            therapeutic response, although the evidence remains inconclusive. Further 
            research may be needed to establish whether these observations represent 
            true biological relationships or methodological artifacts.
            
            The current findings should be interpreted with caution, as sample sizes 
            were limited and confounding variables were not fully controlled. Additional 
            validation studies would likely be required before drawing definitive conclusions.
            """
        }
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of claim extraction capabilities."""
        
        print("=" * 80)
        print("BIOMEDICAL CLAIM EXTRACTOR - COMPREHENSIVE DEMONSTRATION")
        print("=" * 80)
        print()
        
        # Process each demo response
        for response_name, response_text in self.demo_responses.items():
            print(f"Processing: {response_name.replace('_', ' ').title()}")
            print("-" * 60)
            
            # Extract claims
            start_time = time.time()
            claims = await self.extractor.extract_claims(response_text)
            processing_time = (time.time() - start_time) * 1000
            
            print(f"✓ Extracted {len(claims)} claims in {processing_time:.1f}ms")
            
            # Store results
            self.results[response_name] = {
                'total_claims': len(claims),
                'processing_time_ms': processing_time,
                'claims': claims
            }
            
            # Analyze by type
            await self._analyze_claims_by_type(claims)
            
            # Show confidence distribution
            await self._show_confidence_distribution(claims)
            
            # Highlight interesting findings
            await self._highlight_key_findings(claims, response_name)
            
            print()
        
        # Overall analysis
        await self._show_overall_analysis()
        
        # Integration demonstrations
        await self._demonstrate_quality_integration()
        
        # Performance analysis
        await self._show_performance_analysis()
    
    async def _analyze_claims_by_type(self, claims: List[ExtractedClaim]):
        """Analyze and display claims by type."""
        
        classified = await self.extractor.classify_claims_by_type(claims)
        
        print("  Claim Types:")
        for claim_type, type_claims in classified.items():
            avg_confidence = sum(c.confidence.overall_confidence for c in type_claims) / len(type_claims)
            print(f"    • {claim_type.title()}: {len(type_claims)} claims "
                  f"(avg confidence: {avg_confidence:.1f})")
    
    async def _show_confidence_distribution(self, claims: List[ExtractedClaim]):
        """Show confidence score distribution."""
        
        if not claims:
            return
        
        confidences = [c.confidence.overall_confidence for c in claims]
        
        high_conf = len([c for c in confidences if c >= 70])
        med_conf = len([c for c in confidences if 50 <= c < 70])
        low_conf = len([c for c in confidences if c < 50])
        
        print("  Confidence Distribution:")
        print(f"    • High (≥70): {high_conf} claims")
        print(f"    • Medium (50-69): {med_conf} claims")
        print(f"    • Low (<50): {low_conf} claims")
    
    async def _highlight_key_findings(self, claims: List[ExtractedClaim], response_name: str):
        """Highlight key findings from extracted claims."""
        
        # Get top 3 highest confidence claims
        top_claims = sorted(claims, key=lambda c: c.confidence.overall_confidence, reverse=True)[:3]
        
        print("  Key Extracted Claims:")
        for i, claim in enumerate(top_claims, 1):
            # Truncate long claims for display
            display_text = claim.claim_text
            if len(display_text) > 80:
                display_text = display_text[:77] + "..."
            
            print(f"    {i}. [{claim.claim_type.upper()}] {display_text}")
            print(f"       Confidence: {claim.confidence.overall_confidence:.1f}% | "
                  f"Priority: {claim.priority_score:.1f}")
            
            # Show numeric values if present
            if claim.numeric_values:
                values_str = ", ".join(f"{v}{' ' + u if i < len(claim.units) else ''}" 
                                     for i, v in enumerate(claim.numeric_values) 
                                     for u in (claim.units[i:i+1] if i < len(claim.units) else ['']))
                print(f"       Values: {values_str}")
    
    async def _show_overall_analysis(self):
        """Show overall analysis across all responses."""
        
        print("=" * 80)
        print("OVERALL ANALYSIS")
        print("=" * 80)
        
        # Aggregate statistics
        total_claims = sum(r['total_claims'] for r in self.results.values())
        avg_processing_time = sum(r['processing_time_ms'] for r in self.results.values()) / len(self.results)
        
        all_claims = []
        for result in self.results.values():
            all_claims.extend(result['claims'])
        
        # Type distribution
        type_dist = {}
        for claim in all_claims:
            type_dist[claim.claim_type] = type_dist.get(claim.claim_type, 0) + 1
        
        print(f"Total Claims Extracted: {total_claims}")
        print(f"Average Processing Time: {avg_processing_time:.1f}ms")
        print()
        
        print("Claim Type Distribution:")
        for claim_type, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_claims) * 100
            print(f"  • {claim_type.title()}: {count} ({percentage:.1f}%)")
        
        # Confidence analysis
        all_confidences = [c.confidence.overall_confidence for c in all_claims]
        avg_confidence = sum(all_confidences) / len(all_confidences)
        
        print(f"\nOverall Average Confidence: {avg_confidence:.1f}%")
        
        # High-value claims
        high_value_claims = [c for c in all_claims if c.priority_score >= 75]
        print(f"High-Priority Claims: {len(high_value_claims)} ({(len(high_value_claims)/total_claims)*100:.1f}%)")
    
    async def _demonstrate_quality_integration(self):
        """Demonstrate integration with quality assessment systems."""
        
        print("\n" + "=" * 80)
        print("QUALITY ASSESSMENT INTEGRATION")
        print("=" * 80)
        
        # Get all claims
        all_claims = []
        for result in self.results.values():
            all_claims.extend(result['claims'])
        
        # Prepare for quality assessment
        quality_data = await prepare_claims_for_quality_assessment(all_claims, min_confidence=60.0)
        
        print("Quality Assessment Preparation:")
        print(f"  • Total factual claims: {quality_data['claim_count']}")
        print(f"  • High-priority claims: {len(quality_data['high_priority_claims'])}")
        print(f"  • Claims needing verification: {len(quality_data['verification_needed'])}")
        
        # Show verification candidates
        verification_data = await self.extractor.prepare_claims_for_verification(all_claims)
        candidates = verification_data['verification_candidates']
        
        print(f"\nVerification Pipeline:")
        print(f"  • Verification candidates: {len(candidates)}")
        
        if candidates:
            top_candidate = max(candidates, key=lambda c: c['priority_score'])
            print(f"  • Top candidate: {top_candidate['claim_type']} claim")
            print(f"    Priority: {top_candidate['priority_score']:.1f}")
            print(f"    Targets: {', '.join(top_candidate['verification_targets'][:3])}")
        
        # Show confidence distribution
        conf_dist = quality_data['assessment_metadata'].get('confidence_distribution', {})
        if conf_dist:
            print(f"\nConfidence Distribution:")
            for level, count in conf_dist.items():
                print(f"  • {level.replace('_', ' ').title()}: {count}")
    
    async def _show_performance_analysis(self):
        """Show performance analysis and statistics."""
        
        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Get extractor statistics
        stats = self.extractor.get_extraction_statistics()
        
        print("Extraction Statistics:")
        print(f"  • Total extractions: {stats['total_extractions']}")
        print(f"  • Total claims extracted: {stats['total_claims_extracted']}")
        print(f"  • Average claims per extraction: {stats['average_claims_per_extraction']:.1f}")
        
        # Processing time analysis
        times = stats['processing_times']
        if times['count'] > 0:
            print(f"\nProcessing Performance:")
            print(f"  • Average time: {times['average_ms']:.1f}ms")
            print(f"  • Median time: {times['median_ms']:.1f}ms")
            print(f"  • Range: {times['min_ms']:.1f}ms - {times['max_ms']:.1f}ms")
        
        # Response complexity analysis
        print(f"\nResponse Complexity Analysis:")
        for response_name, result in self.results.items():
            complexity_score = result['total_claims'] * (result['processing_time_ms'] / 100)
            print(f"  • {response_name.replace('_', ' ').title()}: "
                  f"{result['total_claims']} claims, {result['processing_time_ms']:.1f}ms "
                  f"(complexity: {complexity_score:.1f})")
    
    async def demonstrate_specific_features(self):
        """Demonstrate specific features of the claim extractor."""
        
        print("\n" + "=" * 80)
        print("SPECIFIC FEATURE DEMONSTRATIONS")
        print("=" * 80)
        
        # Feature 1: Confidence filtering
        print("\n1. Confidence-based Filtering:")
        all_claims = []
        for result in self.results.values():
            all_claims.extend(result['claims'])
        
        high_conf = await self.extractor.filter_high_confidence_claims(all_claims, 80.0)
        med_conf = await self.extractor.filter_high_confidence_claims(all_claims, 60.0)
        
        print(f"   Original claims: {len(all_claims)}")
        print(f"   High confidence (≥80): {len(high_conf)}")
        print(f"   Medium confidence (≥60): {len(med_conf)}")
        
        # Feature 2: Biomedical specialization
        print("\n2. Biomedical Specialization:")
        biomedical_keywords = set()
        for claim in all_claims:
            biomedical_keywords.update(claim.keywords)
        
        bio_terms = [kw for kw in biomedical_keywords 
                    if any(bt in kw.lower() for bt in ['metabol', 'gluc', 'insulin', 'clinical', 'plasma'])]
        
        print(f"   Biomedical terms identified: {len(bio_terms)}")
        if bio_terms:
            print(f"   Examples: {', '.join(list(bio_terms)[:5])}")
        
        # Feature 3: Claim relationships
        print("\n3. Relationship Extraction:")
        relationship_claims = [c for c in all_claims if c.relationships]
        
        print(f"   Claims with relationships: {len(relationship_claims)}")
        if relationship_claims:
            example_rel = relationship_claims[0].relationships[0]
            print(f"   Example: {example_rel.get('subject', 'N/A')} {example_rel.get('predicate', 'N/A')} "
                  f"{example_rel.get('object', 'N/A')}")
    
    async def save_detailed_results(self, output_file: str = "claim_extraction_demo_results.json"):
        """Save detailed results to file for further analysis."""
        
        # Prepare serializable results
        detailed_results = {
            'demo_timestamp': datetime.now().isoformat(),
            'extraction_statistics': self.extractor.get_extraction_statistics(),
            'response_results': {}
        }
        
        for response_name, result in self.results.items():
            detailed_results['response_results'][response_name] = {
                'total_claims': result['total_claims'],
                'processing_time_ms': result['processing_time_ms'],
                'claims_summary': [
                    {
                        'claim_id': claim.claim_id,
                        'claim_type': claim.claim_type,
                        'confidence': claim.confidence.overall_confidence,
                        'priority': claim.priority_score,
                        'text_preview': claim.claim_text[:100] + ("..." if len(claim.claim_text) > 100 else ""),
                        'numeric_values': claim.numeric_values,
                        'units': claim.units,
                        'keywords': claim.keywords[:5]  # Limit for readability
                    }
                    for claim in result['claims']
                ]
            }
        
        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_path.absolute()}")


async def main():
    """Main demonstration function."""
    
    if not CLAIM_EXTRACTOR_AVAILABLE:
        print("Claim extractor is not available. Please ensure the module is properly installed.")
        return
    
    print("Initializing Biomedical Claim Extractor Demo...")
    
    demo = ClaimExtractorDemo()
    
    try:
        # Run comprehensive demonstration
        await demo.run_comprehensive_demo()
        
        # Demonstrate specific features
        await demo.demonstrate_specific_features()
        
        # Save detailed results
        await demo.save_detailed_results()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nThe BiomedicalClaimExtractor has been successfully demonstrated.")
        print("Key capabilities shown:")
        print("  ✓ Multi-type claim extraction (numeric, qualitative, methodological, etc.)")
        print("  ✓ Confidence scoring and filtering")
        print("  ✓ Biomedical domain specialization")
        print("  ✓ Integration with quality assessment pipeline")
        print("  ✓ Performance monitoring and optimization")
        print("  ✓ Verification preparation workflow")
        print("\nThe system is ready for integration with the LightRAG quality validation infrastructure.")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        print(f"\n✗ Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    """Run the demonstration."""
    asyncio.run(main())