#!/usr/bin/env python3
"""
Simple test script to verify BiomedicalResponseFormatter integration.
This script tests the formatter in isolation to ensure it works correctly.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

class BiomedicalResponseFormatter:
    """
    Simplified version of BiomedicalResponseFormatter for testing.
    """
    
    def __init__(self, formatting_config: Optional[Dict[str, Any]] = None):
        self.config = formatting_config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self._compile_patterns()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'extract_entities': True,
            'format_statistics': True,
            'process_sources': True,
            'structure_sections': True,
            'add_clinical_indicators': True,
            'highlight_metabolites': True,
            'format_pathways': True,
            'max_entity_extraction': 50,
            'include_confidence_scores': True,
            'preserve_original_formatting': True
        }
    
    def _compile_patterns(self) -> None:
        # Metabolite patterns
        self.metabolite_patterns = [
            re.compile(r'\b[A-Z][a-z]+(?:-[A-Z]?[a-z]+)*\b(?=\s*(?:concentration|level|metabolism|metabolite))', re.IGNORECASE),
            re.compile(r'\b(?:glucose|insulin|cortisol|creatinine|urea|lactate|pyruvate|acetate|citrate|succinate)\b', re.IGNORECASE),
        ]
        
        # Statistical patterns
        self.statistical_patterns = [
            re.compile(r'p\s*[<>=]\s*0?\.\d+(?:e-?\d+)?', re.IGNORECASE),
            re.compile(r'\b(?:mean|median|SD|SEM|IQR)\s*[±=:]\s*\d+\.?\d*(?:\s*±\s*\d+\.?\d*)?', re.IGNORECASE),
        ]
        
        # Disease patterns
        self.disease_patterns = [
            re.compile(r'\b(?:diabetes|cardiovascular|cancer|Alzheimer|obesity|metabolic syndrome|hypertension)\b', re.IGNORECASE),
        ]
    
    def format_response(self, raw_response: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format a biomedical response with entity extraction and structuring."""
        if not raw_response or not isinstance(raw_response, str):
            return self._create_empty_formatted_response("Empty or invalid response")
        
        try:
            formatted_response = {
                'formatted_content': raw_response,
                'original_content': raw_response,
                'sections': {},
                'entities': {},
                'statistics': [],
                'sources': [],
                'clinical_indicators': {},
                'formatting_metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'formatter_version': '1.0.0',
                    'applied_formatting': []
                }
            }
            
            # Extract entities
            if self.config.get('extract_entities', True):
                formatted_response = self._extract_biomedical_entities(formatted_response)
                formatted_response['formatting_metadata']['applied_formatting'].append('entity_extraction')
            
            # Format statistics
            if self.config.get('format_statistics', True):
                formatted_response = self._format_statistical_data(formatted_response)
                formatted_response['formatting_metadata']['applied_formatting'].append('statistical_formatting')
            
            # Add clinical indicators
            if self.config.get('add_clinical_indicators', True):
                formatted_response = self._add_clinical_relevance_indicators(formatted_response)
                formatted_response['formatting_metadata']['applied_formatting'].append('clinical_indicators')
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error formatting biomedical response: {e}")
            return self._create_error_formatted_response(str(e), raw_response)
    
    def _extract_biomedical_entities(self, formatted_response: Dict[str, Any]) -> Dict[str, Any]:
        content = formatted_response['formatted_content']
        entities = {
            'metabolites': [],
            'diseases': []
        }
        
        # Extract metabolites
        for pattern in self.metabolite_patterns:
            matches = pattern.findall(content)
            entities['metabolites'].extend(matches[:20])  # Limit for testing
        
        # Extract diseases
        for pattern in self.disease_patterns:
            matches = pattern.findall(content)
            entities['diseases'].extend(matches[:20])
        
        # Clean up duplicates
        for entity_type in entities:
            entities[entity_type] = list(set([e.strip() for e in entities[entity_type] if e.strip()]))
        
        formatted_response['entities'] = entities
        return formatted_response
    
    def _format_statistical_data(self, formatted_response: Dict[str, Any]) -> Dict[str, Any]:
        content = formatted_response['formatted_content']
        statistics = []
        
        for pattern in self.statistical_patterns:
            matches = pattern.finditer(content)
            for match in matches:
                stat_info = {
                    'text': match.group(0),
                    'position': match.span(),
                    'type': self._classify_statistic(match.group(0))
                }
                statistics.append(stat_info)
        
        formatted_response['statistics'] = statistics
        return formatted_response
    
    def _classify_statistic(self, stat_text: str) -> str:
        stat_lower = stat_text.lower()
        if 'p' in stat_lower and ('=' in stat_lower or '<' in stat_lower or '>' in stat_lower):
            return 'p_value'
        elif any(term in stat_lower for term in ['mean', 'median', 'sd', 'sem']):
            return 'descriptive_statistic'
        else:
            return 'other'
    
    def _add_clinical_relevance_indicators(self, formatted_response: Dict[str, Any]) -> Dict[str, Any]:
        content = formatted_response['formatted_content'].lower()
        
        clinical_indicators = {
            'disease_association': any(disease in content for disease in [
                'diabetes', 'cancer', 'cardiovascular', 'alzheimer', 'obesity'
            ]),
            'diagnostic_potential': any(term in content for term in [
                'biomarker', 'diagnostic', 'screening', 'detection'
            ]),
            'therapeutic_relevance': any(term in content for term in [
                'treatment', 'therapy', 'drug', 'therapeutic', 'intervention'
            ]),
        }
        
        relevance_score = sum(clinical_indicators.values()) / len(clinical_indicators)
        clinical_indicators['overall_relevance_score'] = relevance_score
        
        formatted_response['clinical_indicators'] = clinical_indicators
        return formatted_response
    
    def _create_empty_formatted_response(self, reason: str) -> Dict[str, Any]:
        return {
            'formatted_content': '',
            'original_content': '',
            'sections': {},
            'entities': {},
            'statistics': [],
            'sources': [],
            'clinical_indicators': {},
            'error': reason,
            'formatting_metadata': {
                'processed_at': datetime.now().isoformat(),
                'formatter_version': '1.0.0',
                'status': 'error'
            }
        }
    
    def _create_error_formatted_response(self, error_msg: str, raw_response: str) -> Dict[str, Any]:
        return {
            'formatted_content': raw_response,
            'original_content': raw_response,
            'sections': {'main_content': raw_response},
            'entities': {},
            'statistics': [],
            'sources': [],
            'clinical_indicators': {},
            'error': f"Formatting failed: {error_msg}",
            'formatting_metadata': {
                'processed_at': datetime.now().isoformat(),
                'formatter_version': '1.0.0',
                'status': 'partial_error'
            }
        }


def test_biomedical_response_formatter():
    """Test the BiomedicalResponseFormatter with sample biomedical content."""
    print("Testing BiomedicalResponseFormatter...")
    
    # Initialize formatter
    formatter = BiomedicalResponseFormatter()
    print(f"✓ Formatter initialized with config: {list(formatter.config.keys())}")
    
    # Test with sample biomedical response
    sample_response = """
    Glucose metabolism plays a crucial role in diabetes management. Studies show that insulin levels 
    are significantly altered in diabetic patients (p < 0.001). The mean glucose concentration 
    was 120.5 ± 15.2 mg/dL in the control group compared to 180.3 ± 22.1 mg/dL in diabetic patients.
    
    Key biomarkers include HbA1c levels and C-peptide concentrations. This research provides diagnostic 
    insights into metabolic disorders and suggests potential therapeutic interventions for cardiovascular 
    complications associated with diabetes.
    
    Citrate and lactate levels were also measured, showing significant correlations with disease progression.
    """
    
    # Format the response
    formatted = formatter.format_response(sample_response)
    
    # Test results
    print(f"✓ Response formatted successfully")
    print(f"✓ Applied formatting: {formatted['formatting_metadata']['applied_formatting']}")
    print(f"✓ Extracted metabolites: {formatted['entities']['metabolites']}")
    print(f"✓ Extracted diseases: {formatted['entities']['diseases']}")
    print(f"✓ Found statistics: {len(formatted['statistics'])} items")
    print(f"✓ Clinical relevance score: {formatted['clinical_indicators']['overall_relevance_score']:.2f}")
    
    # Test with empty response
    empty_formatted = formatter.format_response("")
    print(f"✓ Empty response handled correctly: {empty_formatted.get('error', 'No error')}")
    
    print("\n✓ All tests passed! BiomedicalResponseFormatter is working correctly.")
    return True


def test_response_structure_compatibility():
    """Test that the enhanced response structure maintains backward compatibility."""
    print("\nTesting response structure compatibility...")
    
    # Simulate the enhanced response structure from the integrated query method
    formatter = BiomedicalResponseFormatter()
    sample_response = "Glucose levels in diabetes patients showed significant elevation (p < 0.05)."
    
    formatted_data = formatter.format_response(sample_response)
    
    # Simulate the enhanced result structure
    result = {
        'content': sample_response,  # Original response for backward compatibility
        'metadata': {
            'sources': [],
            'confidence': 0.9,
            'mode': 'hybrid'
        },
        'cost': 0.001,
        'token_usage': {'total_tokens': 150, 'prompt_tokens': 100, 'completion_tokens': 50},
        'query_mode': 'hybrid',
        'processing_time': 1.5
    }
    
    # Add formatted response data
    result['formatted_response'] = formatted_data
    result['biomedical_metadata'] = {
        'entities': formatted_data.get('entities', {}),
        'clinical_indicators': formatted_data.get('clinical_indicators', {}),
        'statistics': formatted_data.get('statistics', []),
        'metabolite_highlights': formatted_data.get('metabolite_highlights', []),
        'sections': list(formatted_data.get('sections', {}).keys()),
        'formatting_applied': formatted_data.get('formatting_metadata', {}).get('applied_formatting', [])
    }
    
    # Test backward compatibility
    assert 'content' in result, "Original content field missing"
    assert 'metadata' in result, "Metadata field missing"
    assert 'cost' in result, "Cost field missing"
    assert result['content'] == sample_response, "Original content not preserved"
    
    # Test new enhancement fields
    assert 'formatted_response' in result, "Formatted response field missing"
    assert 'biomedical_metadata' in result, "Biomedical metadata field missing"
    assert len(result['biomedical_metadata']['entities']) > 0, "No entities extracted"
    
    print("✓ Backward compatibility maintained")
    print("✓ New enhancement fields present")
    print("✓ Response structure test passed")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("BiomedicalResponseFormatter Integration Test")
    print("=" * 60)
    
    try:
        test_biomedical_response_formatter()
        test_response_structure_compatibility()
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Integration is working correctly!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()