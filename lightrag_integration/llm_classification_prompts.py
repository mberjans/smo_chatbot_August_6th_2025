"""
LLM-based Query Classification Prompt Templates for Clinical Metabolomics Oracle

This module provides optimized prompt templates for semantic query classification using LLMs,
designed to integrate with the existing classification infrastructure while adding enhanced
semantic understanding capabilities.

Key Features:
    - Token-efficient prompts optimized for <2 second response times
    - Clinical metabolomics domain-specific understanding
    - Structured JSON output compatible with existing ClassificationResult class
    - Multi-tier prompt strategy for different confidence levels
    - Few-shot examples for each classification category

Prompt Templates:
    - PRIMARY_CLASSIFICATION_PROMPT: Main system prompt for semantic classification
    - FALLBACK_CLASSIFICATION_PROMPT: Simplified prompt for performance-critical cases
    - CONFIDENCE_VALIDATION_PROMPT: Validation prompt for uncertain classifications
    - Category-specific few-shot examples for KNOWLEDGE_GRAPH, REAL_TIME, and GENERAL
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ClassificationCategory(Enum):
    """Classification categories for query routing."""
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    REAL_TIME = "REAL_TIME"  
    GENERAL = "GENERAL"


@dataclass
class ClassificationResult:
    """
    Structured result format for LLM-based classification.
    Compatible with existing routing infrastructure.
    """
    category: str
    confidence: float
    reasoning: str
    alternative_categories: List[Dict[str, float]]
    uncertainty_indicators: List[str]
    biomedical_signals: Dict[str, Any]
    temporal_signals: Dict[str, Any]


class LLMClassificationPrompts:
    """
    Collection of optimized prompts for LLM-based query classification
    in the Clinical Metabolomics Oracle system.
    """
    
    # ============================================================================
    # PRIMARY CLASSIFICATION SYSTEM PROMPT
    # ============================================================================
    
    PRIMARY_CLASSIFICATION_PROMPT = """You are an expert biomedical query classifier for the Clinical Metabolomics Oracle system. Your task is to classify user queries into one of three categories for optimal routing to specialized knowledge systems.

**CLASSIFICATION CATEGORIES:**

1. **KNOWLEDGE_GRAPH** - Route to LightRAG knowledge graph system for:
   - Established relationships between metabolites, pathways, diseases, biomarkers
   - Mechanistic questions about biological processes and pathways
   - Structural queries about molecular connections and interactions
   - Historical/established knowledge in metabolomics and clinical research
   - Pattern matching in known biomedical data

2. **REAL_TIME** - Route to Perplexity API for current information about:
   - Latest research findings, publications, and discoveries (2024+)
   - Recent clinical trials, FDA approvals, regulatory updates
   - Breaking news in metabolomics, drug discovery, diagnostics
   - Current market developments and emerging technologies
   - Time-sensitive or date-specific information needs

3. **GENERAL** - Route flexibly (either system can handle) for:
   - Basic definitional or educational questions
   - Simple explanations of metabolomics concepts
   - Broad introductory topics without specific temporal requirements
   - General methodology or technique inquiries

**BIOMEDICAL CONTEXT:**
Focus on clinical metabolomics including: metabolite identification, pathway analysis, biomarker discovery, drug development, clinical diagnostics, mass spectrometry, NMR, chromatography, systems biology, precision medicine.

**CLASSIFICATION RULES:**
- Prioritize REAL_TIME for queries with temporal indicators: "latest", "recent", "2024", "2025", "new", "current", "breakthrough", "FDA approval", "clinical trial results"
- Prioritize KNOWLEDGE_GRAPH for relationship queries: "connection between", "pathway involving", "mechanism of", "biomarker for", "relationship"
- Use GENERAL for basic "what is", "explain", "define", "how does X work" without specific context

**OUTPUT FORMAT:**
Return ONLY a valid JSON object with this exact structure:
```json
{
  "category": "KNOWLEDGE_GRAPH|REAL_TIME|GENERAL",
  "confidence": 0.85,
  "reasoning": "Brief explanation of classification decision",
  "alternative_categories": [{"category": "REAL_TIME", "confidence": 0.15}],
  "uncertainty_indicators": ["list", "of", "uncertainty", "factors"],
  "biomedical_signals": {
    "entities": ["metabolite", "pathway"],
    "relationships": ["connection", "mechanism"],
    "techniques": ["LC-MS", "NMR"]
  },
  "temporal_signals": {
    "keywords": ["recent", "2024"],
    "patterns": ["latest research"],
    "years": ["2024"]
  }
}
```

**QUERY TO CLASSIFY:**
{query_text}

**CLASSIFICATION:**"""

    # ============================================================================
    # FEW-SHOT EXAMPLES FOR EACH CATEGORY
    # ============================================================================
    
    KNOWLEDGE_GRAPH_EXAMPLES = [
        {
            "query": "What is the relationship between glucose metabolism and insulin signaling in diabetes?",
            "classification": {
                "category": "KNOWLEDGE_GRAPH",
                "confidence": 0.92,
                "reasoning": "Query asks about established metabolic relationships and mechanisms between well-studied biological processes.",
                "alternative_categories": [{"category": "GENERAL", "confidence": 0.08}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["glucose", "insulin", "diabetes"],
                    "relationships": ["relationship between", "signaling"],
                    "techniques": []
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        },
        {
            "query": "How does the citric acid cycle connect to fatty acid biosynthesis?",
            "classification": {
                "category": "KNOWLEDGE_GRAPH",
                "confidence": 0.89,
                "reasoning": "Query about established biochemical pathway connections, ideal for knowledge graph traversal.",
                "alternative_categories": [{"category": "GENERAL", "confidence": 0.11}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["citric acid cycle", "fatty acid biosynthesis"],
                    "relationships": ["connect to"],
                    "techniques": []
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        },
        {
            "query": "Find metabolites associated with Alzheimer's disease biomarkers in cerebrospinal fluid",
            "classification": {
                "category": "KNOWLEDGE_GRAPH",
                "confidence": 0.87,
                "reasoning": "Query seeks established associations between metabolites and disease biomarkers, perfect for knowledge graph relationships.",
                "alternative_categories": [{"category": "REAL_TIME", "confidence": 0.13}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["metabolites", "Alzheimer's disease", "biomarkers", "cerebrospinal fluid"],
                    "relationships": ["associated with"],
                    "techniques": []
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        },
        {
            "query": "What pathways are involved in tryptophan metabolism and serotonin synthesis?",
            "classification": {
                "category": "KNOWLEDGE_GRAPH",
                "confidence": 0.91,
                "reasoning": "Classic biochemical pathway query about established metabolic connections, ideal for knowledge graph.",
                "alternative_categories": [{"category": "GENERAL", "confidence": 0.09}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["tryptophan", "serotonin", "pathways"],
                    "relationships": ["involved in", "synthesis"],
                    "techniques": []
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        },
        {
            "query": "Mechanism of action for metformin in glucose metabolism regulation",
            "classification": {
                "category": "KNOWLEDGE_GRAPH",
                "confidence": 0.88,
                "reasoning": "Query about established drug mechanism and metabolic regulation, well-suited for knowledge graph analysis.",
                "alternative_categories": [{"category": "GENERAL", "confidence": 0.12}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["metformin", "glucose metabolism"],
                    "relationships": ["mechanism of action", "regulation"],
                    "techniques": []
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        }
    ]
    
    REAL_TIME_EXAMPLES = [
        {
            "query": "What are the latest 2024 FDA approvals for metabolomics-based diagnostics?",
            "classification": {
                "category": "REAL_TIME",
                "confidence": 0.95,
                "reasoning": "Query explicitly asks for latest 2024 information about regulatory approvals, requiring current data.",
                "alternative_categories": [{"category": "GENERAL", "confidence": 0.05}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["FDA approvals", "metabolomics", "diagnostics"],
                    "relationships": [],
                    "techniques": ["metabolomics"]
                },
                "temporal_signals": {
                    "keywords": ["latest", "2024"],
                    "patterns": ["latest 2024"],
                    "years": ["2024"]
                }
            }
        },
        {
            "query": "Recent breakthrough discoveries in cancer metabolomics this year",
            "classification": {
                "category": "REAL_TIME",
                "confidence": 0.93,
                "reasoning": "Query seeks recent breakthroughs and discoveries, indicating need for current, time-sensitive information.",
                "alternative_categories": [{"category": "KNOWLEDGE_GRAPH", "confidence": 0.07}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["cancer", "metabolomics"],
                    "relationships": ["discoveries in"],
                    "techniques": ["metabolomics"]
                },
                "temporal_signals": {
                    "keywords": ["recent", "breakthrough", "this year"],
                    "patterns": ["recent breakthrough", "this year"],
                    "years": []
                }
            }
        },
        {
            "query": "Current clinical trials using mass spectrometry for early disease detection",
            "classification": {
                "category": "REAL_TIME",
                "confidence": 0.89,
                "reasoning": "Query about current clinical trials requires up-to-date information about ongoing research.",
                "alternative_categories": [{"category": "KNOWLEDGE_GRAPH", "confidence": 0.11}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["clinical trials", "mass spectrometry", "disease detection"],
                    "relationships": ["using for"],
                    "techniques": ["mass spectrometry"]
                },
                "temporal_signals": {
                    "keywords": ["current"],
                    "patterns": ["current clinical trials"],
                    "years": []
                }
            }
        },
        {
            "query": "New developments in AI-powered metabolomics analysis platforms in 2024",
            "classification": {
                "category": "REAL_TIME",
                "confidence": 0.94,
                "reasoning": "Query specifically asks for new developments in 2024, clearly requiring current information.",
                "alternative_categories": [{"category": "GENERAL", "confidence": 0.06}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["AI", "metabolomics", "analysis platforms"],
                    "relationships": ["developments in"],
                    "techniques": ["metabolomics"]
                },
                "temporal_signals": {
                    "keywords": ["new", "developments", "2024"],
                    "patterns": ["new developments", "in 2024"],
                    "years": ["2024"]
                }
            }
        },
        {
            "query": "What companies just announced metabolomics biomarker partnerships?",
            "classification": {
                "category": "REAL_TIME",
                "confidence": 0.92,
                "reasoning": "Query uses 'just announced' indicating very recent developments, requiring real-time information.",
                "alternative_categories": [{"category": "GENERAL", "confidence": 0.08}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["companies", "metabolomics", "biomarker", "partnerships"],
                    "relationships": ["partnerships"],
                    "techniques": ["metabolomics"]
                },
                "temporal_signals": {
                    "keywords": ["just announced"],
                    "patterns": ["just announced"],
                    "years": []
                }
            }
        }
    ]
    
    GENERAL_EXAMPLES = [
        {
            "query": "What is metabolomics and how does it work?",
            "classification": {
                "category": "GENERAL",
                "confidence": 0.88,
                "reasoning": "Basic definitional query that can be handled by either system, with slight preference for general knowledge.",
                "alternative_categories": [{"category": "KNOWLEDGE_GRAPH", "confidence": 0.12}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["metabolomics"],
                    "relationships": ["how does it work"],
                    "techniques": ["metabolomics"]
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        },
        {
            "query": "Explain the basics of LC-MS analysis for beginners",
            "classification": {
                "category": "GENERAL",
                "confidence": 0.85,
                "reasoning": "Educational query asking for basic explanation, suitable for flexible routing to either system.",
                "alternative_categories": [{"category": "KNOWLEDGE_GRAPH", "confidence": 0.15}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["LC-MS", "analysis"],
                    "relationships": [],
                    "techniques": ["LC-MS"]
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        },
        {
            "query": "How to interpret NMR spectra in metabolomics studies",
            "classification": {
                "category": "GENERAL",
                "confidence": 0.82,
                "reasoning": "Methodological query about general techniques, can be handled flexibly by either system.",
                "alternative_categories": [{"category": "KNOWLEDGE_GRAPH", "confidence": 0.18}],
                "uncertainty_indicators": ["methodological query could benefit from specific examples"],
                "biomedical_signals": {
                    "entities": ["NMR spectra", "metabolomics studies"],
                    "relationships": ["interpret in"],
                    "techniques": ["NMR", "metabolomics"]
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        },
        {
            "query": "What are the main applications of metabolomics in healthcare?",
            "classification": {
                "category": "GENERAL",
                "confidence": 0.86,
                "reasoning": "Broad overview query about applications, suitable for general knowledge with flexible routing.",
                "alternative_categories": [{"category": "KNOWLEDGE_GRAPH", "confidence": 0.14}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["metabolomics", "healthcare"],
                    "relationships": ["applications in"],
                    "techniques": ["metabolomics"]
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        },
        {
            "query": "Define biomarker and its role in personalized medicine",
            "classification": {
                "category": "GENERAL",
                "confidence": 0.87,
                "reasoning": "Definitional query with broad context, appropriate for flexible routing to either system.",
                "alternative_categories": [{"category": "KNOWLEDGE_GRAPH", "confidence": 0.13}],
                "uncertainty_indicators": [],
                "biomedical_signals": {
                    "entities": ["biomarker", "personalized medicine"],
                    "relationships": ["role in"],
                    "techniques": []
                },
                "temporal_signals": {
                    "keywords": [],
                    "patterns": [],
                    "years": []
                }
            }
        }
    ]
    
    # ============================================================================
    # FALLBACK CLASSIFICATION PROMPT (Performance-Optimized)
    # ============================================================================
    
    FALLBACK_CLASSIFICATION_PROMPT = """Classify this metabolomics query into one category:

**CATEGORIES:**
- KNOWLEDGE_GRAPH: relationships, pathways, mechanisms, established connections
- REAL_TIME: latest, recent, 2024+, news, FDA approvals, current trials  
- GENERAL: basic definitions, explanations, how-to questions

**RULES:**
- Recent/latest/2024+ = REAL_TIME
- Relationships/pathways/mechanisms = KNOWLEDGE_GRAPH  
- What is/explain/define = GENERAL

**QUERY:** {query_text}

**RESPONSE (JSON only):**
{{"category": "CATEGORY", "confidence": 0.8, "reasoning": "brief explanation"}}"""

    # ============================================================================
    # CONFIDENCE VALIDATION PROMPT
    # ============================================================================
    
    CONFIDENCE_VALIDATION_PROMPT = """Review this query classification for the Clinical Metabolomics Oracle system and assess its accuracy.

**ORIGINAL QUERY:** {query_text}

**PROPOSED CLASSIFICATION:**
- Category: {predicted_category}
- Confidence: {predicted_confidence}
- Reasoning: {predicted_reasoning}

**VALIDATION CRITERIA:**
1. **Temporal Indicators** - Does the query ask for recent, latest, current, or time-specific information?
2. **Relationship Patterns** - Does the query ask about connections, pathways, mechanisms, or associations?
3. **General Knowledge** - Is this a basic definitional or educational query?

**BIOMEDICAL CONTEXT:**
- KNOWLEDGE_GRAPH best for: established metabolic pathways, biomarker relationships, drug mechanisms
- REAL_TIME best for: recent research, clinical trial updates, regulatory news, emerging technologies
- GENERAL best for: definitions, basic explanations, methodology overviews

**VALIDATION TASK:**
Provide a validation score (0.0-1.0) where:
- 1.0 = Classification is completely correct
- 0.8-0.9 = Classification is mostly correct with minor issues
- 0.6-0.7 = Classification is partially correct but could be improved
- 0.4-0.5 = Classification has significant issues
- 0.0-0.3 = Classification is incorrect

**OUTPUT (JSON only):**
{{
  "validation_score": 0.85,
  "is_correct": true,
  "issues_identified": ["list of any issues found"],
  "suggested_improvements": ["suggestions if validation_score < 0.8"],
  "confidence_adjustment": 0.02,
  "final_reasoning": "updated reasoning if needed"
}}"""

    # ============================================================================
    # PROMPT CONSTRUCTION METHODS
    # ============================================================================
    
    @classmethod
    def build_primary_prompt(cls, query_text: str, include_examples: bool = False) -> str:
        """
        Build the primary classification prompt with optional examples.
        
        Args:
            query_text: The query to classify
            include_examples: Whether to include few-shot examples
            
        Returns:
            Complete prompt string ready for LLM
        """
        base_prompt = cls.PRIMARY_CLASSIFICATION_PROMPT.format(query_text=query_text)
        
        if include_examples:
            # Add few-shot examples for better accuracy (at cost of tokens)
            examples_section = "\n\n**EXAMPLES:**\n\n"
            
            # Include 1 example from each category
            examples_section += "**KNOWLEDGE_GRAPH Example:**\n"
            examples_section += f"Query: {cls.KNOWLEDGE_GRAPH_EXAMPLES[0]['query']}\n"
            examples_section += f"Classification: {json.dumps(cls.KNOWLEDGE_GRAPH_EXAMPLES[0]['classification'], indent=2)}\n\n"
            
            examples_section += "**REAL_TIME Example:**\n"
            examples_section += f"Query: {cls.REAL_TIME_EXAMPLES[0]['query']}\n"
            examples_section += f"Classification: {json.dumps(cls.REAL_TIME_EXAMPLES[0]['classification'], indent=2)}\n\n"
            
            examples_section += "**GENERAL Example:**\n"
            examples_section += f"Query: {cls.GENERAL_EXAMPLES[0]['query']}\n"
            examples_section += f"Classification: {json.dumps(cls.GENERAL_EXAMPLES[0]['classification'], indent=2)}\n\n"
            
            # Insert examples before the query to classify
            base_prompt = base_prompt.replace(
                "**QUERY TO CLASSIFY:**",
                examples_section + "**QUERY TO CLASSIFY:**"
            )
        
        return base_prompt
    
    @classmethod
    def build_fallback_prompt(cls, query_text: str) -> str:
        """Build the performance-optimized fallback prompt."""
        return cls.FALLBACK_CLASSIFICATION_PROMPT.format(query_text=query_text)
    
    @classmethod
    def build_validation_prompt(cls, 
                              query_text: str,
                              predicted_category: str,
                              predicted_confidence: float,
                              predicted_reasoning: str) -> str:
        """Build the confidence validation prompt."""
        return cls.CONFIDENCE_VALIDATION_PROMPT.format(
            query_text=query_text,
            predicted_category=predicted_category,
            predicted_confidence=predicted_confidence,
            predicted_reasoning=predicted_reasoning
        )
    
    @classmethod
    def get_few_shot_examples(cls, category: ClassificationCategory, count: int = 5) -> List[Dict]:
        """
        Get few-shot examples for a specific category.
        
        Args:
            category: The classification category
            count: Number of examples to return
            
        Returns:
            List of example dictionaries
        """
        if category == ClassificationCategory.KNOWLEDGE_GRAPH:
            return cls.KNOWLEDGE_GRAPH_EXAMPLES[:count]
        elif category == ClassificationCategory.REAL_TIME:
            return cls.REAL_TIME_EXAMPLES[:count]
        elif category == ClassificationCategory.GENERAL:
            return cls.GENERAL_EXAMPLES[:count]
        else:
            return []
    
    @classmethod
    def estimate_token_usage(cls, query_text: str, include_examples: bool = False) -> Dict[str, int]:
        """
        Estimate token usage for different prompt configurations.
        
        Args:
            query_text: The query to classify
            include_examples: Whether examples are included
            
        Returns:
            Dict with token usage estimates
        """
        # Rough token estimation (1 token ≈ 4 characters for English)
        base_prompt = cls.build_primary_prompt(query_text, include_examples)
        base_tokens = len(base_prompt) // 4
        
        fallback_prompt = cls.build_fallback_prompt(query_text)
        fallback_tokens = len(fallback_prompt) // 4
        
        return {
            "primary_prompt_tokens": base_tokens,
            "fallback_prompt_tokens": fallback_tokens,
            "examples_overhead": (base_tokens - len(cls.PRIMARY_CLASSIFICATION_PROMPT.format(query_text=query_text)) // 4) if include_examples else 0,
            "estimated_response_tokens": 150  # Typical JSON response
        }


# ============================================================================
# JSON SCHEMA SPECIFICATION
# ============================================================================

CLASSIFICATION_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["KNOWLEDGE_GRAPH", "REAL_TIME", "GENERAL"],
            "description": "Primary classification category"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score for the classification"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the classification decision"
        },
        "alternative_categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["KNOWLEDGE_GRAPH", "REAL_TIME", "GENERAL"]
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["category", "confidence"]
            },
            "description": "Alternative classification options with confidence scores"
        },
        "uncertainty_indicators": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Factors that increase classification uncertainty"
        },
        "biomedical_signals": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Biomedical entities detected in the query"
                },
                "relationships": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Relationship patterns detected"
                },
                "techniques": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Analytical techniques mentioned"
                }
            },
            "required": ["entities", "relationships", "techniques"]
        },
        "temporal_signals": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Temporal keywords detected"
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Temporal patterns detected"
                },
                "years": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific years mentioned"
                }
            },
            "required": ["keywords", "patterns", "years"]
        }
    },
    "required": ["category", "confidence", "reasoning", "alternative_categories", 
                "uncertainty_indicators", "biomedical_signals", "temporal_signals"]
}


# ============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# ============================================================================

class LLMClassificationUsage:
    """
    Examples and documentation for using the LLM classification prompts.
    """
    
    @staticmethod
    def example_basic_usage():
        """Example of basic prompt usage."""
        query = "What is the relationship between glucose metabolism and insulin signaling?"
        
        # Build primary prompt
        prompt = LLMClassificationPrompts.build_primary_prompt(query)
        
        # Estimate tokens
        token_usage = LLMClassificationPrompts.estimate_token_usage(query)
        
        print(f"Prompt ready for LLM API call")
        print(f"Estimated tokens: {token_usage['primary_prompt_tokens']}")
        return prompt
    
    @staticmethod
    def example_with_examples():
        """Example of using prompt with few-shot examples."""
        query = "Latest clinical trials for metabolomics biomarkers in 2024"
        
        # Build prompt with examples for higher accuracy
        prompt = LLMClassificationPrompts.build_primary_prompt(query, include_examples=True)
        
        return prompt
    
    @staticmethod
    def example_fallback_usage():
        """Example of using performance-optimized fallback prompt."""
        query = "How does mass spectrometry work in metabolomics?"
        
        # Use fallback prompt for speed-critical scenarios
        prompt = LLMClassificationPrompts.build_fallback_prompt(query)
        
        return prompt
    
    @staticmethod
    def example_validation():
        """Example of validating a classification result."""
        query = "Recent advances in metabolomics technology"
        predicted_category = "REAL_TIME"
        predicted_confidence = 0.85
        predicted_reasoning = "Query asks for recent advances, indicating temporal focus"
        
        validation_prompt = LLMClassificationPrompts.build_validation_prompt(
            query, predicted_category, predicted_confidence, predicted_reasoning
        )
        
        return validation_prompt


# ============================================================================
# INTEGRATION GUIDELINES
# ============================================================================

INTEGRATION_GUIDELINES = """
INTEGRATION WITH EXISTING INFRASTRUCTURE:

1. **ClassificationResult Compatibility:**
   - The JSON output schema matches existing ClassificationResult dataclass
   - biomedical_signals maps to existing keyword detection
   - temporal_signals maps to existing temporal analysis
   - confidence scores are compatible with existing thresholds

2. **Performance Considerations:**
   - Primary prompt: ~400-600 tokens, target response <2s
   - Fallback prompt: ~100-200 tokens, target response <1s
   - Use fallback for latency-critical scenarios
   - Cache common classifications to reduce API calls

3. **Routing Integration:**
   - KNOWLEDGE_GRAPH → route to LightRAG system
   - REAL_TIME → route to Perplexity API
   - GENERAL → flexible routing (either system)
   - Use alternative_categories for hybrid routing decisions

4. **Confidence Thresholds:**
   - >0.8: High confidence, direct routing
   - 0.6-0.8: Medium confidence, consider alternatives
   - <0.6: Low confidence, use fallback strategies

5. **Error Handling:**
   - Parse JSON response with error handling
   - Fall back to keyword-based classification if LLM fails
   - Log classification results for monitoring and improvement

6. **Cost Optimization:**
   - Use fallback prompt for repeated similar queries
   - Cache classifications for identical queries
   - Monitor token usage and optimize prompt length
   - Consider fine-tuned smaller models for cost reduction

7. **Monitoring and Validation:**
   - Track classification accuracy against user behavior
   - Use validation prompt for uncertain cases
   - Log confidence distributions for threshold tuning
   - A/B test different prompt variants
"""