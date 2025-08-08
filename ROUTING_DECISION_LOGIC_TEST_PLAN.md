# Comprehensive Test Plan for Routing Decision Logic

## Overview

This document provides a detailed test plan for validating the routing decision logic in the Clinical Metabolomics Oracle system. The test plan covers all core routing decisions (LIGHTRAG, PERPLEXITY, EITHER, HYBRID), confidence threshold validation, uncertainty detection, performance requirements, and integration testing.

**Performance Targets:**
- Total routing time: < 50ms
- Analysis time: < 30ms  
- Classification response: < 2 seconds
- Routing accuracy: >90%

## 1. Core Routing Decision Testing

### 1.1 LIGHTRAG Routing Tests

**Test Category:** Knowledge Graph Queries
**Expected Confidence:** High (>0.8)

#### Test Cases:

**LR-001: Biomedical Relationships**
```python
test_cases = [
    {
        "query": "What is the relationship between glucose metabolism and insulin resistance?",
        "expected_route": "LIGHTRAG",
        "expected_confidence": (0.85, 0.95),
        "confidence_factors": ["relationship keywords", "biomedical entities", "mechanism focus"],
        "temporal_indicators": False,
        "ambiguity_score": < 0.3
    },
    {
        "query": "How do metabolic pathways interact in diabetes?",
        "expected_route": "LIGHTRAG", 
        "expected_confidence": (0.8, 0.9),
        "confidence_factors": ["pathway keywords", "disease entity", "interaction focus"],
        "temporal_indicators": False,
        "ambiguity_score": < 0.3
    }
]
```

**LR-002: Pathway and Mechanism Queries**
```python
test_cases = [
    {
        "query": "Explain the glycolysis pathway in cancer cells",
        "expected_route": "LIGHTRAG",
        "expected_confidence": (0.8, 0.92),
        "confidence_factors": ["pathway keyword", "cellular context", "disease context"],
        "biomedical_entities": >= 3
    },
    {
        "query": "Mechanism of action for metformin in glucose homeostasis",
        "expected_route": "LIGHTRAG",
        "expected_confidence": (0.82, 0.94),
        "confidence_factors": ["mechanism keyword", "drug entity", "biological process"],
        "biomedical_entities": >= 2
    }
]
```

**LR-003: Biomarker and Clinical Study Queries**
```python
test_cases = [
    {
        "query": "Metabolomic biomarkers for early detection of Alzheimer's disease",
        "expected_route": "LIGHTRAG",
        "expected_confidence": (0.78, 0.88),
        "confidence_factors": ["biomarker keywords", "analytical method", "disease entity"],
        "domain_specificity": "high"
    }
]
```

### 1.2 PERPLEXITY Routing Tests

**Test Category:** Real-time/Temporal Queries
**Expected Confidence:** High (>0.8)

#### Test Cases:

**PR-001: Temporal Indicators**
```python
test_cases = [
    {
        "query": "What are the latest developments in metabolomics research 2025?",
        "expected_route": "PERPLEXITY",
        "expected_confidence": (0.85, 0.95),
        "confidence_factors": ["latest keyword", "year indicator", "research focus"],
        "temporal_indicators": ["latest", "2025"],
        "ambiguity_score": < 0.2
    },
    {
        "query": "Recent breakthroughs in LC-MS technology this year",
        "expected_route": "PERPLEXITY", 
        "expected_confidence": (0.8, 0.92),
        "confidence_factors": ["recent keyword", "this year", "technology focus"],
        "temporal_indicators": ["recent", "this year"]
    }
]
```

**PR-002: Current Events and News**
```python
test_cases = [
    {
        "query": "Current clinical trials for metabolomic biomarkers",
        "expected_route": "PERPLEXITY",
        "expected_confidence": (0.82, 0.9),
        "confidence_factors": ["current keyword", "ongoing trials"],
        "temporal_indicators": ["current"]
    },
    {
        "query": "Today's advances in personalized metabolomics",
        "expected_route": "PERPLEXITY",
        "expected_confidence": (0.88, 0.96),
        "confidence_factors": ["today keyword", "advances"],
        "temporal_indicators": ["today"]
    }
]
```

### 1.3 EITHER Routing Tests

**Test Category:** General/Flexible Queries
**Expected Confidence:** Medium (0.5-0.7)

#### Test Cases:

**ER-001: Basic Definitions**
```python
test_cases = [
    {
        "query": "What is metabolomics?",
        "expected_route": "EITHER",
        "expected_confidence": (0.5, 0.7),
        "confidence_factors": ["definition request", "basic concept"],
        "flexibility_score": > 0.7,
        "ambiguity_score": 0.4-0.6
    },
    {
        "query": "Define mass spectrometry",
        "expected_route": "EITHER",
        "expected_confidence": (0.52, 0.68),
        "confidence_factors": ["define keyword", "analytical method"],
        "biomedical_entities": 1
    }
]
```

**ER-002: General Inquiries**
```python
test_cases = [
    {
        "query": "How does metabolomics help in healthcare?",
        "expected_route": "EITHER",
        "expected_confidence": (0.45, 0.65),
        "confidence_factors": ["general inquiry", "application focus"],
        "domain_specificity": "medium"
    }
]
```

### 1.4 HYBRID Routing Tests

**Test Category:** Complex Multi-part Queries
**Expected Confidence:** Variable (0.6-0.85)

#### Test Cases:

**HR-001: Multi-faceted Queries**
```python
test_cases = [
    {
        "query": "What are the latest biomarker discoveries and how do they relate to metabolic pathways in diabetes?",
        "expected_route": "HYBRID",
        "expected_confidence": (0.65, 0.8),
        "confidence_factors": ["temporal + relationship", "multiple domains"],
        "temporal_indicators": ["latest"],
        "biomedical_entities": >= 3,
        "query_complexity": "high"
    },
    {
        "query": "Current metabolomic approaches for understanding insulin signaling mechanisms",
        "expected_route": "HYBRID",
        "expected_confidence": (0.68, 0.82),
        "confidence_factors": ["current + mechanism", "analytical + biological"],
        "temporal_indicators": ["current"],
        "mechanism_keywords": True
    }
]
```

## 2. Confidence Threshold Validation Tests

### 2.1 Threshold Boundary Testing

#### Test Cases:

**CT-001: High Confidence Threshold (0.8)**
```python
def test_high_confidence_threshold():
    """Test routing decisions at high confidence threshold"""
    test_cases = [
        {
            "confidence_score": 0.85,
            "expected_action": "direct_routing",
            "fallback_required": False,
            "monitoring_level": "standard"
        },
        {
            "confidence_score": 0.79, # Just below threshold
            "expected_action": "route_with_monitoring", 
            "fallback_required": False,
            "monitoring_level": "enhanced"
        }
    ]
```

**CT-002: Medium Confidence Threshold (0.6)**
```python
def test_medium_confidence_threshold():
    """Test routing with monitoring at medium confidence"""
    test_cases = [
        {
            "confidence_score": 0.65,
            "expected_action": "route_with_monitoring",
            "fallback_required": False,
            "monitoring_level": "enhanced"
        },
        {
            "confidence_score": 0.58, # Just below threshold
            "expected_action": "consider_fallback",
            "fallback_required": True,
            "monitoring_level": "intensive"
        }
    ]
```

**CT-003: Low Confidence Threshold (0.5)**
```python
def test_low_confidence_threshold():
    """Test fallback strategy activation"""
    test_cases = [
        {
            "confidence_score": 0.52,
            "expected_action": "use_fallback_strategies",
            "fallback_required": True,
            "uncertainty_handling": "active"
        },
        {
            "confidence_score": 0.48, # Just below threshold
            "expected_action": "use_fallback_strategies",
            "fallback_required": True,
            "uncertainty_handling": "intensive"
        }
    ]
```

**CT-004: Fallback Threshold (0.2)**
```python
def test_fallback_threshold():
    """Test fallback routing activation"""
    test_cases = [
        {
            "confidence_score": 0.25,
            "expected_action": "use_fallback_routing",
            "fallback_required": True,
            "fallback_strategy": "conservative"
        },
        {
            "confidence_score": 0.15, # Below threshold
            "expected_action": "use_fallback_routing",
            "fallback_required": True,
            "fallback_strategy": "emergency"
        }
    ]
```

### 2.2 Confidence Threshold Adaptation Tests

```python
def test_dynamic_threshold_adjustment():
    """Test adaptive threshold adjustment based on performance"""
    scenarios = [
        {
            "scenario": "high_accuracy_period",
            "recent_accuracy": 0.95,
            "threshold_adjustment": -0.05,  # Lower thresholds
            "confidence_boost": True
        },
        {
            "scenario": "low_accuracy_period", 
            "recent_accuracy": 0.85,
            "threshold_adjustment": +0.05,  # Raise thresholds
            "confidence_penalty": True
        }
    ]
```

## 3. Uncertainty Detection and Handling Tests

### 3.1 Uncertainty Type Detection

#### Test Cases:

**UT-001: Low Confidence Uncertainty**
```python
test_cases = [
    {
        "query": "Something about metabolism maybe?",
        "expected_uncertainty": "LOW_CONFIDENCE",
        "confidence_score": < 0.3,
        "handling_strategy": "CONSERVATIVE_CLASSIFICATION",
        "fallback_level": "KEYWORD_BASED_ONLY"
    }
]
```

**UT-002: High Ambiguity**
```python
test_cases = [
    {
        "query": "MS analysis results interpretation",  # MS = Mass Spec or Multiple Sclerosis?
        "expected_uncertainty": "HIGH_AMBIGUITY",
        "ambiguity_score": > 0.7,
        "alternative_interpretations": >= 2,
        "handling_strategy": "UNCERTAINTY_CLARIFICATION"
    }
]
```

**UT-003: High Conflict**
```python
test_cases = [
    {
        "query": "Latest established metabolic pathways",  # Temporal + Knowledge conflict
        "expected_uncertainty": "HIGH_CONFLICT",
        "conflict_score": > 0.6,
        "conflicting_signals": ["temporal", "knowledge_graph"],
        "handling_strategy": "HYBRID_CONSENSUS"
    }
]
```

**UT-004: Weak Evidence**
```python
test_cases = [
    {
        "query": "Research stuff questions",
        "expected_uncertainty": "WEAK_EVIDENCE",
        "evidence_strength": < 0.3,
        "biomedical_entities": 0,
        "handling_strategy": "CONFIDENCE_BOOSTING"
    }
]
```

### 3.2 Uncertainty Handling Strategy Tests

```python
class UncertaintyHandlingTests:
    
    def test_uncertainty_clarification_strategy(self):
        """Test clarification strategy for ambiguous queries"""
        query = "NMR spectroscopy applications"  # Could be analytical method or specific application
        result = uncertainty_handler.handle_uncertainty(query, UncertaintyType.HIGH_AMBIGUITY)
        
        assert result.strategy == UncertaintyStrategy.UNCERTAINTY_CLARIFICATION
        assert len(result.clarification_questions) >= 2
        assert result.confidence_improvement > 0.2
    
    def test_hybrid_consensus_strategy(self):
        """Test consensus strategy for conflicting signals"""
        query = "Recent metabolomic pathway analysis"
        result = uncertainty_handler.handle_uncertainty(query, UncertaintyType.HIGH_CONFLICT)
        
        assert result.strategy == UncertaintyStrategy.HYBRID_CONSENSUS
        assert len(result.consensus_approaches) >= 2
        assert result.consensus_agreement_score > 0.6
    
    def test_confidence_boosting_strategy(self):
        """Test confidence boosting for weak evidence"""
        query = "metabolism research"
        result = uncertainty_handler.handle_uncertainty(query, UncertaintyType.WEAK_EVIDENCE)
        
        assert result.strategy == UncertaintyStrategy.CONFIDENCE_BOOSTING
        assert result.confidence_improvement > 0.1
        assert result.boosting_factors_applied >= 1
```

## 4. Performance Validation Tests

### 4.1 Response Time Tests

```python
class PerformanceTests:
    
    @pytest.mark.performance
    def test_routing_time_under_50ms(self):
        """Validate total routing time < 50ms"""
        queries = generate_test_queries(100)  # Mix of query types
        
        times = []
        for query in queries:
            start_time = time.perf_counter()
            result = router.route_query(query)
            end_time = time.perf_counter()
            
            routing_time_ms = (end_time - start_time) * 1000
            times.append(routing_time_ms)
            
            assert routing_time_ms < 50, f"Routing time {routing_time_ms}ms exceeds 50ms limit"
        
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
        
        assert avg_time < 30, f"Average routing time {avg_time}ms exceeds 30ms target"
        assert p95_time < 50, f"95th percentile time {p95_time}ms exceeds 50ms limit"
    
    @pytest.mark.performance
    def test_analysis_time_under_30ms(self):
        """Validate analysis time < 30ms"""
        queries = generate_test_queries(50)
        
        for query in queries:
            start_time = time.perf_counter()
            analysis = router.analyze_query(query)
            end_time = time.perf_counter()
            
            analysis_time_ms = (end_time - start_time) * 1000
            assert analysis_time_ms < 30, f"Analysis time {analysis_time_ms}ms exceeds 30ms limit"
    
    @pytest.mark.performance  
    def test_classification_response_under_2_seconds(self):
        """Validate classification response < 2 seconds"""
        queries = generate_complex_queries(25)  # More complex queries
        
        for query in queries:
            start_time = time.perf_counter()
            classification = classifier.classify_query(query)
            end_time = time.perf_counter()
            
            classification_time_ms = (end_time - start_time) * 1000
            assert classification_time_ms < 2000, f"Classification time {classification_time_ms}ms exceeds 2s limit"
```

### 4.2 Throughput Tests

```python
def test_concurrent_routing_performance():
    """Test routing performance under concurrent load"""
    queries = generate_test_queries(200)
    
    def route_query_timed(query):
        start_time = time.perf_counter()
        result = router.route_query(query)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000, result
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(route_query_timed, query) for query in queries]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    times = [result[0] for result in results]
    avg_concurrent_time = statistics.mean(times)
    
    assert avg_concurrent_time < 60, f"Average concurrent routing time {avg_concurrent_time}ms too high"
    assert all(time < 100 for time in times), "Some concurrent requests exceeded 100ms"
```

### 4.3 Memory and Resource Tests

```python
def test_memory_usage_stability():
    """Test memory usage remains stable under load"""
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    queries = generate_test_queries(500)
    for query in queries:
        router.route_query(query)
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 50, f"Memory usage increased by {memory_increase}MB (limit: 50MB)"
```

## 5. Integration Testing

### 5.1 Component Integration Tests

```python
class IntegrationTests:
    
    def test_cascade_system_integration(self):
        """Test integration between routing and cascade system"""
        test_cases = [
            {
                "query": "uncertain metabolomics question",
                "expected_cascade_activation": True,
                "expected_cascade_path": "SKIP_LIGHTRAG",
                "expected_final_route": "PERPLEXITY"
            }
        ]
        
        for case in test_cases:
            result = cascade_system.process_query_with_uncertainty_cascade(case["query"])
            
            assert result.cascade_path_used.value == case["expected_cascade_path"]
            assert result.success == True
            assert result.total_cascade_time_ms < 200
    
    def test_threshold_router_integration(self):
        """Test integration with threshold-based routing"""
        queries_with_expected_thresholds = [
            ("high confidence biomarker query", "HIGH"),
            ("medium confidence pathway question", "MEDIUM"), 
            ("low confidence general inquiry", "LOW"),
            ("very uncertain metabolomics topic", "VERY_LOW")
        ]
        
        for query, expected_level in queries_with_expected_thresholds:
            prediction, uncertainty_analysis = threshold_router.route_with_threshold_awareness(
                query, get_mock_confidence_metrics(query)
            )
            
            confidence_level = prediction.metadata.get("confidence_level")
            assert confidence_level == expected_level
    
    def test_fallback_system_integration(self):
        """Test integration with comprehensive fallback system"""
        failing_query = "completely unrecognizable query content"
        
        result = fallback_integrator.process_with_threshold_awareness(
            failing_query, get_very_low_confidence_metrics()
        )
        
        assert result.success == True  # Fallback should always succeed
        assert result.routing_prediction is not None
        assert result.fallback_level_used != None
```

### 5.2 End-to-End Workflow Tests

```python
def test_complete_routing_workflow():
    """Test complete routing workflow from query to final decision"""
    
    workflow_test_cases = [
        {
            "query": "What metabolomic biomarkers are associated with diabetes pathogenesis?",
            "expected_steps": [
                "query_preprocessing",
                "confidence_calculation", 
                "uncertainty_analysis",
                "threshold_evaluation",
                "routing_decision",
                "confidence_validation"
            ],
            "expected_final_route": "LIGHTRAG",
            "expected_confidence": (0.8, 0.95)
        }
    ]
    
    for case in workflow_test_cases:
        with workflow_tracker() as tracker:
            result = complete_routing_system.route_query(case["query"])
            
        executed_steps = tracker.get_executed_steps()
        assert all(step in executed_steps for step in case["expected_steps"])
        assert result.routing_decision.value == case["expected_final_route"]
        assert case["expected_confidence"][0] <= result.confidence <= case["expected_confidence"][1]
```

## 6. Edge Cases and Error Handling Tests

### 6.1 Edge Case Tests

```python
class EdgeCaseTests:
    
    def test_empty_query_handling(self):
        """Test handling of empty or whitespace queries"""
        edge_queries = ["", "   ", "\n\t", None]
        
        for query in edge_queries:
            result = router.route_query(query)
            assert result.routing_decision == RoutingDecision.EITHER  # Safe default
            assert result.confidence < 0.3  # Low confidence for empty queries
    
    def test_very_long_query_handling(self):
        """Test handling of extremely long queries"""
        long_query = "metabolomics " * 1000  # Very long query
        
        start_time = time.perf_counter()
        result = router.route_query(long_query)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        assert processing_time_ms < 100  # Should still be fast
        assert result.routing_decision != None
    
    def test_special_character_handling(self):
        """Test handling of queries with special characters"""
        special_queries = [
            "What is α-glucose metabolism?",
            "LC-MS/MS analysis (>95% purity)",
            "Metabolomics@clinical-research.org?",
            "β-oxidation pathway vs. γ-secretase"
        ]
        
        for query in special_queries:
            result = router.route_query(query)
            assert result.routing_decision != None
            assert result.confidence > 0.1  # Should have some confidence
    
    def test_multilingual_query_handling(self):
        """Test handling of non-English queries"""
        multilingual_queries = [
            "¿Qué es metabolómica?",  # Spanish
            "Qu'est-ce que la métabolomique?",  # French  
            "什么是代谢组学？"  # Chinese
        ]
        
        for query in multilingual_queries:
            result = router.route_query(query)
            # Should gracefully handle but likely route to EITHER with low confidence
            assert result.routing_decision == RoutingDecision.EITHER
            assert result.confidence < 0.5
```

### 6.2 Error Handling Tests

```python
def test_component_failure_resilience():
    """Test system resilience when components fail"""
    
    # Test with mock component failures
    with patch('lightrag_integration.research_categorizer.ResearchCategorizer.categorize_query') as mock_categorizer:
        mock_categorizer.side_effect = Exception("Categorizer failed")
        
        result = router.route_query("test metabolomics query")
        assert result.success == True  # Should fall back gracefully
        assert result.routing_decision != None
    
    # Test with confidence scorer failure
    with patch('lightrag_integration.comprehensive_confidence_scorer.HybridConfidenceScorer') as mock_scorer:
        mock_scorer.side_effect = Exception("Confidence scorer failed")
        
        result = router.route_query("test query")
        assert result.success == True
        assert result.confidence >= 0.1  # Should have fallback confidence
```

## 7. Accuracy Validation Tests

### 7.1 Routing Accuracy Tests

```python
class AccuracyTests:
    
    @pytest.mark.accuracy
    def test_routing_accuracy_target_90_percent(self):
        """Validate >90% routing accuracy requirement"""
        
        # Load ground truth test dataset
        test_dataset = load_routing_ground_truth_dataset()  # 500+ labeled examples
        
        correct_predictions = 0
        total_predictions = len(test_dataset)
        category_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for test_case in test_dataset:
            result = router.route_query(test_case.query)
            predicted_route = result.routing_decision
            expected_route = test_case.expected_routing
            
            category_accuracies[expected_route]['total'] += 1
            
            if predicted_route == expected_route:
                correct_predictions += 1
                category_accuracies[expected_route]['correct'] += 1
        
        overall_accuracy = correct_predictions / total_predictions
        assert overall_accuracy >= 0.90, f"Overall accuracy {overall_accuracy:.1%} below 90% requirement"
        
        # Test category-specific accuracy
        for category, stats in category_accuracies.items():
            category_accuracy = stats['correct'] / stats['total']
            assert category_accuracy >= 0.85, f"{category} accuracy {category_accuracy:.1%} below 85% minimum"
    
    def test_confidence_calibration_accuracy(self):
        """Test accuracy of confidence predictions"""
        test_cases = generate_confidence_test_cases(200)
        
        confidence_bins = defaultdict(list)  # Group by confidence ranges
        
        for test_case in test_cases:
            result = router.route_query(test_case.query)
            actual_correct = (result.routing_decision == test_case.expected_routing)
            
            confidence_bin = int(result.confidence * 10) / 10  # Round to nearest 0.1
            confidence_bins[confidence_bin].append(actual_correct)
        
        # Check calibration: confidence should match accuracy
        for confidence_level, correct_flags in confidence_bins.items():
            if len(correct_flags) >= 10:  # Sufficient sample size
                actual_accuracy = sum(correct_flags) / len(correct_flags)
                calibration_error = abs(confidence_level - actual_accuracy)
                assert calibration_error < 0.2, f"Poor calibration at confidence {confidence_level}: error {calibration_error}"
```

### 7.2 Domain-Specific Accuracy Tests

```python
def test_clinical_metabolomics_accuracy():
    """Test accuracy specifically for clinical metabolomics domain"""
    
    clinical_test_cases = [
        # Biomarker discovery queries
        ("Metabolomic biomarkers for early cancer detection", "LIGHTRAG", 0.85),
        ("Latest cancer biomarker validation studies 2025", "PERPLEXITY", 0.88), 
        
        # Analytical method queries
        ("LC-MS method optimization for lipidomics", "LIGHTRAG", 0.82),
        ("Current advances in LC-MS technology", "PERPLEXITY", 0.86),
        
        # Clinical application queries  
        ("Personalized medicine using metabolomics", "EITHER", 0.65),
        ("What is precision metabolomics?", "EITHER", 0.60)
    ]
    
    for query, expected_route, min_confidence in clinical_test_cases:
        result = router.route_query(query)
        assert result.routing_decision.value == expected_route
        assert result.confidence >= min_confidence
```

## 8. Stress Testing and Load Testing

### 8.1 High-Volume Stress Tests

```python
@pytest.mark.stress
def test_high_volume_routing_stability():
    """Test stability under high query volume"""
    
    num_queries = 10000
    batch_size = 100
    
    queries = generate_diverse_test_queries(num_queries)
    
    successful_routes = 0
    failed_routes = 0
    response_times = []
    
    for i in range(0, num_queries, batch_size):
        batch = queries[i:i+batch_size]
        
        batch_start_time = time.perf_counter()
        
        for query in batch:
            try:
                start_time = time.perf_counter()
                result = router.route_query(query)
                end_time = time.perf_counter()
                
                if result.success:
                    successful_routes += 1
                    response_times.append((end_time - start_time) * 1000)
                else:
                    failed_routes += 1
            except Exception:
                failed_routes += 1
        
        batch_time = time.perf_counter() - batch_start_time
        print(f"Batch {i//batch_size + 1}: {batch_time:.2f}s for {len(batch)} queries")
    
    success_rate = successful_routes / (successful_routes + failed_routes)
    avg_response_time = statistics.mean(response_times) if response_times else 0
    
    assert success_rate >= 0.99, f"Success rate {success_rate:.1%} below 99% requirement"
    assert avg_response_time < 60, f"Average response time {avg_response_time}ms too high under load"

@pytest.mark.stress
def test_concurrent_request_handling():
    """Test concurrent request handling capability"""
    
    num_concurrent_requests = 50
    queries = generate_test_queries(num_concurrent_requests)
    
    def route_query_with_timing(query):
        start_time = time.perf_counter()
        try:
            result = router.route_query(query)
            end_time = time.perf_counter()
            return {
                'success': result.success,
                'time_ms': (end_time - start_time) * 1000,
                'confidence': result.confidence
            }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'success': False,
                'time_ms': (end_time - start_time) * 1000,
                'error': str(e)
            }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        futures = [executor.submit(route_query_with_timing, query) for query in queries]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    successful_results = [r for r in results if r['success']]
    success_rate = len(successful_results) / len(results)
    
    assert success_rate >= 0.95, f"Concurrent success rate {success_rate:.1%} below 95%"
    
    if successful_results:
        avg_concurrent_time = statistics.mean([r['time_ms'] for r in successful_results])
        assert avg_concurrent_time < 100, f"Average concurrent time {avg_concurrent_time}ms too high"
```

## 9. Test Execution Framework

### 9.1 Test Configuration

```python
# test_config.py
TEST_CONFIG = {
    "performance": {
        "max_routing_time_ms": 50,
        "max_analysis_time_ms": 30,
        "max_classification_time_ms": 2000,
        "target_accuracy": 0.90,
        "min_category_accuracy": 0.85
    },
    "thresholds": {
        "high_confidence": 0.8,
        "medium_confidence": 0.6, 
        "low_confidence": 0.5,
        "fallback_threshold": 0.2
    },
    "stress_testing": {
        "max_concurrent_requests": 50,
        "high_volume_query_count": 10000,
        "max_memory_increase_mb": 50,
        "min_success_rate": 0.95
    }
}
```

### 9.2 Test Data Generation

```python
def generate_test_queries(count: int) -> List[str]:
    """Generate diverse test queries for testing"""
    
    query_templates = {
        "lightrag": [
            "What is the relationship between {entity1} and {entity2}?",
            "How does {pathway} affect {condition}?", 
            "Explain the mechanism of {process}",
            "{biomarker} in {disease} pathogenesis"
        ],
        "perplexity": [
            "Latest {research_area} developments {year}",
            "Current {technology} advances",
            "Recent {clinical_trial} results",
            "Today's {breakthrough} news"
        ],
        "either": [
            "What is {concept}?",
            "Define {term}",
            "How does {process} work?",
            "Explain {method}"
        ]
    }
    
    entities = ["glucose", "insulin", "metabolomics", "LC-MS", "diabetes", "biomarker"]
    pathways = ["glycolysis", "TCA cycle", "lipid metabolism", "amino acid metabolism"]
    # ... generate queries using templates
```

### 9.3 Test Reporting

```python
def generate_test_report(test_results: Dict[str, Any]) -> str:
    """Generate comprehensive test report"""
    
    report = f"""
# Routing Decision Logic Test Report

## Executive Summary
- Overall Accuracy: {test_results['accuracy']['overall']:.1%}
- Performance Target Met: {test_results['performance']['meets_targets']}
- Confidence Calibration: {test_results['confidence']['calibration_score']:.2f}

## Detailed Results

### Routing Accuracy by Category
{format_accuracy_table(test_results['accuracy']['by_category'])}

### Performance Metrics
- Average Routing Time: {test_results['performance']['avg_routing_time_ms']:.1f}ms
- 95th Percentile Time: {test_results['performance']['p95_time_ms']:.1f}ms
- Throughput: {test_results['performance']['throughput_qps']:.1f} QPS

### Confidence Threshold Validation
{format_confidence_results(test_results['confidence']['thresholds'])}

### Uncertainty Handling Results
{format_uncertainty_results(test_results['uncertainty'])}

## Recommendations
{generate_recommendations(test_results)}
"""
    return report
```

## 10. Continuous Testing Strategy

### 10.1 Automated Test Execution

```python
# pytest.ini configuration
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --disable-warnings
testpaths = tests
markers =
    accuracy: Accuracy validation tests
    performance: Performance requirement tests  
    stress: Stress and load testing
    integration: Component integration tests
    edge_cases: Edge case handling tests
```

### 10.2 Performance Regression Detection

```python
def test_performance_regression():
    """Detect performance regressions in routing system"""
    
    baseline_metrics = load_baseline_performance_metrics()
    current_metrics = measure_current_performance()
    
    for metric, baseline_value in baseline_metrics.items():
        current_value = current_metrics[metric]
        regression_threshold = 1.10  # 10% performance degradation threshold
        
        if current_value > baseline_value * regression_threshold:
            pytest.fail(f"Performance regression detected in {metric}: "
                       f"current {current_value} > baseline {baseline_value} * {regression_threshold}")
```

### 10.3 Test Data Quality Validation

```python
def validate_test_dataset_quality():
    """Validate quality and completeness of test dataset"""
    
    test_dataset = load_routing_ground_truth_dataset()
    
    # Check category distribution
    category_counts = Counter(case.expected_routing for case in test_dataset)
    min_category_size = min(category_counts.values())
    
    assert min_category_size >= 50, f"Insufficient test cases for some categories: {category_counts}"
    
    # Check query diversity  
    unique_queries = len(set(case.query for case in test_dataset))
    assert unique_queries == len(test_dataset), "Duplicate queries found in test dataset"
    
    # Validate confidence score ranges
    confidence_scores = [case.expected_confidence for case in test_dataset]
    assert all(0.0 <= score <= 1.0 for score in confidence_scores), "Invalid confidence scores"
```

This comprehensive test plan provides detailed test cases, validation criteria, and implementation guidance to achieve >90% routing decision accuracy while meeting all performance requirements. The test framework is designed to be maintainable, scalable, and capable of detecting regressions in routing performance and accuracy.