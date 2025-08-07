#!/usr/bin/env python3
"""
Advanced Test Data Utilities for Clinical Metabolomics Oracle Testing.

This module provides specialized utilities that work with the test_data_fixtures.py
to provide advanced test data management capabilities. It focuses on complex
data generation, validation, and lifecycle management that extends beyond basic
fixture functionality.

Key Components:
1. TestDataFactory: Advanced data generation with realistic content
2. DataValidationSuite: Comprehensive data integrity and format validation
3. MockDataGenerator: Dynamic mock data creation for complex scenarios
4. TestDataOrchestrator: Coordinated data setup for complex test scenarios
5. PerformanceDataManager: Data management for performance and load testing
6. ErrorTestDataProvider: Specialized data for error handling scenarios

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import json
import logging
import random
import sqlite3
import tempfile
import time
import uuid
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Generator
import hashlib
import shutil
from contextlib import contextmanager
from unittest.mock import Mock, AsyncMock, MagicMock

# Import our base fixtures
from .test_data_fixtures import TestDataManager, TestDataConfig, TestDataInfo, TEST_DATA_ROOT


# =====================================================================
# ADVANCED DATA STRUCTURES
# =====================================================================

@dataclass
class BiochemicalCompound:
    """Represents a biochemical compound for testing."""
    id: str
    name: str
    formula: str
    molecular_weight: float
    kegg_id: Optional[str] = None
    hmdb_id: Optional[str] = None
    pathway: Optional[str] = None
    biological_role: Optional[str] = None
    concentration_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "formula": self.formula,
            "molecular_weight": self.molecular_weight,
            "kegg_id": self.kegg_id,
            "hmdb_id": self.hmdb_id,
            "pathway": self.pathway,
            "biological_role": self.biological_role,
            "concentration_ranges": self.concentration_ranges
        }


@dataclass
class ClinicalStudyData:
    """Represents clinical study data for testing."""
    study_id: str
    title: str
    abstract: str
    methodology: str
    sample_size: int
    duration_months: int
    endpoints: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    statistical_methods: List[str]
    compounds_studied: List[str]
    results_summary: str
    
    def to_research_paper(self) -> str:
        """Convert to research paper format."""
        return f"""CLINICAL RESEARCH STUDY

Study ID: {self.study_id}
Title: {self.title}

ABSTRACT
{self.abstract}

INTRODUCTION
This clinical study investigated {', '.join(self.compounds_studied)} in a {self.sample_size}-participant cohort over {self.duration_months} months.

METHODS
{self.methodology}

Sample Size: {self.sample_size} participants
Duration: {self.duration_months} months
Statistical Methods: {', '.join(self.statistical_methods)}

Inclusion Criteria:
{chr(10).join(f"- {criterion}" for criterion in self.inclusion_criteria)}

Exclusion Criteria:
{chr(10).join(f"- {criterion}" for criterion in self.exclusion_criteria)}

Primary Endpoints:
{chr(10).join(f"- {endpoint}" for endpoint in self.endpoints)}

RESULTS
{self.results_summary}

CONCLUSIONS
The study provides evidence for the clinical significance of metabolomic biomarkers in the studied condition.

KEYWORDS: clinical metabolomics, biomarkers, {', '.join(self.compounds_studied[:3])}
"""


TestScenario = namedtuple('TestScenario', ['name', 'description', 'data_requirements', 'expected_outcomes'])


# =====================================================================
# TEST DATA FACTORY
# =====================================================================

class TestDataFactory:
    """Advanced factory for generating realistic test data."""
    
    # Metabolomics reference data
    COMMON_METABOLITES = [
        ("Glucose", "C6H12O6", 180.16, "C00031", "HMDB0000122"),
        ("Lactate", "C3H6O3", 90.08, "C00186", "HMDB0000190"),
        ("Pyruvate", "C3H4O3", 88.06, "C00022", "HMDB0000243"),
        ("Citrate", "C6H8O7", 192.12, "C00158", "HMDB0000094"),
        ("Succinate", "C4H6O5", 118.09, "C00042", "HMDB0000254"),
        ("Alanine", "C3H7NO2", 89.09, "C00041", "HMDB0000161"),
        ("Glycine", "C2H5NO2", 75.07, "C00037", "HMDB0000123"),
        ("Serine", "C3H7NO3", 105.09, "C00065", "HMDB0000187"),
    ]
    
    METABOLIC_PATHWAYS = [
        "Glycolysis", "TCA cycle", "Amino acid metabolism", 
        "Lipid metabolism", "Nucleotide metabolism", "Pentose phosphate pathway"
    ]
    
    BIOLOGICAL_ROLES = [
        "Energy metabolism intermediate", "Central metabolic intermediate",
        "Amino acid precursor", "Lipid biosynthesis intermediate", 
        "Neurotransmitter precursor", "Osmolyte"
    ]
    
    CLINICAL_CONDITIONS = [
        "Type 2 Diabetes", "Metabolic Syndrome", "Cardiovascular Disease",
        "Alzheimer's Disease", "Cancer", "Kidney Disease"
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize factory with optional random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
        self._compound_id_counter = 1
        self._study_id_counter = 1
        
    def generate_compound(self, 
                         name: Optional[str] = None,
                         include_concentrations: bool = True) -> BiochemicalCompound:
        """Generate a realistic biochemical compound."""
        if name is None:
            compound_data = random.choice(self.COMMON_METABOLITES)
            name, formula, mw, kegg_id, hmdb_id = compound_data
        else:
            # Generate synthetic data for custom names
            formula = self._generate_formula()
            mw = random.uniform(50.0, 500.0)
            kegg_id = f"C{random.randint(10000, 99999):05d}"
            hmdb_id = f"HMDB{random.randint(1000000, 9999999):07d}"
        
        compound_id = f"met_{self._compound_id_counter:03d}"
        self._compound_id_counter += 1
        
        concentration_ranges = {}
        if include_concentrations:
            # Generate realistic concentration ranges
            base_healthy = random.uniform(10.0, 200.0)
            healthy_range = {
                "min": round(base_healthy * 0.8, 2),
                "max": round(base_healthy * 1.2, 2),
                "unit": random.choice(["µM", "nM", "mM", "µg/mL"])
            }
            
            # Disease state typically shows altered levels
            disease_multiplier = random.choice([0.3, 0.5, 1.8, 2.5, 3.0])
            disease_base = base_healthy * disease_multiplier
            disease_range = {
                "min": round(disease_base * 0.7, 2),
                "max": round(disease_base * 1.4, 2),
                "unit": healthy_range["unit"]
            }
            
            concentration_ranges = {
                "plasma_healthy": healthy_range,
                "plasma_diseased": disease_range
            }
        
        return BiochemicalCompound(
            id=compound_id,
            name=name,
            formula=formula,
            molecular_weight=round(mw, 2),
            kegg_id=kegg_id,
            hmdb_id=hmdb_id,
            pathway=random.choice(self.METABOLIC_PATHWAYS),
            biological_role=random.choice(self.BIOLOGICAL_ROLES),
            concentration_ranges=concentration_ranges
        )
    
    def generate_compound_database(self, 
                                  count: int = 20,
                                  include_common: bool = True) -> Dict[str, Any]:
        """Generate a complete compound database."""
        compounds = []
        
        # Include common metabolites if requested
        if include_common and count >= len(self.COMMON_METABOLITES):
            for metabolite_data in self.COMMON_METABOLITES:
                compound = self.generate_compound(name=metabolite_data[0])
                compounds.append(compound.to_dict())
            count -= len(self.COMMON_METABOLITES)
        
        # Generate additional synthetic compounds
        for _ in range(count):
            compound = self.generate_compound()
            compounds.append(compound.to_dict())
        
        return {
            "metabolite_database": {
                "version": "1.0.0",
                "source": "test_data_factory",
                "generated_at": datetime.now().isoformat(),
                "compound_count": len(compounds),
                "metabolites": compounds
            }
        }
    
    def generate_clinical_study(self,
                              condition: Optional[str] = None,
                              compound_count: int = 5) -> ClinicalStudyData:
        """Generate realistic clinical study data."""
        study_id = f"STUDY_{self._study_id_counter:03d}"
        self._study_id_counter += 1
        
        condition = condition or random.choice(self.CLINICAL_CONDITIONS)
        compounds = [comp[0] for comp in random.sample(self.COMMON_METABOLITES, min(compound_count, len(self.COMMON_METABOLITES)))]
        
        sample_size = random.randint(50, 500)
        duration = random.randint(6, 36)
        
        title = f"Metabolomic Analysis of {condition}: A {duration}-Month Clinical Study"
        
        abstract = f"""This study investigated metabolomic profiles in {condition} patients compared to healthy controls. 
We analyzed plasma samples from {sample_size} participants using LC-MS/MS metabolomics. 
Key metabolites including {', '.join(compounds[:3])} showed significant alterations in disease state. 
Results provide insights into disease pathophysiology and potential biomarkers."""
        
        methodology = f"""Participants: {sample_size} {condition} patients and matched controls
Sample Collection: Fasting plasma samples at baseline and follow-up
Analytical Platform: LC-MS/MS using high-resolution mass spectrometry
Data Processing: Peak detection, alignment, and metabolite identification
Quality Control: Pooled samples, blank injections, internal standards"""
        
        endpoints = [
            "Primary: Metabolite concentration differences between groups",
            "Secondary: Correlation with clinical parameters",
            "Exploratory: Pathway enrichment analysis"
        ]
        
        inclusion = [
            f"Diagnosed {condition} according to standard criteria",
            "Age 18-75 years",
            "Stable medication for >3 months",
            "Able to provide informed consent"
        ]
        
        exclusion = [
            "Pregnancy or lactation", 
            "Severe comorbidities",
            "Recent medication changes",
            "Unable to fast for sample collection"
        ]
        
        statistics = [
            "t-tests for group comparisons",
            "Principal component analysis (PCA)",
            "Partial least squares discriminant analysis (PLS-DA)",
            "Pathway enrichment analysis",
            "False discovery rate correction"
        ]
        
        results = f"""Of {len(compounds)} metabolites analyzed, {random.randint(3, len(compounds))} showed significant differences (p < 0.05).
{compounds[0]} was elevated {random.uniform(1.5, 3.0):.1f}-fold in {condition} patients.
{compounds[1]} showed decreased levels ({random.uniform(0.3, 0.7):.1f}-fold reduction).
PCA analysis revealed distinct metabolic signatures between groups.
Pathway analysis identified {random.choice(self.METABOLIC_PATHWAYS)} as most significantly affected."""
        
        return ClinicalStudyData(
            study_id=study_id,
            title=title,
            abstract=abstract,
            methodology=methodology,
            sample_size=sample_size,
            duration_months=duration,
            endpoints=endpoints,
            inclusion_criteria=inclusion,
            exclusion_criteria=exclusion,
            statistical_methods=statistics,
            compounds_studied=compounds,
            results_summary=results
        )
    
    def _generate_formula(self) -> str:
        """Generate a realistic chemical formula."""
        c_count = random.randint(2, 20)
        h_count = random.randint(c_count, c_count * 4)
        
        # Optionally add heteroatoms
        elements = [f"C{c_count}", f"H{h_count}"]
        
        if random.random() < 0.7:  # 70% chance of oxygen
            o_count = random.randint(1, 8)
            elements.append(f"O{o_count}")
        
        if random.random() < 0.3:  # 30% chance of nitrogen
            n_count = random.randint(1, 4)
            elements.append(f"N{n_count}")
        
        if random.random() < 0.1:  # 10% chance of other elements
            other_elements = ["S", "P", "Cl", "F"]
            element = random.choice(other_elements)
            count = random.randint(1, 3)
            elements.append(f"{element}{count}")
        
        return "".join(elements)


# =====================================================================
# DATA VALIDATION SUITE
# =====================================================================

class DataValidationSuite:
    """Comprehensive validation for test data integrity."""
    
    def __init__(self):
        self.validation_results = []
        
    def validate_json_structure(self, 
                               data: Dict[str, Any], 
                               required_fields: List[str],
                               name: str = "data") -> bool:
        """Validate JSON data structure."""
        try:
            for field in required_fields:
                if field not in data:
                    self.validation_results.append({
                        "type": "structure_error",
                        "name": name,
                        "message": f"Missing required field: {field}"
                    })
                    return False
            
            self.validation_results.append({
                "type": "structure_success", 
                "name": name,
                "message": "All required fields present"
            })
            return True
            
        except Exception as e:
            self.validation_results.append({
                "type": "validation_error",
                "name": name, 
                "message": f"Validation failed: {e}"
            })
            return False
    
    def validate_metabolite_data(self, metabolite: Dict[str, Any]) -> bool:
        """Validate individual metabolite data structure."""
        required_fields = ["id", "name", "formula", "molecular_weight"]
        return self.validate_json_structure(metabolite, required_fields, "metabolite")
    
    def validate_database_schema(self, db_path: Path) -> bool:
        """Validate database schema and structure."""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Check if basic tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            if len(tables) == 0:
                self.validation_results.append({
                    "type": "schema_error",
                    "name": str(db_path),
                    "message": "No tables found in database"
                })
                return False
            
            self.validation_results.append({
                "type": "schema_success",
                "name": str(db_path),
                "message": f"Found {len(tables)} tables: {', '.join(tables)}"
            })
            return True
            
        except Exception as e:
            self.validation_results.append({
                "type": "schema_error",
                "name": str(db_path),
                "message": f"Database validation failed: {e}"
            })
            return False
    
    def validate_test_data_directory(self, base_path: Path) -> Dict[str, Any]:
        """Validate entire test data directory structure."""
        validation_summary = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "missing_directories": [],
            "errors": []
        }
        
        expected_dirs = [
            "pdfs/samples", "pdfs/templates", "pdfs/corrupted",
            "databases/schemas", "databases/samples", "databases/test_dbs",
            "mocks/biomedical_data", "mocks/api_responses", "mocks/state_data",
            "logs/templates", "logs/configs", "logs/samples",
            "temp/staging", "temp/processing", "temp/cleanup"
        ]
        
        # Check directory structure
        for expected_dir in expected_dirs:
            dir_path = base_path / expected_dir
            if not dir_path.exists():
                validation_summary["missing_directories"].append(expected_dir)
        
        # Validate files
        for file_path in base_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                validation_summary["total_files"] += 1
                
                try:
                    if file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json.load(f)
                    elif file_path.suffix == '.sql':
                        content = file_path.read_text(encoding='utf-8')
                        if len(content.strip()) == 0:
                            raise ValueError("Empty SQL file")
                    elif file_path.suffix in ['.txt', '.log']:
                        file_path.read_text(encoding='utf-8')
                    
                    validation_summary["valid_files"] += 1
                    
                except Exception as e:
                    validation_summary["invalid_files"] += 1
                    validation_summary["errors"].append({
                        "file": str(file_path.relative_to(base_path)),
                        "error": str(e)
                    })
        
        return validation_summary
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        errors = [r for r in self.validation_results if "error" in r["type"]]
        successes = [r for r in self.validation_results if "success" in r["type"]]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_validations": len(self.validation_results),
            "successful_validations": len(successes),
            "failed_validations": len(errors),
            "success_rate": len(successes) / len(self.validation_results) if self.validation_results else 0,
            "errors": errors,
            "successes": successes
        }


# =====================================================================
# MOCK DATA GENERATOR
# =====================================================================

class MockDataGenerator:
    """Dynamic mock data generation for complex testing scenarios."""
    
    def __init__(self):
        self.factory = TestDataFactory()
        
    def generate_api_response_mock(self, 
                                  response_type: str,
                                  success: bool = True,
                                  latency_ms: int = 100) -> Dict[str, Any]:
        """Generate mock API response data."""
        base_response = {
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": latency_ms,
            "request_id": str(uuid.uuid4()),
        }
        
        if response_type == "openai_chat":
            if success:
                base_response.update({
                    "status": "success",
                    "data": {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "This is a mock response for metabolomics research query."
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": random.randint(50, 200),
                            "completion_tokens": random.randint(100, 500),
                            "total_tokens": random.randint(150, 700)
                        }
                    }
                })
            else:
                base_response.update({
                    "status": "error",
                    "error": {
                        "type": "rate_limit_exceeded",
                        "message": "Rate limit exceeded. Please try again later.",
                        "code": 429
                    }
                })
        
        elif response_type == "embedding":
            if success:
                # Generate realistic embedding vector
                embedding_dim = 1536  # OpenAI ada-002 dimension
                embedding = [random.gauss(0, 1) for _ in range(embedding_dim)]
                
                base_response.update({
                    "status": "success",
                    "data": {
                        "embedding": embedding,
                        "dimension": embedding_dim,
                        "model": "text-embedding-ada-002"
                    }
                })
            else:
                base_response.update({
                    "status": "error",
                    "error": {
                        "type": "invalid_input",
                        "message": "Input text too long for embedding model.",
                        "code": 400
                    }
                })
        
        return base_response
    
    def generate_system_state_mock(self, 
                                  state_type: str,
                                  healthy: bool = True) -> Dict[str, Any]:
        """Generate mock system state data."""
        base_state = {
            "timestamp": datetime.now().isoformat(),
            "system_id": str(uuid.uuid4()),
            "state_type": state_type
        }
        
        if state_type == "cost_monitor":
            if healthy:
                base_state.update({
                    "status": "healthy",
                    "current_cost": round(random.uniform(0.10, 5.00), 4),
                    "daily_budget": 10.00,
                    "utilization_percent": round(random.uniform(10, 70), 1),
                    "requests_processed": random.randint(50, 200),
                    "average_cost_per_request": round(random.uniform(0.001, 0.025), 6)
                })
            else:
                base_state.update({
                    "status": "budget_exceeded",
                    "current_cost": 12.50,
                    "daily_budget": 10.00,
                    "utilization_percent": 125.0,
                    "requests_processed": random.randint(400, 600),
                    "average_cost_per_request": round(random.uniform(0.020, 0.050), 6)
                })
        
        elif state_type == "lightrag_system":
            if healthy:
                base_state.update({
                    "status": "operational",
                    "knowledge_base_size": random.randint(1000, 10000),
                    "indexed_documents": random.randint(50, 500),
                    "query_response_time_ms": random.randint(100, 1000),
                    "memory_usage_mb": random.randint(200, 800),
                    "active_connections": random.randint(1, 10)
                })
            else:
                base_state.update({
                    "status": "degraded",
                    "knowledge_base_size": 0,
                    "indexed_documents": 0,
                    "query_response_time_ms": 30000,
                    "memory_usage_mb": random.randint(1500, 3000),
                    "active_connections": 0,
                    "error_message": "Knowledge base initialization failed"
                })
        
        return base_state
    
    def generate_performance_test_data(self, 
                                     scenario: str,
                                     duration_seconds: int = 60) -> Dict[str, Any]:
        """Generate performance test data for load testing."""
        # Simulate realistic performance metrics over time
        timestamps = []
        response_times = []
        throughput = []
        error_rates = []
        
        start_time = datetime.now()
        
        for i in range(duration_seconds):
            timestamp = start_time + timedelta(seconds=i)
            timestamps.append(timestamp.isoformat())
            
            if scenario == "normal_load":
                response_times.append(random.gauss(250, 50))  # 250ms ± 50ms
                throughput.append(random.gauss(10, 2))  # 10 ± 2 requests/sec
                error_rates.append(random.uniform(0, 0.01))  # 0-1% error rate
            
            elif scenario == "high_load":
                response_times.append(random.gauss(800, 200))  # 800ms ± 200ms
                throughput.append(random.gauss(25, 5))  # 25 ± 5 requests/sec
                error_rates.append(random.uniform(0.02, 0.05))  # 2-5% error rate
            
            elif scenario == "stress_test":
                response_times.append(random.gauss(2000, 500))  # 2s ± 500ms
                throughput.append(random.gauss(50, 10))  # 50 ± 10 requests/sec
                error_rates.append(random.uniform(0.05, 0.15))  # 5-15% error rate
        
        return {
            "scenario": scenario,
            "duration_seconds": duration_seconds,
            "data_points": len(timestamps),
            "metrics": {
                "timestamps": timestamps,
                "response_times_ms": response_times,
                "throughput_rps": throughput,
                "error_rates": error_rates
            },
            "summary": {
                "avg_response_time": sum(response_times) / len(response_times),
                "max_response_time": max(response_times),
                "avg_throughput": sum(throughput) / len(throughput),
                "avg_error_rate": sum(error_rates) / len(error_rates)
            }
        }


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def create_test_scenario(name: str, 
                        description: str,
                        data_requirements: List[str],
                        expected_outcomes: List[str]) -> TestScenario:
    """Create a structured test scenario definition."""
    return TestScenario(name, description, data_requirements, expected_outcomes)


def generate_test_file_batch(output_dir: Path, 
                           file_count: int = 10,
                           content_type: str = "metabolomics") -> List[Path]:
    """Generate a batch of test files."""
    factory = TestDataFactory()
    generated_files = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(file_count):
        if content_type == "metabolomics":
            study = factory.generate_clinical_study()
            content = study.to_research_paper()
            filename = f"generated_study_{i+1:03d}_{study.study_id}.txt"
        elif content_type == "compounds":
            compounds_db = factory.generate_compound_database(count=10)
            content = json.dumps(compounds_db, indent=2)
            filename = f"generated_compounds_{i+1:03d}.json"
        else:
            content = f"Generated test content {i+1} for {content_type}"
            filename = f"generated_{content_type}_{i+1:03d}.txt"
        
        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        generated_files.append(file_path)
    
    return generated_files


@contextmanager
def temporary_test_database(schema_sql: str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript(schema_sql)
        conn.commit()
        yield conn
    finally:
        conn.close()
        Path(db_path).unlink(missing_ok=True)


def calculate_data_checksum(data: Union[str, bytes, Path]) -> str:
    """Calculate checksum for data integrity verification."""
    if isinstance(data, Path):
        with open(data, 'rb') as f:
            data = f.read()
    elif isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()


def cleanup_generated_files(directory: Path, pattern: str = "generated_*") -> int:
    """Clean up generated files in directory."""
    count = 0
    for file_path in directory.glob(pattern):
        try:
            if file_path.is_file():
                file_path.unlink()
                count += 1
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                count += 1
        except Exception as e:
            logging.warning(f"Failed to cleanup {file_path}: {e}")
    
    return count


def load_test_data_safe(file_path: Path, default: Any = None) -> Any:
    """Safely load test data with fallback - utility function for test integration."""
    try:
        if file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return file_path.read_text(encoding="utf-8")
    except Exception as e:
        logging.warning(f"Failed to load test data from {file_path}: {e}")
        return default