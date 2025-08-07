#!/usr/bin/env python3
"""
Comprehensive Test Data Integrity Validation System.

This module provides extensive validation capabilities for ensuring the integrity,
correctness, and completeness of test data across the Clinical Metabolomics Oracle
LightRAG integration system.

Key Features:
1. Multi-layered data integrity validation
2. Biomedical content verification and domain-specific validation
3. Database consistency and schema validation
4. File integrity and corruption detection
5. Mock data validation and structure verification
6. Configuration validation and environment checks
7. Cross-reference validation between data sources
8. Performance impact assessment during validation

Components:
- DataIntegrityValidator: Core validation orchestrator
- BiomedicalContentIntegrityChecker: Domain-specific content validation
- DatabaseIntegrityValidator: Database schema and data consistency
- FileIntegrityChecker: File corruption and format validation
- MockDataValidator: Mock data structure and completeness
- ConfigurationValidator: Configuration and environment validation
- CrossReferenceValidator: Inter-data source consistency
- ValidationPerformanceMonitor: Performance impact tracking

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import psutil
import re
import sqlite3
import statistics
import time
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Dict, List, Set, Any, Optional, Union, Tuple, Callable, 
    Generator, AsyncGenerator, TypeVar, Generic, Pattern
)
import warnings

# Import existing validation infrastructure
try:
    from validation_fixtures import ValidationResult, ValidationReport, ValidationLevel, ValidationType
    from test_data.utilities.validators.test_data_validator import TestDataValidator
    from advanced_cleanup_system import ResourceType, CleanupValidator
except ImportError as e:
    logging.warning(f"Import warning: {e}")
    # Define minimal classes for standalone operation
    
    class ValidationLevel(Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        INFO = "info"
    
    class ValidationType(Enum):
        DATA_INTEGRITY = "data_integrity"
        BIOMEDICAL_ACCURACY = "biomedical_accuracy"
        STRUCTURAL_VALIDATION = "structural_validation"


# =====================================================================
# CORE VALIDATION TYPES AND STRUCTURES
# =====================================================================

class IntegrityValidationType(Enum):
    """Types of integrity validation checks."""
    FILE_INTEGRITY = "file_integrity"
    CONTENT_INTEGRITY = "content_integrity"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    SEMANTIC_INTEGRITY = "semantic_integrity"
    TEMPORAL_INTEGRITY = "temporal_integrity"
    CHECKSUM_VALIDATION = "checksum_validation"
    FORMAT_VALIDATION = "format_validation"
    DOMAIN_VALIDATION = "domain_validation"
    CONSISTENCY_VALIDATION = "consistency_validation"


class DataCategory(Enum):
    """Categories of test data."""
    PDF_DOCUMENTS = "pdf_documents"
    DATABASE_CONTENT = "database_content"
    MOCK_DATA = "mock_data"
    LOG_FILES = "log_files"
    CONFIGURATION = "configuration"
    BIOMEDICAL_CONTENT = "biomedical_content"
    PERFORMANCE_DATA = "performance_data"
    METADATA = "metadata"


class IntegrityLevel(Enum):
    """Levels of integrity checking."""
    BASIC = "basic"           # Quick structural checks
    STANDARD = "standard"     # Comprehensive validation
    DEEP = "deep"            # Extensive validation with cross-references
    EXHAUSTIVE = "exhaustive" # Complete validation including performance impact


@dataclass
class IntegrityValidationResult:
    """Result of an integrity validation check."""
    validation_id: str
    data_path: str
    data_category: DataCategory
    validation_type: IntegrityValidationType
    level: IntegrityLevel
    passed: bool
    confidence: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result['data_category'] = self.data_category.value
        result['validation_type'] = self.validation_type.value
        result['level'] = self.level.value
        return result


@dataclass
class IntegrityReport:
    """Comprehensive integrity validation report."""
    report_id: str
    validation_session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_files_checked: int = 0
    total_validations_performed: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    critical_issues: int = 0
    warnings: int = 0
    overall_integrity_score: float = 0.0
    validation_results: List[IntegrityValidationResult] = field(default_factory=list)
    category_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Calculate validation duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_validations_performed == 0:
            return 0.0
        return self.passed_validations / self.total_validations_performed * 100.0


# =====================================================================
# BIOMEDICAL CONTENT INTEGRITY CHECKER
# =====================================================================

class BiomedicalContentIntegrityChecker:
    """Validates biomedical content integrity and domain accuracy."""
    
    def __init__(self):
        self.biomedical_terms = {
            'metabolomics': ['metabolomics', 'metabolome', 'metabolites', 'metabolic'],
            'clinical': ['clinical', 'patient', 'diagnosis', 'treatment', 'therapeutic'],
            'analytical': ['LC-MS', 'GC-MS', 'NMR', 'mass spectrometry', 'chromatography'],
            'diseases': ['diabetes', 'cardiovascular', 'cancer', 'obesity', 'hypertension'],
            'biomarkers': ['biomarker', 'biomarkers', 'marker', 'indicators', 'signature'],
            'pathways': ['pathway', 'pathways', 'metabolism', 'biosynthesis', 'catabolism']
        }
        
        self.required_patterns = [
            r'\b\d+\.\d+\b',  # Numerical values
            r'\bp[\s<>=]+0\.\d+\b',  # P-values
            r'\b[A-Z]{2,}\b',  # Abbreviations/acronyms
            r'\b\d+\s*[µμ]?[MmLlGg]?\b',  # Concentrations/measurements
        ]
        
        self.validation_cache = {}
    
    def validate_biomedical_content(
        self, 
        content: str, 
        file_path: str, 
        expected_domains: Optional[List[str]] = None
    ) -> IntegrityValidationResult:
        """Validate biomedical content for domain accuracy and completeness."""
        
        validation_id = f"biomed_content_{hash(content) % 10000:04d}"
        
        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.validation_cache:
            cached_result = self.validation_cache[content_hash]
            cached_result.validation_id = validation_id
            return cached_result
        
        start_time = time.time()
        
        # Term frequency analysis
        term_scores = self._analyze_term_frequency(content)
        
        # Pattern validation
        pattern_scores = self._validate_content_patterns(content)
        
        # Domain coherence check
        domain_coherence = self._check_domain_coherence(content, expected_domains)
        
        # Scientific accuracy heuristics
        scientific_accuracy = self._assess_scientific_accuracy(content)
        
        # Calculate overall score
        overall_score = self._calculate_content_score(
            term_scores, pattern_scores, domain_coherence, scientific_accuracy
        )
        
        validation_time = time.time() - start_time
        
        passed = overall_score >= 0.7  # 70% threshold for biomedical content
        confidence = min(overall_score * 1.2, 1.0)
        
        details = {
            'term_analysis': term_scores,
            'pattern_validation': pattern_scores,
            'domain_coherence': domain_coherence,
            'scientific_accuracy': scientific_accuracy,
            'overall_score': overall_score,
            'content_length': len(content),
            'word_count': len(content.split())
        }
        
        evidence = self._generate_evidence(term_scores, pattern_scores, content)
        recommendations = self._generate_recommendations(overall_score, term_scores, pattern_scores)
        
        result = IntegrityValidationResult(
            validation_id=validation_id,
            data_path=file_path,
            data_category=DataCategory.BIOMEDICAL_CONTENT,
            validation_type=IntegrityValidationType.DOMAIN_VALIDATION,
            level=IntegrityLevel.STANDARD,
            passed=passed,
            confidence=confidence,
            message=f"Biomedical content validation {'passed' if passed else 'failed'} with score {overall_score:.2f}",
            details=details,
            evidence=evidence,
            recommendations=recommendations,
            performance_impact={'validation_time_ms': validation_time * 1000}
        )
        
        # Cache result
        self.validation_cache[content_hash] = result
        return result
    
    def _analyze_term_frequency(self, content: str) -> Dict[str, Any]:
        """Analyze frequency of biomedical terms."""
        content_lower = content.lower()
        term_analysis = {}
        
        for category, terms in self.biomedical_terms.items():
            found_terms = []
            total_occurrences = 0
            
            for term in terms:
                count = content_lower.count(term.lower())
                if count > 0:
                    found_terms.append({'term': term, 'count': count})
                    total_occurrences += count
            
            term_analysis[category] = {
                'found_terms': found_terms,
                'unique_terms': len(found_terms),
                'total_occurrences': total_occurrences,
                'coverage_ratio': len(found_terms) / len(terms) if terms else 0
            }
        
        return term_analysis
    
    def _validate_content_patterns(self, content: str) -> Dict[str, Any]:
        """Validate content against expected patterns."""
        pattern_results = {}
        
        for i, pattern in enumerate(self.required_patterns):
            matches = re.findall(pattern, content, re.IGNORECASE)
            pattern_results[f'pattern_{i}'] = {
                'pattern': pattern,
                'matches': len(matches),
                'examples': matches[:5] if matches else []  # First 5 examples
            }
        
        total_patterns = len(self.required_patterns)
        matched_patterns = sum(1 for result in pattern_results.values() if result['matches'] > 0)
        
        pattern_results['summary'] = {
            'total_patterns': total_patterns,
            'matched_patterns': matched_patterns,
            'pattern_score': matched_patterns / total_patterns if total_patterns else 0
        }
        
        return pattern_results
    
    def _check_domain_coherence(self, content: str, expected_domains: Optional[List[str]]) -> Dict[str, Any]:
        """Check domain coherence and consistency."""
        coherence_analysis = {
            'domain_consistency': True,
            'conflicting_information': [],
            'domain_alignment': 1.0
        }
        
        if expected_domains:
            content_lower = content.lower()
            for domain in expected_domains:
                if domain.lower() not in content_lower:
                    coherence_analysis['domain_consistency'] = False
                    coherence_analysis['domain_alignment'] *= 0.8
        
        # Check for common contradictions or inconsistencies
        contradiction_patterns = [
            (r'increases?', r'decreases?'),
            (r'positive', r'negative'),
            (r'high', r'low'),
            (r'significant', r'non-significant')
        ]
        
        for pos_pattern, neg_pattern in contradiction_patterns:
            pos_matches = len(re.findall(pos_pattern, content, re.IGNORECASE))
            neg_matches = len(re.findall(neg_pattern, content, re.IGNORECASE))
            
            if pos_matches > 0 and neg_matches > 0:
                ratio = min(pos_matches, neg_matches) / max(pos_matches, neg_matches)
                if ratio > 0.5:  # High ratio suggests potential contradiction
                    coherence_analysis['conflicting_information'].append({
                        'positive_pattern': pos_pattern,
                        'negative_pattern': neg_pattern,
                        'pos_matches': pos_matches,
                        'neg_matches': neg_matches,
                        'conflict_ratio': ratio
                    })
        
        return coherence_analysis
    
    def _assess_scientific_accuracy(self, content: str) -> Dict[str, Any]:
        """Assess scientific accuracy using heuristics."""
        accuracy_metrics = {
            'has_citations': bool(re.search(r'\[\d+\]|\(\d{4}\)', content)),
            'has_numerical_data': bool(re.search(r'\b\d+\.?\d*\b', content)),
            'has_statistical_measures': bool(re.search(r'\bp[\s<>=]+0\.\d+\b|confidence interval|CI|standard deviation|SD', content, re.IGNORECASE)),
            'has_methodology': bool(re.search(r'method|procedure|protocol|analysis|measurement', content, re.IGNORECASE)),
            'has_results': bool(re.search(r'result|finding|outcome|conclusion|significant', content, re.IGNORECASE))
        }
        
        accuracy_score = sum(accuracy_metrics.values()) / len(accuracy_metrics)
        
        return {
            'metrics': accuracy_metrics,
            'accuracy_score': accuracy_score
        }
    
    def _calculate_content_score(
        self, 
        term_scores: Dict[str, Any], 
        pattern_scores: Dict[str, Any],
        domain_coherence: Dict[str, Any],
        scientific_accuracy: Dict[str, Any]
    ) -> float:
        """Calculate overall content validation score."""
        
        # Term score (40% weight)
        avg_coverage = statistics.mean([
            cat_data['coverage_ratio'] 
            for cat_data in term_scores.values()
        ]) if term_scores else 0
        
        term_weight = 0.4 * avg_coverage
        
        # Pattern score (20% weight)
        pattern_weight = 0.2 * pattern_scores.get('summary', {}).get('pattern_score', 0)
        
        # Domain coherence (20% weight)
        coherence_weight = 0.2 * domain_coherence['domain_alignment']
        
        # Scientific accuracy (20% weight)
        accuracy_weight = 0.2 * scientific_accuracy['accuracy_score']
        
        total_score = term_weight + pattern_weight + coherence_weight + accuracy_weight
        return min(total_score, 1.0)
    
    def _generate_evidence(
        self, 
        term_scores: Dict[str, Any], 
        pattern_scores: Dict[str, Any], 
        content: str
    ) -> List[str]:
        """Generate evidence for validation decision."""
        evidence = []
        
        # Term evidence
        for category, data in term_scores.items():
            if data['unique_terms'] > 0:
                evidence.append(f"Found {data['unique_terms']} unique {category} terms with {data['total_occurrences']} total occurrences")
        
        # Pattern evidence
        pattern_summary = pattern_scores.get('summary', {})
        if pattern_summary.get('matched_patterns', 0) > 0:
            evidence.append(f"Matched {pattern_summary['matched_patterns']} out of {pattern_summary['total_patterns']} expected patterns")
        
        # Content length evidence
        evidence.append(f"Content length: {len(content)} characters, {len(content.split())} words")
        
        return evidence
    
    def _generate_recommendations(
        self, 
        overall_score: float, 
        term_scores: Dict[str, Any], 
        pattern_scores: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving content."""
        recommendations = []
        
        if overall_score < 0.5:
            recommendations.append("Content score is below threshold - consider adding more domain-specific terminology")
        
        # Check for missing term categories
        for category, data in term_scores.items():
            if data['coverage_ratio'] < 0.3:
                recommendations.append(f"Low coverage of {category} terms - consider adding more relevant terminology")
        
        # Check pattern coverage
        pattern_summary = pattern_scores.get('summary', {})
        if pattern_summary.get('pattern_score', 0) < 0.5:
            recommendations.append("Content lacks expected scientific patterns - add numerical data, measurements, or statistical information")
        
        return recommendations


# =====================================================================
# DATABASE INTEGRITY VALIDATOR
# =====================================================================

class DatabaseIntegrityValidator:
    """Validates database schema consistency and data integrity."""
    
    def __init__(self):
        self.connection_pool = {}
        self.schema_cache = {}
    
    @contextmanager
    def get_db_connection(self, db_path: str):
        """Get database connection with connection pooling."""
        try:
            if db_path not in self.connection_pool:
                self.connection_pool[db_path] = sqlite3.connect(db_path)
            
            connection = self.connection_pool[db_path]
            yield connection
        except Exception as e:
            logging.error(f"Database connection error for {db_path}: {e}")
            raise
    
    def validate_database_integrity(self, db_path: str) -> IntegrityValidationResult:
        """Comprehensive database integrity validation."""
        
        validation_id = f"db_integrity_{Path(db_path).stem}"
        start_time = time.time()
        
        try:
            with self.get_db_connection(db_path) as conn:
                # Schema validation
                schema_results = self._validate_schema_structure(conn)
                
                # Data consistency validation
                consistency_results = self._validate_data_consistency(conn)
                
                # Referential integrity
                referential_results = self._validate_referential_integrity(conn)
                
                # Index validation
                index_results = self._validate_indexes(conn)
                
                # Performance metrics
                performance_results = self._assess_database_performance(conn)
                
                validation_time = time.time() - start_time
                
                # Calculate overall score
                overall_score = self._calculate_db_score(
                    schema_results, consistency_results, referential_results, index_results
                )
                
                passed = overall_score >= 0.8
                confidence = min(overall_score * 1.1, 1.0)
                
                details = {
                    'schema_validation': schema_results,
                    'consistency_validation': consistency_results,
                    'referential_integrity': referential_results,
                    'index_validation': index_results,
                    'performance_metrics': performance_results,
                    'overall_score': overall_score
                }
                
                evidence = self._generate_db_evidence(schema_results, consistency_results)
                recommendations = self._generate_db_recommendations(details)
                
                return IntegrityValidationResult(
                    validation_id=validation_id,
                    data_path=db_path,
                    data_category=DataCategory.DATABASE_CONTENT,
                    validation_type=IntegrityValidationType.STRUCTURAL_INTEGRITY,
                    level=IntegrityLevel.STANDARD,
                    passed=passed,
                    confidence=confidence,
                    message=f"Database integrity validation {'passed' if passed else 'failed'} with score {overall_score:.2f}",
                    details=details,
                    evidence=evidence,
                    recommendations=recommendations,
                    performance_impact={'validation_time_ms': validation_time * 1000}
                )
                
        except Exception as e:
            return IntegrityValidationResult(
                validation_id=validation_id,
                data_path=db_path,
                data_category=DataCategory.DATABASE_CONTENT,
                validation_type=IntegrityValidationType.STRUCTURAL_INTEGRITY,
                level=IntegrityLevel.STANDARD,
                passed=False,
                confidence=0.0,
                message=f"Database validation failed: {str(e)}",
                details={'error': str(e)},
                evidence=[f"Database validation error: {str(e)}"],
                recommendations=["Check database file integrity and accessibility"],
                performance_impact={'validation_time_ms': (time.time() - start_time) * 1000}
            )
    
    def _validate_schema_structure(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Validate database schema structure."""
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = {
            'total_tables': len(tables),
            'tables': {},
            'schema_score': 0.0
        }
        
        for table in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            foreign_keys = cursor.fetchall()
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table})")
            indexes = cursor.fetchall()
            
            table_info = {
                'column_count': len(columns),
                'columns': [{'name': col[1], 'type': col[2], 'not_null': bool(col[3]), 'primary_key': bool(col[5])} for col in columns],
                'foreign_keys': len(foreign_keys),
                'indexes': len(indexes),
                'has_primary_key': any(col[5] for col in columns)
            }
            
            schema_info['tables'][table] = table_info
        
        # Calculate schema score
        total_score = 0
        if tables:
            for table_info in schema_info['tables'].values():
                table_score = 0.5  # Base score
                if table_info['has_primary_key']:
                    table_score += 0.3
                if table_info['column_count'] >= 2:
                    table_score += 0.2
                total_score += table_score
            
            schema_info['schema_score'] = min(total_score / len(tables), 1.0)
        
        return schema_info
    
    def _validate_data_consistency(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Validate data consistency within the database."""
        cursor = conn.cursor()
        consistency_results = {
            'null_checks': {},
            'data_type_consistency': {},
            'consistency_score': 1.0
        }
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Check for null values in non-null columns
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            null_issues = []
            for col in columns:
                col_name, col_type, not_null = col[1], col[2], col[3]
                if not_null:
                    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL")
                    null_count = cursor.fetchone()[0]
                    if null_count > 0:
                        null_issues.append({'column': col_name, 'null_count': null_count})
            
            consistency_results['null_checks'][table] = null_issues
            
            if null_issues:
                consistency_results['consistency_score'] *= 0.8
        
        return consistency_results
    
    def _validate_referential_integrity(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Validate referential integrity constraints."""
        cursor = conn.cursor()
        
        # Enable foreign key constraints check
        cursor.execute("PRAGMA foreign_key_check")
        violations = cursor.fetchall()
        
        referential_results = {
            'foreign_key_violations': len(violations),
            'violations': [
                {
                    'table': violation[0],
                    'row_id': violation[1],
                    'parent_table': violation[2],
                    'foreign_key_index': violation[3]
                }
                for violation in violations
            ],
            'referential_integrity_score': 1.0 if len(violations) == 0 else max(0.0, 1.0 - len(violations) * 0.1)
        }
        
        return referential_results
    
    def _validate_indexes(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Validate database indexes."""
        cursor = conn.cursor()
        
        # Get all indexes
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%'")
        indexes = cursor.fetchall()
        
        index_results = {
            'total_indexes': len(indexes),
            'indexes': [{'name': idx[0], 'definition': idx[1]} for idx in indexes],
            'index_score': min(len(indexes) * 0.2, 1.0)  # Reward having indexes
        }
        
        return index_results
    
    def _assess_database_performance(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Assess basic database performance metrics."""
        cursor = conn.cursor()
        
        start_time = time.time()
        
        # Simple query performance test
        cursor.execute("SELECT COUNT(*) FROM sqlite_master")
        master_count = cursor.fetchone()[0]
        
        query_time = time.time() - start_time
        
        # Database size
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        
        db_size = page_size * page_count
        
        return {
            'query_response_time_ms': query_time * 1000,
            'database_size_bytes': db_size,
            'page_size': page_size,
            'page_count': page_count,
            'master_table_entries': master_count
        }
    
    def _calculate_db_score(
        self, 
        schema_results: Dict[str, Any], 
        consistency_results: Dict[str, Any],
        referential_results: Dict[str, Any], 
        index_results: Dict[str, Any]
    ) -> float:
        """Calculate overall database integrity score."""
        
        schema_weight = 0.4 * schema_results.get('schema_score', 0)
        consistency_weight = 0.3 * consistency_results.get('consistency_score', 0)
        referential_weight = 0.2 * referential_results.get('referential_integrity_score', 0)
        index_weight = 0.1 * index_results.get('index_score', 0)
        
        return schema_weight + consistency_weight + referential_weight + index_weight
    
    def _generate_db_evidence(
        self, 
        schema_results: Dict[str, Any], 
        consistency_results: Dict[str, Any]
    ) -> List[str]:
        """Generate evidence for database validation."""
        evidence = []
        
        evidence.append(f"Found {schema_results['total_tables']} tables in database")
        
        primary_key_tables = sum(
            1 for table_info in schema_results['tables'].values() 
            if table_info['has_primary_key']
        )
        evidence.append(f"{primary_key_tables} tables have primary keys")
        
        null_violations = sum(
            len(violations) for violations in consistency_results['null_checks'].values()
        )
        evidence.append(f"Found {null_violations} null constraint violations")
        
        return evidence
    
    def _generate_db_recommendations(self, details: Dict[str, Any]) -> List[str]:
        """Generate recommendations for database improvements."""
        recommendations = []
        
        schema_results = details.get('schema_validation', {})
        consistency_results = details.get('consistency_validation', {})
        referential_results = details.get('referential_integrity', {})
        
        # Schema recommendations
        for table_name, table_info in schema_results.get('tables', {}).items():
            if not table_info['has_primary_key']:
                recommendations.append(f"Add primary key to table '{table_name}'")
            if table_info['indexes'] == 0 and table_info['column_count'] > 2:
                recommendations.append(f"Consider adding indexes to table '{table_name}' for better performance")
        
        # Consistency recommendations
        for table_name, null_issues in consistency_results.get('null_checks', {}).items():
            if null_issues:
                recommendations.append(f"Fix null value violations in table '{table_name}'")
        
        # Referential integrity recommendations
        if referential_results.get('foreign_key_violations', 0) > 0:
            recommendations.append("Fix foreign key constraint violations")
        
        return recommendations


# =====================================================================
# FILE INTEGRITY CHECKER
# =====================================================================

class FileIntegrityChecker:
    """Validates file integrity, format, and corruption detection."""
    
    def __init__(self):
        self.checksum_cache = {}
        self.format_validators = {
            '.json': self._validate_json_format,
            '.sql': self._validate_sql_format,
            '.txt': self._validate_text_format,
            '.log': self._validate_log_format,
            '.py': self._validate_python_format
        }
    
    def validate_file_integrity(
        self, 
        file_path: str, 
        expected_checksum: Optional[str] = None,
        perform_deep_validation: bool = True
    ) -> IntegrityValidationResult:
        """Comprehensive file integrity validation."""
        
        file_path_obj = Path(file_path)
        validation_id = f"file_integrity_{file_path_obj.stem}_{int(time.time()) % 10000}"
        start_time = time.time()
        
        if not file_path_obj.exists():
            return IntegrityValidationResult(
                validation_id=validation_id,
                data_path=file_path,
                data_category=DataCategory.PDF_DOCUMENTS,  # Default, will be corrected
                validation_type=IntegrityValidationType.FILE_INTEGRITY,
                level=IntegrityLevel.BASIC,
                passed=False,
                confidence=0.0,
                message="File does not exist",
                evidence=["File not found at specified path"],
                recommendations=["Verify file path is correct"]
            )
        
        try:
            # Basic file information
            file_stats = file_path_obj.stat()
            file_info = {
                'size_bytes': file_stats.st_size,
                'modified_time': file_stats.st_mtime,
                'created_time': file_stats.st_ctime,
                'permissions': oct(file_stats.st_mode)[-3:]
            }
            
            # Checksum calculation
            checksum = self._calculate_checksum(file_path)
            
            # Format validation
            format_results = self._validate_file_format(file_path, perform_deep_validation)
            
            # Corruption detection
            corruption_results = self._detect_corruption(file_path, file_stats.st_size)
            
            # Accessibility check
            accessibility_results = self._check_file_accessibility(file_path)
            
            validation_time = time.time() - start_time
            
            # Calculate overall score
            overall_score = self._calculate_file_score(
                format_results, corruption_results, accessibility_results, file_info
            )
            
            # Check checksum if provided
            checksum_valid = True
            if expected_checksum:
                checksum_valid = checksum == expected_checksum
                if not checksum_valid:
                    overall_score *= 0.5
            
            passed = overall_score >= 0.8 and checksum_valid
            confidence = min(overall_score * 1.1, 1.0)
            
            details = {
                'file_info': file_info,
                'checksum': checksum,
                'expected_checksum': expected_checksum,
                'checksum_valid': checksum_valid,
                'format_validation': format_results,
                'corruption_detection': corruption_results,
                'accessibility': accessibility_results,
                'overall_score': overall_score
            }
            
            evidence = self._generate_file_evidence(file_info, format_results, corruption_results)
            recommendations = self._generate_file_recommendations(details)
            
            # Determine data category based on file extension
            data_category = self._determine_data_category(file_path)
            
            return IntegrityValidationResult(
                validation_id=validation_id,
                data_path=file_path,
                data_category=data_category,
                validation_type=IntegrityValidationType.FILE_INTEGRITY,
                level=IntegrityLevel.DEEP if perform_deep_validation else IntegrityLevel.STANDARD,
                passed=passed,
                confidence=confidence,
                message=f"File integrity validation {'passed' if passed else 'failed'} with score {overall_score:.2f}",
                details=details,
                evidence=evidence,
                recommendations=recommendations,
                performance_impact={'validation_time_ms': validation_time * 1000}
            )
            
        except Exception as e:
            return IntegrityValidationResult(
                validation_id=validation_id,
                data_path=file_path,
                data_category=DataCategory.METADATA,
                validation_type=IntegrityValidationType.FILE_INTEGRITY,
                level=IntegrityLevel.BASIC,
                passed=False,
                confidence=0.0,
                message=f"File validation failed: {str(e)}",
                details={'error': str(e)},
                evidence=[f"File validation error: {str(e)}"],
                recommendations=["Check file accessibility and format"],
                performance_impact={'validation_time_ms': (time.time() - start_time) * 1000}
            )
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file."""
        if file_path in self.checksum_cache:
            return self.checksum_cache[file_path]
        
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            logging.warning(f"Could not calculate checksum for {file_path}: {e}")
            return ""
        
        checksum = hash_md5.hexdigest()
        self.checksum_cache[file_path] = checksum
        return checksum
    
    def _validate_file_format(self, file_path: str, deep_validation: bool) -> Dict[str, Any]:
        """Validate file format and structure."""
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()
        
        format_results = {
            'detected_extension': file_extension,
            'format_valid': True,
            'format_score': 1.0,
            'format_details': {}
        }
        
        if deep_validation and file_extension in self.format_validators:
            try:
                validator_results = self.format_validators[file_extension](file_path)
                format_results.update(validator_results)
            except Exception as e:
                format_results.update({
                    'format_valid': False,
                    'format_score': 0.0,
                    'format_details': {'error': str(e)}
                })
        
        return format_results
    
    def _validate_json_format(self, file_path: str) -> Dict[str, Any]:
        """Validate JSON file format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                'format_valid': True,
                'format_score': 1.0,
                'format_details': {
                    'json_type': type(data).__name__,
                    'key_count': len(data) if isinstance(data, dict) else None,
                    'item_count': len(data) if isinstance(data, list) else None
                }
            }
        except json.JSONDecodeError as e:
            return {
                'format_valid': False,
                'format_score': 0.0,
                'format_details': {'json_error': str(e)}
            }
    
    def _validate_sql_format(self, file_path: str) -> Dict[str, Any]:
        """Validate SQL file format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic SQL keywords
            sql_keywords = ['CREATE', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'ALTER', 'DROP']
            found_keywords = [kw for kw in sql_keywords if kw in content.upper()]
            
            has_sql_structure = bool(found_keywords)
            has_semicolons = ';' in content
            
            return {
                'format_valid': has_sql_structure,
                'format_score': 1.0 if has_sql_structure else 0.5,
                'format_details': {
                    'found_keywords': found_keywords,
                    'has_semicolons': has_semicolons,
                    'line_count': len(content.splitlines())
                }
            }
        except Exception as e:
            return {
                'format_valid': False,
                'format_score': 0.0,
                'format_details': {'sql_error': str(e)}
            }
    
    def _validate_text_format(self, file_path: str) -> Dict[str, Any]:
        """Validate text file format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic text validation
            is_readable = True
            line_count = len(content.splitlines())
            word_count = len(content.split())
            char_count = len(content)
            
            return {
                'format_valid': is_readable,
                'format_score': 1.0,
                'format_details': {
                    'line_count': line_count,
                    'word_count': word_count,
                    'character_count': char_count,
                    'encoding': 'utf-8'
                }
            }
        except UnicodeDecodeError:
            return {
                'format_valid': False,
                'format_score': 0.3,
                'format_details': {'encoding_error': 'File contains non-UTF-8 characters'}
            }
    
    def _validate_log_format(self, file_path: str) -> Dict[str, Any]:
        """Validate log file format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Look for common log patterns
            log_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # Date pattern
                r'\d{2}:\d{2}:\d{2}',  # Time pattern
                r'\[(INFO|ERROR|DEBUG|WARNING|WARN)\]',  # Log level pattern
                r'(ERROR|INFO|DEBUG|WARNING|WARN)',  # Log level pattern (alternative)
            ]
            
            pattern_matches = {}
            for pattern in log_patterns:
                matches = sum(1 for line in lines if re.search(pattern, line))
                pattern_matches[pattern] = matches
            
            total_matches = sum(pattern_matches.values())
            log_score = min(total_matches / (len(lines) * 2), 1.0) if lines else 0
            
            return {
                'format_valid': log_score > 0.1,
                'format_score': log_score,
                'format_details': {
                    'line_count': len(lines),
                    'pattern_matches': pattern_matches,
                    'appears_to_be_log': log_score > 0.3
                }
            }
        except Exception as e:
            return {
                'format_valid': False,
                'format_score': 0.0,
                'format_details': {'log_error': str(e)}
            }
    
    def _validate_python_format(self, file_path: str) -> Dict[str, Any]:
        """Validate Python file format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for Python syntax elements
            python_indicators = [
                'import ', 'from ', 'def ', 'class ', 'if __name__',
                'print(', '#!/usr/bin/env python', '# -*- coding:'
            ]
            
            found_indicators = [ind for ind in python_indicators if ind in content]
            
            # Try to compile (basic syntax check)
            try:
                compile(content, file_path, 'exec')
                syntax_valid = True
                syntax_error = None
            except SyntaxError as e:
                syntax_valid = False
                syntax_error = str(e)
            
            python_score = len(found_indicators) * 0.2
            if syntax_valid:
                python_score += 0.4
            
            return {
                'format_valid': len(found_indicators) > 0,
                'format_score': min(python_score, 1.0),
                'format_details': {
                    'found_indicators': found_indicators,
                    'syntax_valid': syntax_valid,
                    'syntax_error': syntax_error,
                    'line_count': len(content.splitlines())
                }
            }
        except Exception as e:
            return {
                'format_valid': False,
                'format_score': 0.0,
                'format_details': {'python_error': str(e)}
            }
    
    def _detect_corruption(self, file_path: str, file_size: int) -> Dict[str, Any]:
        """Detect potential file corruption."""
        corruption_indicators = {
            'zero_size': file_size == 0,
            'abnormally_small': file_size < 10,  # Less than 10 bytes might be suspicious
            'read_errors': False,
            'binary_content_in_text': False
        }
        
        try:
            # Try to read file
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)  # Read first 1KB
                
            # Check for binary content in supposed text files
            file_extension = Path(file_path).suffix.lower()
            text_extensions = {'.txt', '.py', '.sql', '.log', '.json', '.md'}
            
            if file_extension in text_extensions:
                # Check for non-printable characters (potential corruption)
                non_printable_count = sum(1 for byte in chunk if byte < 32 and byte not in {9, 10, 13})
                if non_printable_count > len(chunk) * 0.1:  # More than 10% non-printable
                    corruption_indicators['binary_content_in_text'] = True
                    
        except Exception as e:
            corruption_indicators['read_errors'] = True
            corruption_indicators['read_error_details'] = str(e)
        
        # Calculate corruption score
        corruption_score = 1.0
        for indicator, present in corruption_indicators.items():
            if present and indicator != 'read_error_details':
                corruption_score -= 0.25
        
        corruption_score = max(corruption_score, 0.0)
        
        return {
            'corruption_indicators': corruption_indicators,
            'corruption_score': corruption_score,
            'likely_corrupted': corruption_score < 0.5
        }
    
    def _check_file_accessibility(self, file_path: str) -> Dict[str, Any]:
        """Check file accessibility and permissions."""
        file_path_obj = Path(file_path)
        
        accessibility = {
            'readable': os.access(file_path, os.R_OK),
            'writable': os.access(file_path, os.W_OK),
            'executable': os.access(file_path, os.X_OK),
            'exists': file_path_obj.exists(),
            'is_file': file_path_obj.is_file(),
            'is_symlink': file_path_obj.is_symlink()
        }
        
        # Calculate accessibility score
        accessibility_score = 1.0
        if not accessibility['readable']:
            accessibility_score -= 0.5
        if not accessibility['exists'] or not accessibility['is_file']:
            accessibility_score = 0.0
        
        accessibility['accessibility_score'] = accessibility_score
        
        return accessibility
    
    def _calculate_file_score(
        self, 
        format_results: Dict[str, Any], 
        corruption_results: Dict[str, Any],
        accessibility_results: Dict[str, Any], 
        file_info: Dict[str, Any]
    ) -> float:
        """Calculate overall file integrity score."""
        
        format_weight = 0.4 * format_results.get('format_score', 0)
        corruption_weight = 0.3 * corruption_results.get('corruption_score', 0)
        accessibility_weight = 0.2 * accessibility_results.get('accessibility_score', 0)
        
        # Size penalty for zero-size files
        size_weight = 0.1
        if file_info['size_bytes'] == 0:
            size_weight = 0.0
        
        return format_weight + corruption_weight + accessibility_weight + size_weight
    
    def _generate_file_evidence(
        self, 
        file_info: Dict[str, Any], 
        format_results: Dict[str, Any], 
        corruption_results: Dict[str, Any]
    ) -> List[str]:
        """Generate evidence for file validation."""
        evidence = []
        
        evidence.append(f"File size: {file_info['size_bytes']} bytes")
        evidence.append(f"File format validation: {'passed' if format_results['format_valid'] else 'failed'}")
        
        corruption_indicators = corruption_results.get('corruption_indicators', {})
        active_indicators = [k for k, v in corruption_indicators.items() if v and k != 'read_error_details']
        if active_indicators:
            evidence.append(f"Corruption indicators found: {', '.join(active_indicators)}")
        else:
            evidence.append("No corruption indicators detected")
        
        return evidence
    
    def _generate_file_recommendations(self, details: Dict[str, Any]) -> List[str]:
        """Generate recommendations for file improvements."""
        recommendations = []
        
        corruption_results = details.get('corruption_detection', {})
        if corruption_results.get('likely_corrupted', False):
            recommendations.append("File appears corrupted - verify file integrity or regenerate")
        
        accessibility_results = details.get('accessibility', {})
        if not accessibility_results.get('readable', True):
            recommendations.append("File is not readable - check permissions")
        
        format_results = details.get('format_validation', {})
        if not format_results.get('format_valid', True):
            recommendations.append("File format validation failed - verify file format and content")
        
        file_info = details.get('file_info', {})
        if file_info.get('size_bytes', 0) == 0:
            recommendations.append("File is empty - verify file content was written correctly")
        
        return recommendations
    
    def _determine_data_category(self, file_path: str) -> DataCategory:
        """Determine data category based on file path and extension."""
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()
        
        # Check path components
        path_str = str(file_path).lower()
        
        if 'pdf' in path_str or file_extension == '.pdf':
            return DataCategory.PDF_DOCUMENTS
        elif 'database' in path_str or file_extension in {'.db', '.sqlite', '.sql'}:
            return DataCategory.DATABASE_CONTENT
        elif 'mock' in path_str or 'test' in path_str:
            return DataCategory.MOCK_DATA
        elif 'log' in path_str or file_extension == '.log':
            return DataCategory.LOG_FILES
        elif 'config' in path_str or file_extension in {'.conf', '.config', '.ini', '.yaml', '.yml'}:
            return DataCategory.CONFIGURATION
        elif file_extension == '.json' and 'performance' in path_str:
            return DataCategory.PERFORMANCE_DATA
        else:
            return DataCategory.METADATA


# =====================================================================
# MOCK DATA VALIDATOR
# =====================================================================

class MockDataValidator:
    """Validates mock data structure, completeness, and consistency."""
    
    def __init__(self):
        self.mock_data_schemas = {
            'biomedical_data': {
                'required_fields': ['metabolite_id', 'name', 'concentration', 'unit'],
                'optional_fields': ['pathway', 'disease_association', 'reference'],
                'field_types': {
                    'metabolite_id': str,
                    'name': str,
                    'concentration': (int, float),
                    'unit': str
                }
            },
            'api_responses': {
                'required_fields': ['response', 'status_code'],
                'optional_fields': ['headers', 'timestamp', 'request_id'],
                'field_types': {
                    'response': (str, dict, list),
                    'status_code': int
                }
            },
            'state_data': {
                'required_fields': ['state_id', 'timestamp', 'state_data'],
                'optional_fields': ['metadata', 'version'],
                'field_types': {
                    'state_id': str,
                    'timestamp': (int, float),
                    'state_data': dict
                }
            }
        }
    
    def validate_mock_data_integrity(self, data_path: str) -> IntegrityValidationResult:
        """Validate mock data file integrity and structure."""
        
        validation_id = f"mock_data_{Path(data_path).stem}_{int(time.time()) % 10000}"
        start_time = time.time()
        
        try:
            # Determine mock data type from path
            mock_type = self._determine_mock_type(data_path)
            
            # Load and parse mock data
            with open(data_path, 'r', encoding='utf-8') as f:
                mock_data = json.load(f)
            
            # Schema validation
            schema_results = self._validate_mock_schema(mock_data, mock_type)
            
            # Data consistency validation
            consistency_results = self._validate_mock_consistency(mock_data, mock_type)
            
            # Completeness validation
            completeness_results = self._validate_mock_completeness(mock_data, mock_type)
            
            # Realism validation
            realism_results = self._validate_mock_realism(mock_data, mock_type)
            
            validation_time = time.time() - start_time
            
            # Calculate overall score
            overall_score = self._calculate_mock_score(
                schema_results, consistency_results, completeness_results, realism_results
            )
            
            passed = overall_score >= 0.8
            confidence = min(overall_score * 1.1, 1.0)
            
            details = {
                'mock_type': mock_type,
                'data_count': len(mock_data) if isinstance(mock_data, list) else 1,
                'schema_validation': schema_results,
                'consistency_validation': consistency_results,
                'completeness_validation': completeness_results,
                'realism_validation': realism_results,
                'overall_score': overall_score
            }
            
            evidence = self._generate_mock_evidence(mock_data, schema_results, completeness_results)
            recommendations = self._generate_mock_recommendations(details)
            
            return IntegrityValidationResult(
                validation_id=validation_id,
                data_path=data_path,
                data_category=DataCategory.MOCK_DATA,
                validation_type=IntegrityValidationType.STRUCTURAL_INTEGRITY,
                level=IntegrityLevel.STANDARD,
                passed=passed,
                confidence=confidence,
                message=f"Mock data validation {'passed' if passed else 'failed'} with score {overall_score:.2f}",
                details=details,
                evidence=evidence,
                recommendations=recommendations,
                performance_impact={'validation_time_ms': validation_time * 1000}
            )
            
        except json.JSONDecodeError as e:
            return IntegrityValidationResult(
                validation_id=validation_id,
                data_path=data_path,
                data_category=DataCategory.MOCK_DATA,
                validation_type=IntegrityValidationType.FORMAT_VALIDATION,
                level=IntegrityLevel.BASIC,
                passed=False,
                confidence=0.0,
                message=f"Invalid JSON format: {str(e)}",
                details={'json_error': str(e)},
                evidence=[f"JSON parsing error: {str(e)}"],
                recommendations=["Fix JSON format errors"],
                performance_impact={'validation_time_ms': (time.time() - start_time) * 1000}
            )
        except Exception as e:
            return IntegrityValidationResult(
                validation_id=validation_id,
                data_path=data_path,
                data_category=DataCategory.MOCK_DATA,
                validation_type=IntegrityValidationType.STRUCTURAL_INTEGRITY,
                level=IntegrityLevel.BASIC,
                passed=False,
                confidence=0.0,
                message=f"Mock data validation failed: {str(e)}",
                details={'error': str(e)},
                evidence=[f"Validation error: {str(e)}"],
                recommendations=["Check mock data file format and structure"],
                performance_impact={'validation_time_ms': (time.time() - start_time) * 1000}
            )
    
    def _determine_mock_type(self, data_path: str) -> str:
        """Determine type of mock data based on path."""
        path_str = str(data_path).lower()
        
        if 'biomedical' in path_str or 'metabolite' in path_str:
            return 'biomedical_data'
        elif 'api' in path_str or 'response' in path_str:
            return 'api_responses'
        elif 'state' in path_str:
            return 'state_data'
        else:
            return 'unknown'
    
    def _validate_mock_schema(self, mock_data: Any, mock_type: str) -> Dict[str, Any]:
        """Validate mock data against expected schema."""
        if mock_type not in self.mock_data_schemas:
            return {
                'schema_valid': True,  # No schema to validate against
                'schema_score': 1.0,
                'missing_fields': [],
                'type_errors': []
            }
        
        schema = self.mock_data_schemas[mock_type]
        
        # Handle both single objects and arrays
        data_items = mock_data if isinstance(mock_data, list) else [mock_data]
        
        missing_fields = []
        type_errors = []
        valid_items = 0
        
        for i, item in enumerate(data_items):
            if not isinstance(item, dict):
                type_errors.append(f"Item {i} is not a dictionary")
                continue
            
            # Check required fields
            item_missing = []
            for field in schema['required_fields']:
                if field not in item:
                    item_missing.append(field)
            
            if item_missing:
                missing_fields.append(f"Item {i} missing: {', '.join(item_missing)}")
            
            # Check field types
            for field, expected_types in schema['field_types'].items():
                if field in item:
                    if not isinstance(expected_types, tuple):
                        expected_types = (expected_types,)
                    
                    if not isinstance(item[field], expected_types):
                        type_errors.append(f"Item {i}.{field}: expected {expected_types}, got {type(item[field])}")
                    else:
                        valid_items += 1
        
        schema_score = valid_items / (len(data_items) * len(schema['field_types'])) if data_items else 0
        
        return {
            'schema_valid': len(missing_fields) == 0 and len(type_errors) == 0,
            'schema_score': schema_score,
            'missing_fields': missing_fields,
            'type_errors': type_errors,
            'validated_items': len(data_items)
        }
    
    def _validate_mock_consistency(self, mock_data: Any, mock_type: str) -> Dict[str, Any]:
        """Validate consistency within mock data."""
        consistency_results = {
            'consistent': True,
            'consistency_score': 1.0,
            'issues': []
        }
        
        if isinstance(mock_data, list) and len(mock_data) > 1:
            # Check field consistency across items
            all_fields = set()
            for item in mock_data:
                if isinstance(item, dict):
                    all_fields.update(item.keys())
            
            # Check if all items have similar structure
            field_counts = {field: 0 for field in all_fields}
            for item in mock_data:
                if isinstance(item, dict):
                    for field in item.keys():
                        field_counts[field] += 1
            
            total_items = len(mock_data)
            inconsistent_fields = []
            
            for field, count in field_counts.items():
                if count < total_items * 0.8:  # Less than 80% coverage
                    inconsistent_fields.append(f"{field}: {count}/{total_items} items")
            
            if inconsistent_fields:
                consistency_results['consistent'] = False
                consistency_results['consistency_score'] *= 0.7
                consistency_results['issues'].extend(inconsistent_fields)
        
        # Type-specific consistency checks
        if mock_type == 'biomedical_data':
            consistency_results = self._check_biomedical_consistency(mock_data, consistency_results)
        elif mock_type == 'api_responses':
            consistency_results = self._check_api_response_consistency(mock_data, consistency_results)
        
        return consistency_results
    
    def _check_biomedical_consistency(self, mock_data: Any, consistency_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency specific to biomedical data."""
        if isinstance(mock_data, list):
            concentrations = []
            units = set()
            
            for item in mock_data:
                if isinstance(item, dict):
                    if 'concentration' in item and isinstance(item['concentration'], (int, float)):
                        concentrations.append(item['concentration'])
                    if 'unit' in item and isinstance(item['unit'], str):
                        units.add(item['unit'])
            
            # Check for reasonable concentration ranges
            if concentrations:
                min_conc = min(concentrations)
                max_conc = max(concentrations)
                
                if min_conc < 0:
                    consistency_results['issues'].append("Negative concentrations found")
                    consistency_results['consistent'] = False
                
                if max_conc > min_conc * 10000:  # Very large range might be suspicious
                    consistency_results['issues'].append("Extremely large concentration range detected")
            
            # Check unit consistency
            if len(units) > 5:  # Too many different units might indicate inconsistency
                consistency_results['issues'].append(f"Many different units used: {len(units)}")
        
        return consistency_results
    
    def _check_api_response_consistency(self, mock_data: Any, consistency_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency specific to API response data."""
        if isinstance(mock_data, list):
            status_codes = []
            
            for item in mock_data:
                if isinstance(item, dict) and 'status_code' in item:
                    if isinstance(item['status_code'], int):
                        status_codes.append(item['status_code'])
            
            # Check for valid HTTP status codes
            invalid_codes = [code for code in status_codes if not (100 <= code <= 599)]
            if invalid_codes:
                consistency_results['issues'].append(f"Invalid HTTP status codes: {invalid_codes}")
                consistency_results['consistent'] = False
        
        return consistency_results
    
    def _validate_mock_completeness(self, mock_data: Any, mock_type: str) -> Dict[str, Any]:
        """Validate completeness of mock data."""
        completeness_results = {
            'complete': True,
            'completeness_score': 1.0,
            'coverage_analysis': {}
        }
        
        data_items = mock_data if isinstance(mock_data, list) else [mock_data]
        
        if not data_items:
            completeness_results.update({
                'complete': False,
                'completeness_score': 0.0,
                'coverage_analysis': {'empty_dataset': True}
            })
            return completeness_results
        
        # Analyze field coverage
        if mock_type in self.mock_data_schemas:
            schema = self.mock_data_schemas[mock_type]
            all_fields = schema['required_fields'] + schema['optional_fields']
            
            field_coverage = {}
            for field in all_fields:
                present_count = sum(1 for item in data_items if isinstance(item, dict) and field in item)
                field_coverage[field] = {
                    'present_count': present_count,
                    'coverage_ratio': present_count / len(data_items) if data_items else 0
                }
            
            completeness_results['coverage_analysis'] = field_coverage
            
            # Calculate completeness score
            avg_coverage = statistics.mean([
                info['coverage_ratio'] for info in field_coverage.values()
            ]) if field_coverage else 0
            
            completeness_results['completeness_score'] = avg_coverage
            completeness_results['complete'] = avg_coverage >= 0.8
        
        # Check for minimum data volume
        min_items = 5  # Minimum expected items for meaningful mock data
        if len(data_items) < min_items:
            completeness_results['complete'] = False
            completeness_results['completeness_score'] *= 0.5
            completeness_results['coverage_analysis']['insufficient_volume'] = {
                'current_count': len(data_items),
                'minimum_expected': min_items
            }
        
        return completeness_results
    
    def _validate_mock_realism(self, mock_data: Any, mock_type: str) -> Dict[str, Any]:
        """Validate realism of mock data values."""
        realism_results = {
            'realistic': True,
            'realism_score': 1.0,
            'realism_issues': []
        }
        
        if mock_type == 'biomedical_data':
            realism_results = self._assess_biomedical_realism(mock_data, realism_results)
        elif mock_type == 'api_responses':
            realism_results = self._assess_api_response_realism(mock_data, realism_results)
        
        return realism_results
    
    def _assess_biomedical_realism(self, mock_data: Any, realism_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess realism of biomedical mock data."""
        data_items = mock_data if isinstance(mock_data, list) else [mock_data]
        
        for item in data_items:
            if not isinstance(item, dict):
                continue
            
            # Check metabolite names
            if 'name' in item:
                name = item['name']
                if len(name) < 3:  # Very short names might be unrealistic
                    realism_results['realism_issues'].append(f"Very short metabolite name: {name}")
                elif len(name) > 50:  # Very long names might be unrealistic
                    realism_results['realism_issues'].append(f"Very long metabolite name: {name}")
            
            # Check concentration values
            if 'concentration' in item and isinstance(item['concentration'], (int, float)):
                conc = item['concentration']
                if conc <= 0:
                    realism_results['realism_issues'].append(f"Non-positive concentration: {conc}")
                elif conc > 1e6:  # Extremely high concentrations
                    realism_results['realism_issues'].append(f"Extremely high concentration: {conc}")
            
            # Check units
            if 'unit' in item:
                unit = item['unit'].lower()
                valid_units = ['µm', 'mm', 'ng/ml', 'µg/ml', 'mg/ml', 'pmol/l', 'nmol/l', 'µmol/l', 'mmol/l']
                if not any(valid_unit in unit for valid_unit in valid_units):
                    realism_results['realism_issues'].append(f"Unusual unit: {item['unit']}")
        
        # Calculate realism score
        if realism_results['realism_issues']:
            issue_penalty = min(len(realism_results['realism_issues']) * 0.1, 0.5)
            realism_results['realism_score'] = max(1.0 - issue_penalty, 0.0)
            realism_results['realistic'] = realism_results['realism_score'] >= 0.7
        
        return realism_results
    
    def _assess_api_response_realism(self, mock_data: Any, realism_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess realism of API response mock data."""
        data_items = mock_data if isinstance(mock_data, list) else [mock_data]
        
        for item in data_items:
            if not isinstance(item, dict):
                continue
            
            # Check status codes
            if 'status_code' in item:
                code = item['status_code']
                if isinstance(code, int):
                    if code < 100 or code >= 600:
                        realism_results['realism_issues'].append(f"Invalid HTTP status code: {code}")
                    elif code >= 500:  # Too many server errors might be unrealistic
                        realism_results['realism_issues'].append(f"Server error status code: {code}")
            
            # Check response structure
            if 'response' in item:
                response = item['response']
                if isinstance(response, str) and len(response) == 0:
                    realism_results['realism_issues'].append("Empty response string")
                elif isinstance(response, dict) and len(response) == 0:
                    realism_results['realism_issues'].append("Empty response object")
        
        # Calculate realism score
        if realism_results['realism_issues']:
            issue_penalty = min(len(realism_results['realism_issues']) * 0.1, 0.5)
            realism_results['realism_score'] = max(1.0 - issue_penalty, 0.0)
            realism_results['realistic'] = realism_results['realism_score'] >= 0.7
        
        return realism_results
    
    def _calculate_mock_score(
        self, 
        schema_results: Dict[str, Any], 
        consistency_results: Dict[str, Any],
        completeness_results: Dict[str, Any], 
        realism_results: Dict[str, Any]
    ) -> float:
        """Calculate overall mock data score."""
        
        schema_weight = 0.3 * schema_results.get('schema_score', 0)
        consistency_weight = 0.3 * consistency_results.get('consistency_score', 0)
        completeness_weight = 0.25 * completeness_results.get('completeness_score', 0)
        realism_weight = 0.15 * realism_results.get('realism_score', 0)
        
        return schema_weight + consistency_weight + completeness_weight + realism_weight
    
    def _generate_mock_evidence(
        self, 
        mock_data: Any, 
        schema_results: Dict[str, Any], 
        completeness_results: Dict[str, Any]
    ) -> List[str]:
        """Generate evidence for mock data validation."""
        evidence = []
        
        data_count = len(mock_data) if isinstance(mock_data, list) else 1
        evidence.append(f"Mock data contains {data_count} items")
        
        if schema_results.get('schema_valid', False):
            evidence.append("Schema validation passed")
        else:
            evidence.append(f"Schema validation failed: {len(schema_results.get('missing_fields', []))} missing field issues, {len(schema_results.get('type_errors', []))} type errors")
        
        completeness_score = completeness_results.get('completeness_score', 0)
        evidence.append(f"Data completeness score: {completeness_score:.2f}")
        
        return evidence
    
    def _generate_mock_recommendations(self, details: Dict[str, Any]) -> List[str]:
        """Generate recommendations for mock data improvements."""
        recommendations = []
        
        schema_results = details.get('schema_validation', {})
        if schema_results.get('missing_fields'):
            recommendations.append("Add missing required fields to mock data items")
        if schema_results.get('type_errors'):
            recommendations.append("Fix data type mismatches in mock data")
        
        completeness_results = details.get('completeness_validation', {})
        if not completeness_results.get('complete', True):
            recommendations.append("Increase mock data coverage and volume")
        
        consistency_results = details.get('consistency_validation', {})
        if not consistency_results.get('consistent', True):
            recommendations.append("Improve consistency across mock data items")
        
        realism_results = details.get('realism_validation', {})
        if realism_results.get('realism_issues'):
            recommendations.append("Address realism issues in mock data values")
        
        return recommendations


# =====================================================================
# MAIN DATA INTEGRITY VALIDATOR
# =====================================================================

class DataIntegrityValidator:
    """Main orchestrator for comprehensive test data integrity validation."""
    
    def __init__(self):
        self.biomedical_checker = BiomedicalContentIntegrityChecker()
        self.database_validator = DatabaseIntegrityValidator()
        self.file_checker = FileIntegrityChecker()
        self.mock_validator = MockDataValidator()
        
        # Performance monitoring
        self.performance_monitor = {
            'validation_count': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'memory_usage': []
        }
        
        self.validation_cache = {}
        
    def validate_test_data_integrity(
        self, 
        test_data_path: str,
        integrity_level: IntegrityLevel = IntegrityLevel.STANDARD,
        categories_to_validate: Optional[List[DataCategory]] = None
    ) -> IntegrityReport:
        """Perform comprehensive test data integrity validation."""
        
        session_id = f"integrity_session_{int(time.time())}"
        report_id = f"integrity_report_{int(time.time())}"
        start_time = time.time()
        
        logging.info(f"Starting test data integrity validation session: {session_id}")
        
        # Initialize report
        report = IntegrityReport(
            report_id=report_id,
            validation_session_id=session_id,
            start_time=start_time
        )
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        try:
            test_data_path_obj = Path(test_data_path)
            
            if not test_data_path_obj.exists():
                report.validation_results.append(
                    IntegrityValidationResult(
                        validation_id="path_check",
                        data_path=test_data_path,
                        data_category=DataCategory.METADATA,
                        validation_type=IntegrityValidationType.FILE_INTEGRITY,
                        level=integrity_level,
                        passed=False,
                        confidence=0.0,
                        message="Test data path does not exist",
                        evidence=["Path not found"],
                        recommendations=["Verify test data path is correct"]
                    )
                )
                report.end_time = time.time()
                return report
            
            # Discover all files to validate
            files_to_validate = self._discover_validation_targets(
                test_data_path_obj, categories_to_validate
            )
            
            report.total_files_checked = len(files_to_validate)
            
            # Perform validations based on integrity level
            if integrity_level == IntegrityLevel.BASIC:
                validation_results = self._perform_basic_validation(files_to_validate)
            elif integrity_level == IntegrityLevel.STANDARD:
                validation_results = self._perform_standard_validation(files_to_validate)
            elif integrity_level == IntegrityLevel.DEEP:
                validation_results = self._perform_deep_validation(files_to_validate)
            else:  # EXHAUSTIVE
                validation_results = self._perform_exhaustive_validation(files_to_validate)
            
            report.validation_results.extend(validation_results)
            
            # Calculate summary statistics
            report.total_validations_performed = len(validation_results)
            report.passed_validations = sum(1 for r in validation_results if r.passed)
            report.failed_validations = report.total_validations_performed - report.passed_validations
            report.critical_issues = sum(1 for r in validation_results if not r.passed and r.confidence < 0.3)
            report.warnings = sum(1 for r in validation_results if r.passed and r.confidence < 0.8)
            
            # Calculate overall integrity score
            if validation_results:
                confidence_scores = [r.confidence for r in validation_results]
                report.overall_integrity_score = statistics.mean(confidence_scores) * 100
            
            # Generate category summaries
            report.category_summaries = self._generate_category_summaries(validation_results)
            
            # Monitor performance
            final_memory = process.memory_info().rss
            memory_delta = final_memory - initial_memory
            
            report.performance_metrics = {
                'validation_duration_seconds': time.time() - start_time,
                'memory_usage_delta_bytes': memory_delta,
                'average_validation_time_ms': statistics.mean([
                    r.performance_impact.get('validation_time_ms', 0) 
                    for r in validation_results if r.performance_impact
                ]) if validation_results else 0,
                'files_per_second': len(files_to_validate) / (time.time() - start_time) if time.time() > start_time else 0
            }
            
            # Generate recommendations
            report.recommendations = self._generate_overall_recommendations(validation_results, report)
            
            report.end_time = time.time()
            
            # Update performance monitoring
            self._update_performance_monitoring(report.duration, memory_delta)
            
            logging.info(f"Completed integrity validation session: {session_id} in {report.duration:.2f}s")
            
            return report
            
        except Exception as e:
            logging.error(f"Integrity validation failed: {e}")
            
            report.validation_results.append(
                IntegrityValidationResult(
                    validation_id="validation_error",
                    data_path=test_data_path,
                    data_category=DataCategory.METADATA,
                    validation_type=IntegrityValidationType.STRUCTURAL_INTEGRITY,
                    level=integrity_level,
                    passed=False,
                    confidence=0.0,
                    message=f"Validation failed: {str(e)}",
                    evidence=[f"Exception occurred: {str(e)}"],
                    recommendations=["Check test data structure and accessibility"]
                )
            )
            
            report.end_time = time.time()
            report.failed_validations = 1
            report.critical_issues = 1
            
            return report
    
    def _discover_validation_targets(
        self, 
        test_data_path: Path, 
        categories_to_validate: Optional[List[DataCategory]]
    ) -> List[Tuple[str, DataCategory]]:
        """Discover files to validate based on categories."""
        
        targets = []
        
        category_paths = {
            DataCategory.PDF_DOCUMENTS: ['pdfs'],
            DataCategory.DATABASE_CONTENT: ['databases'],
            DataCategory.MOCK_DATA: ['mocks'],
            DataCategory.LOG_FILES: ['logs'],
            DataCategory.CONFIGURATION: ['.', 'config'],  # Root and config dirs
            DataCategory.PERFORMANCE_DATA: ['reports/performance'],
            DataCategory.METADATA: ['utilities', 'reports']
        }
        
        # If no specific categories, validate all
        if not categories_to_validate:
            categories_to_validate = list(DataCategory)
        
        for category in categories_to_validate:
            if category in category_paths:
                for path_segment in category_paths[category]:
                    search_path = test_data_path / path_segment
                    if search_path.exists():
                        for file_path in search_path.rglob('*'):
                            if file_path.is_file() and not file_path.name.startswith('.'):
                                targets.append((str(file_path), category))
        
        return targets
    
    def _perform_basic_validation(self, files_to_validate: List[Tuple[str, DataCategory]]) -> List[IntegrityValidationResult]:
        """Perform basic integrity validation."""
        results = []
        
        for file_path, category in files_to_validate:
            # Basic file existence and accessibility check
            result = self.file_checker.validate_file_integrity(
                file_path, perform_deep_validation=False
            )
            results.append(result)
        
        return results
    
    def _perform_standard_validation(self, files_to_validate: List[Tuple[str, DataCategory]]) -> List[IntegrityValidationResult]:
        """Perform standard integrity validation."""
        results = []
        
        for file_path, category in files_to_validate:
            if category == DataCategory.DATABASE_CONTENT and file_path.endswith(('.db', '.sqlite')):
                result = self.database_validator.validate_database_integrity(file_path)
            elif category == DataCategory.MOCK_DATA and file_path.endswith('.json'):
                result = self.mock_validator.validate_mock_data_integrity(file_path)
            elif category == DataCategory.BIOMEDICAL_CONTENT or 'biomedical' in file_path.lower():
                # Read content and validate
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    result = self.biomedical_checker.validate_biomedical_content(content, file_path)
                except Exception as e:
                    result = IntegrityValidationResult(
                        validation_id=f"biomed_error_{int(time.time())}",
                        data_path=file_path,
                        data_category=category,
                        validation_type=IntegrityValidationType.CONTENT_INTEGRITY,
                        level=IntegrityLevel.STANDARD,
                        passed=False,
                        confidence=0.0,
                        message=f"Content validation failed: {str(e)}",
                        evidence=[f"Error reading file: {str(e)}"],
                        recommendations=["Check file accessibility and format"]
                    )
            else:
                result = self.file_checker.validate_file_integrity(file_path)
            
            results.append(result)
        
        return results
    
    def _perform_deep_validation(self, files_to_validate: List[Tuple[str, DataCategory]]) -> List[IntegrityValidationResult]:
        """Perform deep integrity validation with cross-references."""
        results = self._perform_standard_validation(files_to_validate)
        
        # Add cross-reference validation
        cross_ref_results = self._perform_cross_reference_validation(files_to_validate)
        results.extend(cross_ref_results)
        
        return results
    
    def _perform_exhaustive_validation(self, files_to_validate: List[Tuple[str, DataCategory]]) -> List[IntegrityValidationResult]:
        """Perform exhaustive validation with performance impact analysis."""
        results = self._perform_deep_validation(files_to_validate)
        
        # Add performance impact analysis
        performance_results = self._analyze_validation_performance_impact(files_to_validate)
        results.extend(performance_results)
        
        return results
    
    def _perform_cross_reference_validation(self, files_to_validate: List[Tuple[str, DataCategory]]) -> List[IntegrityValidationResult]:
        """Perform cross-reference validation between related files."""
        results = []
        
        # Group files by category
        files_by_category = defaultdict(list)
        for file_path, category in files_to_validate:
            files_by_category[category].append(file_path)
        
        # Check for expected file relationships
        validation_id = f"cross_ref_{int(time.time())}"
        
        # Example: Check if biomedical PDFs have corresponding mock data
        pdf_files = files_by_category.get(DataCategory.PDF_DOCUMENTS, [])
        mock_files = files_by_category.get(DataCategory.MOCK_DATA, [])
        
        if pdf_files and not mock_files:
            results.append(IntegrityValidationResult(
                validation_id=f"{validation_id}_pdf_mock",
                data_path="cross_reference_check",
                data_category=DataCategory.METADATA,
                validation_type=IntegrityValidationType.REFERENTIAL_INTEGRITY,
                level=IntegrityLevel.DEEP,
                passed=False,
                confidence=0.6,
                message="PDF documents found but no corresponding mock data",
                evidence=[f"Found {len(pdf_files)} PDF files but no mock data"],
                recommendations=["Create mock data to support PDF document testing"]
            ))
        
        # Check database-mock data alignment
        db_files = files_by_category.get(DataCategory.DATABASE_CONTENT, [])
        if db_files and mock_files:
            # This is a good alignment
            results.append(IntegrityValidationResult(
                validation_id=f"{validation_id}_db_mock",
                data_path="cross_reference_check",
                data_category=DataCategory.METADATA,
                validation_type=IntegrityValidationType.REFERENTIAL_INTEGRITY,
                level=IntegrityLevel.DEEP,
                passed=True,
                confidence=0.9,
                message="Good alignment between database files and mock data",
                evidence=[f"Found {len(db_files)} database files and {len(mock_files)} mock data files"],
                recommendations=[]
            ))
        
        return results
    
    def _analyze_validation_performance_impact(self, files_to_validate: List[Tuple[str, DataCategory]]) -> List[IntegrityValidationResult]:
        """Analyze performance impact of validation process."""
        results = []
        
        validation_id = f"perf_impact_{int(time.time())}"
        
        # Estimate validation complexity
        total_files = len(files_to_validate)
        large_files = sum(1 for file_path, _ in files_to_validate if Path(file_path).stat().st_size > 1024*1024)  # > 1MB
        
        complexity_score = min((total_files * 0.1 + large_files * 0.5), 10.0)
        
        if complexity_score > 5.0:
            results.append(IntegrityValidationResult(
                validation_id=validation_id,
                data_path="performance_analysis",
                data_category=DataCategory.PERFORMANCE_DATA,
                validation_type=IntegrityValidationType.SEMANTIC_INTEGRITY,
                level=IntegrityLevel.EXHAUSTIVE,
                passed=True,
                confidence=0.8,
                message=f"High validation complexity detected (score: {complexity_score:.1f})",
                evidence=[f"Total files: {total_files}", f"Large files (>1MB): {large_files}"],
                recommendations=["Consider parallel processing for large validation tasks"]
            ))
        
        return results
    
    def _generate_category_summaries(self, validation_results: List[IntegrityValidationResult]) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics by data category."""
        summaries = {}
        
        # Group results by category
        results_by_category = defaultdict(list)
        for result in validation_results:
            results_by_category[result.data_category.value].append(result)
        
        for category, results in results_by_category.items():
            total_validations = len(results)
            passed_validations = sum(1 for r in results if r.passed)
            failed_validations = total_validations - passed_validations
            
            avg_confidence = statistics.mean([r.confidence for r in results]) if results else 0
            avg_time = statistics.mean([
                r.performance_impact.get('validation_time_ms', 0) 
                for r in results if r.performance_impact
            ]) if results else 0
            
            summaries[category] = {
                'total_validations': total_validations,
                'passed_validations': passed_validations,
                'failed_validations': failed_validations,
                'success_rate': (passed_validations / total_validations * 100) if total_validations else 0,
                'average_confidence': avg_confidence,
                'average_validation_time_ms': avg_time
            }
        
        return summaries
    
    def _generate_overall_recommendations(
        self, 
        validation_results: List[IntegrityValidationResult], 
        report: IntegrityReport
    ) -> List[str]:
        """Generate overall recommendations based on validation results."""
        recommendations = []
        
        # Analyze failure patterns
        failed_results = [r for r in validation_results if not r.passed]
        
        if failed_results:
            failure_types = defaultdict(int)
            for result in failed_results:
                failure_types[result.validation_type.value] += 1
            
            most_common_failure = max(failure_types.items(), key=lambda x: x[1])
            recommendations.append(f"Address {most_common_failure[0]} issues ({most_common_failure[1]} occurrences)")
        
        # Check overall success rate
        if report.success_rate < 80:
            recommendations.append("Overall validation success rate is below 80% - review test data quality")
        
        # Performance recommendations
        if report.performance_metrics.get('validation_duration_seconds', 0) > 60:
            recommendations.append("Validation took over 60 seconds - consider optimization")
        
        # Memory usage recommendations  
        memory_delta = report.performance_metrics.get('memory_usage_delta_bytes', 0)
        if memory_delta > 100 * 1024 * 1024:  # 100MB
            recommendations.append("High memory usage during validation - optimize for large datasets")
        
        return recommendations
    
    def _update_performance_monitoring(self, duration: float, memory_delta: int):
        """Update performance monitoring statistics."""
        self.performance_monitor['validation_count'] += 1
        self.performance_monitor['total_time'] += duration
        self.performance_monitor['average_time'] = (
            self.performance_monitor['total_time'] / self.performance_monitor['validation_count']
        )
        self.performance_monitor['memory_usage'].append(memory_delta)
        
        # Keep only last 10 memory measurements
        if len(self.performance_monitor['memory_usage']) > 10:
            self.performance_monitor['memory_usage'] = self.performance_monitor['memory_usage'][-10:]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_monitor.copy()
        
        if stats['memory_usage']:
            stats['average_memory_usage'] = statistics.mean(stats['memory_usage'])
            stats['peak_memory_usage'] = max(stats['memory_usage'])
        
        return stats
    
    def generate_integrity_report_summary(self, report: IntegrityReport) -> str:
        """Generate a human-readable summary of the integrity report."""
        
        summary = f"""
TEST DATA INTEGRITY VALIDATION REPORT
{"="*50}

Session ID: {report.validation_session_id}
Report ID: {report.report_id}
Validation Duration: {report.duration:.2f} seconds
Overall Integrity Score: {report.overall_integrity_score:.1f}%

FILES ANALYZED:
- Total files checked: {report.total_files_checked}
- Total validations performed: {report.total_validations_performed}

RESULTS SUMMARY:
- Passed validations: {report.passed_validations} ({report.success_rate:.1f}%)
- Failed validations: {report.failed_validations}
- Critical issues: {report.critical_issues}
- Warnings: {report.warnings}

CATEGORY BREAKDOWN:
"""
        
        for category, summary_data in report.category_summaries.items():
            summary += f"  {category.replace('_', ' ').title()}:\n"
            summary += f"    - Success rate: {summary_data['success_rate']:.1f}%\n"
            summary += f"    - Average confidence: {summary_data['average_confidence']:.2f}\n"
            summary += f"    - Validations: {summary_data['total_validations']}\n"
        
        summary += f"\nPERFORMANCE METRICS:\n"
        summary += f"- Validation speed: {report.performance_metrics.get('files_per_second', 0):.2f} files/second\n"
        summary += f"- Average validation time: {report.performance_metrics.get('average_validation_time_ms', 0):.2f}ms\n"
        summary += f"- Memory usage delta: {report.performance_metrics.get('memory_usage_delta_bytes', 0) / (1024*1024):.2f}MB\n"
        
        if report.recommendations:
            summary += f"\nRECOMMENDATIONS:\n"
            for i, rec in enumerate(report.recommendations, 1):
                summary += f"  {i}. {rec}\n"
        
        summary += f"\n{'='*50}\n"
        
        return summary
    
    def save_integrity_report(self, report: IntegrityReport, output_path: Optional[str] = None) -> str:
        """Save integrity report to file."""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_data_integrity_report_{timestamp}.json"
        
        # Convert report to dictionary
        report_dict = asdict(report)
        
        # Convert enums to strings
        for result_dict in report_dict['validation_results']:
            result_dict['data_category'] = result_dict['data_category'].value if hasattr(result_dict['data_category'], 'value') else result_dict['data_category']
            result_dict['validation_type'] = result_dict['validation_type'].value if hasattr(result_dict['validation_type'], 'value') else result_dict['validation_type']
            result_dict['level'] = result_dict['level'].value if hasattr(result_dict['level'], 'value') else result_dict['level']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logging.info(f"Integrity report saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    validator = DataIntegrityValidator()
    
    # Example validation
    test_data_path = "./test_data"  # Adjust path as needed
    
    print("Running comprehensive test data integrity validation...")
    report = validator.validate_test_data_integrity(
        test_data_path, 
        integrity_level=IntegrityLevel.STANDARD
    )
    
    print(validator.generate_integrity_report_summary(report))
    
    # Save report
    report_path = validator.save_integrity_report(report)
    print(f"Detailed report saved to: {report_path}")
    
    # Performance statistics
    perf_stats = validator.get_performance_statistics()
    print(f"\nPerformance Statistics:")
    print(f"- Validations performed: {perf_stats['validation_count']}")
    print(f"- Average time: {perf_stats['average_time']:.2f}s")
    print(f"- Total time: {perf_stats['total_time']:.2f}s")