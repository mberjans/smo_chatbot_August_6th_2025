#!/usr/bin/env python3
"""
Test Data Validator

Validates the integrity and structure of test data for the LightRAG integration project.
Ensures test data meets requirements and is properly formatted.

Usage:
    python test_data_validator.py [options]
"""

import os
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestDataValidator:
    """Validates test data structure and integrity"""
    
    def __init__(self, test_data_path: str):
        self.test_data_path = Path(test_data_path)
        self.validation_results = {
            'structure_validation': {},
            'content_validation': {},
            'integrity_validation': {},
            'errors': [],
            'warnings': []
        }
        
        # Expected directory structure
        self.expected_structure = {
            'pdfs': ['samples', 'templates', 'corrupted'],
            'databases': ['schemas', 'samples', 'test_dbs'],
            'logs': ['templates', 'configs', 'samples'],
            'mocks': ['biomedical_data', 'api_responses', 'state_data'],
            'temp': ['staging', 'processing', 'cleanup'],
            'utilities': ['cleanup_scripts', 'data_generators', 'validators'],
            'reports': ['performance', 'validation', 'cleanup']
        }
        
    def validate_directory_structure(self) -> bool:
        """Validate that expected directory structure exists"""
        logger.info("Validating directory structure...")
        
        structure_valid = True
        
        # Check main test_data directory exists
        if not self.test_data_path.exists():
            self.validation_results['errors'].append(
                f"Test data directory does not exist: {self.test_data_path}"
            )
            return False
            
        # Check expected subdirectories
        for main_dir, subdirs in self.expected_structure.items():
            main_path = self.test_data_path / main_dir
            if not main_path.exists():
                self.validation_results['errors'].append(
                    f"Missing main directory: {main_path}"
                )
                structure_valid = False
                continue
                
            for subdir in subdirs:
                sub_path = main_path / subdir
                if not sub_path.exists():
                    self.validation_results['warnings'].append(
                        f"Missing subdirectory: {sub_path}"
                    )
                    
        self.validation_results['structure_validation']['status'] = structure_valid
        return structure_valid
        
    def validate_pdf_samples(self) -> bool:
        """Validate PDF sample files"""
        logger.info("Validating PDF samples...")
        
        pdf_path = self.test_data_path / 'pdfs'
        validation_results = {
            'samples_count': 0,
            'templates_count': 0,
            'corrupted_count': 0,
            'valid_content': 0,
            'errors': []
        }
        
        # Check samples directory
        samples_path = pdf_path / 'samples'
        if samples_path.exists():
            sample_files = list(samples_path.glob('*.txt'))  # Using .txt for testing
            validation_results['samples_count'] = len(sample_files)
            
            for sample_file in sample_files:
                if self._validate_biomedical_content(sample_file):
                    validation_results['valid_content'] += 1
                    
        # Check templates directory
        templates_path = pdf_path / 'templates'
        if templates_path.exists():
            template_files = list(templates_path.glob('*.txt'))
            validation_results['templates_count'] = len(template_files)
            
        # Check corrupted samples
        corrupted_path = pdf_path / 'corrupted'
        if corrupted_path.exists():
            corrupted_files = list(corrupted_path.glob('*.txt'))
            validation_results['corrupted_count'] = len(corrupted_files)
            
        self.validation_results['content_validation']['pdfs'] = validation_results
        
        # Validation criteria
        min_samples = 2
        min_templates = 1
        
        valid = (validation_results['samples_count'] >= min_samples and 
                validation_results['templates_count'] >= min_templates)
                
        if not valid:
            self.validation_results['errors'].append(
                f"Insufficient PDF samples: need {min_samples} samples, {min_templates} templates"
            )
            
        return valid
        
    def validate_databases(self) -> bool:
        """Validate database schemas and samples"""
        logger.info("Validating databases...")
        
        db_path = self.test_data_path / 'databases'
        validation_results = {
            'schemas_count': 0,
            'sample_dbs_count': 0,
            'valid_schemas': 0,
            'valid_dbs': 0,
            'errors': []
        }
        
        # Check schema files
        schemas_path = db_path / 'schemas'
        if schemas_path.exists():
            schema_files = list(schemas_path.glob('*.sql'))
            validation_results['schemas_count'] = len(schema_files)
            
            for schema_file in schema_files:
                if self._validate_sql_schema(schema_file):
                    validation_results['valid_schemas'] += 1
                    
        # Check sample databases
        samples_path = db_path / 'samples'
        test_dbs_path = db_path / 'test_dbs'
        
        db_files = []
        if samples_path.exists():
            db_files.extend(list(samples_path.glob('*.db')))
        if test_dbs_path.exists():
            db_files.extend(list(test_dbs_path.glob('*.db')))
            
        validation_results['sample_dbs_count'] = len(db_files)
        
        for db_file in db_files:
            if self._validate_sqlite_database(db_file):
                validation_results['valid_dbs'] += 1
                
        self.validation_results['content_validation']['databases'] = validation_results
        
        # Validation criteria
        min_schemas = 1
        valid = validation_results['schemas_count'] >= min_schemas
        
        if not valid:
            self.validation_results['errors'].append(
                f"Insufficient database schemas: need at least {min_schemas}"
            )
            
        return valid
        
    def validate_mock_data(self) -> bool:
        """Validate mock data files"""
        logger.info("Validating mock data...")
        
        mocks_path = self.test_data_path / 'mocks'
        validation_results = {
            'biomedical_files': 0,
            'api_response_files': 0,
            'state_files': 0,
            'valid_json_files': 0,
            'errors': []
        }
        
        # Check biomedical data
        bio_path = mocks_path / 'biomedical_data'
        if bio_path.exists():
            bio_files = list(bio_path.glob('*.json'))
            validation_results['biomedical_files'] = len(bio_files)
            
            for bio_file in bio_files:
                if self._validate_json_file(bio_file):
                    validation_results['valid_json_files'] += 1
                    
        # Check API responses
        api_path = mocks_path / 'api_responses'
        if api_path.exists():
            api_files = list(api_path.glob('*.json'))
            validation_results['api_response_files'] = len(api_files)
            
            for api_file in api_files:
                if self._validate_json_file(api_file):
                    validation_results['valid_json_files'] += 1
                    
        # Check state data
        state_path = mocks_path / 'state_data'
        if state_path.exists():
            state_files = list(state_path.glob('*.json'))
            validation_results['state_files'] = len(state_files)
            
            for state_file in state_files:
                if self._validate_json_file(state_file):
                    validation_results['valid_json_files'] += 1
                    
        self.validation_results['content_validation']['mocks'] = validation_results
        
        total_files = (validation_results['biomedical_files'] + 
                      validation_results['api_response_files'] + 
                      validation_results['state_files'])
                      
        valid = total_files > 0 and validation_results['valid_json_files'] == total_files
        
        if not valid:
            self.validation_results['errors'].append(
                "Mock data validation failed: missing or invalid JSON files"
            )
            
        return valid
        
    def validate_utilities(self) -> bool:
        """Validate utility scripts"""
        logger.info("Validating utilities...")
        
        utilities_path = self.test_data_path / 'utilities'
        validation_results = {
            'cleanup_scripts': 0,
            'data_generators': 0,
            'validators': 0,
            'executable_scripts': 0
        }
        
        # Check cleanup scripts
        cleanup_path = utilities_path / 'cleanup_scripts'
        if cleanup_path.exists():
            cleanup_files = list(cleanup_path.glob('*.py'))
            validation_results['cleanup_scripts'] = len(cleanup_files)
            
            for script in cleanup_files:
                if self._validate_python_script(script):
                    validation_results['executable_scripts'] += 1
                    
        # Check data generators
        generators_path = utilities_path / 'data_generators'
        if generators_path.exists():
            generator_files = list(generators_path.glob('*.py'))
            validation_results['data_generators'] = len(generator_files)
            
            for script in generator_files:
                if self._validate_python_script(script):
                    validation_results['executable_scripts'] += 1
                    
        # Check validators
        validators_path = utilities_path / 'validators'
        if validators_path.exists():
            validator_files = list(validators_path.glob('*.py'))
            validation_results['validators'] = len(validator_files)
            
            for script in validator_files:
                if self._validate_python_script(script):
                    validation_results['executable_scripts'] += 1
                    
        self.validation_results['content_validation']['utilities'] = validation_results
        
        min_utilities = 1
        valid = (validation_results['cleanup_scripts'] >= min_utilities or
                validation_results['data_generators'] >= min_utilities)
                
        return valid
        
    def _validate_biomedical_content(self, file_path: Path) -> bool:
        """Validate biomedical content file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for key biomedical terms
            biomedical_terms = [
                'metabolomics', 'metabolite', 'clinical', 'biomarker',
                'diabetes', 'cardiovascular', 'LC-MS', 'analysis'
            ]
            
            term_count = sum(1 for term in biomedical_terms if term.lower() in content.lower())
            
            # Must contain at least 3 biomedical terms
            return term_count >= 3
            
        except Exception as e:
            self.validation_results['errors'].append(f"Error validating {file_path}: {e}")
            return False
            
    def _validate_sql_schema(self, file_path: Path) -> bool:
        """Validate SQL schema file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for basic SQL keywords
            required_keywords = ['CREATE TABLE', 'PRIMARY KEY']
            return all(keyword in content.upper() for keyword in required_keywords)
            
        except Exception as e:
            self.validation_results['errors'].append(f"Error validating schema {file_path}: {e}")
            return False
            
    def _validate_sqlite_database(self, file_path: Path) -> bool:
        """Validate SQLite database"""
        try:
            conn = sqlite3.connect(str(file_path))
            cursor = conn.cursor()
            
            # Check if database has tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            conn.close()
            return len(tables) > 0
            
        except Exception as e:
            self.validation_results['errors'].append(f"Error validating database {file_path}: {e}")
            return False
            
    def _validate_json_file(self, file_path: Path) -> bool:
        """Validate JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
            
        except json.JSONDecodeError as e:
            self.validation_results['errors'].append(f"Invalid JSON in {file_path}: {e}")
            return False
        except Exception as e:
            self.validation_results['errors'].append(f"Error validating JSON {file_path}: {e}")
            return False
            
    def _validate_python_script(self, file_path: Path) -> bool:
        """Validate Python script"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic validation - check for Python syntax elements
            python_indicators = ['import ', 'def ', 'class ', 'if __name__']
            return any(indicator in content for indicator in python_indicators)
            
        except Exception as e:
            self.validation_results['errors'].append(f"Error validating script {file_path}: {e}")
            return False
            
    def calculate_data_integrity_checksums(self) -> Dict[str, str]:
        """Calculate checksums for data integrity verification"""
        logger.info("Calculating data integrity checksums...")
        
        checksums = {}
        
        for root, dirs, files in os.walk(self.test_data_path):
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            checksum = hashlib.md5(content).hexdigest()
                            relative_path = str(file_path.relative_to(self.test_data_path))
                            checksums[relative_path] = checksum
                    except Exception as e:
                        self.validation_results['warnings'].append(
                            f"Could not calculate checksum for {file_path}: {e}"
                        )
                        
        self.validation_results['integrity_validation']['checksums'] = checksums
        return checksums
        
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("Starting full validation of test data...")
        
        validation_passed = True
        
        # Run all validation checks
        structure_valid = self.validate_directory_structure()
        pdfs_valid = self.validate_pdf_samples()
        dbs_valid = self.validate_databases()
        mocks_valid = self.validate_mock_data()
        utilities_valid = self.validate_utilities()
        
        # Calculate integrity checksums
        self.calculate_data_integrity_checksums()
        
        validation_passed = all([
            structure_valid, pdfs_valid, dbs_valid, mocks_valid, utilities_valid
        ])
        
        self.validation_results['overall_status'] = 'PASSED' if validation_passed else 'FAILED'
        self.validation_results['validation_timestamp'] = str(datetime.now())
        
        return self.validation_results
        
    def save_validation_report(self, output_path: str = None) -> str:
        """Save validation report to file"""
        if output_path is None:
            output_path = self.test_data_path.parent / 'test_data_validation_report.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
            
        logger.info(f"Validation report saved to: {output_path}")
        return str(output_path)
        
    def print_validation_summary(self):
        """Print validation summary to console"""
        print("\n" + "="*60)
        print("TEST DATA VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall Status: {self.validation_results.get('overall_status', 'UNKNOWN')}")
        print(f"Validation Time: {self.validation_results.get('validation_timestamp', 'Unknown')}")
        
        print(f"\nErrors: {len(self.validation_results['errors'])}")
        for error in self.validation_results['errors']:
            print(f"  ERROR: {error}")
            
        print(f"\nWarnings: {len(self.validation_results['warnings'])}")
        for warning in self.validation_results['warnings']:
            print(f"  WARNING: {warning}")
            
        # Print content validation summary
        content_val = self.validation_results.get('content_validation', {})
        if 'pdfs' in content_val:
            pdf_stats = content_val['pdfs']
            print(f"\nPDF Files: {pdf_stats['samples_count']} samples, "
                  f"{pdf_stats['templates_count']} templates, "
                  f"{pdf_stats['corrupted_count']} corrupted")
                  
        if 'databases' in content_val:
            db_stats = content_val['databases']
            print(f"Databases: {db_stats['schemas_count']} schemas, "
                  f"{db_stats['sample_dbs_count']} sample DBs")
                  
        if 'mocks' in content_val:
            mock_stats = content_val['mocks']
            total_mock_files = (mock_stats['biomedical_files'] + 
                               mock_stats['api_response_files'] + 
                               mock_stats['state_files'])
            print(f"Mock Data: {total_mock_files} files, "
                  f"{mock_stats['valid_json_files']} valid JSON")
                  
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Validate test data for LightRAG integration')
    parser.add_argument('--test-data-path', 
                       default='./test_data', 
                       help='Path to test data directory')
    parser.add_argument('--report-output', 
                       help='Output path for validation report')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    validator = TestDataValidator(args.test_data_path)
    results = validator.run_full_validation()
    
    if not args.quiet:
        validator.print_validation_summary()
        
    # Save report
    report_path = validator.save_validation_report(args.report_output)
    
    # Return appropriate exit code
    return 0 if results['overall_status'] == 'PASSED' else 1


if __name__ == '__main__':
    exit(main())