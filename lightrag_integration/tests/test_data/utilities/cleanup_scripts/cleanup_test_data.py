#!/usr/bin/env python3
"""
Test Data Cleanup Utility

This script provides comprehensive cleanup functionality for test data management
in the Clinical Metabolomics Oracle LightRAG integration project.

Usage:
    python cleanup_test_data.py [options]
    
Options:
    --mode: Cleanup mode (temp_only, databases, all, selective)
    --age: Clean files older than N hours (default: 24)
    --dry-run: Show what would be cleaned without actually cleaning
    --verbose: Show detailed cleanup information
"""

import os
import sys
import glob
import shutil
import argparse
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class TestDataCleanup:
    """Manages cleanup of test data across the project"""
    
    def __init__(self, base_path: str, dry_run: bool = False, verbose: bool = False):
        self.base_path = Path(base_path)
        self.dry_run = dry_run
        self.verbose = verbose
        self.cleanup_stats = {
            'files_removed': 0,
            'directories_removed': 0,
            'databases_cleaned': 0,
            'size_freed_mb': 0.0
        }
        
    def cleanup_temporary_files(self, max_age_hours: int = 24) -> None:
        """Clean temporary files and directories"""
        temp_patterns = [
            'temp/**/*',
            'tmp_*',
            '*.tmp',
            '*.temp',
            '__pycache__/**/*',
            '.pytest_cache/**/*',
            '*.pyc',
            '.coverage'
        ]
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for pattern in temp_patterns:
            files = glob.glob(str(self.base_path / pattern), recursive=True)
            for file_path in files:
                self._remove_if_old(file_path, cutoff_time)
                
    def cleanup_test_databases(self) -> None:
        """Clean test databases and reset to initial state"""
        db_patterns = [
            'databases/test_dbs/*.db',
            'databases/samples/*.db',
            '**/test_*.db',
            'cost_tracking.db'
        ]
        
        for pattern in db_patterns:
            db_files = glob.glob(str(self.base_path / pattern), recursive=True)
            for db_path in db_files:
                self._cleanup_database(db_path)
                
    def cleanup_log_files(self, max_age_hours: int = 48) -> None:
        """Clean old log files"""
        log_patterns = [
            'logs/**/*.log',
            'logs/**/*.log.*',  # Rotated logs
            '**/*.log',
            'reports/**/*.json',
            'reports/**/*.txt'
        ]
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for pattern in log_patterns:
            log_files = glob.glob(str(self.base_path / pattern), recursive=True)
            for log_path in log_files:
                if not self._is_template_file(log_path):
                    self._remove_if_old(log_path, cutoff_time)
                    
    def cleanup_generated_pdfs(self) -> None:
        """Clean generated PDF files, keeping templates"""
        pdf_patterns = [
            'pdfs/samples/generated_*.txt',
            'pdfs/samples/test_*.txt',
            '**/generated_*.pdf',
            '**/test_*.pdf'
        ]
        
        for pattern in pdf_patterns:
            pdf_files = glob.glob(str(self.base_path / pattern), recursive=True)
            for pdf_path in pdf_files:
                if not self._is_template_file(pdf_path):
                    self._remove_file(pdf_path)
                    
    def reset_mock_data_states(self) -> None:
        """Reset mock data to initial states"""
        state_files = [
            'mocks/state_data/*.json',
            'mocks/api_responses/dynamic_*.json'
        ]
        
        for pattern in state_files:
            files = glob.glob(str(self.base_path / pattern))
            for file_path in files:
                if 'mock_system_states.json' in file_path:
                    self._reset_system_states(file_path)
                    
    def _remove_if_old(self, file_path: str, cutoff_time: datetime) -> None:
        """Remove file if older than cutoff time"""
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return
                
            file_time = datetime.fromtimestamp(path_obj.stat().st_mtime)
            if file_time < cutoff_time:
                self._remove_file(file_path)
        except Exception as e:
            logger.warning(f"Error checking file age {file_path}: {e}")
            
    def _remove_file(self, file_path: str) -> None:
        """Remove a single file"""
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return
                
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would remove: {file_path}")
                return
                
            if path_obj.is_file():
                path_obj.unlink()
                self.cleanup_stats['files_removed'] += 1
            elif path_obj.is_dir():
                shutil.rmtree(file_path)
                self.cleanup_stats['directories_removed'] += 1
                
            self.cleanup_stats['size_freed_mb'] += size_mb
            
            if self.verbose:
                logger.info(f"Removed: {file_path} ({size_mb:.2f} MB)")
                
        except Exception as e:
            logger.error(f"Error removing {file_path}: {e}")
            
    def _cleanup_database(self, db_path: str) -> None:
        """Clean database by removing non-essential data"""
        try:
            if not Path(db_path).exists():
                return
                
            if self.dry_run:
                logger.info(f"[DRY RUN] Would clean database: {db_path}")
                return
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Clean test data from tables
            for (table,) in tables:
                if table.startswith('test_'):
                    cursor.execute(f"DELETE FROM {table} WHERE created_at < datetime('now', '-1 day');")
                elif table in ['cost_tracking', 'api_metrics']:
                    cursor.execute(f"DELETE FROM {table} WHERE timestamp < datetime('now', '-7 days');")
                    
            # Vacuum database to reclaim space
            cursor.execute("VACUUM;")
            conn.commit()
            conn.close()
            
            self.cleanup_stats['databases_cleaned'] += 1
            
            if self.verbose:
                logger.info(f"Cleaned database: {db_path}")
                
        except Exception as e:
            logger.error(f"Error cleaning database {db_path}: {e}")
            
    def _is_template_file(self, file_path: str) -> bool:
        """Check if file is a template that should be preserved"""
        template_indicators = [
            'template',
            'sample_',
            'mock_',
            '/templates/',
            '/samples/',
            'README',
            '.md'
        ]
        
        return any(indicator in file_path for indicator in template_indicators)
        
    def _reset_system_states(self, file_path: str) -> None:
        """Reset system states to healthy defaults"""
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would reset states in: {file_path}")
                return
                
            # Load current states
            with open(file_path, 'r') as f:
                states = json.load(f)
                
            # Reset to healthy state
            if 'system_states' in states:
                for state_name in states['system_states']:
                    if state_name != 'healthy_system':
                        # Reset timestamps to current time
                        states['system_states'][state_name]['timestamp'] = datetime.now().isoformat() + 'Z'
                        
            # Write back
            with open(file_path, 'w') as f:
                json.dump(states, f, indent=2)
                
            if self.verbose:
                logger.info(f"Reset system states in: {file_path}")
                
        except Exception as e:
            logger.error(f"Error resetting states {file_path}: {e}")
            
    def run_cleanup(self, mode: str = 'temp_only', max_age_hours: int = 24) -> Dict[str, Any]:
        """Run cleanup based on mode"""
        logger.info(f"Starting cleanup in mode: {mode}")
        
        if mode in ['temp_only', 'all']:
            self.cleanup_temporary_files(max_age_hours)
            
        if mode in ['databases', 'all']:
            self.cleanup_test_databases()
            
        if mode in ['logs', 'all']:
            self.cleanup_log_files(max_age_hours)
            
        if mode in ['pdfs', 'all']:
            self.cleanup_generated_pdfs()
            
        if mode in ['states', 'all']:
            self.reset_mock_data_states()
            
        return self.cleanup_stats
        
    def print_cleanup_report(self) -> None:
        """Print cleanup statistics"""
        print("\n" + "="*50)
        print("TEST DATA CLEANUP REPORT")
        print("="*50)
        print(f"Files removed: {self.cleanup_stats['files_removed']}")
        print(f"Directories removed: {self.cleanup_stats['directories_removed']}")
        print(f"Databases cleaned: {self.cleanup_stats['databases_cleaned']}")
        print(f"Space freed: {self.cleanup_stats['size_freed_mb']:.2f} MB")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Clean test data for LightRAG integration')
    parser.add_argument('--mode', choices=['temp_only', 'databases', 'logs', 'pdfs', 'states', 'all'], 
                       default='temp_only', help='Cleanup mode')
    parser.add_argument('--age', type=int, default=24, help='Clean files older than N hours')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--base-path', default='.', help='Base path for cleanup')
    
    args = parser.parse_args()
    
    cleaner = TestDataCleanup(args.base_path, args.dry_run, args.verbose)
    stats = cleaner.run_cleanup(args.mode, args.age)
    cleaner.print_cleanup_report()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())