"""
Performance Benchmark Script for BiomedicalPDFProcessor.

This comprehensive benchmark script tests all key aspects of the BiomedicalPDFProcessor
with detailed performance metrics, memory monitoring, and error simulation testing.

Features:
- Processing time measurement per PDF and per operation
- Memory usage monitoring with peak memory tracking
- Text extraction quality assessment
- Error handling robustness testing
- Metadata extraction accuracy validation
- Preprocessing effectiveness measurement
- Comprehensive reporting (JSON and human-readable formats)
- Scalable design for 1 to 5+ PDF files

The script generates detailed performance reports and includes suggestions for
obtaining additional biomedical PDF samples for more comprehensive testing.

Author: Clinical Metabolomics Oracle System
Date: August 6, 2025
"""

import asyncio
import json
import logging
import psutil
import signal
import statistics
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
import sys

# Import our PDF processor
from lightrag_integration.pdf_processor import (
    BiomedicalPDFProcessor,
    BiomedicalPDFProcessorError,
    PDFValidationError,
    PDFProcessingTimeoutError,
    PDFMemoryError,
    PDFFileAccessError,
    PDFContentError
)


class MemoryProfiler:
    """Memory usage profiler for detailed monitoring during benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = 0
        self.peak_memory = 0
        self.memory_samples = []
        self.monitoring = False
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.memory_samples = [self.initial_memory]
        self.monitoring = True
    
    def sample_memory(self):
        """Take a memory sample."""
        if not self.monitoring:
            return
        
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_samples.append(current_memory)
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return memory statistics."""
        if not self.monitoring:
            return {}
        
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_samples.append(final_memory)
        self.monitoring = False
        
        return {
            'initial_mb': round(self.initial_memory, 2),
            'final_mb': round(final_memory, 2),
            'peak_mb': round(self.peak_memory, 2),
            'increase_mb': round(final_memory - self.initial_memory, 2),
            'peak_increase_mb': round(self.peak_memory - self.initial_memory, 2),
            'average_mb': round(statistics.mean(self.memory_samples), 2),
            'samples_count': len(self.memory_samples)
        }


class PDFProcessingBenchmark:
    """
    Comprehensive benchmark suite for BiomedicalPDFProcessor.
    
    This class provides a complete testing framework that evaluates:
    - Processing performance and timing
    - Memory usage patterns
    - Text extraction quality
    - Error handling robustness
    - Metadata extraction accuracy
    - Preprocessing effectiveness
    """
    
    def __init__(self, 
                 papers_dir: Union[str, Path] = "papers/",
                 output_dir: Union[str, Path] = "benchmark_results/",
                 verbose: bool = True):
        """
        Initialize the benchmark suite.
        
        Args:
            papers_dir: Directory containing PDF files to benchmark
            output_dir: Directory to save benchmark results
            verbose: Enable verbose logging
        """
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.processor = BiomedicalPDFProcessor(logger=self.logger)
        self.memory_profiler = MemoryProfiler()
        
        # Benchmark results storage
        self.results = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'papers_directory': str(self.papers_dir.absolute()),
                'processor_config': self.processor.get_processing_stats()
            },
            'pdf_files': [],
            'performance_metrics': {},
            'quality_metrics': {},
            'error_handling_tests': {},
            'recommendations': [],
            'summary': {}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"{__name__}.benchmark")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.
        
        Returns:
            Dict[str, Any]: Complete benchmark results
        """
        self.logger.info("Starting comprehensive PDF processing benchmark")
        self.logger.info(f"Papers directory: {self.papers_dir.absolute()}")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
        
        try:
            # Step 1: Discover and validate PDF files
            pdf_files = await self._discover_pdf_files()
            if not pdf_files:
                self.logger.error("No PDF files found for benchmarking")
                return self.results
            
            # Step 2: Basic file validation and information gathering
            await self._validate_pdf_files(pdf_files)
            
            # Step 3: Performance benchmarks
            await self._run_performance_benchmarks(pdf_files)
            
            # Step 4: Quality assessment tests
            await self._run_quality_assessments(pdf_files)
            
            # Step 5: Error handling and robustness tests
            await self._run_error_handling_tests(pdf_files)
            
            # Step 6: Memory stress tests
            await self._run_memory_stress_tests(pdf_files)
            
            # Step 7: Generate summary and recommendations
            await self._generate_summary_and_recommendations()
            
            # Step 8: Save results
            await self._save_results()
            
            self.logger.info("Comprehensive benchmark completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Benchmark failed with error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.results['error'] = str(e)
            self.results['traceback'] = traceback.format_exc()
            return self.results
    
    async def _discover_pdf_files(self) -> List[Path]:
        """Discover and list all PDF files in the papers directory."""
        self.logger.info(f"Discovering PDF files in {self.papers_dir}")
        
        if not self.papers_dir.exists():
            self.logger.warning(f"Papers directory does not exist: {self.papers_dir}")
            return []
        
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            self.logger.info(f"  - {pdf_file.name} ({pdf_file.stat().st_size / 1024 / 1024:.2f} MB)")
        
        return pdf_files
    
    async def _validate_pdf_files(self, pdf_files: List[Path]):
        """Validate PDF files and collect basic information."""
        self.logger.info("Validating PDF files and collecting basic information")
        
        for pdf_file in pdf_files:
            self.logger.info(f"Validating {pdf_file.name}")
            
            start_time = time.time()
            try:
                validation_result = self.processor.validate_pdf(pdf_file)
                validation_time = time.time() - start_time
                
                file_info = {
                    'filename': pdf_file.name,
                    'file_path': str(pdf_file.absolute()),
                    'file_size_mb': round(pdf_file.stat().st_size / 1024 / 1024, 2),
                    'validation_time_seconds': round(validation_time, 3),
                    'validation_result': validation_result,
                    'valid': validation_result.get('valid', False),
                    'pages': validation_result.get('pages', 0),
                    'encrypted': validation_result.get('encrypted', False),
                    'validation_error': validation_result.get('error')
                }
                
                if validation_result.get('metadata'):
                    file_info['metadata'] = validation_result['metadata']
                
                self.results['pdf_files'].append(file_info)
                
                if file_info['valid']:
                    self.logger.info(f"✓ {pdf_file.name} - Valid ({file_info['pages']} pages)")
                else:
                    self.logger.warning(f"✗ {pdf_file.name} - Invalid: {file_info['validation_error']}")
                    
            except Exception as e:
                self.logger.error(f"Validation failed for {pdf_file.name}: {e}")
                file_info = {
                    'filename': pdf_file.name,
                    'file_path': str(pdf_file.absolute()),
                    'valid': False,
                    'validation_error': str(e),
                    'validation_time_seconds': round(time.time() - start_time, 3)
                }
                self.results['pdf_files'].append(file_info)
    
    async def _run_performance_benchmarks(self, pdf_files: List[Path]):
        """Run comprehensive performance benchmarks."""
        self.logger.info("Running performance benchmarks")
        
        valid_files = [f for f in pdf_files if self._is_valid_pdf(f)]
        if not valid_files:
            self.logger.warning("No valid PDF files for performance testing")
            return
        
        performance_results = {
            'processing_times': [],
            'throughput_metrics': {},
            'page_processing_rates': [],
            'memory_usage_patterns': [],
            'preprocessing_impact': {}
        }
        
        # Test each PDF with different configurations
        for pdf_file in valid_files:
            self.logger.info(f"Performance testing {pdf_file.name}")
            
            # Test 1: Full processing with preprocessing
            await self._benchmark_single_pdf(pdf_file, preprocess=True, performance_results=performance_results)
            
            # Test 2: Processing without preprocessing (for comparison)
            await self._benchmark_single_pdf(pdf_file, preprocess=False, performance_results=performance_results)
            
            # Test 3: Page range processing (if PDF has multiple pages)
            if self._get_pdf_page_count(pdf_file) > 5:
                await self._benchmark_page_range_processing(pdf_file, performance_results)
        
        # Calculate aggregate metrics
        if performance_results['processing_times']:
            processing_times = [t['total_time'] for t in performance_results['processing_times']]
            performance_results['aggregate_metrics'] = {
                'average_processing_time': round(statistics.mean(processing_times), 3),
                'median_processing_time': round(statistics.median(processing_times), 3),
                'min_processing_time': round(min(processing_times), 3),
                'max_processing_time': round(max(processing_times), 3),
                'std_dev_processing_time': round(statistics.stdev(processing_times) if len(processing_times) > 1 else 0, 3)
            }
        
        self.results['performance_metrics'] = performance_results
    
    async def _benchmark_single_pdf(self, pdf_file: Path, preprocess: bool, performance_results: Dict):
        """Benchmark processing of a single PDF."""
        config_name = "with_preprocessing" if preprocess else "without_preprocessing"
        
        self.memory_profiler.start_monitoring()
        start_time = time.time()
        
        try:
            result = self.processor.extract_text_from_pdf(
                pdf_file, 
                preprocess_text=preprocess
            )
            
            end_time = time.time()
            memory_stats = self.memory_profiler.stop_monitoring()
            
            processing_time = end_time - start_time
            chars_per_second = len(result['text']) / processing_time if processing_time > 0 else 0
            pages_per_second = result['metadata']['pages'] / processing_time if processing_time > 0 else 0
            
            benchmark_result = {
                'pdf_name': pdf_file.name,
                'configuration': config_name,
                'total_time': round(processing_time, 3),
                'characters_extracted': len(result['text']),
                'pages_processed': result['metadata']['pages'],
                'chars_per_second': round(chars_per_second, 2),
                'pages_per_second': round(pages_per_second, 3),
                'memory_stats': memory_stats,
                'preprocessing_applied': preprocess,
                'processing_info': result['processing_info']
            }
            
            performance_results['processing_times'].append(benchmark_result)
            performance_results['page_processing_rates'].append(pages_per_second)
            performance_results['memory_usage_patterns'].append(memory_stats)
            
            self.logger.info(f"  {config_name}: {processing_time:.3f}s, {chars_per_second:.0f} chars/sec")
            
        except Exception as e:
            self.logger.error(f"Performance test failed for {pdf_file.name} ({config_name}): {e}")
    
    async def _benchmark_page_range_processing(self, pdf_file: Path, performance_results: Dict):
        """Benchmark processing of specific page ranges."""
        page_count = self._get_pdf_page_count(pdf_file)
        if page_count < 6:
            return
        
        # Test processing first 3 pages vs last 3 pages
        ranges_to_test = [
            (0, 3, "first_3_pages"),
            (page_count - 3, page_count, "last_3_pages"),
            (page_count // 2 - 1, page_count // 2 + 2, "middle_3_pages")
        ]
        
        for start_page, end_page, range_name in ranges_to_test:
            self.memory_profiler.start_monitoring()
            start_time = time.time()
            
            try:
                result = self.processor.extract_text_from_pdf(
                    pdf_file,
                    start_page=start_page,
                    end_page=end_page,
                    preprocess_text=True
                )
                
                end_time = time.time()
                memory_stats = self.memory_profiler.stop_monitoring()
                
                processing_time = end_time - start_time
                pages_processed = end_page - start_page
                
                benchmark_result = {
                    'pdf_name': pdf_file.name,
                    'configuration': f"page_range_{range_name}",
                    'page_range': f"{start_page}-{end_page}",
                    'total_time': round(processing_time, 3),
                    'characters_extracted': len(result['text']),
                    'pages_processed': pages_processed,
                    'pages_per_second': round(pages_processed / processing_time, 3),
                    'memory_stats': memory_stats
                }
                
                performance_results['processing_times'].append(benchmark_result)
                
            except Exception as e:
                self.logger.error(f"Page range test failed for {pdf_file.name} ({range_name}): {e}")
    
    async def _run_quality_assessments(self, pdf_files: List[Path]):
        """Run text extraction quality assessments."""
        self.logger.info("Running text extraction quality assessments")
        
        valid_files = [f for f in pdf_files if self._is_valid_pdf(f)]
        if not valid_files:
            self.logger.warning("No valid PDF files for quality testing")
            return
        
        quality_results = {
            'text_completeness': [],
            'preprocessing_effectiveness': [],
            'metadata_completeness': [],
            'encoding_issues': [],
            'content_analysis': []
        }
        
        for pdf_file in valid_files:
            self.logger.info(f"Quality assessment for {pdf_file.name}")
            
            try:
                # Extract text with and without preprocessing for comparison
                result_with_preprocessing = self.processor.extract_text_from_pdf(pdf_file, preprocess_text=True)
                result_without_preprocessing = self.processor.extract_text_from_pdf(pdf_file, preprocess_text=False)
                
                # Analyze text completeness
                completeness_metrics = self._analyze_text_completeness(result_with_preprocessing)
                quality_results['text_completeness'].append({
                    'pdf_name': pdf_file.name,
                    **completeness_metrics
                })
                
                # Analyze preprocessing effectiveness
                preprocessing_metrics = self._analyze_preprocessing_effectiveness(
                    result_without_preprocessing['text'],
                    result_with_preprocessing['text']
                )
                quality_results['preprocessing_effectiveness'].append({
                    'pdf_name': pdf_file.name,
                    **preprocessing_metrics
                })
                
                # Analyze metadata completeness
                metadata_metrics = self._analyze_metadata_completeness(result_with_preprocessing['metadata'])
                quality_results['metadata_completeness'].append({
                    'pdf_name': pdf_file.name,
                    **metadata_metrics
                })
                
                # Check for encoding issues
                encoding_metrics = self._analyze_encoding_quality(result_with_preprocessing['text'])
                quality_results['encoding_issues'].append({
                    'pdf_name': pdf_file.name,
                    **encoding_metrics
                })
                
                # Content-specific analysis (biomedical terms, scientific notation, etc.)
                content_metrics = self._analyze_biomedical_content(result_with_preprocessing['text'])
                quality_results['content_analysis'].append({
                    'pdf_name': pdf_file.name,
                    **content_metrics
                })
                
            except Exception as e:
                self.logger.error(f"Quality assessment failed for {pdf_file.name}: {e}")
        
        self.results['quality_metrics'] = quality_results
    
    async def _run_error_handling_tests(self, pdf_files: List[Path]):
        """Run comprehensive error handling and robustness tests."""
        self.logger.info("Running error handling and robustness tests")
        
        error_test_results = {
            'timeout_handling': [],
            'memory_limit_handling': [],
            'file_corruption_simulation': [],
            'invalid_input_handling': [],
            'edge_case_handling': []
        }
        
        valid_files = [f for f in pdf_files if self._is_valid_pdf(f)]
        
        # Test 1: Timeout handling with very short timeouts
        await self._test_timeout_handling(valid_files, error_test_results)
        
        # Test 2: Memory limit handling with very low limits
        await self._test_memory_limit_handling(valid_files, error_test_results)
        
        # Test 3: Invalid input handling
        await self._test_invalid_input_handling(error_test_results)
        
        # Test 4: Edge case handling
        await self._test_edge_cases(valid_files, error_test_results)
        
        self.results['error_handling_tests'] = error_test_results
    
    async def _test_timeout_handling(self, valid_files: List[Path], error_test_results: Dict):
        """Test timeout handling with artificially short timeouts."""
        if not valid_files:
            return
        
        # Create processor with very short timeout
        timeout_processor = BiomedicalPDFProcessor(
            logger=self.logger,
            processing_timeout=1  # 1 second timeout
        )
        
        for pdf_file in valid_files:
            try:
                start_time = time.time()
                result = timeout_processor.extract_text_from_pdf(pdf_file)
                processing_time = time.time() - start_time
                
                error_test_results['timeout_handling'].append({
                    'pdf_name': pdf_file.name,
                    'expected_timeout': True,
                    'actual_timeout': False,
                    'processing_time': round(processing_time, 3),
                    'result': 'Unexpected success'
                })
                
            except PDFProcessingTimeoutError as e:
                processing_time = time.time() - start_time
                error_test_results['timeout_handling'].append({
                    'pdf_name': pdf_file.name,
                    'expected_timeout': True,
                    'actual_timeout': True,
                    'processing_time': round(processing_time, 3),
                    'error_message': str(e),
                    'result': 'Correct timeout handling'
                })
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_test_results['timeout_handling'].append({
                    'pdf_name': pdf_file.name,
                    'expected_timeout': True,
                    'actual_timeout': False,
                    'processing_time': round(processing_time, 3),
                    'error_message': str(e),
                    'result': 'Unexpected error type'
                })
    
    async def _test_memory_limit_handling(self, valid_files: List[Path], error_test_results: Dict):
        """Test memory limit handling with very low memory limits."""
        if not valid_files:
            return
        
        # Create processor with very low memory limit
        memory_processor = BiomedicalPDFProcessor(
            logger=self.logger,
            memory_limit_mb=1  # 1 MB limit (very low)
        )
        
        for pdf_file in valid_files:
            try:
                start_time = time.time()
                result = memory_processor.extract_text_from_pdf(pdf_file)
                processing_time = time.time() - start_time
                
                error_test_results['memory_limit_handling'].append({
                    'pdf_name': pdf_file.name,
                    'memory_limit_mb': 1,
                    'processing_time': round(processing_time, 3),
                    'result': 'Processing succeeded (may have high memory usage warnings)'
                })
                
            except PDFMemoryError as e:
                processing_time = time.time() - start_time
                error_test_results['memory_limit_handling'].append({
                    'pdf_name': pdf_file.name,
                    'memory_limit_mb': 1,
                    'processing_time': round(processing_time, 3),
                    'error_message': str(e),
                    'result': 'Correct memory limit handling'
                })
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_test_results['memory_limit_handling'].append({
                    'pdf_name': pdf_file.name,
                    'memory_limit_mb': 1,
                    'processing_time': round(processing_time, 3),
                    'error_message': str(e),
                    'result': 'Unexpected error type'
                })
    
    async def _test_invalid_input_handling(self, error_test_results: Dict):
        """Test handling of various invalid inputs."""
        invalid_inputs = [
            ("/nonexistent/file.pdf", "Non-existent file"),
            ("", "Empty path"),
            ("/dev/null", "Non-PDF file"),
            (str(self.papers_dir), "Directory instead of file")
        ]
        
        for invalid_path, description in invalid_inputs:
            try:
                start_time = time.time()
                result = self.processor.extract_text_from_pdf(invalid_path)
                processing_time = time.time() - start_time
                
                error_test_results['invalid_input_handling'].append({
                    'input_path': invalid_path,
                    'description': description,
                    'processing_time': round(processing_time, 3),
                    'result': 'Unexpected success',
                    'expected_error': True,
                    'actual_error': False
                })
                
            except (FileNotFoundError, PDFValidationError, PDFFileAccessError) as e:
                processing_time = time.time() - start_time
                error_test_results['invalid_input_handling'].append({
                    'input_path': invalid_path,
                    'description': description,
                    'processing_time': round(processing_time, 3),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'result': 'Correct error handling',
                    'expected_error': True,
                    'actual_error': True
                })
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_test_results['invalid_input_handling'].append({
                    'input_path': invalid_path,
                    'description': description,
                    'processing_time': round(processing_time, 3),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'result': 'Unexpected error type',
                    'expected_error': True,
                    'actual_error': True
                })
    
    async def _test_edge_cases(self, valid_files: List[Path], error_test_results: Dict):
        """Test various edge cases."""
        if not valid_files:
            return
        
        # Test invalid page ranges
        pdf_file = valid_files[0]
        page_count = self._get_pdf_page_count(pdf_file)
        
        edge_cases = [
            (-1, 1, "Negative start page"),
            (page_count, page_count + 1, "Start page beyond PDF"),
            (0, page_count + 10, "End page beyond PDF"),
            (5, 2, "Start page > end page")
        ]
        
        for start_page, end_page, description in edge_cases:
            try:
                start_time = time.time()
                result = self.processor.extract_text_from_pdf(
                    pdf_file, 
                    start_page=start_page, 
                    end_page=end_page
                )
                processing_time = time.time() - start_time
                
                error_test_results['edge_case_handling'].append({
                    'pdf_name': pdf_file.name,
                    'test_case': description,
                    'start_page': start_page,
                    'end_page': end_page,
                    'processing_time': round(processing_time, 3),
                    'result': 'Unexpected success',
                    'expected_error': True,
                    'actual_error': False
                })
                
            except BiomedicalPDFProcessorError as e:
                processing_time = time.time() - start_time
                error_test_results['edge_case_handling'].append({
                    'pdf_name': pdf_file.name,
                    'test_case': description,
                    'start_page': start_page,
                    'end_page': end_page,
                    'processing_time': round(processing_time, 3),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'result': 'Correct error handling',
                    'expected_error': True,
                    'actual_error': True
                })
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_test_results['edge_case_handling'].append({
                    'pdf_name': pdf_file.name,
                    'test_case': description,
                    'start_page': start_page,
                    'end_page': end_page,
                    'processing_time': round(processing_time, 3),
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'result': 'Unexpected error type',
                    'expected_error': True,
                    'actual_error': True
                })
    
    async def _run_memory_stress_tests(self, pdf_files: List[Path]):
        """Run memory stress tests with multiple concurrent processings."""
        self.logger.info("Running memory stress tests")
        
        valid_files = [f for f in pdf_files if self._is_valid_pdf(f)]
        if not valid_files:
            self.logger.warning("No valid PDF files for memory stress testing")
            return
        
        stress_test_results = {
            'concurrent_processing': [],
            'repeated_processing': [],
            'memory_leak_detection': []
        }
        
        # Test 1: Repeated processing of the same file
        if valid_files:
            await self._test_repeated_processing(valid_files[0], stress_test_results)
        
        # Test 2: Memory leak detection over multiple cycles
        if valid_files:
            await self._test_memory_leak_detection(valid_files, stress_test_results)
        
        self.results['memory_stress_tests'] = stress_test_results
    
    async def _test_repeated_processing(self, pdf_file: Path, stress_test_results: Dict):
        """Test repeated processing of the same PDF to detect memory issues."""
        num_iterations = 5
        memory_samples = []
        
        self.logger.info(f"Running repeated processing test ({num_iterations} iterations)")
        
        for i in range(num_iterations):
            self.memory_profiler.start_monitoring()
            start_time = time.time()
            
            try:
                result = self.processor.extract_text_from_pdf(pdf_file)
                processing_time = time.time() - start_time
                memory_stats = self.memory_profiler.stop_monitoring()
                
                memory_samples.append(memory_stats['peak_mb'])
                
                stress_test_results['repeated_processing'].append({
                    'iteration': i + 1,
                    'pdf_name': pdf_file.name,
                    'processing_time': round(processing_time, 3),
                    'memory_stats': memory_stats,
                    'characters_extracted': len(result['text'])
                })
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Repeated processing iteration {i+1} failed: {e}")
        
        # Analyze memory trend
        if len(memory_samples) > 1:
            memory_trend = "increasing" if memory_samples[-1] > memory_samples[0] * 1.1 else "stable"
            stress_test_results['memory_analysis'] = {
                'initial_memory_mb': memory_samples[0],
                'final_memory_mb': memory_samples[-1],
                'max_memory_mb': max(memory_samples),
                'memory_trend': memory_trend,
                'potential_memory_leak': memory_trend == "increasing"
            }
    
    async def _test_memory_leak_detection(self, valid_files: List[Path], stress_test_results: Dict):
        """Test for memory leaks by processing multiple files in sequence."""
        if len(valid_files) < 1:
            return
        
        # Process each file multiple times
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_progression = [initial_memory]
        
        for cycle in range(3):  # 3 cycles through all files
            for pdf_file in valid_files:
                try:
                    result = self.processor.extract_text_from_pdf(pdf_file)
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_progression.append(current_memory)
                    
                    stress_test_results['memory_leak_detection'].append({
                        'cycle': cycle + 1,
                        'pdf_name': pdf_file.name,
                        'memory_mb': round(current_memory, 2),
                        'memory_increase_mb': round(current_memory - initial_memory, 2)
                    })
                    
                    # Small delay
                    await asyncio.sleep(0.05)
                    
                except Exception as e:
                    self.logger.error(f"Memory leak test failed for {pdf_file.name}: {e}")
        
        # Analysis
        if len(memory_progression) > 1:
            final_memory = memory_progression[-1]
            memory_increase = final_memory - initial_memory
            
            stress_test_results['memory_leak_analysis'] = {
                'initial_memory_mb': round(initial_memory, 2),
                'final_memory_mb': round(final_memory, 2),
                'total_increase_mb': round(memory_increase, 2),
                'max_memory_mb': round(max(memory_progression), 2),
                'potential_leak': memory_increase > 50,  # More than 50MB increase
                'memory_progression': [round(m, 2) for m in memory_progression[-10:]]  # Last 10 samples
            }
    
    async def _generate_summary_and_recommendations(self):
        """Generate benchmark summary and recommendations."""
        self.logger.info("Generating summary and recommendations")
        
        # Count valid vs invalid files
        valid_count = sum(1 for f in self.results['pdf_files'] if f.get('valid', False))
        total_count = len(self.results['pdf_files'])
        
        # Performance summary
        performance_summary = {}
        if self.results['performance_metrics'].get('processing_times'):
            times = [t['total_time'] for t in self.results['performance_metrics']['processing_times']]
            performance_summary = {
                'total_files_tested': len(set(t['pdf_name'] for t in self.results['performance_metrics']['processing_times'])),
                'average_processing_time': round(statistics.mean(times), 3),
                'fastest_processing_time': round(min(times), 3),
                'slowest_processing_time': round(max(times), 3)
            }
        
        # Error handling summary
        error_summary = {}
        if self.results.get('error_handling_tests'):
            error_tests = self.results['error_handling_tests']
            error_summary = {
                'timeout_tests': len(error_tests.get('timeout_handling', [])),
                'memory_tests': len(error_tests.get('memory_limit_handling', [])),
                'invalid_input_tests': len(error_tests.get('invalid_input_handling', [])),
                'edge_case_tests': len(error_tests.get('edge_case_handling', []))
            }
        
        self.results['summary'] = {
            'files_discovered': total_count,
            'files_valid': valid_count,
            'files_invalid': total_count - valid_count,
            'performance_summary': performance_summary,
            'error_handling_summary': error_summary,
            'benchmark_completed': datetime.now().isoformat()
        }
        
        # Generate recommendations
        recommendations = []
        
        if total_count < 5:
            recommendations.append({
                'type': 'sample_size',
                'priority': 'high',
                'recommendation': f"Only {total_count} PDF(s) found. For comprehensive benchmarking, obtain 5+ diverse biomedical PDFs.",
                'suggested_sources': [
                    "PubMed Central (PMC) - https://www.ncbi.nlm.nih.gov/pmc/",
                    "bioRxiv preprints - https://www.biorxiv.org/",
                    "medRxiv preprints - https://www.medrxiv.org/",
                    "PLOS journals - https://plos.org/",
                    "Nature journals - https://www.nature.com/",
                    "Clinical metabolomics journals from major publishers"
                ]
            })
        
        if valid_count < total_count:
            recommendations.append({
                'type': 'file_validation',
                'priority': 'medium',
                'recommendation': f"{total_count - valid_count} PDF(s) failed validation. Review and fix corrupted or invalid files.",
                'action': "Check file integrity and permissions"
            })
        
        # Performance recommendations based on processing times
        if performance_summary.get('slowest_processing_time', 0) > 30:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium',
                'recommendation': "Some PDFs take >30 seconds to process. Consider optimizing for large files or increasing timeout limits.",
                'action': "Review timeout settings and memory limits"
            })
        
        # Memory recommendations
        if self.results.get('memory_stress_tests', {}).get('memory_leak_analysis', {}).get('potential_leak', False):
            recommendations.append({
                'type': 'memory',
                'priority': 'high',
                'recommendation': "Potential memory leak detected. Review memory management in PDF processing.",
                'action': "Investigate memory usage patterns and implement garbage collection"
            })
        
        # Quality recommendations
        if self.results.get('quality_metrics', {}).get('encoding_issues'):
            encoding_issues = [e for e in self.results['quality_metrics']['encoding_issues'] if e.get('issues_detected', 0) > 0]
            if encoding_issues:
                recommendations.append({
                    'type': 'text_quality',
                    'priority': 'medium',
                    'recommendation': f"{len(encoding_issues)} PDF(s) have text encoding issues. Review character encoding handling.",
                    'action': "Enhance Unicode and special character processing"
                })
        
        self.results['recommendations'] = recommendations
    
    async def _save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save human-readable report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_human_readable_report())
        
        self.logger.info(f"Results saved to:")
        self.logger.info(f"  JSON: {json_file}")
        self.logger.info(f"  Report: {report_file}")
    
    def _generate_human_readable_report(self) -> str:
        """Generate a human-readable benchmark report."""
        report = []
        report.append("="*80)
        report.append("BIOMEDICAL PDF PROCESSOR - COMPREHENSIVE BENCHMARK REPORT")
        report.append("="*80)
        report.append(f"Generated: {self.results['benchmark_info']['timestamp']}")
        report.append(f"Papers Directory: {self.results['benchmark_info']['papers_directory']}")
        report.append("")
        
        # Summary section
        summary = self.results.get('summary', {})
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total PDF files discovered: {summary.get('files_discovered', 0)}")
        report.append(f"Valid PDF files: {summary.get('files_valid', 0)}")
        report.append(f"Invalid PDF files: {summary.get('files_invalid', 0)}")
        report.append("")
        
        # Performance summary
        perf_summary = summary.get('performance_summary', {})
        if perf_summary:
            report.append("PERFORMANCE HIGHLIGHTS")
            report.append("-" * 40)
            report.append(f"Files tested: {perf_summary.get('total_files_tested', 0)}")
            report.append(f"Average processing time: {perf_summary.get('average_processing_time', 'N/A')} seconds")
            report.append(f"Fastest processing: {perf_summary.get('fastest_processing_time', 'N/A')} seconds")
            report.append(f"Slowest processing: {perf_summary.get('slowest_processing_time', 'N/A')} seconds")
            report.append("")
        
        # Detailed file information
        if self.results.get('pdf_files'):
            report.append("PDF FILES ANALYZED")
            report.append("-" * 40)
            for file_info in self.results['pdf_files']:
                status = "✓ VALID" if file_info.get('valid', False) else "✗ INVALID"
                report.append(f"{file_info.get('filename', 'Unknown')}: {status}")
                if file_info.get('file_size_mb'):
                    report.append(f"  Size: {file_info['file_size_mb']} MB")
                if file_info.get('pages'):
                    report.append(f"  Pages: {file_info['pages']}")
                if not file_info.get('valid', False) and file_info.get('validation_error'):
                    report.append(f"  Error: {file_info['validation_error']}")
                report.append("")
        
        # Recommendations
        if self.results.get('recommendations'):
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(self.results['recommendations'], 1):
                priority = rec.get('priority', 'medium').upper()
                report.append(f"{i}. [{priority}] {rec.get('recommendation', '')}")
                if rec.get('action'):
                    report.append(f"   Action: {rec['action']}")
                if rec.get('suggested_sources'):
                    report.append("   Suggested sources:")
                    for source in rec['suggested_sources']:
                        report.append(f"     • {source}")
                report.append("")
        
        # Detailed performance metrics
        if self.results.get('performance_metrics', {}).get('processing_times'):
            report.append("DETAILED PERFORMANCE RESULTS")
            report.append("-" * 40)
            for result in self.results['performance_metrics']['processing_times']:
                report.append(f"File: {result.get('pdf_name', 'Unknown')}")
                report.append(f"Configuration: {result.get('configuration', 'Unknown')}")
                report.append(f"Processing time: {result.get('total_time', 'N/A')} seconds")
                report.append(f"Characters extracted: {result.get('characters_extracted', 'N/A'):,}")
                report.append(f"Pages processed: {result.get('pages_processed', 'N/A')}")
                chars_per_sec = result.get('chars_per_second', 'N/A')
                if isinstance(chars_per_sec, (int, float)):
                    report.append(f"Throughput: {chars_per_sec:.0f} chars/sec")
                else:
                    report.append(f"Throughput: {chars_per_sec} chars/sec")
                if result.get('memory_stats'):
                    memory = result['memory_stats']
                    report.append(f"Memory usage: {memory.get('initial_mb', 'N/A')} → {memory.get('peak_mb', 'N/A')} MB (peak)")
                report.append("")
        
        # Error handling results
        if self.results.get('error_handling_tests'):
            report.append("ERROR HANDLING TEST RESULTS")
            report.append("-" * 40)
            error_tests = self.results['error_handling_tests']
            
            for test_type, results in error_tests.items():
                if results:
                    report.append(f"{test_type.replace('_', ' ').title()}: {len(results)} tests")
                    
            report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)
    
    # Utility methods
    def _is_valid_pdf(self, pdf_file: Path) -> bool:
        """Check if PDF file is valid based on our validation results."""
        for file_info in self.results['pdf_files']:
            if Path(file_info['file_path']).name == pdf_file.name:
                return file_info.get('valid', False)
        return False
    
    def _get_pdf_page_count(self, pdf_file: Path) -> int:
        """Get page count for a PDF file."""
        for file_info in self.results['pdf_files']:
            if Path(file_info['file_path']).name == pdf_file.name:
                return file_info.get('pages', 0)
        return 0
    
    def _analyze_text_completeness(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text extraction completeness."""
        text = result.get('text', '')
        page_texts = result.get('page_texts', [])
        
        # Basic text statistics
        total_chars = len(text)
        total_words = len(text.split()) if text else 0
        total_lines = len(text.splitlines()) if text else 0
        
        # Page-level analysis
        non_empty_pages = sum(1 for page_text in page_texts if page_text.strip())
        empty_pages = len(page_texts) - non_empty_pages
        
        # Content density analysis
        chars_per_page = total_chars / len(page_texts) if page_texts else 0
        
        return {
            'total_characters': total_chars,
            'total_words': total_words,
            'total_lines': total_lines,
            'pages_with_content': non_empty_pages,
            'empty_pages': empty_pages,
            'average_chars_per_page': round(chars_per_page, 1),
            'content_extraction_rate': round(non_empty_pages / len(page_texts) * 100, 1) if page_texts else 0
        }
    
    def _analyze_preprocessing_effectiveness(self, raw_text: str, processed_text: str) -> Dict[str, Any]:
        """Analyze the effectiveness of preprocessing."""
        if not raw_text or not processed_text:
            return {'preprocessing_effective': False, 'reason': 'Empty text'}
        
        # Character count changes
        raw_chars = len(raw_text)
        processed_chars = len(processed_text)
        char_reduction = raw_chars - processed_chars
        char_reduction_percent = (char_reduction / raw_chars * 100) if raw_chars > 0 else 0
        
        # Line break analysis
        raw_lines = len(raw_text.splitlines())
        processed_lines = len(processed_text.splitlines())
        
        # Word count changes
        raw_words = len(raw_text.split())
        processed_words = len(processed_text.split())
        
        # Scientific notation preservation check
        scientific_patterns = [
            r'p\s*[<>=]\s*0\.\d+',  # p-values
            r'\d+\.?\d*\s*[×x]\s*10\s*[⁻−-]\s*\d+',  # scientific notation
            r'[A-Z][a-z]?\d+',  # chemical formulas
        ]
        
        scientific_preservation = 0
        for pattern in scientific_patterns:
            import re
            raw_matches = len(re.findall(pattern, raw_text, re.IGNORECASE))
            processed_matches = len(re.findall(pattern, processed_text, re.IGNORECASE))
            if raw_matches > 0:
                preservation_rate = processed_matches / raw_matches
                scientific_preservation += preservation_rate
        
        scientific_preservation = scientific_preservation / len(scientific_patterns) if scientific_patterns else 0
        
        return {
            'character_reduction': char_reduction,
            'character_reduction_percent': round(char_reduction_percent, 2),
            'raw_line_count': raw_lines,
            'processed_line_count': processed_lines,
            'raw_word_count': raw_words,
            'processed_word_count': processed_words,
            'scientific_notation_preservation': round(scientific_preservation * 100, 1),
            'preprocessing_effective': char_reduction_percent > 5 and scientific_preservation > 0.8
        }
    
    def _analyze_metadata_completeness(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metadata extraction completeness."""
        required_fields = ['filename', 'file_path', 'pages', 'file_size_bytes']
        optional_fields = ['title', 'author', 'subject', 'creator', 'producer', 'creation_date', 'modification_date']
        
        required_present = sum(1 for field in required_fields if metadata.get(field) is not None)
        optional_present = sum(1 for field in optional_fields if metadata.get(field) is not None)
        
        completeness_score = (required_present / len(required_fields)) * 0.7 + (optional_present / len(optional_fields)) * 0.3
        
        return {
            'required_fields_present': required_present,
            'required_fields_total': len(required_fields),
            'optional_fields_present': optional_present,
            'optional_fields_total': len(optional_fields),
            'completeness_score': round(completeness_score * 100, 1),
            'has_title': bool(metadata.get('title')),
            'has_author': bool(metadata.get('author')),
            'has_dates': bool(metadata.get('creation_date') or metadata.get('modification_date'))
        }
    
    def _analyze_encoding_quality(self, text: str) -> Dict[str, Any]:
        """Analyze text encoding quality and detect issues."""
        if not text:
            return {'issues_detected': 0}
        
        # Check for common encoding issues
        encoding_issues = 0
        issue_types = []
        
        # Control characters (excluding normal whitespace)
        import re
        control_chars = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', text))
        if control_chars > 0:
            encoding_issues += control_chars
            issue_types.append(f'control_characters: {control_chars}')
        
        # Replacement characters (indicating encoding problems)
        replacement_chars = text.count('\ufffd')
        if replacement_chars > 0:
            encoding_issues += replacement_chars
            issue_types.append(f'replacement_characters: {replacement_chars}')
        
        # Suspicious character sequences
        suspicious_patterns = [
            r'â€™',  # Common smart quote encoding issue
            r'â€œ',  # Common quote encoding issue
            r'â€',   # Common em dash encoding issue
            r'Ã¡',   # Common accented character issue
            r'Ã©',   # Common accented character issue
        ]
        
        suspicious_count = 0
        for pattern in suspicious_patterns:
            matches = len(re.findall(pattern, text))
            if matches > 0:
                suspicious_count += matches
                issue_types.append(f'{pattern}: {matches}')
        
        encoding_issues += suspicious_count
        
        return {
            'issues_detected': encoding_issues,
            'control_characters': control_chars,
            'replacement_characters': replacement_chars,
            'suspicious_sequences': suspicious_count,
            'issue_details': issue_types,
            'text_length': len(text),
            'issue_rate_percent': round((encoding_issues / len(text)) * 100, 4) if text else 0
        }
    
    def _analyze_biomedical_content(self, text: str) -> Dict[str, Any]:
        """Analyze biomedical content quality and preservation."""
        if not text:
            return {'biomedical_content_detected': False}
        
        import re
        
        # Biomedical term patterns
        patterns = {
            'p_values': r'\bp\s*[<>=]\s*0\.\d+',
            'statistical_values': r'\b(mean|median|std|CI|confidence)\b',
            'chemical_formulas': r'\b[A-Z][a-z]?\d+',
            'units': r'\b\d+\.?\d*\s*(mg|μg|ng|ml|μl|mM|μM|nM|°C|pH)\b',
            'techniques': r'\b(HPLC|LC-MS|GC-MS|NMR|PCR|qPCR|ELISA|Western)\b',
            'genes_proteins': r'\b[A-Z]{2,}[-_]?\d*\b',
            'citations': r'\[\s*\d+\s*[-,]?\s*\d*\s*\]',
            'figures_tables': r'\b(Figure|Fig\.|Table)\s+\d+',
        }
        
        content_analysis = {}
        total_biomedical_terms = 0
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            count = len(matches)
            content_analysis[category] = {
                'count': count,
                'examples': matches[:5] if matches else []  # First 5 examples
            }
            total_biomedical_terms += count
        
        # Overall biomedical content score
        text_words = len(text.split()) if text else 0
        biomedical_density = (total_biomedical_terms / text_words * 100) if text_words > 0 else 0
        
        return {
            'biomedical_content_detected': total_biomedical_terms > 0,
            'total_biomedical_terms': total_biomedical_terms,
            'biomedical_density_percent': round(biomedical_density, 2),
            'content_categories': content_analysis,
            'content_quality_score': min(100, biomedical_density * 10)  # Scale to 0-100
        }


async def main():
    """Main function to run the benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark BiomedicalPDFProcessor')
    parser.add_argument('--papers-dir', default='papers/', help='Directory containing PDF files')
    parser.add_argument('--output-dir', default='benchmark_results/', help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = PDFProcessingBenchmark(
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Run comprehensive benchmark
    results = await benchmark.run_comprehensive_benchmark()
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETED")
    print("="*80)
    
    summary = results.get('summary', {})
    print(f"Files processed: {summary.get('files_valid', 0)}/{summary.get('files_discovered', 0)}")
    
    perf_summary = summary.get('performance_summary', {})
    if perf_summary:
        print(f"Average processing time: {perf_summary.get('average_processing_time', 'N/A')} seconds")
    
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations: {len(recommendations)} items")
        for rec in recommendations[:3]:  # Show top 3
            print(f"  • [{rec.get('priority', 'medium').upper()}] {rec.get('recommendation', '')}")
    
    print(f"\nDetailed results saved to: {benchmark.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())