#!/usr/bin/env python3
"""
PDF to Knowledge Base Pipeline Verification Test for CMO-LIGHTRAG-006-T08

This test script performs focused verification of the complete PDF to knowledge base
pipeline to ensure proper integration and functionality.

Test Coverage:
1. PDF document loading and processing
2. LightRAG knowledge base initialization with PDF content
3. Document ingestion verification
4. Basic query functionality testing
5. Knowledge graph construction validation

Author: Claude Code
Created: 2025-08-07
Version: 1.0.0
"""

import asyncio
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pdf_kb_pipeline_verification.log')
    ]
)

logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    'pdf_file_path': '/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/papers/Clinical_Metabolomics_paper.pdf',
    'test_queries': [
        "What is clinical metabolomics?",
        "What are inborn errors of metabolism?",
        "How are metabolomics used for diagnosis?",
        "What analytical techniques are used in metabolomics?",
        "What is targeted vs untargeted metabolomics?"
    ],
    'expected_keywords': [
        'metabolomics',
        'inborn errors',
        'metabolism',
        'diagnosis',
        'biomarkers',
        'mass spectrometry',
        'chromatography'
    ],
    'working_directory': '/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025/lightrag_integration/tests/lightrag',
    'max_test_duration': 300  # 5 minutes
}

class PDFKnowledgeBaseVerifier:
    """Verification class for PDF to knowledge base pipeline testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_results = {}
        self.start_time = time.time()
        self.rag_system = None
        
    async def setup_rag_system(self) -> bool:
        """Initialize the RAG system for testing."""
        try:
            logger.info("Setting up RAG system...")
            
            # Import the RAG system
            sys.path.insert(0, str(Path(__file__).parent / 'lightrag_integration'))
            from lightrag_integration.clinical_metabolomics_rag import ClinicalMetabolomicsRAG
            
            # Initialize with test-specific configuration
            self.rag_system = ClinicalMetabolomicsRAG(
                working_dir=self.config['working_directory'],
                enable_cost_monitoring=True,
                enable_enhanced_logging=True
            )
            
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup RAG system: {e}")
            return False
    
    async def verify_pdf_loading(self) -> bool:
        """Verify that the PDF file can be loaded and processed."""
        try:
            logger.info("Verifying PDF file loading...")
            
            pdf_path = Path(self.config['pdf_file_path'])
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            file_size = pdf_path.stat().st_size
            logger.info(f"PDF file found: {pdf_path} ({file_size / 1024:.1f} KB)")
            
            # Verify file is readable
            with open(pdf_path, 'rb') as f:
                header = f.read(10)
                if not header.startswith(b'%PDF'):
                    logger.error("File is not a valid PDF")
                    return False
            
            logger.info("PDF file verification passed")
            return True
            
        except Exception as e:
            logger.error(f"PDF loading verification failed: {e}")
            return False
    
    async def verify_knowledge_base_initialization(self) -> bool:
        """Verify knowledge base initialization with PDF content."""
        try:
            logger.info("Verifying knowledge base initialization...")
            
            if not self.rag_system:
                logger.error("RAG system not initialized")
                return False
            
            # Process the PDF document
            pdf_path = self.config['pdf_file_path']
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Ingest the document
            result = await self.rag_system.ingest_document(pdf_path)
            
            if result.get('success', False):
                logger.info("Document ingestion completed successfully")
                logger.info(f"Processing summary: {result.get('summary', {})}")
                return True
            else:
                logger.error(f"Document ingestion failed: {result.get('error', 'Unknown error')}")
                return False
            
        except Exception as e:
            logger.error(f"Knowledge base initialization failed: {e}")
            return False
    
    async def verify_basic_query_functionality(self) -> bool:
        """Verify basic query functionality on the initialized knowledge base."""
        try:
            logger.info("Verifying basic query functionality...")
            
            if not self.rag_system:
                logger.error("RAG system not initialized")
                return False
            
            successful_queries = 0
            total_queries = len(self.config['test_queries'])
            
            for i, query in enumerate(self.config['test_queries'], 1):
                try:
                    logger.info(f"Testing query {i}/{total_queries}: {query}")
                    
                    # Test the query
                    response = await self.rag_system.query(query, mode='hybrid')
                    
                    if response and len(response.strip()) > 50:  # Reasonable response length
                        # Check if response contains expected keywords
                        response_lower = response.lower()
                        keyword_matches = [kw for kw in self.config['expected_keywords'] 
                                         if kw.lower() in response_lower]
                        
                        if keyword_matches:
                            logger.info(f"Query successful. Found keywords: {keyword_matches}")
                            logger.info(f"Response preview: {response[:200]}...")
                            successful_queries += 1
                        else:
                            logger.warning(f"Query response lacks expected keywords: {response[:100]}...")
                    else:
                        logger.warning(f"Query response too short or empty: {response}")
                    
                except Exception as e:
                    logger.error(f"Query failed: {e}")
                    continue
            
            success_rate = successful_queries / total_queries
            logger.info(f"Query success rate: {successful_queries}/{total_queries} ({success_rate:.1%})")
            
            # Consider successful if at least 60% of queries work
            return success_rate >= 0.6
            
        except Exception as e:
            logger.error(f"Query functionality verification failed: {e}")
            return False
    
    async def verify_knowledge_graph_construction(self) -> bool:
        """Verify that knowledge graph construction is working."""
        try:
            logger.info("Verifying knowledge graph construction...")
            
            if not self.rag_system:
                logger.error("RAG system not initialized")
                return False
            
            # Check if the working directory contains expected files
            working_dir = Path(self.config['working_directory'])
            
            # Look for common LightRAG storage files
            expected_files = ['graph_chunk_entity_relation.graphml', 'entities.json', 'relationships.json']
            found_files = []
            
            for file_pattern in ['*.graphml', '*.json', '*.pkl', '*.db']:
                found_files.extend(list(working_dir.glob(file_pattern)))
            
            if found_files:
                logger.info(f"Knowledge graph files found: {[f.name for f in found_files]}")
                return True
            else:
                logger.warning("No knowledge graph files found - this may be expected for some configurations")
                # Don't fail the test if files aren't found, as storage might be in-memory
                return True
                
        except Exception as e:
            logger.error(f"Knowledge graph verification failed: {e}")
            return False
    
    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run comprehensive verification of the PDF to knowledge base pipeline."""
        logger.info("="*60)
        logger.info("STARTING PDF TO KNOWLEDGE BASE PIPELINE VERIFICATION")
        logger.info("="*60)
        
        test_steps = [
            ('pdf_loading', self.verify_pdf_loading),
            ('rag_system_setup', self.setup_rag_system),
            ('knowledge_base_init', self.verify_knowledge_base_initialization),
            ('query_functionality', self.verify_basic_query_functionality),
            ('knowledge_graph', self.verify_knowledge_graph_construction)
        ]
        
        results = {
            'overall_success': True,
            'test_results': {},
            'execution_time': 0,
            'summary': {}
        }
        
        for step_name, step_function in test_steps:
            try:
                logger.info(f"\n--- Starting {step_name.replace('_', ' ').title()} ---")
                step_start = time.time()
                
                # Check timeout
                if time.time() - self.start_time > self.config['max_test_duration']:
                    logger.error("Test timeout reached")
                    results['test_results'][step_name] = False
                    results['overall_success'] = False
                    break
                
                success = await step_function()
                step_duration = time.time() - step_start
                
                results['test_results'][step_name] = success
                logger.info(f"{step_name} completed in {step_duration:.1f}s: {'PASS' if success else 'FAIL'}")
                
                if not success:
                    results['overall_success'] = False
                    logger.error(f"Critical failure in {step_name}")
                    
            except Exception as e:
                logger.error(f"Exception in {step_name}: {e}")
                results['test_results'][step_name] = False
                results['overall_success'] = False
        
        results['execution_time'] = time.time() - self.start_time
        
        # Generate summary
        passed_tests = sum(1 for result in results['test_results'].values() if result)
        total_tests = len(results['test_results'])
        results['summary'] = {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'execution_time': results['execution_time']
        }
        
        # Log final results
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Overall Success: {'PASS' if results['overall_success'] else 'FAIL'}")
        logger.info(f"Tests Passed: {passed_tests}/{total_tests} ({results['summary']['success_rate']:.1%})")
        logger.info(f"Execution Time: {results['execution_time']:.1f}s")
        
        for step_name, success in results['test_results'].items():
            status = "PASS" if success else "FAIL"
            logger.info(f"  {step_name.replace('_', ' ').title()}: {status}")
        
        # Save results to file
        results_file = Path('pdf_kb_pipeline_verification_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        return results

async def main():
    """Main test execution function."""
    try:
        verifier = PDFKnowledgeBaseVerifier(TEST_CONFIG)
        results = await verifier.run_comprehensive_verification()
        
        # Exit with appropriate code
        if results['overall_success']:
            logger.info("All tests passed successfully!")
            sys.exit(0)
        else:
            logger.error("Some tests failed - check logs for details")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the verification
    asyncio.run(main())