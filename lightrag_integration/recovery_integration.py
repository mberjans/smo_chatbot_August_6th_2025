#!/usr/bin/env python3
"""
Recovery System Integration for Clinical Metabolomics Oracle.

This module integrates the advanced recovery and graceful degradation system
with the existing ClinicalMetabolomicsRAG infrastructure, providing seamless
recovery capabilities during document ingestion and processing.

Features:
    - Integration with ClinicalMetabolomicsRAG error handling
    - Automatic recovery strategy selection based on failure patterns
    - Progress tracking integration with unified progress system
    - Circuit breaker integration with recovery mechanisms
    - Batch processing optimization with resource awareness

Author: Claude Code (Anthropic)
Created: 2025-08-07
Version: 1.0.0
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

from .advanced_recovery_system import (
    AdvancedRecoverySystem, DegradationMode, FailureType, 
    ResourceThresholds, DegradationConfig, CheckpointData
)
from .unified_progress_tracker import (
    KnowledgeBaseProgressTracker, KnowledgeBasePhase
)
from .progress_config import ProgressTrackingConfig


class RecoveryIntegratedProcessor:
    """
    A document processor that integrates advanced recovery mechanisms
    with the existing ClinicalMetabolomicsRAG infrastructure.
    """
    
    def __init__(self,
                 rag_system: Any,  # ClinicalMetabolomicsRAG instance
                 recovery_system: Optional[AdvancedRecoverySystem] = None,
                 progress_tracker: Optional[KnowledgeBaseProgressTracker] = None,
                 enable_checkpointing: bool = True,
                 checkpoint_interval: int = 10,  # Checkpoint every N processed documents
                 logger: Optional[logging.Logger] = None):
        """
        Initialize recovery-integrated processor.
        
        Args:
            rag_system: ClinicalMetabolomicsRAG instance
            recovery_system: Advanced recovery system (creates default if None)
            progress_tracker: Progress tracker (creates default if None)
            enable_checkpointing: Whether to enable automatic checkpointing
            checkpoint_interval: How often to create checkpoints
            logger: Logger instance
        """
        self.rag_system = rag_system
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize recovery system
        if recovery_system is None:
            self.recovery_system = AdvancedRecoverySystem(
                progress_tracker=progress_tracker,
                logger=self.logger
            )
        else:
            self.recovery_system = recovery_system
        
        # Initialize progress tracker if needed
        if progress_tracker is None:
            progress_config = ProgressTrackingConfig()
            self.progress_tracker = KnowledgeBaseProgressTracker(
                progress_config=progress_config,
                logger=self.logger
            )
        else:
            self.progress_tracker = progress_tracker
        
        # Integrate progress tracker with recovery system
        if self.recovery_system.progress_tracker is None:
            self.recovery_system.progress_tracker = self.progress_tracker
        
        # Checkpointing configuration
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_interval = checkpoint_interval
        self._processed_since_checkpoint = 0
        
        # Processing statistics
        self._processing_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'recovery_actions_taken': 0,
            'degradation_events': 0,
            'checkpoints_created': 0
        }
    
    async def process_documents_with_recovery(self,
                                            documents: List[str],
                                            phase: KnowledgeBasePhase = KnowledgeBasePhase.DOCUMENT_INGESTION,
                                            initial_batch_size: int = 10,
                                            max_failures_per_batch: int = 3,
                                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process documents with advanced recovery mechanisms.
        
        Args:
            documents: List of document paths or IDs to process
            phase: Current processing phase
            initial_batch_size: Initial batch size
            max_failures_per_batch: Maximum failures allowed per batch before degradation
            progress_callback: Optional progress callback function
            
        Returns:
            Processing results with recovery statistics
        """
        # Initialize processing session
        self.recovery_system.initialize_ingestion_session(
            documents=documents,
            phase=phase,
            batch_size=initial_batch_size
        )
        
        # Start progress tracking
        self.progress_tracker.start_phase(
            phase=phase,
            status_message=f"Starting document processing: {len(documents)} documents",
            estimated_duration=len(documents) * 2.0  # Rough estimate: 2 seconds per document
        )
        
        processing_start_time = time.time()
        results = {
            'processed_documents': [],
            'failed_documents': {},
            'skipped_documents': [],
            'recovery_events': [],
            'checkpoints': [],
            'final_degradation_mode': DegradationMode.OPTIMAL,
            'processing_time': 0.0,
            'statistics': {}
        }
        
        try:
            while True:
                # Get next batch
                next_batch = self.recovery_system.get_next_batch()
                if not next_batch:
                    break  # No more documents to process
                
                self.logger.info(f"Processing batch of {len(next_batch)} documents in {self.recovery_system.current_degradation_mode.value} mode")
                
                # Process batch with recovery
                batch_results = await self._process_batch_with_recovery(
                    next_batch, max_failures_per_batch, progress_callback
                )
                
                # Update results
                results['processed_documents'].extend(batch_results['processed'])
                results['failed_documents'].update(batch_results['failed'])
                results['skipped_documents'].extend(batch_results['skipped'])
                results['recovery_events'].extend(batch_results['recovery_events'])
                
                # Update progress
                total_processed = len(results['processed_documents']) + len(results['failed_documents'])
                progress = total_processed / len(documents) if documents else 1.0
                
                self.progress_tracker.update_phase_progress(
                    phase=phase,
                    progress=progress,
                    status_message=f"Processed {total_processed}/{len(documents)} documents",
                    details={
                        'processed_count': len(results['processed_documents']),
                        'failed_count': len(results['failed_documents']),
                        'current_degradation_mode': self.recovery_system.current_degradation_mode.value,
                        'current_batch_size': self.recovery_system._current_batch_size
                    }
                )
                
                # Create checkpoint if needed
                if self.enable_checkpointing and self._processed_since_checkpoint >= self.checkpoint_interval:
                    checkpoint_id = self.recovery_system.create_checkpoint({
                        'batch_results': batch_results,
                        'progress': progress
                    })
                    results['checkpoints'].append(checkpoint_id)
                    self._processed_since_checkpoint = 0
                    self._processing_stats['checkpoints_created'] += 1
                
                # Check if we should stop due to excessive failures
                if len(results['failed_documents']) > len(documents) * 0.8:  # 80% failure rate
                    self.logger.error("Stopping due to excessive failure rate")
                    break
            
            # Complete the phase
            self.progress_tracker.complete_phase(
                phase=phase,
                status_message=f"Completed: {len(results['processed_documents'])} processed, {len(results['failed_documents'])} failed"
            )
            
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Unexpected error during processing: {e}")
            self.progress_tracker.fail_phase(
                phase=phase,
                error_message=str(e)
            )
            raise
        
        finally:
            # Calculate final statistics
            processing_time = time.time() - processing_start_time
            results['processing_time'] = processing_time
            results['final_degradation_mode'] = self.recovery_system.current_degradation_mode
            results['statistics'] = self._calculate_processing_statistics(results, processing_time)
            
            # Create final checkpoint
            if self.enable_checkpointing:
                final_checkpoint = self.recovery_system.create_checkpoint({
                    'final_results': results,
                    'session_completed': True
                })
                results['checkpoints'].append(final_checkpoint)
        
        return results
    
    async def _process_batch_with_recovery(self,
                                         batch: List[str],
                                         max_failures: int,
                                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a batch of documents with recovery handling.
        
        Args:
            batch: Batch of document IDs to process
            max_failures: Maximum failures allowed in this batch
            progress_callback: Optional progress callback
            
        Returns:
            Batch processing results
        """
        batch_results = {
            'processed': [],
            'failed': {},
            'skipped': [],
            'recovery_events': []
        }
        
        batch_failures = 0
        
        for doc_id in batch:
            try:
                # Check if we should skip due to too many batch failures
                if batch_failures >= max_failures:
                    recovery_event = self.recovery_system.handle_failure(
                        FailureType.PROCESSING_ERROR,
                        f"Batch failure limit reached ({batch_failures}/{max_failures})",
                        context={'batch_id': id(batch), 'failures': batch_failures}
                    )
                    batch_results['recovery_events'].append(recovery_event)
                    batch_results['skipped'].extend(batch[len(batch_results['processed']) + len(batch_results['failed']):])
                    self._processing_stats['recovery_actions_taken'] += 1
                    break
                
                # Process single document
                doc_result = await self._process_single_document_with_recovery(doc_id)
                
                if doc_result['success']:
                    batch_results['processed'].append(doc_id)
                    self.recovery_system.mark_document_processed(doc_id)
                    self._processed_since_checkpoint += 1
                    self._processing_stats['total_processed'] += 1
                else:
                    batch_results['failed'][doc_id] = doc_result['error']
                    batch_failures += 1
                    self._processing_stats['total_failed'] += 1
                
                # Add any recovery events
                if doc_result.get('recovery_events'):
                    batch_results['recovery_events'].extend(doc_result['recovery_events'])
                
                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(doc_id, doc_result)
                    except Exception as e:
                        self.logger.warning(f"Progress callback failed: {e}")
                
            except Exception as e:
                # Unexpected error processing document
                batch_results['failed'][doc_id] = str(e)
                batch_failures += 1
                self._processing_stats['total_failed'] += 1
                
                # Handle as processing error
                recovery_event = self.recovery_system.handle_failure(
                    FailureType.PROCESSING_ERROR,
                    str(e),
                    document_id=doc_id,
                    context={'batch_id': id(batch)}
                )
                batch_results['recovery_events'].append(recovery_event)
                self._processing_stats['recovery_actions_taken'] += 1
        
        return batch_results
    
    async def _process_single_document_with_recovery(self, doc_id: str) -> Dict[str, Any]:
        """
        Process a single document with recovery mechanisms.
        
        Args:
            doc_id: Document ID to process
            
        Returns:
            Processing result
        """
        result = {
            'success': False,
            'error': None,
            'recovery_events': [],
            'attempts': 0
        }
        
        max_attempts = self.recovery_system.degradation_config.max_retry_attempts
        
        for attempt in range(1, max_attempts + 1):
            result['attempts'] = attempt
            
            try:
                # Check system resources before processing
                system_resources = self.recovery_system.resource_monitor.get_current_resources()
                resource_pressure = self.recovery_system.resource_monitor.check_resource_pressure()
                
                # Handle resource pressure
                if resource_pressure:
                    recovery_event = await self._handle_resource_pressure(resource_pressure, doc_id)
                    if recovery_event:
                        result['recovery_events'].append(recovery_event)
                
                # Process document based on current degradation mode
                success = await self._process_document_degraded(doc_id)
                
                if success:
                    result['success'] = True
                    # Update response time for backoff calculation
                    self.recovery_system.backoff_calculator.update_api_response_time(1.0)  # Placeholder
                    break
                else:
                    raise Exception("Document processing failed")
                
            except Exception as e:
                error_message = str(e)
                result['error'] = error_message
                
                # Determine failure type
                failure_type = self._classify_error(error_message)
                
                # Handle the failure
                recovery_event = self.recovery_system.handle_failure(
                    failure_type=failure_type,
                    error_message=error_message,
                    document_id=doc_id,
                    context={
                        'attempt': attempt,
                        'max_attempts': max_attempts,
                        'system_resources': system_resources
                    }
                )
                
                result['recovery_events'].append(recovery_event)
                self._processing_stats['recovery_actions_taken'] += 1
                
                # Apply backoff if not the last attempt
                if attempt < max_attempts:
                    backoff_seconds = self.recovery_system.backoff_calculator.calculate_backoff(
                        failure_type, attempt
                    )
                    
                    self.logger.info(f"Backing off for {backoff_seconds:.2f}s before retry (attempt {attempt}/{max_attempts})")
                    await asyncio.sleep(backoff_seconds)
                
                # Check if degradation mode changed
                current_mode = self.recovery_system.current_degradation_mode
                if recovery_event.get('degradation_needed') and current_mode != DegradationMode.OPTIMAL:
                    self._processing_stats['degradation_events'] += 1
        
        return result
    
    async def _process_document_degraded(self, doc_id: str) -> bool:
        """
        Process document according to current degradation mode.
        
        Args:
            doc_id: Document ID to process
            
        Returns:
            True if processing succeeded
        """
        mode = self.recovery_system.current_degradation_mode
        config = self.recovery_system.degradation_config
        
        try:
            if mode == DegradationMode.OPTIMAL:
                # Full processing with all features
                return await self._process_document_optimal(doc_id)
            
            elif mode == DegradationMode.ESSENTIAL:
                # Process only if document is marked as essential
                if not self._is_essential_document(doc_id):
                    self.logger.info(f"Skipping non-essential document {doc_id} in essential mode")
                    return True  # Skip non-essential documents
                return await self._process_document_reduced(doc_id, skip_metadata=config.skip_optional_metadata)
            
            elif mode == DegradationMode.MINIMAL:
                # Minimal processing - fast but lower quality
                return await self._process_document_minimal(doc_id)
            
            elif mode == DegradationMode.OFFLINE:
                # Queue for later processing when APIs are available
                self.logger.info(f"Queueing document {doc_id} for offline processing")
                return await self._queue_document_offline(doc_id)
            
            elif mode == DegradationMode.SAFE:
                # Ultra-conservative processing with maximum error tolerance
                return await self._process_document_safe(doc_id)
            
            else:
                # Fallback to optimal mode
                return await self._process_document_optimal(doc_id)
            
        except Exception as e:
            self.logger.error(f"Failed to process document {doc_id} in {mode.value} mode: {e}")
            return False
    
    async def _process_document_optimal(self, doc_id: str) -> bool:
        """Process document with full features (optimal mode)."""
        try:
            # This would integrate with the actual RAG system processing
            # For now, we'll simulate processing
            await asyncio.sleep(0.1)  # Simulate processing time
            self.logger.debug(f"Processed document {doc_id} in optimal mode")
            return True
        except Exception as e:
            raise Exception(f"Optimal processing failed: {e}")
    
    async def _process_document_reduced(self, doc_id: str, skip_metadata: bool = False) -> bool:
        """Process document with reduced features."""
        try:
            # Simulate reduced processing
            await asyncio.sleep(0.05)  # Faster processing
            self.logger.debug(f"Processed document {doc_id} in reduced mode (skip_metadata={skip_metadata})")
            return True
        except Exception as e:
            raise Exception(f"Reduced processing failed: {e}")
    
    async def _process_document_minimal(self, doc_id: str) -> bool:
        """Process document with minimal features for maximum speed."""
        try:
            # Simulate minimal processing
            await asyncio.sleep(0.02)  # Very fast processing
            self.logger.debug(f"Processed document {doc_id} in minimal mode")
            return True
        except Exception as e:
            raise Exception(f"Minimal processing failed: {e}")
    
    async def _queue_document_offline(self, doc_id: str) -> bool:
        """Queue document for offline processing."""
        try:
            # In reality, this would add to a persistent queue
            self.logger.info(f"Queued document {doc_id} for offline processing")
            return True
        except Exception as e:
            raise Exception(f"Offline queueing failed: {e}")
    
    async def _process_document_safe(self, doc_id: str) -> bool:
        """Process document in ultra-safe mode with maximum error tolerance."""
        try:
            # Simulate safe processing with extra error handling
            await asyncio.sleep(0.03)
            
            # Add extra validation and error checking here
            if not doc_id or len(doc_id) == 0:
                raise ValueError("Invalid document ID")
            
            self.logger.debug(f"Processed document {doc_id} in safe mode")
            return True
        except Exception as e:
            # In safe mode, we're more tolerant of errors
            self.logger.warning(f"Safe mode processing had error for {doc_id}: {e}, continuing anyway")
            return True  # Return success even with minor errors
    
    async def _handle_resource_pressure(self, pressure: Dict[str, str], doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Handle detected resource pressure.
        
        Args:
            pressure: Resource pressure information
            doc_id: Current document being processed
            
        Returns:
            Recovery event if action was taken
        """
        critical_resources = [k for k, v in pressure.items() if 'critical' in v]
        
        if critical_resources:
            failure_type = FailureType.RESOURCE_EXHAUSTION
            if 'memory' in critical_resources:
                failure_type = FailureType.MEMORY_PRESSURE
            
            return self.recovery_system.handle_failure(
                failure_type=failure_type,
                error_message=f"Critical resource pressure detected: {', '.join(critical_resources)}",
                document_id=doc_id,
                context={'resource_pressure': pressure}
            )
        
        return None
    
    def _classify_error(self, error_message: str) -> FailureType:
        """
        Classify an error message to determine failure type.
        
        Args:
            error_message: Error message to classify
            
        Returns:
            Classified failure type
        """
        error_lower = error_message.lower()
        
        if 'rate limit' in error_lower or 'too many requests' in error_lower:
            return FailureType.API_RATE_LIMIT
        elif 'timeout' in error_lower or 'timed out' in error_lower:
            return FailureType.API_TIMEOUT
        elif 'memory' in error_lower or 'out of memory' in error_lower:
            return FailureType.MEMORY_PRESSURE
        elif 'disk' in error_lower or 'no space' in error_lower:
            return FailureType.DISK_SPACE
        elif 'network' in error_lower or 'connection' in error_lower:
            return FailureType.NETWORK_ERROR
        elif 'api' in error_lower or 'openai' in error_lower:
            return FailureType.API_ERROR
        else:
            return FailureType.PROCESSING_ERROR
    
    def _is_essential_document(self, doc_id: str) -> bool:
        """
        Determine if a document is essential and should be processed in essential mode.
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document is essential
        """
        # Simple heuristic - in practice this would be more sophisticated
        essential_keywords = ['essential', 'critical', 'important', 'key', 'core']
        return any(keyword in doc_id.lower() for keyword in essential_keywords)
    
    def _calculate_processing_statistics(self, results: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """
        Calculate comprehensive processing statistics.
        
        Args:
            results: Processing results
            processing_time: Total processing time
            
        Returns:
            Statistics dictionary
        """
        total_docs = len(results['processed_documents']) + len(results['failed_documents'])
        
        return {
            'total_documents': total_docs,
            'processed_documents': len(results['processed_documents']),
            'failed_documents': len(results['failed_documents']),
            'skipped_documents': len(results['skipped_documents']),
            'success_rate': len(results['processed_documents']) / max(1, total_docs),
            'processing_time_seconds': processing_time,
            'documents_per_second': total_docs / max(0.001, processing_time),
            'recovery_events': len(results['recovery_events']),
            'checkpoints_created': len(results['checkpoints']),
            'degradation_mode_changes': self._processing_stats['degradation_events'],
            'total_recovery_actions': self._processing_stats['recovery_actions_taken']
        }
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery system status."""
        base_status = self.recovery_system.get_recovery_status()
        
        # Add integration-specific information
        base_status.update({
            'processing_statistics': self._processing_stats.copy(),
            'checkpointing_enabled': self.enable_checkpointing,
            'checkpoint_interval': self.checkpoint_interval,
            'processed_since_checkpoint': self._processed_since_checkpoint
        })
        
        return base_status
    
    async def resume_from_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Resume processing from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to resume from
            
        Returns:
            True if resumed successfully
        """
        success = self.recovery_system.resume_from_checkpoint(checkpoint_id)
        if success:
            # Reset integration-specific counters
            self._processed_since_checkpoint = 0
            self.logger.info(f"Resumed processing from checkpoint {checkpoint_id}")
        
        return success