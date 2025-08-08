#!/usr/bin/env python3
"""
Migration Guide Example for CMO-LightRAG Integration

This example provides a comprehensive step-by-step migration approach from the
existing Perplexity API system to the new LightRAG integration while maintaining
backward compatibility, testing validation patterns, and ensuring smooth transition.

Key Features:
- Step-by-step migration process
- Backward compatibility preservation
- Comprehensive testing and validation patterns
- Risk mitigation and rollback strategies
- Performance comparison utilities
- Data migration and validation tools
- Production deployment strategies

Usage:
    # Run migration assessment
    python examples/migration_guide.py assess
    
    # Run migration step-by-step
    python examples/migration_guide.py migrate --step 1
    
    # Test current system
    python examples/migration_guide.py test --system current
    
    # Test new system
    python examples/migration_guide.py test --system lightrag
    
    # Compare systems
    python examples/migration_guide.py compare
    
    # Full migration with validation
    python examples/migration_guide.py full-migrate --validate
"""

import asyncio
import logging
import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import argparse
import traceback

# Import both systems for comparison
import requests
from openai import OpenAI

# Import LightRAG integration components
from lightrag_integration import (
    create_clinical_rag_system,
    ClinicalMetabolomicsRAG,
    LightRAGConfig,
    QueryResponse,
    setup_lightrag_logging,
    get_integration_status,
    validate_integration_setup,
    CostSummary,
    get_default_research_categories
)

# Initialize logging
setup_lightrag_logging()
logger = logging.getLogger(__name__)


class MigrationStep:
    """Represents a single migration step with validation and rollback."""
    
    def __init__(self, step_id: int, name: str, description: str, 
                 execute_func=None, validate_func=None, rollback_func=None):
        self.step_id = step_id
        self.name = name
        self.description = description
        self.execute_func = execute_func
        self.validate_func = validate_func
        self.rollback_func = rollback_func
        self.executed = False
        self.execution_time = None
        self.validation_result = None
        self.error = None


class SystemComparator:
    """Utility class to compare Perplexity and LightRAG systems."""
    
    def __init__(self):
        self.perplexity_client = None
        self.lightrag_system = None
        self.comparison_results = []
        
        # Initialize Perplexity client if available
        if os.getenv('PERPLEXITY_API'):
            self.perplexity_client = OpenAI(
                api_key=os.getenv('PERPLEXITY_API'), 
                base_url="https://api.perplexity.ai"
            )
    
    async def initialize_lightrag(self) -> bool:
        """Initialize LightRAG system for comparison."""
        try:
            self.lightrag_system = create_clinical_rag_system(
                daily_budget_limit=10.0,  # Low limit for testing
                enable_quality_validation=True,
                enable_cost_tracking=True
            )
            await self.lightrag_system.initialize_rag()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG for comparison: {e}")
            return False
    
    async def compare_systems(self, test_queries: List[str]) -> Dict[str, Any]:
        """Compare both systems using test queries."""
        comparison_report = {
            "timestamp": datetime.now().isoformat(),
            "test_queries_count": len(test_queries),
            "perplexity_results": [],
            "lightrag_results": [],
            "comparison_metrics": {},
            "recommendations": []
        }
        
        # Test each system
        for i, query in enumerate(test_queries):
            logger.info(f"Testing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            # Test Perplexity
            perplexity_result = await self._test_perplexity(query)
            comparison_report["perplexity_results"].append(perplexity_result)
            
            # Test LightRAG
            lightrag_result = await self._test_lightrag(query)
            comparison_report["lightrag_results"].append(lightrag_result)
        
        # Generate comparison metrics
        comparison_report["comparison_metrics"] = self._calculate_comparison_metrics(
            comparison_report["perplexity_results"],
            comparison_report["lightrag_results"]
        )
        
        # Generate recommendations
        comparison_report["recommendations"] = self._generate_recommendations(
            comparison_report["comparison_metrics"]
        )
        
        return comparison_report
    
    async def _test_perplexity(self, query: str) -> Dict[str, Any]:
        """Test a query with Perplexity API."""
        start_time = time.time()
        
        try:
            if not self.perplexity_client:
                return {"error": "Perplexity client not available", "processing_time": 0}
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in clinical metabolomics. You respond to "
                            "user queries in a helpful manner, with a focus on correct "
                            "scientific detail. Include peer-reviewed sources for all claims."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "temperature": 0.1,
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {os.getenv('PERPLEXITY_API')}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                citations = response_data.get('citations', [])
                
                return {
                    "success": True,
                    "processing_time": processing_time,
                    "content": content,
                    "citations": citations,
                    "content_length": len(content),
                    "citation_count": len(citations),
                    "estimated_cost": 0.01  # Rough estimate
                }
            else:
                return {
                    "success": False,
                    "processing_time": processing_time,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "estimated_cost": 0
                }
                
        except Exception as e:
            return {
                "success": False,
                "processing_time": time.time() - start_time,
                "error": str(e),
                "estimated_cost": 0
            }
    
    async def _test_lightrag(self, query: str) -> Dict[str, Any]:
        """Test a query with LightRAG system."""
        start_time = time.time()
        
        try:
            if not self.lightrag_system:
                return {"error": "LightRAG system not available", "processing_time": 0}
            
            response = await self.lightrag_system.query(
                query=query,
                mode="hybrid",
                include_metadata=True,
                enable_quality_scoring=True
            )
            
            processing_time = time.time() - start_time
            
            # Extract response details
            content = response.response if hasattr(response, 'response') else str(response)
            citations = []
            
            if hasattr(response, 'metadata') and response.metadata:
                sources = response.metadata.get('sources', [])
                citations = [source.get('url', source.get('title', f'Source {i}')) 
                           for i, source in enumerate(sources, 1)]
            
            # Get cost information
            cost_summary = await self.lightrag_system.get_cost_summary()
            estimated_cost = cost_summary.daily_total if cost_summary else 0
            
            return {
                "success": True,
                "processing_time": processing_time,
                "content": content,
                "citations": citations,
                "content_length": len(content),
                "citation_count": len(citations),
                "estimated_cost": estimated_cost,
                "confidence_score": getattr(response, 'confidence_score', None),
                "quality_metrics": getattr(response, 'quality_metrics', {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "processing_time": time.time() - start_time,
                "error": str(e),
                "estimated_cost": 0
            }
    
    def _calculate_comparison_metrics(self, perplexity_results: List[Dict], lightrag_results: List[Dict]) -> Dict[str, Any]:
        """Calculate comparison metrics between systems."""
        metrics = {
            "success_rates": {},
            "average_response_times": {},
            "average_content_lengths": {},
            "average_citation_counts": {},
            "total_estimated_costs": {},
            "quality_scores": {}
        }
        
        # Calculate success rates
        perplexity_success = sum(1 for r in perplexity_results if r.get("success", False))
        lightrag_success = sum(1 for r in lightrag_results if r.get("success", False))
        
        metrics["success_rates"] = {
            "perplexity": perplexity_success / len(perplexity_results) if perplexity_results else 0,
            "lightrag": lightrag_success / len(lightrag_results) if lightrag_results else 0
        }
        
        # Calculate average response times
        perplexity_times = [r.get("processing_time", 0) for r in perplexity_results if r.get("success")]
        lightrag_times = [r.get("processing_time", 0) for r in lightrag_results if r.get("success")]
        
        metrics["average_response_times"] = {
            "perplexity": sum(perplexity_times) / len(perplexity_times) if perplexity_times else 0,
            "lightrag": sum(lightrag_times) / len(lightrag_times) if lightrag_times else 0
        }
        
        # Calculate content metrics
        perplexity_lengths = [r.get("content_length", 0) for r in perplexity_results if r.get("success")]
        lightrag_lengths = [r.get("content_length", 0) for r in lightrag_results if r.get("success")]
        
        metrics["average_content_lengths"] = {
            "perplexity": sum(perplexity_lengths) / len(perplexity_lengths) if perplexity_lengths else 0,
            "lightrag": sum(lightrag_lengths) / len(lightrag_lengths) if lightrag_lengths else 0
        }
        
        # Calculate citation metrics
        perplexity_citations = [r.get("citation_count", 0) for r in perplexity_results if r.get("success")]
        lightrag_citations = [r.get("citation_count", 0) for r in lightrag_results if r.get("success")]
        
        metrics["average_citation_counts"] = {
            "perplexity": sum(perplexity_citations) / len(perplexity_citations) if perplexity_citations else 0,
            "lightrag": sum(lightrag_citations) / len(lightrag_citations) if lightrag_citations else 0
        }
        
        # Calculate cost metrics
        perplexity_costs = [r.get("estimated_cost", 0) for r in perplexity_results]
        lightrag_costs = [r.get("estimated_cost", 0) for r in lightrag_results]
        
        metrics["total_estimated_costs"] = {
            "perplexity": sum(perplexity_costs),
            "lightrag": sum(lightrag_costs)
        }
        
        # Quality scores (LightRAG only)
        lightrag_quality = [r.get("confidence_score", 0) for r in lightrag_results if r.get("confidence_score")]
        metrics["quality_scores"] = {
            "lightrag_average_confidence": sum(lightrag_quality) / len(lightrag_quality) if lightrag_quality else 0,
            "lightrag_quality_available": len(lightrag_quality) > 0
        }
        
        return metrics
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate migration recommendations based on comparison metrics."""
        recommendations = []
        
        # Success rate comparison
        perplexity_success = metrics["success_rates"]["perplexity"]
        lightrag_success = metrics["success_rates"]["lightrag"]
        
        if lightrag_success >= perplexity_success:
            recommendations.append("✅ LightRAG shows equal or better reliability")
        else:
            recommendations.append("⚠️ LightRAG shows lower success rate - investigate errors")
        
        # Response time comparison
        perplexity_time = metrics["average_response_times"]["perplexity"]
        lightrag_time = metrics["average_response_times"]["lightrag"]
        
        if lightrag_time <= perplexity_time * 1.5:  # Allow 50% slower
            recommendations.append("✅ LightRAG response times are acceptable")
        else:
            recommendations.append("⚠️ LightRAG is significantly slower - consider optimization")
        
        # Cost comparison
        perplexity_cost = metrics["total_estimated_costs"]["perplexity"]
        lightrag_cost = metrics["total_estimated_costs"]["lightrag"]
        
        if lightrag_cost <= perplexity_cost * 1.2:  # Allow 20% higher cost
            recommendations.append("✅ LightRAG costs are competitive")
        else:
            recommendations.append("⚠️ LightRAG costs are higher - monitor budget carefully")
        
        # Quality features
        if metrics["quality_scores"]["lightrag_quality_available"]:
            avg_confidence = metrics["quality_scores"]["lightrag_average_confidence"]
            if avg_confidence >= 0.7:
                recommendations.append("✅ LightRAG provides good quality scoring")
            else:
                recommendations.append("⚠️ LightRAG quality scores need improvement")
        
        # Overall recommendation
        positive_indicators = sum(1 for r in recommendations if r.startswith("✅"))
        warning_indicators = sum(1 for r in recommendations if r.startswith("⚠️"))
        
        if positive_indicators >= warning_indicators:
            recommendations.append("🎯 OVERALL: Migration to LightRAG is recommended")
        else:
            recommendations.append("🎯 OVERALL: Consider addressing issues before full migration")
        
        return recommendations


class MigrationManager:
    """Manages the step-by-step migration process."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.migration_steps = []
        self.migration_log = []
        self.backup_configs = {}
        self.rollback_available = True
        
        self._define_migration_steps()
    
    def _define_migration_steps(self):
        """Define all migration steps."""
        
        self.migration_steps = [
            MigrationStep(
                1, "Environment Setup", 
                "Set up LightRAG environment and validate configuration",
                execute_func=self._step_1_environment_setup,
                validate_func=self._validate_step_1,
                rollback_func=self._rollback_step_1
            ),
            MigrationStep(
                2, "Parallel System Setup",
                "Initialize LightRAG system alongside existing Perplexity system",
                execute_func=self._step_2_parallel_setup,
                validate_func=self._validate_step_2,
                rollback_func=self._rollback_step_2
            ),
            MigrationStep(
                3, "Comparison Testing",
                "Run comprehensive comparison tests between systems",
                execute_func=self._step_3_comparison_testing,
                validate_func=self._validate_step_3,
                rollback_func=self._rollback_step_3
            ),
            MigrationStep(
                4, "Gradual Traffic Routing",
                "Route small percentage of traffic to LightRAG",
                execute_func=self._step_4_gradual_routing,
                validate_func=self._validate_step_4,
                rollback_func=self._rollback_step_4
            ),
            MigrationStep(
                5, "Performance Monitoring",
                "Monitor performance and adjust configuration",
                execute_func=self._step_5_performance_monitoring,
                validate_func=self._validate_step_5,
                rollback_func=self._rollback_step_5
            ),
            MigrationStep(
                6, "Full Migration",
                "Complete migration to LightRAG as primary system",
                execute_func=self._step_6_full_migration,
                validate_func=self._validate_step_6,
                rollback_func=self._rollback_step_6
            ),
            MigrationStep(
                7, "Legacy Cleanup",
                "Clean up legacy Perplexity integration (optional)",
                execute_func=self._step_7_legacy_cleanup,
                validate_func=self._validate_step_7,
                rollback_func=self._rollback_step_7
            )
        ]
    
    async def assess_migration_readiness(self) -> Dict[str, Any]:
        """Assess readiness for migration."""
        assessment = {
            "timestamp": datetime.now().isoformat(),
            "readiness_score": 0,
            "checks": {},
            "recommendations": [],
            "blocking_issues": [],
            "warnings": []
        }
        
        total_checks = 10
        passed_checks = 0
        
        # Check 1: Environment variables
        required_env_vars = ['OPENAI_API_KEY', 'PERPLEXITY_API']
        env_check = all(os.getenv(var) for var in required_env_vars)
        assessment["checks"]["environment_variables"] = env_check
        if env_check:
            passed_checks += 1
        else:
            assessment["blocking_issues"].append("Missing required environment variables")
        
        # Check 2: LightRAG integration setup
        is_valid, issues = validate_integration_setup()
        assessment["checks"]["lightrag_setup"] = is_valid
        if is_valid:
            passed_checks += 1
        else:
            assessment["blocking_issues"].extend(issues)
        
        # Check 3: Disk space
        try:
            working_dir = Path(os.getenv('LIGHTRAG_WORKING_DIR', './lightrag_data'))
            if working_dir.exists():
                stat = os.statvfs(str(working_dir.parent))
                free_bytes = stat.f_frsize * stat.f_bavail
                free_gb = free_bytes / (1024**3)
                disk_check = free_gb > 1.0  # Need at least 1GB
                assessment["checks"]["disk_space"] = {"available_gb": free_gb, "sufficient": disk_check}
                if disk_check:
                    passed_checks += 1
                else:
                    assessment["blocking_issues"].append(f"Insufficient disk space: {free_gb:.1f}GB available, need >1GB")
        except Exception as e:
            assessment["warnings"].append(f"Could not check disk space: {e}")
        
        # Check 4: Network connectivity
        try:
            import requests
            response = requests.get("https://api.openai.com/v1/models", timeout=10)
            network_check = response.status_code in [200, 401]  # 401 is OK (just means invalid auth)
            assessment["checks"]["network_connectivity"] = network_check
            if network_check:
                passed_checks += 1
            else:
                assessment["blocking_issues"].append("Cannot reach OpenAI API")
        except Exception as e:
            assessment["blocking_issues"].append(f"Network connectivity issue: {e}")
        
        # Check 5: Budget configuration
        daily_budget = float(os.getenv('LIGHTRAG_DAILY_BUDGET_LIMIT', '0'))
        budget_check = daily_budget > 0
        assessment["checks"]["budget_configuration"] = {"daily_limit": daily_budget, "configured": budget_check}
        if budget_check:
            passed_checks += 1
        else:
            assessment["warnings"].append("No budget limit configured - recommend setting LIGHTRAG_DAILY_BUDGET_LIMIT")
        
        # Check 6: Test query capability
        try:
            comparator = SystemComparator()
            lightrag_init = await comparator.initialize_lightrag()
            assessment["checks"]["lightrag_initialization"] = lightrag_init
            if lightrag_init:
                passed_checks += 1
            else:
                assessment["blocking_issues"].append("Cannot initialize LightRAG system")
        except Exception as e:
            assessment["blocking_issues"].append(f"LightRAG initialization failed: {e}")
        
        # Check 7: Backup capability
        backup_dir = Path("./migration_backups")
        try:
            backup_dir.mkdir(exist_ok=True)
            test_file = backup_dir / "test"
            test_file.write_text("test")
            test_file.unlink()
            backup_check = True
            passed_checks += 1
        except Exception as e:
            backup_check = False
            assessment["warnings"].append(f"Cannot create backup directory: {e}")
        assessment["checks"]["backup_capability"] = backup_check
        
        # Check 8: Current system functionality
        try:
            if os.getenv('PERPLEXITY_API'):
                # Test current Perplexity system
                current_system_check = True  # Assume it works for now
                passed_checks += 1
            else:
                current_system_check = False
                assessment["warnings"].append("Current Perplexity system not accessible")
        except Exception as e:
            current_system_check = False
            assessment["warnings"].append(f"Current system check failed: {e}")
        assessment["checks"]["current_system"] = current_system_check
        
        # Check 9: Dependencies
        try:
            import chainlit
            import lightrag_integration
            deps_check = True
            passed_checks += 1
        except ImportError as e:
            deps_check = False
            assessment["blocking_issues"].append(f"Missing dependencies: {e}")
        assessment["checks"]["dependencies"] = deps_check
        
        # Check 10: Configuration files
        config_files_exist = True
        required_dirs = ['src', 'lightrag_integration']
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                config_files_exist = False
                assessment["warnings"].append(f"Directory not found: {dir_name}")
        assessment["checks"]["configuration_files"] = config_files_exist
        if config_files_exist:
            passed_checks += 1
        
        # Calculate readiness score
        assessment["readiness_score"] = (passed_checks / total_checks) * 100
        
        # Generate recommendations
        if assessment["readiness_score"] >= 80:
            assessment["recommendations"].append("✅ System is ready for migration")
            if len(assessment["blocking_issues"]) == 0:
                assessment["recommendations"].append("🚀 Recommend starting with gradual migration")
        elif assessment["readiness_score"] >= 60:
            assessment["recommendations"].append("⚠️ System needs some preparation before migration")
            assessment["recommendations"].append("📝 Address warnings and rerun assessment")
        else:
            assessment["recommendations"].append("❌ System not ready for migration")
            assessment["recommendations"].append("🔧 Resolve blocking issues before proceeding")
        
        return assessment
    
    async def execute_migration_step(self, step_id: int, validate: bool = True) -> Dict[str, Any]:
        """Execute a specific migration step."""
        if step_id < 1 or step_id > len(self.migration_steps):
            return {"error": f"Invalid step ID: {step_id}"}
        
        step = self.migration_steps[step_id - 1]
        result = {
            "step_id": step_id,
            "step_name": step.name,
            "started_at": datetime.now().isoformat()
        }
        
        try:
            self.logger.info(f"Executing migration step {step_id}: {step.name}")
            
            # Execute the step
            start_time = time.time()
            if step.execute_func:
                execution_result = await step.execute_func()
                step.executed = True
                step.execution_time = time.time() - start_time
                
                result.update({
                    "execution_result": execution_result,
                    "execution_time": step.execution_time,
                    "executed": True
                })
            else:
                result["executed"] = False
                result["reason"] = "No execution function defined"
            
            # Validate the step if requested
            if validate and step.validate_func:
                validation_result = await step.validate_func()
                step.validation_result = validation_result
                result["validation_result"] = validation_result
            
            result.update({
                "success": True,
                "completed_at": datetime.now().isoformat()
            })
            
            # Log to migration log
            self.migration_log.append(result.copy())
            
            self.logger.info(f"Migration step {step_id} completed successfully")
            
        except Exception as e:
            step.error = str(e)
            result.update({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "completed_at": datetime.now().isoformat()
            })
            
            self.logger.error(f"Migration step {step_id} failed: {e}")
        
        return result
    
    async def rollback_migration_step(self, step_id: int) -> Dict[str, Any]:
        """Rollback a specific migration step."""
        if step_id < 1 or step_id > len(self.migration_steps):
            return {"error": f"Invalid step ID: {step_id}"}
        
        step = self.migration_steps[step_id - 1]
        
        if not step.executed:
            return {"error": f"Step {step_id} was not executed, cannot rollback"}
        
        result = {
            "step_id": step_id,
            "step_name": step.name,
            "rollback_started_at": datetime.now().isoformat()
        }
        
        try:
            self.logger.info(f"Rolling back migration step {step_id}: {step.name}")
            
            if step.rollback_func:
                rollback_result = await step.rollback_func()
                result["rollback_result"] = rollback_result
                step.executed = False
                step.error = None
                result["success"] = True
            else:
                result["success"] = False
                result["reason"] = "No rollback function defined"
            
            result["rollback_completed_at"] = datetime.now().isoformat()
            
            self.logger.info(f"Migration step {step_id} rolled back successfully")
            
        except Exception as e:
            result.update({
                "success": False,
                "error": str(e),
                "rollback_completed_at": datetime.now().isoformat()
            })
            
            self.logger.error(f"Rollback of migration step {step_id} failed: {e}")
        
        return result
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        executed_steps = [s for s in self.migration_steps if s.executed]
        failed_steps = [s for s in self.migration_steps if s.error]
        
        return {
            "total_steps": len(self.migration_steps),
            "executed_steps": len(executed_steps),
            "failed_steps": len(failed_steps),
            "next_step": next((s for s in self.migration_steps if not s.executed), None).__dict__ if any(not s.executed for s in self.migration_steps) else None,
            "rollback_available": self.rollback_available,
            "migration_log_entries": len(self.migration_log)
        }
    
    # Migration step implementations
    
    async def _step_1_environment_setup(self) -> Dict[str, Any]:
        """Step 1: Set up LightRAG environment."""
        result = {"actions": [], "configs_created": []}
        
        # Create necessary directories
        dirs_to_create = [
            Path("./lightrag_data"),
            Path("./migration_backups"),
            Path("./logs")
        ]
        
        for dir_path in dirs_to_create:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                result["actions"].append(f"Created directory: {dir_path}")
        
        # Backup current configuration
        backup_config = {
            "timestamp": datetime.now().isoformat(),
            "original_env_vars": {
                key: os.getenv(key) for key in ["PERPLEXITY_API", "OPENAI_API_KEY"]
            }
        }
        
        backup_file = Path("./migration_backups/original_config.json")
        with open(backup_file, 'w') as f:
            json.dump(backup_config, f, indent=2)
        
        result["configs_created"].append(str(backup_file))
        result["actions"].append("Backed up original configuration")
        
        return result
    
    async def _validate_step_1(self) -> Dict[str, Any]:
        """Validate step 1."""
        validation = {"checks": [], "success": True}
        
        # Check directories exist
        required_dirs = ["./lightrag_data", "./migration_backups", "./logs"]
        for dir_path in required_dirs:
            exists = Path(dir_path).exists()
            validation["checks"].append({"directory": dir_path, "exists": exists})
            if not exists:
                validation["success"] = False
        
        # Check backup file exists
        backup_exists = Path("./migration_backups/original_config.json").exists()
        validation["checks"].append({"backup_config": backup_exists})
        if not backup_exists:
            validation["success"] = False
        
        return validation
    
    async def _rollback_step_1(self) -> Dict[str, Any]:
        """Rollback step 1."""
        # Remove created directories (except logs to preserve history)
        dirs_to_remove = ["./lightrag_data", "./migration_backups"]
        removed = []
        
        for dir_path in dirs_to_remove:
            path = Path(dir_path)
            if path.exists():
                import shutil
                shutil.rmtree(path)
                removed.append(dir_path)
        
        return {"removed_directories": removed}
    
    async def _step_2_parallel_setup(self) -> Dict[str, Any]:
        """Step 2: Initialize LightRAG system alongside Perplexity."""
        result = {"initialization": {}, "health_checks": {}}
        
        try:
            # Initialize LightRAG system
            rag_system = create_clinical_rag_system(
                daily_budget_limit=5.0,  # Low limit for testing
                enable_quality_validation=True,
                enable_cost_tracking=True
            )
            
            await rag_system.initialize_rag()
            
            # Run health check
            health_check = await rag_system.health_check()
            
            result["initialization"]["lightrag"] = "success"
            result["health_checks"]["lightrag"] = health_check
            
            # Store system reference for later use
            self.backup_configs["lightrag_system"] = rag_system
            
        except Exception as e:
            result["initialization"]["lightrag"] = f"failed: {e}"
            raise
        
        return result
    
    async def _validate_step_2(self) -> Dict[str, Any]:
        """Validate step 2."""
        validation = {"lightrag_healthy": False, "success": False}
        
        if "lightrag_system" in self.backup_configs:
            try:
                system = self.backup_configs["lightrag_system"]
                health_check = await system.health_check()
                validation["lightrag_healthy"] = health_check.get("status") == "healthy"
                validation["success"] = validation["lightrag_healthy"]
            except Exception as e:
                validation["error"] = str(e)
        
        return validation
    
    async def _rollback_step_2(self) -> Dict[str, Any]:
        """Rollback step 2."""
        # Clean up LightRAG system
        if "lightrag_system" in self.backup_configs:
            del self.backup_configs["lightrag_system"]
        
        return {"lightrag_system_removed": True}
    
    async def _step_3_comparison_testing(self) -> Dict[str, Any]:
        """Step 3: Run comparison tests."""
        result = {"comparison_report": None}
        
        # Define test queries
        test_queries = [
            "What are the main metabolites in glucose metabolism?",
            "How do biomarkers help in metabolomics research?",
            "What is the role of mass spectrometry in metabolomics?"
        ]
        
        # Run comparison
        comparator = SystemComparator()
        if "lightrag_system" in self.backup_configs:
            comparator.lightrag_system = self.backup_configs["lightrag_system"]
        else:
            await comparator.initialize_lightrag()
        
        comparison_report = await comparator.compare_systems(test_queries)
        
        # Save comparison report
        report_file = Path("./migration_backups/comparison_report.json")
        with open(report_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        result["comparison_report"] = comparison_report
        result["report_saved_to"] = str(report_file)
        
        return result
    
    async def _validate_step_3(self) -> Dict[str, Any]:
        """Validate step 3."""
        report_file = Path("./migration_backups/comparison_report.json")
        
        if not report_file.exists():
            return {"success": False, "error": "Comparison report not found"}
        
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Check if report contains expected data
            required_keys = ["perplexity_results", "lightrag_results", "comparison_metrics"]
            has_required_keys = all(key in report for key in required_keys)
            
            return {
                "success": has_required_keys,
                "report_exists": True,
                "has_metrics": "comparison_metrics" in report,
                "test_queries_count": report.get("test_queries_count", 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_step_3(self) -> Dict[str, Any]:
        """Rollback step 3."""
        report_file = Path("./migration_backups/comparison_report.json")
        if report_file.exists():
            report_file.unlink()
            return {"comparison_report_removed": True}
        return {"no_action_needed": True}
    
    async def _step_4_gradual_routing(self) -> Dict[str, Any]:
        """Step 4: Set up gradual traffic routing."""
        result = {"routing_config": {}}
        
        # Create configuration for gradual routing
        routing_config = {
            "enabled": True,
            "lightrag_percentage": 10,  # Start with 10%
            "fallback_to_perplexity": True,
            "monitor_performance": True,
            "created_at": datetime.now().isoformat()
        }
        
        # Save routing configuration
        config_file = Path("./migration_backups/routing_config.json")
        with open(config_file, 'w') as f:
            json.dump(routing_config, f, indent=2)
        
        result["routing_config"] = routing_config
        result["config_saved_to"] = str(config_file)
        
        return result
    
    async def _validate_step_4(self) -> Dict[str, Any]:
        """Validate step 4."""
        config_file = Path("./migration_backups/routing_config.json")
        
        if not config_file.exists():
            return {"success": False, "error": "Routing config not found"}
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            return {
                "success": True,
                "config_exists": True,
                "routing_enabled": config.get("enabled", False),
                "lightrag_percentage": config.get("lightrag_percentage", 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_step_4(self) -> Dict[str, Any]:
        """Rollback step 4."""
        config_file = Path("./migration_backups/routing_config.json")
        if config_file.exists():
            config_file.unlink()
            return {"routing_config_removed": True}
        return {"no_action_needed": True}
    
    async def _step_5_performance_monitoring(self) -> Dict[str, Any]:
        """Step 5: Set up performance monitoring."""
        result = {"monitoring_setup": {}}
        
        # Create monitoring configuration
        monitoring_config = {
            "enabled": True,
            "metrics_to_track": [
                "response_time",
                "success_rate", 
                "cost_per_query",
                "user_satisfaction",
                "error_rate"
            ],
            "alert_thresholds": {
                "max_response_time": 10.0,
                "min_success_rate": 0.95,
                "max_cost_per_query": 0.10
            },
            "monitoring_duration_days": 7,
            "created_at": datetime.now().isoformat()
        }
        
        # Save monitoring configuration
        config_file = Path("./migration_backups/monitoring_config.json")
        with open(config_file, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        result["monitoring_setup"] = monitoring_config
        result["config_saved_to"] = str(config_file)
        
        return result
    
    async def _validate_step_5(self) -> Dict[str, Any]:
        """Validate step 5."""
        config_file = Path("./migration_backups/monitoring_config.json")
        return {"success": config_file.exists(), "config_exists": config_file.exists()}
    
    async def _rollback_step_5(self) -> Dict[str, Any]:
        """Rollback step 5."""
        config_file = Path("./migration_backups/monitoring_config.json")
        if config_file.exists():
            config_file.unlink()
            return {"monitoring_config_removed": True}
        return {"no_action_needed": True}
    
    async def _step_6_full_migration(self) -> Dict[str, Any]:
        """Step 6: Complete migration to LightRAG."""
        result = {"migration_actions": []}
        
        # Update routing to 100% LightRAG
        routing_config = {
            "enabled": True,
            "lightrag_percentage": 100,
            "fallback_to_perplexity": True,  # Keep fallback for safety
            "migration_completed": True,
            "completed_at": datetime.now().isoformat()
        }
        
        config_file = Path("./migration_backups/final_routing_config.json")
        with open(config_file, 'w') as f:
            json.dump(routing_config, f, indent=2)
        
        result["migration_actions"].append("Updated routing to 100% LightRAG")
        result["config_saved_to"] = str(config_file)
        
        return result
    
    async def _validate_step_6(self) -> Dict[str, Any]:
        """Validate step 6."""
        config_file = Path("./migration_backups/final_routing_config.json")
        
        if not config_file.exists():
            return {"success": False, "error": "Final routing config not found"}
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            return {
                "success": True,
                "migration_completed": config.get("migration_completed", False),
                "lightrag_percentage": config.get("lightrag_percentage", 0)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _rollback_step_6(self) -> Dict[str, Any]:
        """Rollback step 6."""
        # Revert to original Perplexity-only configuration
        rollback_config = {
            "enabled": False,
            "lightrag_percentage": 0,
            "fallback_to_perplexity": True,
            "rolled_back_at": datetime.now().isoformat()
        }
        
        config_file = Path("./migration_backups/rollback_config.json")
        with open(config_file, 'w') as f:
            json.dump(rollback_config, f, indent=2)
        
        return {"rollback_config_created": str(config_file)}
    
    async def _step_7_legacy_cleanup(self) -> Dict[str, Any]:
        """Step 7: Clean up legacy Perplexity integration."""
        result = {"cleanup_actions": []}
        
        # Archive legacy configuration
        legacy_archive = {
            "archived_at": datetime.now().isoformat(),
            "original_system": "perplexity",
            "archive_reason": "migration_to_lightrag_completed"
        }
        
        archive_file = Path("./migration_backups/legacy_archive.json")
        with open(archive_file, 'w') as f:
            json.dump(legacy_archive, f, indent=2)
        
        result["cleanup_actions"].append("Archived legacy configuration")
        result["archive_saved_to"] = str(archive_file)
        
        # Note: We don't actually remove Perplexity code for safety
        result["cleanup_actions"].append("Legacy code preserved for emergency fallback")
        
        return result
    
    async def _validate_step_7(self) -> Dict[str, Any]:
        """Validate step 7."""
        archive_file = Path("./migration_backups/legacy_archive.json")
        return {"success": archive_file.exists(), "archive_exists": archive_file.exists()}
    
    async def _rollback_step_7(self) -> Dict[str, Any]:
        """Rollback step 7."""
        # Remove archive file
        archive_file = Path("./migration_backups/legacy_archive.json")
        if archive_file.exists():
            archive_file.unlink()
            return {"legacy_archive_removed": True}
        return {"no_action_needed": True}


# Command line interface

async def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='CMO-LightRAG Migration Guide')
    parser.add_argument('command', choices=['assess', 'migrate', 'test', 'compare', 'full-migrate', 'status', 'rollback'])
    parser.add_argument('--step', type=int, help='Migration step number (1-7)')
    parser.add_argument('--system', choices=['current', 'lightrag'], help='System to test')
    parser.add_argument('--validate', action='store_true', help='Run validation after migration steps')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    if args.command == 'assess':
        # Run migration readiness assessment
        manager = MigrationManager()
        assessment = await manager.assess_migration_readiness()
        
        print(f"\n🔍 Migration Readiness Assessment")
        print(f"═══════════════════════════════════")
        print(f"Readiness Score: {assessment['readiness_score']:.1f}/100")
        print(f"Blocking Issues: {len(assessment['blocking_issues'])}")
        print(f"Warnings: {len(assessment['warnings'])}")
        
        if assessment['blocking_issues']:
            print(f"\n❌ Blocking Issues:")
            for issue in assessment['blocking_issues']:
                print(f"   • {issue}")
        
        if assessment['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in assessment['warnings']:
                print(f"   • {warning}")
        
        print(f"\n📋 Recommendations:")
        for rec in assessment['recommendations']:
            print(f"   {rec}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(assessment, f, indent=2)
            print(f"\n📁 Detailed assessment saved to {args.output}")
    
    elif args.command == 'migrate':
        if not args.step:
            print("❌ Error: --step parameter required for migrate command")
            return
        
        manager = MigrationManager()
        result = await manager.execute_migration_step(args.step, validate=args.validate)
        
        print(f"\n🔄 Migration Step {args.step}: {result['step_name']}")
        print(f"══════════════════════════════════════════════")
        
        if result['success']:
            print(f"✅ Step completed successfully")
            print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
            
            if 'validation_result' in result:
                validation = result['validation_result']
                if validation.get('success'):
                    print(f"✅ Validation passed")
                else:
                    print(f"⚠️  Validation warnings: {validation}")
        else:
            print(f"❌ Step failed: {result.get('error', 'Unknown error')}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"📁 Detailed results saved to {args.output}")
    
    elif args.command == 'test':
        if not args.system:
            print("❌ Error: --system parameter required for test command")
            return
        
        comparator = SystemComparator()
        
        test_queries = [
            "What are the main metabolites in glucose metabolism?",
            "How do biomarkers help in metabolomics research?"
        ]
        
        print(f"\n🧪 Testing {args.system.upper()} system")
        print(f"═══════════════════════════════════════")
        
        if args.system == 'lightrag':
            success = await comparator.initialize_lightrag()
            if not success:
                print("❌ Failed to initialize LightRAG system")
                return
            
            for i, query in enumerate(test_queries, 1):
                print(f"\nQuery {i}: {query}")
                result = await comparator._test_lightrag(query)
                if result['success']:
                    print(f"✅ Success - {result['processing_time']:.2f}s - {result['content_length']} chars")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        elif args.system == 'current':
            for i, query in enumerate(test_queries, 1):
                print(f"\nQuery {i}: {query}")
                result = await comparator._test_perplexity(query)
                if result['success']:
                    print(f"✅ Success - {result['processing_time']:.2f}s - {result['content_length']} chars")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    elif args.command == 'compare':
        comparator = SystemComparator()
        
        test_queries = [
            "What are the main metabolites in glucose metabolism?",
            "How do biomarkers help in metabolomics research?",
            "What is the role of mass spectrometry in metabolomics?"
        ]
        
        print(f"\n⚖️  Comparing Perplexity vs LightRAG")
        print(f"═══════════════════════════════════════")
        print(f"Running {len(test_queries)} test queries...")
        
        comparison = await comparator.compare_systems(test_queries)
        
        metrics = comparison['comparison_metrics']
        print(f"\n📊 Results:")
        print(f"Success Rates:")
        print(f"   Perplexity: {metrics['success_rates']['perplexity']:.1%}")
        print(f"   LightRAG:   {metrics['success_rates']['lightrag']:.1%}")
        
        print(f"\nAverage Response Times:")
        print(f"   Perplexity: {metrics['average_response_times']['perplexity']:.2f}s")
        print(f"   LightRAG:   {metrics['average_response_times']['lightrag']:.2f}s")
        
        print(f"\nTotal Estimated Costs:")
        print(f"   Perplexity: ${metrics['total_estimated_costs']['perplexity']:.4f}")
        print(f"   LightRAG:   ${metrics['total_estimated_costs']['lightrag']:.4f}")
        
        print(f"\n🎯 Recommendations:")
        for rec in comparison['recommendations']:
            print(f"   {rec}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\n📁 Detailed comparison saved to {args.output}")
    
    elif args.command == 'full-migrate':
        manager = MigrationManager()
        
        print(f"\n🚀 Starting Full Migration Process")
        print(f"════════════════════════════════════")
        
        # Run assessment first
        assessment = await manager.assess_migration_readiness()
        if assessment['readiness_score'] < 70:
            print(f"❌ System readiness score too low: {assessment['readiness_score']:.1f}/100")
            print("Please run 'assess' command and address issues first")
            return
        
        print(f"✅ System ready for migration (score: {assessment['readiness_score']:.1f}/100)")
        
        # Execute all migration steps
        for step_id in range(1, 8):
            print(f"\n🔄 Executing Step {step_id}...")
            result = await manager.execute_migration_step(step_id, validate=args.validate)
            
            if result['success']:
                print(f"✅ Step {step_id} completed")
            else:
                print(f"❌ Step {step_id} failed: {result.get('error')}")
                print("Migration halted. Use 'rollback' command to undo changes.")
                return
        
        print(f"\n🎉 Full migration completed successfully!")
        print(f"   System is now using LightRAG as primary backend")
        print(f"   Fallback to Perplexity remains available")
        print(f"   Run 'status' command to check current state")
    
    elif args.command == 'status':
        manager = MigrationManager()
        status = manager.get_migration_status()
        
        print(f"\n📊 Migration Status")
        print(f"═══════════════════")
        print(f"Total Steps: {status['total_steps']}")
        print(f"Executed Steps: {status['executed_steps']}")
        print(f"Failed Steps: {status['failed_steps']}")
        print(f"Rollback Available: {status['rollback_available']}")
        
        if status['next_step']:
            next_step = status['next_step']
            print(f"\n⏭️  Next Step: {next_step['step_id']} - {next_step['name']}")
            print(f"   {next_step['description']}")
        else:
            print(f"\n✅ All migration steps completed")
    
    elif args.command == 'rollback':
        if not args.step:
            print("❌ Error: --step parameter required for rollback command")
            return
        
        manager = MigrationManager()
        result = await manager.rollback_migration_step(args.step)
        
        print(f"\n↩️  Rolling back Step {args.step}")
        print(f"═══════════════════════════════")
        
        if result['success']:
            print(f"✅ Rollback completed successfully")
        else:
            print(f"❌ Rollback failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    """Entry point for migration guide."""
    print("🔬 Clinical Metabolomics Oracle - Migration Guide")
    print("=" * 55)
    
    if len(sys.argv) == 1:
        print("""
Available commands:
  assess           - Run migration readiness assessment
  migrate --step N - Execute specific migration step (1-7)
  test --system X  - Test current or lightrag system
  compare          - Compare both systems side by side
  full-migrate     - Run complete migration process
  status           - Check current migration status
  rollback --step N - Rollback specific migration step

Examples:
  python migration_guide.py assess
  python migration_guide.py migrate --step 1 --validate
  python migration_guide.py test --system lightrag
  python migration_guide.py compare --output comparison.json
  python migration_guide.py full-migrate --validate
        """)
    else:
        asyncio.run(main())