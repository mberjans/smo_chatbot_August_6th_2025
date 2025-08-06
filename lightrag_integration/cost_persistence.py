"""
Cost Persistence Layer for Clinical Metabolomics Oracle LightRAG Integration

This module provides comprehensive cost tracking persistence with database schema,
historical data management, and research-specific categorization capabilities.

Classes:
    - CostRecord: Data model for individual cost entries
    - ResearchCategory: Enum for metabolomics research categories
    - CostDatabase: SQLite database management for cost tracking
    - CostPersistence: High-level interface for cost data persistence

The persistence system supports:
    - Historical cost tracking with detailed metadata
    - Research category-specific cost analysis
    - Audit trail capabilities
    - Data retention policies
    - Thread-safe database operations
"""

import sqlite3
import threading
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import logging


class ResearchCategory(Enum):
    """Research-specific categories for metabolomics cost tracking."""
    
    # Core metabolomics research areas
    METABOLITE_IDENTIFICATION = "metabolite_identification"
    PATHWAY_ANALYSIS = "pathway_analysis"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    DRUG_DISCOVERY = "drug_discovery"
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"
    
    # Data processing categories
    DATA_PREPROCESSING = "data_preprocessing"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    LITERATURE_SEARCH = "literature_search"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    
    # Integration and validation
    DATABASE_INTEGRATION = "database_integration"
    EXPERIMENTAL_VALIDATION = "experimental_validation"
    
    # General categories
    GENERAL_QUERY = "general_query"
    SYSTEM_MAINTENANCE = "system_maintenance"


@dataclass
class CostRecord:
    """
    Data model for individual cost tracking entries.
    
    This dataclass represents a single cost event with comprehensive metadata
    for analysis and audit purposes.
    """
    
    id: Optional[int] = None
    timestamp: float = None
    date_str: str = None
    session_id: str = None
    operation_type: str = None  # llm, embedding, hybrid
    model_name: str = None
    cost_usd: float = None
    
    # Token usage details
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    total_tokens: int = 0
    
    # Research categorization
    research_category: str = ResearchCategory.GENERAL_QUERY.value
    query_type: Optional[str] = None
    subject_area: Optional[str] = None
    
    # Performance metrics
    response_time_seconds: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    
    # Audit information
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = time.time()
        
        if self.date_str is None:
            self.date_str = datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat()
        
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens + self.embedding_tokens
        
        # Validate research category
        valid_categories = [cat.value for cat in ResearchCategory]
        if self.research_category not in valid_categories:
            self.research_category = ResearchCategory.GENERAL_QUERY.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert metadata to JSON string if it's a dict
        if isinstance(result.get('metadata'), dict):
            result['metadata'] = json.dumps(result['metadata'])
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostRecord':
        """Create instance from dictionary."""
        # Parse metadata from JSON string if needed
        if isinstance(data.get('metadata'), str):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except json.JSONDecodeError:
                data['metadata'] = {}
        
        return cls(**data)


class CostDatabase:
    """
    SQLite database management for cost tracking with thread safety.
    
    This class handles all database operations for cost tracking including
    schema creation, data insertion, querying, and maintenance operations.
    """
    
    def __init__(self, db_path: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the cost database.
        
        Args:
            db_path: Path to the SQLite database file
            logger: Logger instance for database operations
        """
        self.db_path = Path(db_path)
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._initialize_schema()
    
    @contextmanager
    def _get_connection(self):
        """Get a thread-safe database connection."""
        with self._lock:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,  # 30 second timeout
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
            finally:
                conn.close()
    
    def _initialize_schema(self) -> None:
        """Initialize database schema with all required tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main cost records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cost_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    date_str TEXT NOT NULL,
                    session_id TEXT,
                    operation_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    cost_usd REAL NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    embedding_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    research_category TEXT DEFAULT 'general_query',
                    query_type TEXT,
                    subject_area TEXT,
                    response_time_seconds REAL,
                    success BOOLEAN DEFAULT 1,
                    error_type TEXT,
                    user_id TEXT,
                    project_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Budget tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budget_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_type TEXT NOT NULL,  -- 'daily', 'monthly'
                    period_key TEXT NOT NULL,   -- '2025-08-06' or '2025-08'
                    total_cost REAL DEFAULT 0.0,
                    record_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(period_type, period_key)
                )
            """)
            
            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    action TEXT NOT NULL,
                    table_name TEXT,
                    record_id INTEGER,
                    old_values TEXT,
                    new_values TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_records_timestamp ON cost_records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_records_date ON cost_records(date_str)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_records_category ON cost_records(research_category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cost_records_session ON cost_records(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_budget_tracking_period ON budget_tracking(period_type, period_key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)")
            
            conn.commit()
            self.logger.debug(f"Database schema initialized at {self.db_path}")
    
    def insert_cost_record(self, record: CostRecord) -> int:
        """
        Insert a cost record into the database.
        
        Args:
            record: CostRecord instance to insert
            
        Returns:
            int: The ID of the inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            record_dict = record.to_dict()
            record_dict.pop('id', None)  # Remove id if present
            
            placeholders = ', '.join(['?'] * len(record_dict))
            columns = ', '.join(record_dict.keys())
            
            cursor.execute(
                f"INSERT INTO cost_records ({columns}) VALUES ({placeholders})",
                list(record_dict.values())
            )
            
            record_id = cursor.lastrowid
            conn.commit()
            
            # Update budget tracking
            self._update_budget_tracking(conn, record)
            
            # Log audit entry
            self._log_audit_action(
                conn, 
                "INSERT", 
                "cost_records", 
                record_id,
                new_values=record_dict,
                session_id=record.session_id
            )
            
            self.logger.debug(f"Inserted cost record with ID {record_id}")
            return record_id
    
    def _update_budget_tracking(self, conn: sqlite3.Connection, record: CostRecord) -> None:
        """Update budget tracking tables with new cost record."""
        cursor = conn.cursor()
        
        # Get date components
        dt = datetime.fromtimestamp(record.timestamp, timezone.utc)
        daily_key = dt.strftime('%Y-%m-%d')
        monthly_key = dt.strftime('%Y-%m')
        
        # Update daily budget
        cursor.execute("""
            INSERT OR IGNORE INTO budget_tracking (period_type, period_key, total_cost, record_count)
            VALUES (?, ?, 0.0, 0)
        """, ('daily', daily_key))
        
        cursor.execute("""
            UPDATE budget_tracking 
            SET total_cost = total_cost + ?, 
                record_count = record_count + 1,
                last_updated = CURRENT_TIMESTAMP
            WHERE period_type = ? AND period_key = ?
        """, (record.cost_usd, 'daily', daily_key))
        
        # Update monthly budget
        cursor.execute("""
            INSERT OR IGNORE INTO budget_tracking (period_type, period_key, total_cost, record_count)
            VALUES (?, ?, 0.0, 0)
        """, ('monthly', monthly_key))
        
        cursor.execute("""
            UPDATE budget_tracking 
            SET total_cost = total_cost + ?, 
                record_count = record_count + 1,
                last_updated = CURRENT_TIMESTAMP
            WHERE period_type = ? AND period_key = ?
        """, (record.cost_usd, 'monthly', monthly_key))
        
        conn.commit()
    
    def _log_audit_action(self, 
                         conn: sqlite3.Connection,
                         action: str,
                         table_name: str,
                         record_id: Optional[int] = None,
                         old_values: Optional[Dict] = None,
                         new_values: Optional[Dict] = None,
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> None:
        """Log an audit action."""
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_log 
            (timestamp, action, table_name, record_id, old_values, new_values, user_id, session_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            action,
            table_name,
            record_id,
            json.dumps(old_values) if old_values else None,
            json.dumps(new_values) if new_values else None,
            user_id,
            session_id,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
    
    def get_budget_summary(self, 
                          period_type: str = 'daily',
                          period_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get budget summary for a specific period.
        
        Args:
            period_type: 'daily' or 'monthly'
            period_key: Specific period key (e.g., '2025-08-06' for daily)
                       If None, uses current period
                       
        Returns:
            Dict containing budget summary information
        """
        if period_key is None:
            dt = datetime.now(timezone.utc)
            if period_type == 'daily':
                period_key = dt.strftime('%Y-%m-%d')
            else:  # monthly
                period_key = dt.strftime('%Y-%m')
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM budget_tracking 
                WHERE period_type = ? AND period_key = ?
            """, (period_type, period_key))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'period_type': row['period_type'],
                    'period_key': row['period_key'],
                    'total_cost': row['total_cost'],
                    'record_count': row['record_count'],
                    'last_updated': row['last_updated']
                }
            else:
                return {
                    'period_type': period_type,
                    'period_key': period_key,
                    'total_cost': 0.0,
                    'record_count': 0,
                    'last_updated': None
                }
    
    def get_cost_records(self,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        research_category: Optional[str] = None,
                        session_id: Optional[str] = None,
                        limit: int = 1000) -> List[CostRecord]:
        """
        Retrieve cost records with optional filtering.
        
        Args:
            start_time: Unix timestamp for start of time range
            end_time: Unix timestamp for end of time range
            research_category: Filter by research category
            session_id: Filter by session ID
            limit: Maximum number of records to return
            
        Returns:
            List of CostRecord instances
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM cost_records WHERE 1=1"
            params = []
            
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            if research_category:
                query += " AND research_category = ?"
                params.append(research_category)
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                record_dict = dict(row)
                # Parse metadata if it's a JSON string
                if record_dict.get('metadata'):
                    try:
                        record_dict['metadata'] = json.loads(record_dict['metadata'])
                    except json.JSONDecodeError:
                        record_dict['metadata'] = {}
                
                records.append(CostRecord.from_dict(record_dict))
            
            return records
    
    def get_research_category_summary(self, 
                                    start_time: Optional[float] = None,
                                    end_time: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Get cost summary by research category.
        
        Args:
            start_time: Unix timestamp for start of time range
            end_time: Unix timestamp for end of time range
            
        Returns:
            Dict mapping category names to cost and count summaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    research_category,
                    COUNT(*) as record_count,
                    SUM(cost_usd) as total_cost,
                    AVG(cost_usd) as avg_cost,
                    SUM(total_tokens) as total_tokens
                FROM cost_records 
                WHERE 1=1
            """
            params = []
            
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " GROUP BY research_category ORDER BY total_cost DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            summary = {}
            for row in rows:
                summary[row['research_category']] = {
                    'record_count': row['record_count'],
                    'total_cost': row['total_cost'],
                    'avg_cost': row['avg_cost'],
                    'total_tokens': row['total_tokens']
                }
            
            return summary
    
    def cleanup_old_records(self, retention_days: int = 365) -> int:
        """
        Clean up old cost records based on retention policy.
        
        Args:
            retention_days: Number of days to retain records
            
        Returns:
            int: Number of records deleted
        """
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # First, get count of records to be deleted
            cursor.execute("SELECT COUNT(*) FROM cost_records WHERE timestamp < ?", (cutoff_time,))
            delete_count = cursor.fetchone()[0]
            
            if delete_count > 0:
                # Log audit action
                self._log_audit_action(
                    conn,
                    "CLEANUP",
                    "cost_records",
                    metadata={
                        "retention_days": retention_days,
                        "records_deleted": delete_count,
                        "cutoff_timestamp": cutoff_time
                    }
                )
                
                # Delete old records
                cursor.execute("DELETE FROM cost_records WHERE timestamp < ?", (cutoff_time,))
                conn.commit()
                
                self.logger.info(f"Cleaned up {delete_count} cost records older than {retention_days} days")
            
            return delete_count


class CostPersistence:
    """
    High-level interface for cost data persistence and management.
    
    This class provides a comprehensive interface for cost tracking persistence,
    integrating database operations with business logic for budget management,
    research categorization, and audit capabilities.
    """
    
    def __init__(self, 
                 db_path: Path,
                 retention_days: int = 365,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize cost persistence layer.
        
        Args:
            db_path: Path to the cost tracking database
            retention_days: Number of days to retain cost records
            logger: Logger instance for operations
        """
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize database
        self.db = CostDatabase(db_path, logger)
        
        self.logger.info(f"Cost persistence initialized with database at {db_path}")
    
    def record_cost(self,
                   cost_usd: float,
                   operation_type: str,
                   model_name: str,
                   token_usage: Dict[str, int],
                   session_id: Optional[str] = None,
                   research_category: ResearchCategory = ResearchCategory.GENERAL_QUERY,
                   query_type: Optional[str] = None,
                   subject_area: Optional[str] = None,
                   response_time: Optional[float] = None,
                   success: bool = True,
                   error_type: Optional[str] = None,
                   user_id: Optional[str] = None,
                   project_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Record a cost entry with comprehensive metadata.
        
        Args:
            cost_usd: Cost in USD
            operation_type: Type of operation (llm, embedding, hybrid)
            model_name: Name of the model used
            token_usage: Dictionary with token counts
            session_id: Session identifier
            research_category: Research category for the operation
            query_type: Type of query performed
            subject_area: Subject area of the research
            response_time: Response time in seconds
            success: Whether the operation was successful
            error_type: Type of error if not successful
            user_id: User identifier
            project_id: Project identifier
            metadata: Additional metadata
            
        Returns:
            int: Record ID of the inserted cost entry
        """
        record = CostRecord(
            cost_usd=cost_usd,
            operation_type=operation_type,
            model_name=model_name,
            prompt_tokens=token_usage.get('prompt_tokens', 0),
            completion_tokens=token_usage.get('completion_tokens', 0),
            embedding_tokens=token_usage.get('embedding_tokens', 0),
            session_id=session_id,
            research_category=research_category.value,
            query_type=query_type,
            subject_area=subject_area,
            response_time_seconds=response_time,
            success=success,
            error_type=error_type,
            user_id=user_id,
            project_id=project_id,
            metadata=metadata or {}
        )
        
        record_id = self.db.insert_cost_record(record)
        
        self.logger.debug(f"Recorded cost entry: ${cost_usd:.4f} for {operation_type} operation")
        return record_id
    
    def get_daily_budget_status(self, 
                               date: Optional[datetime] = None,
                               budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Get current daily budget status.
        
        Args:
            date: Date to check (defaults to today)
            budget_limit: Daily budget limit for comparison
            
        Returns:
            Dict containing budget status information
        """
        if date is None:
            date = datetime.now(timezone.utc)
        
        period_key = date.strftime('%Y-%m-%d')
        summary = self.db.get_budget_summary('daily', period_key)
        
        result = {
            'date': period_key,
            'total_cost': summary['total_cost'],
            'record_count': summary['record_count'],
            'last_updated': summary['last_updated']
        }
        
        if budget_limit is not None:
            result['budget_limit'] = budget_limit
            result['remaining_budget'] = budget_limit - summary['total_cost']
            result['percentage_used'] = (summary['total_cost'] / budget_limit) * 100 if budget_limit > 0 else 0
            result['over_budget'] = summary['total_cost'] > budget_limit
        
        return result
    
    def get_monthly_budget_status(self,
                                 date: Optional[datetime] = None,
                                 budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Get current monthly budget status.
        
        Args:
            date: Date to check (defaults to current month)
            budget_limit: Monthly budget limit for comparison
            
        Returns:
            Dict containing budget status information
        """
        if date is None:
            date = datetime.now(timezone.utc)
        
        period_key = date.strftime('%Y-%m')
        summary = self.db.get_budget_summary('monthly', period_key)
        
        result = {
            'month': period_key,
            'total_cost': summary['total_cost'],
            'record_count': summary['record_count'],
            'last_updated': summary['last_updated']
        }
        
        if budget_limit is not None:
            result['budget_limit'] = budget_limit
            result['remaining_budget'] = budget_limit - summary['total_cost']
            result['percentage_used'] = (summary['total_cost'] / budget_limit) * 100 if budget_limit > 0 else 0
            result['over_budget'] = summary['total_cost'] > budget_limit
        
        return result
    
    def get_research_analysis(self,
                            days: int = 30) -> Dict[str, Any]:
        """
        Get research-specific cost analysis.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict containing research cost analysis
        """
        start_time = time.time() - (days * 24 * 60 * 60)
        
        category_summary = self.db.get_research_category_summary(start_time)
        total_cost = sum(cat['total_cost'] for cat in category_summary.values())
        total_records = sum(cat['record_count'] for cat in category_summary.values())
        
        # Calculate percentages
        for category_data in category_summary.values():
            category_data['percentage_of_total'] = (category_data['total_cost'] / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'period_days': days,
            'total_cost': total_cost,
            'total_records': total_records,
            'average_cost_per_record': total_cost / total_records if total_records > 0 else 0,
            'categories': category_summary,
            'top_categories': sorted(
                category_summary.items(),
                key=lambda x: x[1]['total_cost'],
                reverse=True
            )[:5]  # Top 5 categories by cost
        }
    
    def generate_cost_report(self,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive cost report.
        
        Args:
            start_date: Start date for the report
            end_date: End date for the report
            
        Returns:
            Dict containing comprehensive cost report
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # Default to 30 days
        
        start_time = start_date.timestamp()
        end_time = end_date.timestamp()
        
        # Get records for the period
        records = self.db.get_cost_records(start_time, end_time, limit=10000)
        
        if not records:
            return {
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_cost': 0.0,
                'total_records': 0,
                'message': 'No cost records found for the specified period'
            }
        
        # Calculate summary statistics
        total_cost = sum(record.cost_usd for record in records)
        total_tokens = sum(record.total_tokens for record in records)
        successful_records = [r for r in records if r.success]
        failed_records = [r for r in records if not r.success]
        
        # Group by operation type
        operation_summary = {}
        for record in records:
            if record.operation_type not in operation_summary:
                operation_summary[record.operation_type] = {
                    'count': 0,
                    'total_cost': 0.0,
                    'total_tokens': 0
                }
            
            op_summary = operation_summary[record.operation_type]
            op_summary['count'] += 1
            op_summary['total_cost'] += record.cost_usd
            op_summary['total_tokens'] += record.total_tokens
        
        # Group by model
        model_summary = {}
        for record in records:
            if record.model_name not in model_summary:
                model_summary[record.model_name] = {
                    'count': 0,
                    'total_cost': 0.0,
                    'total_tokens': 0
                }
            
            model_sum = model_summary[record.model_name]
            model_sum['count'] += 1
            model_sum['total_cost'] += record.cost_usd
            model_sum['total_tokens'] += record.total_tokens
        
        # Research category analysis
        research_summary = self.db.get_research_category_summary(start_time, end_time)
        
        return {
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'summary': {
                'total_cost': total_cost,
                'total_records': len(records),
                'successful_operations': len(successful_records),
                'failed_operations': len(failed_records),
                'success_rate': len(successful_records) / len(records) * 100,
                'total_tokens': total_tokens,
                'average_cost_per_operation': total_cost / len(records),
                'average_tokens_per_operation': total_tokens / len(records)
            },
            'operation_breakdown': operation_summary,
            'model_breakdown': model_summary,
            'research_categories': research_summary,
            'daily_costs': self._get_daily_cost_breakdown(records),
            'error_analysis': self._get_error_analysis(failed_records) if failed_records else {}
        }
    
    def _get_daily_cost_breakdown(self, records: List[CostRecord]) -> Dict[str, float]:
        """Get daily cost breakdown from records."""
        daily_costs = {}
        for record in records:
            date_str = datetime.fromtimestamp(record.timestamp, timezone.utc).strftime('%Y-%m-%d')
            daily_costs[date_str] = daily_costs.get(date_str, 0.0) + record.cost_usd
        
        return daily_costs
    
    def _get_error_analysis(self, failed_records: List[CostRecord]) -> Dict[str, Any]:
        """Analyze error patterns in failed records."""
        if not failed_records:
            return {}
        
        error_types = {}
        for record in failed_records:
            error_type = record.error_type or 'unknown'
            if error_type not in error_types:
                error_types[error_type] = {
                    'count': 0,
                    'total_cost': 0.0,
                    'operations': []
                }
            
            error_types[error_type]['count'] += 1
            error_types[error_type]['total_cost'] += record.cost_usd
            error_types[error_type]['operations'].append(record.operation_type)
        
        return {
            'total_failed_operations': len(failed_records),
            'error_types': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1]['count'])[0] if error_types else None
        }
    
    def cleanup_old_data(self) -> int:
        """
        Clean up old cost records based on retention policy.
        
        Returns:
            int: Number of records deleted
        """
        return self.db.cleanup_old_records(self.retention_days)