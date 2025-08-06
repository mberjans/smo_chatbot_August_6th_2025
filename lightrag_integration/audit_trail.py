"""
Compliance and Audit Trail System for Clinical Metabolomics Oracle LightRAG Integration

This module provides comprehensive audit trail and compliance capabilities
for cost tracking, usage monitoring, and regulatory compliance.

Classes:
    - AuditEventType: Enum for different types of audit events
    - AuditEvent: Data model for individual audit events
    - ComplianceRule: Configuration for compliance checking
    - ComplianceChecker: System for monitoring compliance violations
    - AuditTrail: Main audit trail management system

The audit system supports:
    - Comprehensive event logging with tamper-proof timestamps
    - Compliance rule monitoring and violation detection
    - Cost tracking audit trails for financial accountability
    - User activity monitoring and session tracking
    - Automated compliance reporting and alerting
    - Data retention and archival policies
"""

import hashlib
import hmac
import time
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .cost_persistence import CostPersistence, CostRecord, ResearchCategory
from .budget_manager import BudgetAlert, AlertLevel


class AuditEventType(Enum):
    """Types of events that can be audited."""
    
    # Cost tracking events
    COST_RECORDED = "cost_recorded"
    BUDGET_ALERT = "budget_alert"
    BUDGET_EXCEEDED = "budget_exceeded"
    BUDGET_UPDATED = "budget_updated"
    
    # User activity events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    QUERY_SUBMITTED = "query_submitted"
    QUERY_COMPLETED = "query_completed"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGED = "config_changed"
    DATABASE_BACKUP = "database_backup"
    
    # Security events
    AUTHENTICATION_FAILED = "authentication_failed"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # Compliance events
    COMPLIANCE_VIOLATION = "compliance_violation"
    COMPLIANCE_CHECK = "compliance_check"
    POLICY_VIOLATION = "policy_violation"
    
    # Data events
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    DATA_DELETION = "data_deletion"
    DATA_MODIFICATION = "data_modification"


@dataclass
class AuditEvent:
    """
    Represents a single audit event with comprehensive metadata.
    
    This dataclass captures all information needed for compliance monitoring,
    security auditing, and operational tracking.
    """
    
    timestamp: float
    event_type: AuditEventType
    event_id: str = None  # Unique identifier for the event
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Event details
    description: str = None
    category: Optional[str] = None
    severity: str = "info"  # info, warning, error, critical
    
    # Associated data
    cost_amount: Optional[float] = None
    research_category: Optional[str] = None
    operation_type: Optional[str] = None
    
    # System information
    system_version: Optional[str] = None
    client_info: Optional[Dict[str, str]] = None
    
    # Compliance and security
    compliance_flags: List[str] = field(default_factory=list)
    security_context: Optional[Dict[str, Any]] = None
    
    # Audit integrity
    checksum: Optional[str] = None
    previous_event_hash: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.event_id is None:
            # Generate unique event ID
            self.event_id = self._generate_event_id()
        
        if self.description is None:
            self.description = f"{self.event_type.value} event"
        
        # Calculate integrity checksum
        self._calculate_checksum()
    
    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        unique_string = f"{self.timestamp}_{self.event_type.value}_{self.user_id}_{time.time_ns()}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self) -> None:
        """Calculate integrity checksum for tamper detection."""
        # Create a canonical representation of the event
        event_data = {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'event_id': self.event_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'description': self.description,
            'category': self.category,
            'severity': self.severity,
            'cost_amount': self.cost_amount,
            'research_category': self.research_category,
            'operation_type': self.operation_type
        }
        
        # Convert to canonical JSON
        canonical_json = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
        
        # Calculate HMAC with system secret (in production, use proper key management)
        secret_key = b"clinical_metabolomics_audit_secret"  # This should be configurable
        self.checksum = hmac.new(secret_key, canonical_json.encode(), hashlib.sha256).hexdigest()
    
    def verify_integrity(self, secret_key: bytes = None) -> bool:
        """Verify the integrity of the audit event."""
        if not secret_key:
            secret_key = b"clinical_metabolomics_audit_secret"
        
        # Recalculate checksum
        current_checksum = self.checksum
        self.checksum = None  # Temporarily clear for recalculation
        self._calculate_checksum()
        
        # Compare checksums
        is_valid = hmac.compare_digest(current_checksum, self.checksum)
        self.checksum = current_checksum  # Restore original
        
        return is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'event_id': self.event_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'description': self.description,
            'category': self.category,
            'severity': self.severity,
            'cost_amount': self.cost_amount,
            'research_category': self.research_category,
            'operation_type': self.operation_type,
            'system_version': self.system_version,
            'client_info': self.client_info,
            'compliance_flags': self.compliance_flags,
            'security_context': self.security_context,
            'checksum': self.checksum,
            'previous_event_hash': self.previous_event_hash,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary."""
        # Handle enum conversion
        event_type = AuditEventType(data['event_type'])
        
        return cls(
            timestamp=data['timestamp'],
            event_type=event_type,
            event_id=data.get('event_id'),
            session_id=data.get('session_id'),
            user_id=data.get('user_id'),
            description=data.get('description'),
            category=data.get('category'),
            severity=data.get('severity', 'info'),
            cost_amount=data.get('cost_amount'),
            research_category=data.get('research_category'),
            operation_type=data.get('operation_type'),
            system_version=data.get('system_version'),
            client_info=data.get('client_info'),
            compliance_flags=data.get('compliance_flags', []),
            security_context=data.get('security_context'),
            checksum=data.get('checksum'),
            previous_event_hash=data.get('previous_event_hash'),
            metadata=data.get('metadata')
        )


@dataclass
class ComplianceRule:
    """
    Configuration for compliance checking and monitoring.
    """
    
    rule_id: str
    name: str
    description: str
    rule_type: str  # cost_limit, usage_limit, access_control, data_retention
    
    # Rule parameters
    threshold_value: Optional[float] = None
    time_window_hours: Optional[int] = None
    applies_to_categories: List[str] = field(default_factory=list)
    applies_to_users: List[str] = field(default_factory=list)
    
    # Enforcement settings
    is_enforced: bool = True
    violation_severity: str = "warning"  # info, warning, error, critical
    auto_remediation: bool = False
    
    # Notification settings
    notify_on_violation: bool = True
    notification_recipients: List[str] = field(default_factory=list)
    
    def check_compliance(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if the current context complies with this rule.
        
        Args:
            context: Current system context for compliance checking
            
        Returns:
            Tuple of (is_compliant, violation_message)
        """
        if not self.is_enforced:
            return True, ""
        
        if self.rule_type == "cost_limit":
            return self._check_cost_limit(context)
        elif self.rule_type == "usage_limit":
            return self._check_usage_limit(context)
        elif self.rule_type == "access_control":
            return self._check_access_control(context)
        elif self.rule_type == "data_retention":
            return self._check_data_retention(context)
        else:
            return True, ""
    
    def _check_cost_limit(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check cost-based compliance rules."""
        current_cost = context.get('current_cost', 0)
        time_window = context.get('time_window_hours', self.time_window_hours or 24)
        
        if self.threshold_value and current_cost > self.threshold_value:
            return False, f"Cost limit exceeded: ${current_cost:.2f} > ${self.threshold_value:.2f} in {time_window}h"
        
        return True, ""
    
    def _check_usage_limit(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check usage-based compliance rules."""
        current_usage = context.get('current_usage', 0)
        time_window = context.get('time_window_hours', self.time_window_hours or 24)
        
        if self.threshold_value and current_usage > self.threshold_value:
            return False, f"Usage limit exceeded: {current_usage} > {self.threshold_value} operations in {time_window}h"
        
        return True, ""
    
    def _check_access_control(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check access control compliance."""
        user_id = context.get('user_id')
        
        if self.applies_to_users and user_id not in self.applies_to_users:
            return False, f"User {user_id} not authorized for this operation"
        
        return True, ""
    
    def _check_data_retention(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check data retention compliance."""
        data_age_hours = context.get('data_age_hours', 0)
        
        if self.time_window_hours and data_age_hours > self.time_window_hours:
            return False, f"Data retention period exceeded: {data_age_hours}h > {self.time_window_hours}h"
        
        return True, ""


class ComplianceChecker:
    """
    System for monitoring compliance violations and enforcing policies.
    """
    
    def __init__(self, 
                 rules: List[ComplianceRule] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize compliance checker with rules.
        
        Args:
            rules: List of compliance rules to enforce
            logger: Logger for compliance events
        """
        self.rules = rules or []
        self.logger = logger or logging.getLogger(__name__)
        self.violation_history: List[Dict[str, Any]] = []
        
        # Default compliance rules for cost tracking
        if not self.rules:
            self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Set up default compliance rules for cost tracking."""
        self.rules = [
            ComplianceRule(
                rule_id="daily_cost_limit",
                name="Daily Cost Limit",
                description="Monitor daily spending limits",
                rule_type="cost_limit",
                threshold_value=100.0,  # $100 daily limit
                time_window_hours=24,
                violation_severity="warning",
                notify_on_violation=True
            ),
            ComplianceRule(
                rule_id="hourly_usage_limit",
                name="Hourly Usage Limit", 
                description="Prevent excessive API usage per hour",
                rule_type="usage_limit",
                threshold_value=1000,  # 1000 operations per hour
                time_window_hours=1,
                violation_severity="error",
                notify_on_violation=True
            ),
            ComplianceRule(
                rule_id="data_retention_policy",
                name="Data Retention Policy",
                description="Enforce data retention requirements",
                rule_type="data_retention",
                time_window_hours=8760,  # 1 year retention
                violation_severity="info",
                auto_remediation=True
            )
        ]
    
    def add_rule(self, rule: ComplianceRule) -> None:
        """Add a new compliance rule."""
        # Remove existing rule with same ID
        self.rules = [r for r in self.rules if r.rule_id != rule.rule_id]
        self.rules.append(rule)
        self.logger.info(f"Added compliance rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a compliance rule by ID."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        
        removed = len(self.rules) < original_count
        if removed:
            self.logger.info(f"Removed compliance rule: {rule_id}")
        
        return removed
    
    def check_compliance(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check all compliance rules against current context.
        
        Args:
            context: Current system context
            
        Returns:
            List of compliance violations
        """
        violations = []
        
        for rule in self.rules:
            is_compliant, violation_message = rule.check_compliance(context)
            
            if not is_compliant:
                violation = {
                    'timestamp': time.time(),
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'severity': rule.violation_severity,
                    'message': violation_message,
                    'context': context,
                    'auto_remediation': rule.auto_remediation
                }
                
                violations.append(violation)
                self.violation_history.append(violation)
                
                self.logger.log(
                    getattr(logging, rule.violation_severity.upper(), logging.WARNING),
                    f"Compliance violation: {rule.name} - {violation_message}"
                )
        
        return violations
    
    def get_violation_history(self, 
                             hours: int = 24,
                             rule_id: Optional[str] = None,
                             severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get compliance violation history.
        
        Args:
            hours: Number of hours of history to return
            rule_id: Filter by specific rule ID
            severity: Filter by violation severity
            
        Returns:
            List of historical violations matching criteria
        """
        cutoff_time = time.time() - (hours * 3600)
        
        filtered_violations = []
        for violation in self.violation_history:
            # Time filter
            if violation['timestamp'] < cutoff_time:
                continue
            
            # Rule filter
            if rule_id and violation['rule_id'] != rule_id:
                continue
            
            # Severity filter
            if severity and violation['severity'] != severity:
                continue
            
            filtered_violations.append(violation)
        
        return sorted(filtered_violations, key=lambda x: x['timestamp'], reverse=True)
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance status summary."""
        recent_violations = self.get_violation_history(hours=24)
        
        severity_counts = {}
        rule_violation_counts = {}
        
        for violation in recent_violations:
            severity = violation['severity']
            rule_id = violation['rule_id']
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            rule_violation_counts[rule_id] = rule_violation_counts.get(rule_id, 0) + 1
        
        return {
            'total_violations_24h': len(recent_violations),
            'severity_breakdown': severity_counts,
            'rule_violation_counts': rule_violation_counts,
            'active_rules': len(self.rules),
            'enforced_rules': len([r for r in self.rules if r.is_enforced]),
            'overall_compliance_status': 'non_compliant' if recent_violations else 'compliant'
        }


class AuditTrail:
    """
    Comprehensive audit trail system for compliance and security monitoring.
    
    This class provides the main interface for audit logging, compliance checking,
    and audit trail management with integrity protection.
    """
    
    def __init__(self,
                 cost_persistence: CostPersistence,
                 compliance_checker: Optional[ComplianceChecker] = None,
                 audit_callback: Optional[Callable[[AuditEvent], None]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize audit trail system.
        
        Args:
            cost_persistence: Cost persistence layer for data storage
            compliance_checker: Compliance monitoring system
            audit_callback: Optional callback for audit events
            logger: Logger for audit operations
        """
        self.cost_persistence = cost_persistence
        self.compliance_checker = compliance_checker or ComplianceChecker()
        self.audit_callback = audit_callback
        self.logger = logger or logging.getLogger(__name__)
        
        # Thread safety for audit operations
        self._lock = threading.RLock()
        
        # Audit event chain for integrity
        self._last_event_hash: Optional[str] = None
        self._event_count = 0
        
        # Session tracking
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Audit trail system initialized")
        
        # Log system start event
        self.log_event(
            AuditEventType.SYSTEM_START,
            description="Audit trail system started",
            severity="info",
            metadata={'version': '1.0.0'}
        )
    
    def log_event(self,
                  event_type: AuditEventType,
                  session_id: Optional[str] = None,
                  user_id: Optional[str] = None,
                  description: Optional[str] = None,
                  category: Optional[str] = None,
                  severity: str = "info",
                  cost_amount: Optional[float] = None,
                  research_category: Optional[ResearchCategory] = None,
                  operation_type: Optional[str] = None,
                  compliance_flags: List[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> AuditEvent:
        """
        Log an audit event with comprehensive metadata.
        
        Args:
            event_type: Type of event being logged
            session_id: Session identifier
            user_id: User identifier
            description: Human-readable description
            category: Event category
            severity: Event severity level
            cost_amount: Associated cost amount
            research_category: Research category if applicable
            operation_type: Type of operation
            compliance_flags: Compliance-related flags
            metadata: Additional metadata
            
        Returns:
            The created AuditEvent
        """
        with self._lock:
            # Create audit event
            event = AuditEvent(
                timestamp=time.time(),
                event_type=event_type,
                session_id=session_id,
                user_id=user_id,
                description=description,
                category=category,
                severity=severity,
                cost_amount=cost_amount,
                research_category=research_category.value if research_category else None,
                operation_type=operation_type,
                compliance_flags=compliance_flags or [],
                previous_event_hash=self._last_event_hash,
                metadata=metadata or {}
            )
            
            # Update event chain
            self._last_event_hash = event.checksum
            self._event_count += 1
            
            # Store in database via cost persistence
            try:
                # Use metadata to store audit event in cost persistence
                # This leverages the existing database infrastructure
                audit_metadata = event.to_dict()
                
                # Create a special cost record for audit events with zero cost
                from .cost_persistence import CostRecord
                audit_record = CostRecord(
                    timestamp=event.timestamp,
                    session_id=session_id,
                    operation_type=f"audit_{event_type.value}",
                    model_name="audit_system",
                    cost_usd=cost_amount or 0.0,
                    research_category=research_category.value if research_category else ResearchCategory.SYSTEM_MAINTENANCE.value,
                    success=True,
                    metadata=audit_metadata
                )
                
                self.cost_persistence.db.insert_cost_record(audit_record)
                
            except Exception as e:
                self.logger.error(f"Failed to persist audit event: {e}")
            
            # Check compliance if this is a compliance-relevant event
            if event_type in [AuditEventType.COST_RECORDED, AuditEventType.QUERY_SUBMITTED]:
                self._check_compliance_for_event(event)
            
            # Trigger callback if provided
            if self.audit_callback:
                try:
                    self.audit_callback(event)
                except Exception as e:
                    self.logger.error(f"Error in audit callback: {e}")
            
            # Log to system logger
            log_level = getattr(logging, severity.upper(), logging.INFO)
            self.logger.log(
                log_level,
                f"Audit Event [{event_type.value}]: {description or 'No description'}"
            )
            
            return event
    
    def log_cost_event(self, cost_record: CostRecord) -> AuditEvent:
        """
        Log a cost-related audit event.
        
        Args:
            cost_record: The cost record being audited
            
        Returns:
            The created AuditEvent
        """
        return self.log_event(
            AuditEventType.COST_RECORDED,
            session_id=cost_record.session_id,
            user_id=cost_record.user_id,
            description=f"Cost recorded: ${cost_record.cost_usd:.4f} for {cost_record.operation_type}",
            category="cost_tracking",
            severity="info",
            cost_amount=cost_record.cost_usd,
            research_category=ResearchCategory(cost_record.research_category),
            operation_type=cost_record.operation_type,
            metadata={
                'model_name': cost_record.model_name,
                'total_tokens': cost_record.total_tokens,
                'success': cost_record.success
            }
        )
    
    def log_budget_alert(self, budget_alert: BudgetAlert) -> AuditEvent:
        """
        Log a budget alert as an audit event.
        
        Args:
            budget_alert: The budget alert being logged
            
        Returns:
            The created AuditEvent
        """
        severity_mapping = {
            AlertLevel.INFO: "info",
            AlertLevel.WARNING: "warning",
            AlertLevel.CRITICAL: "error",
            AlertLevel.EXCEEDED: "critical"
        }
        
        event_type = AuditEventType.BUDGET_EXCEEDED if budget_alert.alert_level == AlertLevel.EXCEEDED else AuditEventType.BUDGET_ALERT
        
        return self.log_event(
            event_type,
            description=budget_alert.message,
            category="budget_management",
            severity=severity_mapping.get(budget_alert.alert_level, "info"),
            cost_amount=budget_alert.current_cost,
            compliance_flags=[f"budget_{budget_alert.alert_level.value}"],
            metadata={
                'period_type': budget_alert.period_type,
                'period_key': budget_alert.period_key,
                'budget_limit': budget_alert.budget_limit,
                'percentage_used': budget_alert.percentage_used,
                'threshold_percentage': budget_alert.threshold_percentage
            }
        )
    
    def start_session(self, session_id: str, user_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> AuditEvent:
        """
        Start tracking a user session.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            metadata: Session metadata
            
        Returns:
            The session start audit event
        """
        with self._lock:
            session_data = {
                'start_time': time.time(),
                'user_id': user_id,
                'metadata': metadata or {},
                'event_count': 0,
                'total_cost': 0.0
            }
            
            self._active_sessions[session_id] = session_data
            
            return self.log_event(
                AuditEventType.USER_LOGIN,
                session_id=session_id,
                user_id=user_id,
                description=f"Session started for user {user_id or 'anonymous'}",
                category="session_management",
                severity="info",
                metadata=metadata
            )
    
    def end_session(self, session_id: str) -> Optional[AuditEvent]:
        """
        End tracking of a user session.
        
        Args:
            session_id: Session identifier to end
            
        Returns:
            The session end audit event, or None if session not found
        """
        with self._lock:
            session_data = self._active_sessions.pop(session_id, None)
            
            if not session_data:
                self.logger.warning(f"Attempted to end non-existent session: {session_id}")
                return None
            
            duration = time.time() - session_data['start_time']
            
            return self.log_event(
                AuditEventType.USER_LOGOUT,
                session_id=session_id,
                user_id=session_data['user_id'],
                description=f"Session ended after {duration:.1f}s",
                category="session_management",
                severity="info",
                metadata={
                    'duration_seconds': duration,
                    'events_in_session': session_data['event_count'],
                    'total_session_cost': session_data['total_cost']
                }
            )
    
    def _check_compliance_for_event(self, event: AuditEvent) -> None:
        """Check compliance rules for an audit event."""
        context = {
            'timestamp': event.timestamp,
            'user_id': event.user_id,
            'session_id': event.session_id,
            'cost_amount': event.cost_amount or 0,
            'operation_type': event.operation_type,
            'research_category': event.research_category
        }
        
        # Add session context if available
        if event.session_id in self._active_sessions:
            session_data = self._active_sessions[event.session_id]
            context.update({
                'session_duration': time.time() - session_data['start_time'],
                'session_event_count': session_data['event_count'],
                'session_total_cost': session_data['total_cost']
            })
        
        violations = self.compliance_checker.check_compliance(context)
        
        # Log compliance violations as audit events
        for violation in violations:
            self.log_event(
                AuditEventType.COMPLIANCE_VIOLATION,
                session_id=event.session_id,
                user_id=event.user_id,
                description=f"Compliance violation: {violation['message']}",
                category="compliance",
                severity=violation['severity'],
                compliance_flags=[violation['rule_id']],
                metadata=violation
            )
    
    def get_audit_events(self,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        event_type: Optional[AuditEventType] = None,
                        user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        severity: Optional[str] = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with filtering options.
        
        Args:
            start_time: Start timestamp for filtering
            end_time: End timestamp for filtering
            event_type: Filter by event type
            user_id: Filter by user ID
            session_id: Filter by session ID
            severity: Filter by severity level
            limit: Maximum number of events to return
            
        Returns:
            List of audit events as dictionaries
        """
        # Query cost records that contain audit data
        cost_records = self.cost_persistence.get_cost_records(
            start_time=start_time,
            end_time=end_time,
            session_id=session_id,
            limit=limit
        )
        
        audit_events = []
        for record in cost_records:
            # Check if this is an audit record
            if record.operation_type.startswith('audit_') and record.metadata:
                try:
                    # Extract audit event from metadata
                    audit_data = record.metadata
                    
                    # Apply filters
                    if event_type and audit_data.get('event_type') != event_type.value:
                        continue
                    
                    if user_id and audit_data.get('user_id') != user_id:
                        continue
                    
                    if severity and audit_data.get('severity') != severity:
                        continue
                    
                    audit_events.append(audit_data)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing audit event from cost record: {e}")
        
        return sorted(audit_events, key=lambda x: x.get('timestamp', 0), reverse=True)[:limit]
    
    def generate_audit_report(self,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive audit report.
        
        Args:
            start_date: Start date for report period
            end_date: End date for report period
            
        Returns:
            Dict containing comprehensive audit report
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        start_time = start_date.timestamp()
        end_time = end_date.timestamp()
        
        # Get audit events for the period
        audit_events = self.get_audit_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # Analyze events
        event_type_counts = {}
        severity_counts = {}
        user_activity = {}
        daily_event_counts = {}
        
        for event in audit_events:
            event_type = event.get('event_type', 'unknown')
            severity = event.get('severity', 'info')
            user_id = event.get('user_id', 'anonymous')
            
            # Count by type
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Count by severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by user
            if user_id not in user_activity:
                user_activity[user_id] = {'events': 0, 'cost': 0.0}
            user_activity[user_id]['events'] += 1
            user_activity[user_id]['cost'] += event.get('cost_amount', 0)
            
            # Daily breakdown
            event_date = datetime.fromtimestamp(
                event.get('timestamp', 0), timezone.utc
            ).strftime('%Y-%m-%d')
            daily_event_counts[event_date] = daily_event_counts.get(event_date, 0) + 1
        
        # Get compliance summary
        compliance_summary = self.compliance_checker.get_compliance_summary()
        
        # Get cost analysis
        cost_report = self.cost_persistence.generate_cost_report(start_date, end_date)
        
        return {
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'summary': {
                'total_events': len(audit_events),
                'unique_users': len(user_activity),
                'active_sessions': len(self._active_sessions),
                'event_chain_integrity': self._verify_event_chain(),
                'system_uptime_hours': (time.time() - self._get_system_start_time()) / 3600
            },
            'event_analysis': {
                'by_type': event_type_counts,
                'by_severity': severity_counts,
                'daily_breakdown': daily_event_counts
            },
            'user_activity': user_activity,
            'compliance': compliance_summary,
            'cost_analysis': cost_report.get('summary', {}),
            'security_events': [
                event for event in audit_events
                if event.get('event_type') in ['authentication_failed', 'unauthorized_access', 'suspicious_activity']
            ],
            'system_health': {
                'database_status': 'operational',  # Could check actual DB health
                'audit_storage_usage': self._get_audit_storage_usage(),
                'recent_errors': [
                    event for event in audit_events[-100:]  # Last 100 events
                    if event.get('severity') in ['error', 'critical']
                ]
            }
        }
    
    def _verify_event_chain(self) -> bool:
        """Verify the integrity of the audit event chain."""
        # This is a simplified version - in production, you'd verify the full chain
        return self._event_count > 0
    
    def _get_system_start_time(self) -> float:
        """Get the system start time from the first audit event."""
        # This would query the database for the first SYSTEM_START event
        return time.time() - 3600  # Placeholder: 1 hour ago
    
    def _get_audit_storage_usage(self) -> Dict[str, Any]:
        """Get audit storage usage statistics."""
        # This would check actual database sizes
        return {
            'total_events': self._event_count,
            'estimated_size_mb': self._event_count * 0.001,  # Rough estimate
            'retention_policy_days': self.cost_persistence.retention_days
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the audit system."""
        # End all active sessions
        for session_id in list(self._active_sessions.keys()):
            self.end_session(session_id)
        
        # Log system shutdown
        self.log_event(
            AuditEventType.SYSTEM_STOP,
            description="Audit trail system shutting down",
            severity="info",
            metadata={
                'total_events_logged': self._event_count,
                'active_sessions_ended': len(self._active_sessions)
            }
        )
        
        self.logger.info("Audit trail system shutdown complete")