"""Compliance and regulatory features for quantum scheduler."""

import json
import logging
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock
from pathlib import Path

logger = logging.getLogger(__name__)


class ComplianceRegulation(Enum):
    """Supported compliance regulations."""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act
    PDPA = "pdpa"           # Personal Data Protection Act (Singapore/Thailand)
    HIPAA = "hipaa"         # Health Insurance Portability and Accountability Act
    SOX = "sox"             # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"     # Payment Card Industry Data Security Standard


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    SCHEDULING = "scheduling"
    ANALYTICS = "analytics"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    SECURITY = "security"
    RESEARCH = "research"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    record_id: str
    timestamp: datetime
    data_subject: Optional[str]
    data_categories: List[str]
    processing_purpose: ProcessingPurpose
    legal_basis: str
    data_classification: DataClassification
    retention_period_days: int
    processor_id: str
    controller_id: Optional[str]
    third_party_transfers: List[str]
    security_measures: List[str]
    consent_required: bool = False
    consent_obtained: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['processing_purpose'] = self.processing_purpose.value
        result['data_classification'] = self.data_classification.value
        return result


class ComplianceManager:
    """Manages compliance and regulatory requirements."""
    
    def __init__(
        self,
        enabled_regulations: Optional[List[ComplianceRegulation]] = None,
        data_retention_days: int = 730,  # 2 years default
        audit_log_enabled: bool = True
    ):
        """Initialize compliance manager.
        
        Args:
            enabled_regulations: List of regulations to comply with
            data_retention_days: Default data retention period
            audit_log_enabled: Whether to enable audit logging
        """
        self.enabled_regulations = enabled_regulations or [
            ComplianceRegulation.GDPR,
            ComplianceRegulation.CCPA
        ]
        self.data_retention_days = data_retention_days
        self.audit_log_enabled = audit_log_enabled
        
        self._processing_records: List[DataProcessingRecord] = []
        self._consent_records: Dict[str, Dict[str, Any]] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = Lock()
        
        # Privacy settings per regulation
        self._privacy_settings = self._initialize_privacy_settings()
        
        logger.info(f"Compliance manager initialized with regulations: {[r.value for r in self.enabled_regulations]}")
    
    def _initialize_privacy_settings(self) -> Dict[ComplianceRegulation, Dict[str, Any]]:
        """Initialize privacy settings for each regulation."""
        settings = {}
        
        for regulation in self.enabled_regulations:
            if regulation == ComplianceRegulation.GDPR:
                settings[regulation] = {
                    'data_minimization': True,
                    'purpose_limitation': True,
                    'storage_limitation': True,
                    'accuracy_requirement': True,
                    'integrity_confidentiality': True,
                    'accountability': True,
                    'consent_required_for_sensitive': True,
                    'right_to_erasure': True,
                    'right_to_portability': True,
                    'right_to_rectification': True,
                    'data_protection_impact_assessment': True,
                    'privacy_by_design': True
                }
            elif regulation == ComplianceRegulation.CCPA:
                settings[regulation] = {
                    'right_to_know': True,
                    'right_to_delete': True,
                    'right_to_opt_out': True,
                    'right_to_non_discrimination': True,
                    'notice_at_collection': True,
                    'notice_of_sale': False,  # We don't sell data
                    'consumer_request_verification': True
                }
            elif regulation == ComplianceRegulation.PDPA:
                settings[regulation] = {
                    'consent_required': True,
                    'purpose_limitation': True,
                    'notification_requirement': True,
                    'access_correction_rights': True,
                    'data_breach_notification': True,
                    'cross_border_transfer_restrictions': True
                }
        
        return settings
    
    def record_data_processing(
        self,
        data_subject: Optional[str],
        data_categories: List[str],
        purpose: ProcessingPurpose,
        legal_basis: str,
        classification: DataClassification = DataClassification.INTERNAL,
        retention_days: Optional[int] = None,
        processor_id: str = "quantum_scheduler",
        controller_id: Optional[str] = None,
        third_party_transfers: Optional[List[str]] = None,
        security_measures: Optional[List[str]] = None
    ) -> str:
        """Record a data processing activity.
        
        Args:
            data_subject: Identifier of the data subject (if applicable)
            data_categories: Categories of data being processed
            purpose: Purpose for processing
            legal_basis: Legal basis for processing
            classification: Data classification level
            retention_days: Data retention period (uses default if None)
            processor_id: Identifier of the processor
            controller_id: Identifier of the data controller
            third_party_transfers: List of third parties data is shared with
            security_measures: List of security measures applied
            
        Returns:
            Record ID for tracking
        """
        with self._lock:
            record_id = str(uuid.uuid4())
            
            # Determine if consent is required
            consent_required = self._is_consent_required(purpose, classification)
            
            record = DataProcessingRecord(
                record_id=record_id,
                timestamp=datetime.now(),
                data_subject=data_subject,
                data_categories=data_categories,
                processing_purpose=purpose,
                legal_basis=legal_basis,
                data_classification=classification,
                retention_period_days=retention_days or self.data_retention_days,
                processor_id=processor_id,
                controller_id=controller_id,
                third_party_transfers=third_party_transfers or [],
                security_measures=security_measures or [],
                consent_required=consent_required,
                consent_obtained=False  # Will be updated separately
            )
            
            self._processing_records.append(record)
            
            # Log the activity
            if self.audit_log_enabled:
                self._log_audit_event("data_processing_recorded", {
                    "record_id": record_id,
                    "purpose": purpose.value,
                    "data_categories": data_categories,
                    "classification": classification.value
                })
            
            logger.debug(f"Recorded data processing: {record_id}")
            return record_id
    
    def record_consent(
        self,
        data_subject: str,
        processing_purposes: List[ProcessingPurpose],
        consent_given: bool,
        consent_method: str = "explicit",
        expiry_date: Optional[datetime] = None
    ) -> str:
        """Record consent from a data subject.
        
        Args:
            data_subject: Identifier of the data subject
            processing_purposes: Purposes for which consent is given/withdrawn
            consent_given: Whether consent was given or withdrawn
            consent_method: Method of consent collection
            expiry_date: When the consent expires (if applicable)
            
        Returns:
            Consent record ID
        """
        with self._lock:
            consent_id = str(uuid.uuid4())
            
            consent_record = {
                'consent_id': consent_id,
                'data_subject': data_subject,
                'purposes': [p.value for p in processing_purposes],
                'consent_given': consent_given,
                'consent_method': consent_method,
                'timestamp': datetime.now().isoformat(),
                'expiry_date': expiry_date.isoformat() if expiry_date else None,
                'ip_address': None,  # Could be captured from request context
                'user_agent': None   # Could be captured from request context
            }
            
            if data_subject not in self._consent_records:
                self._consent_records[data_subject] = {}
            
            self._consent_records[data_subject][consent_id] = consent_record
            
            # Update related processing records
            self._update_processing_consent(data_subject, processing_purposes, consent_given)
            
            # Log the activity
            if self.audit_log_enabled:
                self._log_audit_event("consent_recorded", {
                    "consent_id": consent_id,
                    "data_subject": data_subject,
                    "consent_given": consent_given,
                    "purposes": [p.value for p in processing_purposes]
                })
            
            logger.info(f"Recorded consent: {consent_id} for subject {data_subject}")
            return consent_id
    
    def _update_processing_consent(
        self,
        data_subject: str,
        purposes: List[ProcessingPurpose],
        consent_given: bool
    ):
        """Update processing records with consent status."""
        for record in self._processing_records:
            if (record.data_subject == data_subject and
                record.processing_purpose in purposes and
                record.consent_required):
                record.consent_obtained = consent_given
    
    def _is_consent_required(
        self,
        purpose: ProcessingPurpose,
        classification: DataClassification
    ) -> bool:
        """Determine if consent is required for this processing."""
        # Consent is generally required for:
        # - Confidential or restricted data
        # - Research purposes
        # - Analytics (depending on regulation)
        
        if classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
            return True
        
        if purpose in [ProcessingPurpose.RESEARCH, ProcessingPurpose.ANALYTICS]:
            return True
        
        # Check regulation-specific requirements
        for regulation in self.enabled_regulations:
            settings = self._privacy_settings.get(regulation, {})
            if settings.get('consent_required_for_sensitive', False):
                return True
        
        return False
    
    def handle_data_subject_request(
        self,
        request_type: str,
        data_subject: str,
        verification_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle data subject rights requests.
        
        Args:
            request_type: Type of request (access, delete, portability, etc.)
            data_subject: Identifier of the data subject
            verification_data: Data for verifying the request
            
        Returns:
            Response to the request
        """
        with self._lock:
            # Verify the request if verification data is provided
            if verification_data:
                verified = self._verify_data_subject_request(data_subject, verification_data)
                if not verified:
                    return {"status": "error", "message": "Request verification failed"}
            
            response = {"status": "success", "data": {}}
            
            if request_type == "access":
                # Right to access (GDPR Art. 15, CCPA right to know)
                response["data"] = self._get_data_subject_data(data_subject)
                
            elif request_type == "delete":
                # Right to erasure/deletion (GDPR Art. 17, CCPA right to delete)
                deleted_count = self._delete_data_subject_data(data_subject)
                response["data"] = {"deleted_records": deleted_count}
                
            elif request_type == "portability":
                # Right to data portability (GDPR Art. 20)
                if ComplianceRegulation.GDPR in self.enabled_regulations:
                    response["data"] = self._export_data_subject_data(data_subject)
                else:
                    response = {"status": "error", "message": "Data portability not available"}
                    
            elif request_type == "rectification":
                # Right to rectification (GDPR Art. 16)
                response = {"status": "pending", "message": "Rectification request recorded"}
                
            elif request_type == "opt_out":
                # Right to opt out (CCPA)
                if ComplianceRegulation.CCPA in self.enabled_regulations:
                    self._opt_out_data_subject(data_subject)
                    response["data"] = {"opted_out": True}
                else:
                    response = {"status": "error", "message": "Opt-out not available"}
            
            # Log the request
            if self.audit_log_enabled:
                self._log_audit_event("data_subject_request", {
                    "request_type": request_type,
                    "data_subject": data_subject,
                    "status": response["status"]
                })
            
            return response
    
    def _verify_data_subject_request(
        self,
        data_subject: str,
        verification_data: Dict[str, Any]
    ) -> bool:
        """Verify a data subject request."""
        # Implement verification logic based on your requirements
        # This could involve checking email, phone, government ID, etc.
        required_fields = ["email", "verification_code"]
        
        for field in required_fields:
            if field not in verification_data:
                return False
        
        # In a real implementation, you would verify the email and code
        # For now, return True if basic fields are present
        return True
    
    def _get_data_subject_data(self, data_subject: str) -> Dict[str, Any]:
        """Get all data for a data subject."""
        data = {
            "processing_records": [],
            "consent_records": self._consent_records.get(data_subject, {}),
            "data_categories": set()
        }
        
        for record in self._processing_records:
            if record.data_subject == data_subject:
                data["processing_records"].append(record.to_dict())
                data["data_categories"].update(record.data_categories)
        
        data["data_categories"] = list(data["data_categories"])
        return data
    
    def _delete_data_subject_data(self, data_subject: str) -> int:
        """Delete all data for a data subject."""
        deleted_count = 0
        
        # Remove processing records
        original_count = len(self._processing_records)
        self._processing_records = [
            r for r in self._processing_records if r.data_subject != data_subject
        ]
        deleted_count += original_count - len(self._processing_records)
        
        # Remove consent records
        if data_subject in self._consent_records:
            deleted_count += len(self._consent_records[data_subject])
            del self._consent_records[data_subject]
        
        return deleted_count
    
    def _export_data_subject_data(self, data_subject: str) -> Dict[str, Any]:
        """Export data for portability."""
        return {
            "format": "json",
            "data": self._get_data_subject_data(data_subject),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def _opt_out_data_subject(self, data_subject: str):
        """Opt out a data subject from data processing."""
        # Mark all processing records as opted out
        for record in self._processing_records:
            if record.data_subject == data_subject:
                # Add opt-out flag or modify processing
                pass
    
    def check_data_retention(self) -> List[str]:
        """Check for data that needs to be deleted due to retention policies."""
        with self._lock:
            expired_records = []
            current_time = datetime.now()
            
            for record in self._processing_records:
                retention_end = record.timestamp + timedelta(days=record.retention_period_days)
                if current_time > retention_end:
                    expired_records.append(record.record_id)
            
            return expired_records
    
    def delete_expired_data(self) -> int:
        """Delete data that has exceeded retention period."""
        with self._lock:
            expired_record_ids = self.check_data_retention()
            
            if not expired_record_ids:
                return 0
            
            # Remove expired records
            original_count = len(self._processing_records)
            self._processing_records = [
                r for r in self._processing_records 
                if r.record_id not in expired_record_ids
            ]
            deleted_count = original_count - len(self._processing_records)
            
            # Log the deletion
            if self.audit_log_enabled:
                self._log_audit_event("expired_data_deleted", {
                    "deleted_count": deleted_count,
                    "record_ids": expired_record_ids
                })
            
            logger.info(f"Deleted {deleted_count} expired data records")
            return deleted_count
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log an audit event."""
        audit_entry = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "source": "compliance_manager"
        }
        
        self._audit_log.append(audit_entry)
        
        # Keep audit log size manageable
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]  # Keep last 5000 entries
    
    def generate_compliance_report(
        self,
        regulation: ComplianceRegulation,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate a compliance report for a specific regulation."""
        with self._lock:
            end_date = end_date or datetime.now()
            start_date = start_date or (end_date - timedelta(days=30))
            
            # Filter records by date range
            filtered_records = [
                r for r in self._processing_records
                if start_date <= r.timestamp <= end_date
            ]
            
            report = {
                "regulation": regulation.value,
                "report_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "summary": {
                    "total_processing_activities": len(filtered_records),
                    "data_subjects_count": len(set(r.data_subject for r in filtered_records if r.data_subject)),
                    "consent_required_activities": len([r for r in filtered_records if r.consent_required]),
                    "consent_obtained_rate": 0.0
                },
                "compliance_status": {},
                "recommendations": []
            }
            
            # Calculate consent obtained rate
            consent_required = [r for r in filtered_records if r.consent_required]
            if consent_required:
                consent_obtained = len([r for r in consent_required if r.consent_obtained])
                report["summary"]["consent_obtained_rate"] = consent_obtained / len(consent_required)
            
            # Regulation-specific compliance checks
            if regulation == ComplianceRegulation.GDPR:
                report["compliance_status"]["lawful_basis_documented"] = all(
                    r.legal_basis for r in filtered_records
                )
                report["compliance_status"]["data_categories_documented"] = all(
                    r.data_categories for r in filtered_records
                )
                report["compliance_status"]["retention_periods_defined"] = all(
                    r.retention_period_days > 0 for r in filtered_records
                )
                
                # Recommendations
                if report["summary"]["consent_obtained_rate"] < 0.9:
                    report["recommendations"].append(
                        "Improve consent collection process - currently at "
                        f"{report['summary']['consent_obtained_rate']:.1%}"
                    )
            
            elif regulation == ComplianceRegulation.CCPA:
                report["compliance_status"]["consumer_rights_implemented"] = True
                report["compliance_status"]["privacy_notice_provided"] = True
                
            return report
    
    def get_audit_log(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        with self._lock:
            if event_type:
                filtered_log = [entry for entry in self._audit_log if entry["event_type"] == event_type]
            else:
                filtered_log = self._audit_log.copy()
            
            # Return most recent entries
            return sorted(filtered_log, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_processing_records(
        self,
        data_subject: Optional[str] = None,
        purpose: Optional[ProcessingPurpose] = None
    ) -> List[DataProcessingRecord]:
        """Get processing records with optional filters."""
        with self._lock:
            records = self._processing_records.copy()
            
            if data_subject:
                records = [r for r in records if r.data_subject == data_subject]
            
            if purpose:
                records = [r for r in records if r.processing_purpose == purpose]
            
            return records
    
    def export_compliance_data(self, file_path: Optional[str] = None) -> str:
        """Export compliance data for backup or transfer."""
        with self._lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "regulations": [r.value for r in self.enabled_regulations],
                "processing_records": [r.to_dict() for r in self._processing_records],
                "consent_records": self._consent_records,
                "audit_log": self._audit_log[-1000:]  # Last 1000 entries
            }
            
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)
                logger.info(f"Compliance data exported to {file_path}")
            
            return json_data


# Global compliance manager instance
_compliance_manager: Optional[ComplianceManager] = None
_compliance_lock = Lock()


def get_compliance_manager() -> Optional[ComplianceManager]:
    """Get the global compliance manager."""
    return _compliance_manager


def initialize_compliance(
    regulations: Optional[List[ComplianceRegulation]] = None,
    **kwargs
) -> ComplianceManager:
    """Initialize global compliance manager."""
    global _compliance_manager
    
    with _compliance_lock:
        if _compliance_manager is None:
            _compliance_manager = ComplianceManager(
                enabled_regulations=regulations,
                **kwargs
            )
        return _compliance_manager


def record_processing(
    data_categories: List[str],
    purpose: ProcessingPurpose,
    legal_basis: str,
    **kwargs
) -> Optional[str]:
    """Record data processing if compliance is enabled."""
    if _compliance_manager:
        return _compliance_manager.record_data_processing(
            data_categories=data_categories,
            purpose=purpose,
            legal_basis=legal_basis,
            **kwargs
        )
    return None


__all__ = [
    "ComplianceRegulation",
    "DataClassification", 
    "ProcessingPurpose",
    "DataProcessingRecord",
    "ComplianceManager",
    "get_compliance_manager",
    "initialize_compliance",
    "record_processing"
]