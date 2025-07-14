"""Logging configuration for the AI Vehicle Monitoring System."""

import logging
import sys
from typing import Optional
import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(
    log_level: str = "INFO",
    enable_json: bool = True,
    enable_audit: bool = True
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to use JSON formatting
        enable_audit: Whether to enable audit logging
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if enable_json else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Setup audit logger if enabled
    if enable_audit:
        audit_logger = logging.getLogger("audit")
        audit_handler = logging.FileHandler("logs/audit.log")
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (defaults to calling module)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def get_audit_logger() -> logging.Logger:
    """
    Get the audit logger for security and compliance logging.
    
    Returns:
        Standard logging.Logger for audit events
    """
    return logging.getLogger("audit")


class SecurityAuditLogger:
    """Security audit logger for Ghana data protection compliance."""
    
    def __init__(self):
        self.logger = get_audit_logger()
    
    def log_user_access(self, user_id: str, resource: str, action: str, success: bool = True):
        """Log user access events."""
        self.logger.info(
            f"USER_ACCESS: user={user_id} resource={resource} action={action} success={success}"
        )
    
    def log_data_access(self, user_id: str, data_type: str, record_id: str, action: str):
        """Log data access events."""
        self.logger.info(
            f"DATA_ACCESS: user={user_id} data_type={data_type} record_id={record_id} action={action}"
        )
    
    def log_query_execution(self, user_id: str, query_type: str, query_id: str, processing_time: int):
        """Log query execution events."""
        self.logger.info(
            f"QUERY_EXECUTION: user={user_id} query_type={query_type} query_id={query_id} processing_time={processing_time}ms"
        )
    
    def log_security_event(self, event_type: str, user_id: str, details: str, severity: str = "INFO"):
        """Log security events."""
        self.logger.log(
            getattr(logging, severity.upper()),
            f"SECURITY_EVENT: type={event_type} user={user_id} details={details}"
        )
    
    def log_system_event(self, event_type: str, component: str, details: str, severity: str = "INFO"):
        """Log system events."""
        self.logger.log(
            getattr(logging, severity.upper()),
            f"SYSTEM_EVENT: type={event_type} component={component} details={details}"
        )


# Global audit logger instance
audit_logger = SecurityAuditLogger()
