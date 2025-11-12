"""
Custom exceptions for RefScore academic application.

This module defines custom exception classes used throughout
the RefScore application for better error handling and reporting.
"""

from __future__ import annotations


class RefScoreError(Exception):
    """
    Base exception class for RefScore application.
    
    All custom exceptions in the RefScore application should
    inherit from this base class.
    """
    
    def __init__(self, message: str = "", details: str = "") -> None:
        """
        Initialize exception.
        
        Args:
            message: Main error message
            details: Additional error details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """String representation of the exception."""
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message


class ValidationError(RefScoreError):
    """
    Exception raised for input validation errors.
    
    This exception is used when user input or file content
    fails validation checks.
    """
    
    def __init__(self, message: str = "Validation failed", field: str = "", value: str = "") -> None:
        """
        Initialize validation error.
        
        Args:
            message: Validation error message
            field: Field name that failed validation
            value: Value that failed validation
        """
        details = ""
        if field:
            if value:
                details += f"field: {field}"
            else:
                # If only a single positional detail was provided (not a field name), treat as plain details
                if any(ch.isspace() for ch in field):
                    details += field
                else:
                    details += f"field: {field}"
        if value:
            if details:
                details += ", "
            details += f"value: {value}"
        
        super().__init__(message, details)
        self.field = field
        self.value = value


class ProcessingError(RefScoreError):
    """
    Exception raised for processing errors.
    
    This exception is used when document parsing, source loading,
    or scoring computation fails.
    """
    
    def __init__(self, message: str = "Processing failed", operation: str = "", 
                 original_error: str = "") -> None:
        """
        Initialize processing error.
        
        Args:
            message: Processing error message
            operation: Operation that failed
            original_error: Original error message
        """
        details = ""
        if operation:
            details += f"operation: {operation}"
        if original_error:
            if details:
                details += ", "
            details += f"error: {original_error}"
        
        super().__init__(message, details)
        self.operation = operation
        self.original_error = original_error


class FileFormatError(RefScoreError):
    """
    Exception raised for unsupported or invalid file formats.
    
    This exception is used when a file format is not supported
    or the file content is invalid for the expected format.
    """
    
    def __init__(self, message: str = "File format error", file_path: str = "",
                 expected_format: str = "", actual_format: str = "") -> None:
        """
        Initialize file format error.
        
        Args:
            message: File format error message
            file_path: Path to the problematic file
            expected_format: Expected file format
            actual_format: Actual file format detected
        """
        details = ""
        if file_path:
            details += f"file: {file_path}"
        if expected_format:
            if details:
                details += ", "
            details += f"expected: {expected_format}"
        if actual_format:
            if details:
                details += ", "
            details += f"actual: {actual_format}"
        
        super().__init__(message, details)
        self.file_path = file_path
        self.expected_format = expected_format
        self.actual_format = actual_format


class ConfigurationError(RefScoreError):
    """
    Exception raised for configuration errors.
    
    This exception is used when configuration files are invalid,
    missing, or contain invalid settings.
    """
    
    def __init__(self, message: str = "Configuration error", config_key: str = "",
                 config_value: str = "") -> None:
        """
        Initialize configuration error.
        
        Args:
            message: Configuration error message
            config_key: Configuration key that caused the error
            config_value: Invalid configuration value
        """
        details = ""
        if config_key:
            details += f"key: {config_key}"
        if config_value:
            if details:
                details += ", "
            details += f"value: {config_value}"
        
        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


class NetworkError(RefScoreError):
    """
    Exception raised for network-related errors.
    
    This exception is used when network operations fail,
    such as API calls to Crossref or other external services.
    """
    
    def __init__(self, message: str = "Network error", url: str = "",
                 status_code: int = 0, response_text: str = "") -> None:
        """
        Initialize network error.
        
        Args:
            message: Network error message
            url: URL that caused the error
            status_code: HTTP status code
            response_text: Response text
        """
        details = ""
        if url:
            details += f"url: {url}"
        if status_code:
            if details:
                details += ", "
            details += f"status: {status_code}"
        if response_text:
            if details:
                details += ", "
            details += f"response: {response_text[:100]}..."
        
        super().__init__(message, details)
        self.url = url
        self.status_code = status_code
        self.response_text = response_text


class DependencyError(RefScoreError):
    """
    Exception raised for missing or invalid dependencies.
    
    This exception is used when optional dependencies are missing
    or when dependency versions are incompatible.
    """
    
    def __init__(self, message: str = "Dependency error", dependency_name: str = "",
                 required_version: str = "", installed_version: str = "") -> None:
        """
        Initialize dependency error.
        
        Args:
            message: Dependency error message
            dependency_name: Name of the problematic dependency
            required_version: Required version
            installed_version: Installed version
        """
        details = ""
        if dependency_name:
            details += f"dependency: {dependency_name}"
        if required_version:
            if details:
                details += ", "
            details += f"required: {required_version}"
        if installed_version:
            if details:
                details += ", "
            details += f"installed: {installed_version}"
        
        super().__init__(message, details)
        self.dependency_name = dependency_name
        self.required_version = required_version
        self.installed_version = installed_version


class GUIError(RefScoreError):
    """
    Exception raised for GUI-related errors.
    
    This exception is used when GUI operations fail,
    such as widget creation, event handling, or display issues.
    """
    
    def __init__(self, message: str = "GUI error", widget_name: str = "",
                 operation: str = "") -> None:
        """
        Initialize GUI error.
        
        Args:
            message: GUI error message
            widget_name: Name of the problematic widget
            operation: Operation that failed
        """
        details = ""
        if widget_name:
            details += f"widget: {widget_name}"
        if operation:
            if details:
                details += ", "
            details += f"operation: {operation}"
        
        super().__init__(message, details)
        self.widget_name = widget_name
        self.operation = operation


class DatabaseError(RefScoreError):
    """
    Exception raised for database-related errors.
    
    This exception is used when database operations fail,
    such as connection issues, query errors, or data integrity problems.
    """
    
    def __init__(self, message: str = "Database error", table_name: str = "",
                 operation: str = "", query: str = "") -> None:
        """
        Initialize database error.
        
        Args:
            message: Database error message
            table_name: Name of the problematic table
            operation: Database operation that failed
            query: SQL query that caused the error
        """
        details = ""
        if table_name:
            details += f"table: {table_name}"
        if operation:
            if details:
                details += ", "
            details += f"operation: {operation}"
        if query:
            if details:
                details += ", "
            details += f"query: {query[:100]}..."
        
        super().__init__(message, details)
        self.table_name = table_name
        self.operation = operation
        self.query = query


class NetworkError(RefScoreError):
    """
    Exception raised for network-related errors (HTTP, timeouts, DNS).
    """
    def __init__(self, message: str = "Network error", url: str = "", status_code: int = 0, response_text: str = "") -> None:
        details = ""
        if url:
            details += f"url: {url}"
        if status_code:
            if details:
                details += ", "
            details += f"status: {status_code}"
        if response_text:
            if details:
                details += ", "
            details += f"response: {response_text}"
        super().__init__(message, details)
        self.url = url
        self.status_code = status_code
        self.response_text = response_text


def get_error_context(exception: Exception) -> str:
    """
    Get a formatted error context string for logging.
    
    Args:
        exception: Exception to format
        
    Returns:
        Formatted error context string
    """
    if isinstance(exception, RefScoreError):
        return str(exception)
    else:
        return f"{type(exception).__name__}: {str(exception)}"


def log_exception(logger, exception: Exception, context: str = "") -> None:
    """
    Log an exception with appropriate level and context.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context information
    """
    error_message = get_error_context(exception)
    
    if context:
        full_message = f"{context} - {error_message}"
    else:
        full_message = error_message
    
    if isinstance(exception, (ValidationError, ConfigurationError)):
        logger.warning(full_message)
    elif isinstance(exception, (ProcessingError, FileFormatError, NetworkError)):
        logger.error(full_message)
    elif isinstance(exception, (DependencyError, GUIError, DatabaseError)):
        logger.critical(full_message)
    else:
        logger.error(full_message)
