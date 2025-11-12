"""
Input validation utilities for RefScore academic application.

This module provides comprehensive input validation for files,
directories, and data structures used throughout the application.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Union

from ..utils.exceptions import ValidationError


class InputValidator:
    """
    Input validation utilities for RefScore application.
    
    Provides validation methods for:
    - File paths and formats
    - Directory paths
    - DOI formats
    - Email addresses
    - Author names
    - Publication years
    """
    
    # Regular expressions for validation
    DOI_PATTERN = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    YEAR_PATTERN = re.compile(r'^\d{4}$')
    
    # Supported file extensions
    SUPPORTED_DOCUMENT_FORMATS = ['.tex', '.pdf']
    SUPPORTED_SOURCE_FORMATS = ['.bib', '.json', '.csv', '.txt']
    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
    
    @classmethod
    def validate_file_path(cls, file_path: Union[str, Path], must_exist: bool = True,
                          extensions: Optional[List[str]] = None) -> Path:
        """
        Validate a file path.
        
        Args:
            file_path: Path to validate
            must_exist: Whether the file must exist
            extensions: List of allowed extensions (with dots)
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid
        """
        try:
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            if not isinstance(file_path, Path):
                raise ValidationError("File path must be a string or Path object")
            
            # Check if path is absolute or relative
            if not file_path.is_absolute():
                file_path = file_path.resolve()
            
            # Check existence if required
            if must_exist:
                if not file_path.exists():
                    raise ValidationError(f"File does not exist: {file_path}")
                
                if not file_path.is_file():
                    raise ValidationError(f"Path is not a file: {file_path}")
            
            # Check extension if provided
            if extensions:
                if file_path.suffix.lower() not in [ext.lower() for ext in extensions]:
                    valid_exts = ', '.join(extensions)
                    raise ValidationError(f"File must have one of these extensions: {valid_exts}")
            
            return file_path
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Invalid file path: {e}")
    
    @classmethod
    def validate_directory_path(cls, dir_path: Union[str, Path], must_exist: bool = True,
                               create_if_missing: bool = False) -> Path:
        """
        Validate a directory path.
        
        Args:
            dir_path: Path to validate
            must_exist: Whether the directory must exist
            create_if_missing: Whether to create the directory if it doesn't exist
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid
        """
        try:
            if isinstance(dir_path, str):
                dir_path = Path(dir_path)
            
            if not isinstance(dir_path, Path):
                raise ValidationError("Directory path must be a string or Path object")
            
            # Check if path is absolute or relative
            if not dir_path.is_absolute():
                dir_path = dir_path.resolve()
            
            # Handle missing directory
            if not dir_path.exists():
                if create_if_missing:
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        log.info(f"Created directory: {dir_path}")
                    except Exception as e:
                        raise ValidationError(f"Failed to create directory: {e}")
                elif must_exist:
                    raise ValidationError(f"Directory does not exist: {dir_path}")
            else:
                # Check if it's actually a directory
                if not dir_path.is_dir():
                    raise ValidationError(f"Path is not a directory: {dir_path}")
            
            return dir_path
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Invalid directory path: {e}")
    
    @classmethod
    def validate_document_file(cls, file_path: Union[str, Path]) -> Path:
        """
        Validate a document file path.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If file is invalid
        """
        return cls.validate_file_path(file_path, must_exist=True,
                                     extensions=cls.SUPPORTED_DOCUMENT_FORMATS)
    
    @classmethod
    def validate_source_file(cls, file_path: Union[str, Path]) -> Path:
        """
        Validate a source file path.
        
        Args:
            file_path: Path to source file
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If file is invalid
        """
        return cls.validate_file_path(file_path, must_exist=True,
                                     extensions=cls.SUPPORTED_SOURCE_FORMATS)
    
    @classmethod
    def validate_doi(cls, doi: str) -> str:
        """
        Validate a DOI string.
        
        Args:
            doi: DOI string to validate
            
        Returns:
            Validated DOI string
            
        Raises:
            ValidationError: If DOI is invalid
        """
        if not doi or not isinstance(doi, str):
            raise ValidationError("DOI must be a non-empty string")
        
        # Clean DOI first
        cleaned_doi = cls.clean_doi(doi)
        
        if not cls.DOI_PATTERN.match(cleaned_doi):
            raise ValidationError(f"Invalid DOI format: {doi}")
        
        return cleaned_doi
    
    @classmethod
    def validate_email(cls, email: str) -> str:
        """
        Validate an email address.
        
        Args:
            email: Email address to validate
            
        Returns:
            Validated email address
            
        Raises:
            ValidationError: If email is invalid
        """
        if not email or not isinstance(email, str):
            raise ValidationError("Email must be a non-empty string")
        
        email = email.strip()
        
        if not cls.EMAIL_PATTERN.match(email):
            raise ValidationError(f"Invalid email format: {email}")
        
        return email
    
    @classmethod
    def validate_year(cls, year: Union[str, int]) -> int:
        """
        Validate a publication year.
        
        Args:
            year: Year to validate
            
        Returns:
            Validated year as integer
            
        Raises:
            ValidationError: If year is invalid
        """
        try:
            if isinstance(year, str):
                year = int(year)
            elif not isinstance(year, int):
                raise ValidationError("Year must be a string or integer")
            
            current_year = 2024  # Could be made dynamic
            
            if year < 1000 or year > current_year + 10:
                raise ValidationError(f"Year must be between 1000 and {current_year + 10}")
            
            return year
            
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid year: {year}")
    
    @classmethod
    def validate_author_name(cls, author_name: str) -> str:
        """
        Validate an author name.
        
        Args:
            author_name: Author name to validate
            
        Returns:
            Validated author name
            
        Raises:
            ValidationError: If name is invalid
        """
        if not author_name or not isinstance(author_name, str):
            raise ValidationError("Author name must be a non-empty string")
        
        author_name = author_name.strip()
        
        if len(author_name) < 2:
            raise ValidationError("Author name must be at least 2 characters long")
        
        if len(author_name) > 200:
            raise ValidationError("Author name is too long (max 200 characters)")
        
        # Check for invalid characters
        invalid_chars = re.findall(r'[^\w\s\-\.\,\']', author_name)
        if invalid_chars:
            raise ValidationError(f"Author name contains invalid characters: {invalid_chars}")
        
        return author_name
    
    @classmethod
    def validate_title(cls, title: str) -> str:
        """
        Validate a publication title.
        
        Args:
            title: Title to validate
            
        Returns:
            Validated title
            
        Raises:
            ValidationError: If title is invalid
        """
        if not isinstance(title, str):
            raise ValidationError("Title must be a string")
        
        title = title.strip()
        
        if not title:
            return title  # Empty titles are allowed but logged as warnings elsewhere
        
        if len(title) > 1000:
            raise ValidationError("Title is too long (max 1000 characters)")
        
        return title
    
    @classmethod
    def validate_abstract(cls, abstract: str) -> str:
        """
        Validate an abstract.
        
        Args:
            abstract: Abstract to validate
            
        Returns:
            Validated abstract
            
        Raises:
            ValidationError: If abstract is invalid
        """
        if not isinstance(abstract, str):
            raise ValidationError("Abstract must be a string")
        
        abstract = abstract.strip()
        
        if len(abstract) > 10000:
            raise ValidationError("Abstract is too long (max 10000 characters)")
        
        return abstract
    
    @classmethod
    def validate_venue(cls, venue: str) -> str:
        """
        Validate a publication venue.
        
        Args:
            venue: Venue to validate
            
        Returns:
            Validated venue
            
        Raises:
            ValidationError: If venue is invalid
        """
        if not isinstance(venue, str):
            raise ValidationError("Venue must be a string")
        
        venue = venue.strip()
        
        if len(venue) > 500:
            raise ValidationError("Venue is too long (max 500 characters)")
        
        return venue
    
    @classmethod
    def validate_source_id(cls, source_id: str) -> str:
        """
        Validate a source ID.
        
        Args:
            source_id: Source ID to validate
            
        Returns:
            Validated source ID
            
        Raises:
            ValidationError: If source ID is invalid
        """
        if not source_id or not isinstance(source_id, str):
            raise ValidationError("Source ID must be a non-empty string")
        
        source_id = source_id.strip()
        
        if len(source_id) < 1:
            raise ValidationError("Source ID cannot be empty")
        
        if len(source_id) > 100:
            raise ValidationError("Source ID is too long (max 100 characters)")
        
        # Check for invalid characters
        invalid_chars = re.findall(r'[^\w\-\.\_]', source_id)
        if invalid_chars:
            raise ValidationError(f"Source ID contains invalid characters: {invalid_chars}")
        
        return source_id
    
    @classmethod
    def clean_doi(cls, doi: str) -> str:
        """
        Clean and normalize a DOI string.
        
        Args:
            doi: DOI string to clean
            
        Returns:
            Cleaned DOI string
        """
        if not doi or not isinstance(doi, str):
            return ""
        
        doi = doi.strip()
        
        # Remove common prefixes and URL components
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        doi = re.sub(r'^doi:', '', doi)
        
        return doi
    
    @classmethod
    def validate_file_list(cls, file_paths: List[Union[str, Path]], 
                          extensions: Optional[List[str]] = None,
                          must_exist: bool = True) -> List[Path]:
        """
        Validate a list of file paths.
        
        Args:
            file_paths: List of file paths to validate
            extensions: List of allowed extensions
            must_exist: Whether files must exist
            
        Returns:
            List of validated Path objects
            
        Raises:
            ValidationError: If any file is invalid
        """
        if not file_paths:
            raise ValidationError("File list cannot be empty")
        
        validated_paths = []
        
        for file_path in file_paths:
            try:
                validated_path = cls.validate_file_path(file_path, must_exist, extensions)
                validated_paths.append(validated_path)
            except ValidationError as e:
                raise ValidationError(f"Invalid file in list: {e}")
        
        return validated_paths
    
    @classmethod
    def validate_scoring_weights(cls, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate scoring weights.
        
        Args:
            weights: Dictionary of scoring weights
            
        Returns:
            Validated weights dictionary
            
        Raises:
            ValidationError: If weights are invalid
        """
        if not isinstance(weights, dict):
            raise ValidationError("Weights must be a dictionary")
        
        valid_keys = {
            "alignment", "number_unit", "entities", "method_metric", "recency", "authority"
        }
        
        # Check for unknown keys
        unknown_keys = set(weights.keys()) - valid_keys
        if unknown_keys:
            raise ValidationError(f"Unknown scoring weight keys: {unknown_keys}")
        
        # Check for missing keys
        missing_keys = valid_keys - set(weights.keys())
        if missing_keys:
            raise ValidationError(f"Missing scoring weight keys: {missing_keys}")
        
        # Validate individual weights
        for key, value in weights.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Weight '{key}' must be a number")
            
            if not (0.0 <= value <= 1.0):
                raise ValidationError(f"Weight '{key}' must be between 0.0 and 1.0")
        
        # Check that weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
            raise ValidationError(f"Weights must sum to 1.0 (current sum: {total_weight:.3f})")
        
        return weights
    
    @classmethod
    def is_valid_file_path(cls, file_path: Union[str, Path], must_exist: bool = True,
                          extensions: Optional[List[str]] = None) -> bool:
        """
        Check if a file path is valid without raising exceptions.
        
        Args:
            file_path: Path to check
            must_exist: Whether the file must exist
            extensions: List of allowed extensions
            
        Returns:
            True if path is valid, False otherwise
        """
        try:
            cls.validate_file_path(file_path, must_exist, extensions)
            return True
        except ValidationError:
            return False
    
    @classmethod
    def is_valid_doi(cls, doi: str) -> bool:
        """
        Check if a DOI is valid without raising exceptions.
        
        Args:
            doi: DOI to check
            
        Returns:
            True if DOI is valid, False otherwise
        """
        try:
            cls.validate_doi(doi)
            return True
        except ValidationError:
            return False
    
    @classmethod
    def is_valid_year(cls, year: Union[str, int]) -> bool:
        """
        Check if a year is valid without raising exceptions.
        
        Args:
            year: Year to check
            
        Returns:
            True if year is valid, False otherwise
        """
        try:
            cls.validate_year(year)
            return True
        except ValidationError:
            return False


# Import logging for the module
import logging
log = logging.getLogger(__name__)