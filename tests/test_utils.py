"""
Test suite for RefScore utilities.

This module contains tests for utility functions, validators, and configuration.
"""

import pytest
import tempfile
from pathlib import Path

from refscore.utils.config import Config, Settings
from refscore.utils.validators import InputValidator
from refscore.utils.exceptions import (
    ValidationError,
    ProcessingError,
    ConfigurationError,
    FileFormatError,
    NetworkError,
    DependencyError,
    GUIError,
    DatabaseError,
)


class TestInputValidator:
    """Test cases for InputValidator."""
    
    def test_validate_file_path_success(self):
        """Test successful file path validation."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            validator = InputValidator()
            validated_path = validator.validate_file_path(str(tmp_path))
            
            assert isinstance(validated_path, Path)
            assert validated_path == tmp_path.resolve()
            
        finally:
            tmp_path.unlink()
    
    def test_validate_file_path_nonexistent(self):
        """Test file path validation with nonexistent file."""
        validator = InputValidator()
        
        with pytest.raises(ValidationError, match="File does not exist"):
            validator.validate_file_path("nonexistent_file.txt")
    
    def test_validate_file_path_invalid_extension(self):
        """Test file path validation with invalid extension."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            validator = InputValidator()
            
            with pytest.raises(ValidationError, match="File must have one of these extensions"):
                validator.validate_file_path(str(tmp_path), extensions=['.txt', '.pdf'])
                
        finally:
            tmp_path.unlink()
    
    def test_validate_directory_path_success(self):
        """Test successful directory path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            validator = InputValidator()
            validated_path = validator.validate_directory_path(str(tmp_path))
            
            assert isinstance(validated_path, Path)
            assert validated_path == tmp_path.resolve()
    
    def test_validate_directory_path_create_if_missing(self):
        """Test directory path validation with creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir)
            new_dir_path = parent_path / "new_directory"
            
            validator = InputValidator()
            validated_path = validator.validate_directory_path(
                str(new_dir_path), create_if_missing=True
            )
            
            assert validated_path.exists()
            assert validated_path.is_dir()
    
    def test_validate_doi_success(self):
        """Test successful DOI validation."""
        validator = InputValidator()
        
        valid_doi = "10.1234/example.2023"
        validated_doi = validator.validate_doi(valid_doi)
        
        assert validated_doi == valid_doi
    
    def test_validate_doi_with_prefixes(self):
        """Test DOI validation with common prefixes."""
        validator = InputValidator()
        
        # Test with URL prefix
        doi_with_url = "https://doi.org/10.1234/example.2023"
        validated_doi = validator.validate_doi(doi_with_url)
        assert validated_doi == "10.1234/example.2023"
        
        # Test with DOI prefix
        doi_with_prefix = "doi:10.1234/example.2023"
        validated_doi = validator.validate_doi(doi_with_prefix)
        assert validated_doi == "10.1234/example.2023"
    
    def test_validate_doi_invalid(self):
        """Test DOI validation with invalid format."""
        validator = InputValidator()
        
        with pytest.raises(ValidationError, match="Invalid DOI format"):
            validator.validate_doi("invalid_doi_format")
    
    def test_validate_year_success(self):
        """Test successful year validation."""
        validator = InputValidator()
        
        # Test with integer
        validated_year = validator.validate_year(2023)
        assert validated_year == 2023
        
        # Test with string
        validated_year = validator.validate_year("2023")
        assert validated_year == 2023
    
    def test_validate_year_invalid_range(self):
        """Test year validation with invalid range."""
        validator = InputValidator()
        
        # Too old
        with pytest.raises(ValidationError, match="Year must be between 1000 and"):
            validator.validate_year(999)
        
        # Too far in future
        with pytest.raises(ValidationError, match="Year must be between 1000 and"):
            validator.validate_year(2035)
    
    def test_validate_author_name_success(self):
        """Test successful author name validation."""
        validator = InputValidator()
        
        valid_names = [
            "John Doe",
            "Jane Smith",
            "Dr. John Doe",
            "John O'Connor",
            "Jean-Pierre Dubois"
        ]
        
        for name in valid_names:
            validated_name = validator.validate_author_name(name)
            assert isinstance(validated_name, str)
    
    def test_validate_author_name_invalid(self):
        """Test author name validation with invalid input."""
        validator = InputValidator()
        
        # Too short
        with pytest.raises(ValidationError, match="Author name must be at least 2 characters"):
            validator.validate_author_name("J")
        
        # Invalid characters
        with pytest.raises(ValidationError, match="Author name contains invalid characters"):
            validator.validate_author_name("John@Doe")
    
    def test_validate_title_success(self):
        """Test successful title validation."""
        validator = InputValidator()
        
        valid_titles = [
            "Test Paper Title",
            "A Very Long Paper Title with Many Words and Special Characters: A Comprehensive Study",
            ""  # Empty title should be allowed
        ]
        
        for title in valid_titles:
            validated_title = validator.validate_title(title)
            assert isinstance(validated_title, str)
    
    def test_validate_title_too_long(self):
        """Test title validation with title that's too long."""
        validator = InputValidator()
        
        long_title = "A" * 1001  # Exceeds 1000 character limit
        
        with pytest.raises(ValidationError, match="Title is too long"):
            validator.validate_title(long_title)
    
    def test_validate_scoring_weights_success(self):
        """Test successful scoring weights validation."""
        validator = InputValidator()
        
        valid_weights = {
            "alignment": 0.45,
            "entities": 0.15,
            "number_unit": 0.20,
            "method_metric": 0.10,
            "recency": 0.07,
            "authority": 0.03,
        }
        
        validated_weights = validator.validate_scoring_weights(valid_weights)
        assert validated_weights == valid_weights
    
    def test_validate_scoring_weights_invalid_sum(self):
        """Test scoring weights validation with invalid sum."""
        validator = InputValidator()
        
        # Weights don't sum to 1.0
        invalid_weights = {
            "alignment": 0.5,
            "entities": 0.5,
            "number_unit": 0.5,
            "method_metric": 0.5,
            "recency": 0.5,
            "authority": 0.5,
        }
        
        with pytest.raises(ValidationError, match="Weights must sum to 1.0"):
            validator.validate_scoring_weights(invalid_weights)
    
    def test_validate_scoring_weights_missing_keys(self):
        """Test scoring weights validation with missing keys."""
        validator = InputValidator()
        
        # Missing some required keys
        incomplete_weights = {
            "alignment": 0.5,
            "entities": 0.3,
            "number_unit": 0.2,
        }
        
        with pytest.raises(ValidationError, match="Missing scoring weight keys"):
            validator.validate_scoring_weights(incomplete_weights)
    
    def test_is_valid_file_path(self):
        """Test file path validation without exceptions."""
        validator = InputValidator()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            assert validator.is_valid_file_path(str(tmp_path)) is True
            assert validator.is_valid_file_path("nonexistent_file.txt") is False
            
        finally:
            tmp_path.unlink()
    
    def test_is_valid_doi(self):
        """Test DOI validation without exceptions."""
        validator = InputValidator()
        
        assert validator.is_valid_doi("10.1234/example.2023") is True
        assert validator.is_valid_doi("invalid_doi") is False
        assert validator.is_valid_doi("") is False
    
    def test_is_valid_year(self):
        """Test year validation without exceptions."""
        validator = InputValidator()
        
        assert validator.is_valid_year(2023) is True
        assert validator.is_valid_year("2023") is True
        assert validator.is_valid_year(999) is False
        assert validator.is_valid_year("invalid") is False
    
    def test_clean_doi(self):
        """Test DOI cleaning function."""
        validator = InputValidator()
        
        # Test various DOI formats
        test_cases = [
            ("10.1234/example.2023", "10.1234/example.2023"),
            ("https://doi.org/10.1234/example.2023", "10.1234/example.2023"),
            ("http://dx.doi.org/10.1234/example.2023", "10.1234/example.2023"),
            ("doi:10.1234/example.2023", "10.1234/example.2023"),
            ("  10.1234/example.2023  ", "10.1234/example.2023"),  # Whitespace
            ("", ""),  # Empty string
        ]
        
        for input_doi, expected in test_cases:
            cleaned = validator.clean_doi(input_doi)
            assert cleaned == expected


class TestConfig:
    """Test cases for Config."""
    
    def test_config_initialization_default(self):
        """Test configuration initialization with default settings."""
        config = Config()
        
        assert config.settings is not None
        assert isinstance(config.settings, Settings)
        assert config.config_file is not None
        assert config.validate_configuration() is True
    
    def test_config_initialization_with_file(self, temp_dir):
        """Test configuration initialization with custom file."""
        config_file = temp_dir / "test_config.json"
        
        # Create a test configuration file
        test_config = {
            "scoring_weights": {
                "alignment": 0.5,
                "entities": 0.3,
                "number_unit": 0.2,
                "method_metric": 0.0,
                "recency": 0.0,
                "authority": 0.0,
            }
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        config = Config(config_file)
        
        # Check that custom weights were loaded
        weights = config.get_scoring_weights()
        assert weights["alignment"] == 0.5
        assert weights["entities"] == 0.3
        assert weights["number_unit"] == 0.2
    
    def test_set_scoring_weights(self):
        """Test setting scoring weights."""
        config = Config()
        
        new_weights = {
            "alignment": 0.6,
            "entities": 0.2,
            "number_unit": 0.1,
            "method_metric": 0.05,
            "recency": 0.03,
            "authority": 0.02,
        }
        
        config.set_scoring_weights(new_weights)
        
        weights = config.get_scoring_weights()
        assert weights == new_weights
    
    def test_set_scoring_weights_invalid(self):
        """Test setting invalid scoring weights."""
        config = Config()
        
        # Invalid weight value
        invalid_weights = {
            "alignment": 1.5,  # Out of range
            "entities": 0.2,
            "number_unit": 0.1,
            "method_metric": 0.05,
            "recency": 0.03,
            "authority": 0.02,
        }
        
        with pytest.raises(ValueError, match="Weight for alignment must be between 0.0 and 1.0"):
            config.set_scoring_weights(invalid_weights)
    
    def test_save_and_load_configuration(self, temp_dir):
        """Test saving and loading configuration."""
        config = Config()
        
        # Modify some settings
        new_weights = {
            "alignment": 0.55,
            "entities": 0.25,
            "number_unit": 0.15,
            "method_metric": 0.03,
            "recency": 0.01,
            "authority": 0.01,
        }
        config.set_scoring_weights(new_weights)
        
        # Save configuration
        save_file = temp_dir / "saved_config.json"
        config.save_configuration(save_file)
        
        # Load configuration in new instance
        new_config = Config(save_file)
        loaded_weights = new_config.get_scoring_weights()
        
        assert loaded_weights == new_weights
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults."""
        config = Config()
        
        # Modify settings
        original_weights = config.get_scoring_weights()
        modified_weights = original_weights.copy()
        modified_weights["alignment"] = 0.99
        config.set_scoring_weights(modified_weights)
        
        # Reset to defaults
        config.reset_to_defaults()
        
        # Check that weights are back to original
        current_weights = config.get_scoring_weights()
        assert current_weights == original_weights
    
    def test_get_config_summary(self):
        """Test getting configuration summary."""
        config = Config()
        
        summary = config.get_config_summary()
        
        assert isinstance(summary, str)
        assert "RefScore Configuration Summary" in summary
        assert "Scoring Weights:" in summary
        assert "NLP Models:" in summary
        assert "Processing:" in summary
        assert "Output:" in summary
        assert "GUI:" in summary
    
    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        config = Config()
        assert config.validate_configuration() is True
    
    def test_validate_configuration_invalid_weights(self):
        """Test configuration validation with invalid weights."""
        config = Config()
        
        # Corrupt the weights
        config.settings.scoring_weights = "invalid"  # Should be dict
        
        assert config.validate_configuration() is False
    
    def test_get_nlp_model_config(self):
        """Test getting NLP model configuration."""
        config = Config()
        
        nlp_config = config.get_nlp_model_config()
        assert isinstance(nlp_config, dict)
        assert "sentence_transformer" in nlp_config
        assert "spacy_model" in nlp_config
    
    def test_get_processing_config(self):
        """Test getting processing configuration."""
        config = Config()
        
        processing_config = config.get_processing_config()
        assert isinstance(processing_config, dict)
        assert "min_sentence_length" in processing_config
        assert "max_sentence_length" in processing_config
    
    def test_get_output_config(self):
        """Test getting output configuration."""
        config = Config()
        
        output_config = config.get_output_config()
        assert isinstance(output_config, dict)
        assert "default_output_dir" in output_config
        assert "save_json" in output_config
    
    def test_get_gui_config(self):
        """Test getting GUI configuration."""
        config = Config()
        
        gui_config = config.get_gui_config()
        assert isinstance(gui_config, dict)
        assert "theme" in gui_config
        assert "font_size" in gui_config
    
    def test_set_gui_config(self):
        """Test setting GUI configuration."""
        config = Config()
        
        new_gui_config = {
            "theme": "dark",
            "font_size": 12,
            "show_tooltips": False,
        }
        
        config.set_gui_config(new_gui_config)
        
        gui_config = config.get_gui_config()
        assert gui_config["theme"] == "dark"
        assert gui_config["font_size"] == 12
        assert gui_config["show_tooltips"] is False


class TestExceptions:
    """Test cases for custom exceptions."""
    
    def test_refscore_error_base(self):
        """Test base RefScoreError."""
        error = ValidationError("Test error", "Test details")
        
        assert str(error) == "Test error (Test details)"
        assert error.message == "Test error"
        assert error.details == "Test details"
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Validation failed", field="test_field", value="invalid_value")
        
        assert "field: test_field" in str(error)
        assert "value: invalid_value" in str(error)
        assert error.field == "test_field"
        assert error.value == "invalid_value"
    
    def test_processing_error(self):
        """Test ProcessingError."""
        error = ProcessingError("Processing failed", operation="parse_document", 
                               original_error="File not found")
        
        assert "operation: parse_document" in str(error)
        assert "error: File not found" in str(error)
        assert error.operation == "parse_document"
        assert error.original_error == "File not found"
    
    def test_file_format_error(self):
        """Test FileFormatError."""
        error = FileFormatError("Invalid format", file_path="test.doc", 
                               expected_format="PDF", actual_format="DOC")
        
        assert "file: test.doc" in str(error)
        assert "expected: PDF" in str(error)
        assert "actual: DOC" in str(error)
        assert error.file_path == "test.doc"
        assert error.expected_format == "PDF"
        assert error.actual_format == "DOC"
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Config error", config_key="timeout", 
                                config_value="invalid")
        
        assert "key: timeout" in str(error)
        assert "value: invalid" in str(error)
        assert error.config_key == "timeout"
        assert error.config_value == "invalid"
    
    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Network error", url="https://api.example.com", 
                           status_code=404, response_text="Not found")
        
        assert "url: https://api.example.com" in str(error)
        assert "status: 404" in str(error)
        assert "response: Not found" in str(error)
        assert error.url == "https://api.example.com"
        assert error.status_code == 404
        assert error.response_text == "Not found"
    
    def test_dependency_error(self):
        """Test DependencyError."""
        error = DependencyError("Missing dependency", dependency_name="numpy", 
                              required_version=">=1.20.0", installed_version="1.19.0")
        
        assert "dependency: numpy" in str(error)
        assert "required: >=1.20.0" in str(error)
        assert "installed: 1.19.0" in str(error)
        assert error.dependency_name == "numpy"
        assert error.required_version == ">=1.20.0"
        assert error.installed_version == "1.19.0"
    
    def test_gui_error(self):
        """Test GUIError."""
        error = GUIError("GUI error", widget_name="MainWindow", operation="show")
        
        assert "widget: MainWindow" in str(error)
        assert "operation: show" in str(error)
        assert error.widget_name == "MainWindow"
        assert error.operation == "show"
    
    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Database error", table_name="users", 
                            operation="SELECT", query="SELECT * FROM users")
        
        assert "table: users" in str(error)
        assert "operation: SELECT" in str(error)
        assert "query: SELECT * FROM users" in str(error)
        assert error.table_name == "users"
        assert error.operation == "SELECT"
        assert error.query == "SELECT * FROM users"
