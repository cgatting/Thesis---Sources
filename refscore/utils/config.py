"""
Configuration management for RefScore academic application.

This module provides configuration management, settings handling,
and application-wide parameter storage.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict


log = logging.getLogger(__name__)


@dataclass
class Settings:
    """
    Application settings and configuration parameters.
    
    Attributes:
        scoring_weights: Weights for different scoring dimensions
        nlp_models: Configuration for NLP models
        processing: Processing-related settings
        output: Output format and location settings
        gui: GUI appearance and behavior settings
    """
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "alignment": 0.45,
        "number_unit": 0.20,
        "entities": 0.15,
        "method_metric": 0.10,
        "recency": 0.07,
        "authority": 0.03,
    })
    
    nlp_models: Dict[str, Any] = field(default_factory=lambda: {
        "sentence_transformer": "all-MiniLM-L6-v2",
        "spacy_model": "en_core_web_sm",
        "fallback_to_blank_spacy": True,
        "use_tfidf_fallback": True,
        "use_jaccard_fallback": True,
    })
    
    processing: Dict[str, Any] = field(default_factory=lambda: {
        "min_sentence_length": 6,
        "max_sentence_length": 1000,
        "enable_multiprocessing": True,
        "max_workers": 4,
        "timeout_seconds": 300,
    })
    
    output: Dict[str, Any] = field(default_factory=lambda: {
        "default_output_dir": "refscore_results",
        "save_json": True,
        "save_csv": True,
        "save_visualizations": True,
        "generate_report": True,
    })
    
    gui: Dict[str, Any] = field(default_factory=lambda: {
        "theme": "light",
        "window_size": [1200, 800],
        "font_size": 10,
        "show_tooltips": True,
        "auto_save_settings": True,
    })
    quality: Dict[str, Any] = field(default_factory=lambda: {
        "document_type": "research_paper",
        "language": "auto",
        "criteria": {
            "coverage_thresholds": [0.6, 0.3],
            "strength_support_threshold": 0.65,
            "weak_support_threshold": 0.35,
            "min_examples_per_section": 2
        }
    })


class Config:
    """
    Configuration manager for RefScore application.
    
    Provides centralized configuration management with support for:
    - Default settings
    - User configuration files
    - Environment variables
    - Runtime configuration changes
    """
    
    DEFAULT_CONFIG_FILE = "refscore_config.json"
    USER_CONFIG_DIR = Path.home() / ".config" / "refscore"
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.settings = Settings()
        self.config_file = config_file or self.USER_CONFIG_DIR / self.DEFAULT_CONFIG_FILE
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from file if it exists."""
        try:
            if isinstance(self.config_file, str):
                self.config_file = Path(self.config_file)
            
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Update settings with loaded configuration
                self._update_settings_from_dict(config_data)
                log.info(f"Configuration loaded from {self.config_file}")
            else:
                log.info("Using default configuration")
                
        except Exception as e:
            log.warning(f"Failed to load configuration from {self.config_file}: {e}")
            log.info("Using default configuration")
    
    def _update_settings_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update settings from dictionary."""
        if "scoring_weights" in config_dict:
            self.settings.scoring_weights.update(config_dict["scoring_weights"])
        
        if "nlp_models" in config_dict:
            self.settings.nlp_models.update(config_dict["nlp_models"])
        
        if "processing" in config_dict:
            self.settings.processing.update(config_dict["processing"])
        
        if "output" in config_dict:
            self.settings.output.update(config_dict["output"])
        
        if "gui" in config_dict:
            self.settings.gui.update(config_dict["gui"])
        if "quality" in config_dict:
            self.settings.quality.update(config_dict["quality"])
    
    def save_configuration(self, config_file: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_file: Optional path to save configuration file
        """
        try:
            save_path = config_file or self.config_file
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert settings to dictionary
            config_dict = asdict(self.settings)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            log.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            log.error(f"Failed to save configuration: {e}")
            raise
    
    def get_scoring_weights(self) -> Dict[str, float]:
        """Get current scoring weights."""
        return self.settings.scoring_weights.copy()
    
    def set_scoring_weights(self, weights: Dict[str, float]) -> None:
        """
        Set scoring weights.
        
        Args:
            weights: Dictionary of scoring weights
        """
        # Validate weights
        for key, value in weights.items():
            if key not in self.settings.scoring_weights:
                log.warning(f"Unknown scoring weight key: {key}")
                continue
            
            if not isinstance(value, (int, float)):
                raise ValueError(f"Weight for {key} must be a number")
            
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Weight for {key} must be between 0.0 and 1.0")
            
            self.settings.scoring_weights[key] = float(value)
        
        log.info("Scoring weights updated")
    
    def get_nlp_model_config(self) -> Dict[str, Any]:
        """Get NLP model configuration."""
        return self.settings.nlp_models.copy()
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.settings.processing.copy()
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.settings.output.copy()
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration."""
        return self.settings.gui.copy()
    
    def set_gui_config(self, config: Dict[str, Any]) -> None:
        """
        Set GUI configuration.
        
        Args:
            config: GUI configuration dictionary
        """
        self.settings.gui.update(config)
        log.info("GUI configuration updated")
        
        if self.settings.gui.get("auto_save_settings", True):
            self.save_configuration()

    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality evaluation configuration."""
        return self.settings.quality.copy()

    def set_quality_config(self, config: Dict[str, Any]) -> None:
        """Set quality evaluation configuration."""
        self.settings.quality.update(config)
        if self.settings.gui.get("auto_save_settings", True):
            self.save_configuration()
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self.settings = Settings()
        log.info("Configuration reset to defaults")
    
    def get_config_summary(self) -> str:
        """
        Get a summary of current configuration.
        
        Returns:
            Formatted configuration summary
        """
        summary = []
        summary.append("=== RefScore Configuration Summary ===")
        summary.append(f"Config file: {self.config_file}")
        summary.append("")
        
        # Scoring weights
        summary.append("Scoring Weights:")
        for key, value in self.settings.scoring_weights.items():
            summary.append(f"  {key}: {value:.3f}")
        summary.append("")
        
        # NLP models
        summary.append("NLP Models:")
        for key, value in self.settings.nlp_models.items():
            summary.append(f"  {key}: {value}")
        summary.append("")
        
        # Processing
        summary.append("Processing:")
        for key, value in self.settings.processing.items():
            summary.append(f"  {key}: {value}")
        summary.append("")
        
        # Output
        summary.append("Output:")
        for key, value in self.settings.output.items():
            summary.append(f"  {key}: {value}")
        summary.append("")
        
        # GUI
        summary.append("GUI:")
        for key, value in self.settings.gui.items():
            summary.append(f"  {key}: {value}")
        
        return "\n".join(summary)
    
    def validate_configuration(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate scoring weights
            weights = self.settings.scoring_weights
            if not isinstance(weights, dict):
                return False
            
            for key, value in weights.items():
                if not isinstance(value, (int, float)):
                    return False
                if not (0.0 <= value <= 1.0):
                    return False
            
            # Validate other settings
            if not isinstance(self.settings.nlp_models, dict):
                return False
            
            if not isinstance(self.settings.processing, dict):
                return False
            
            if not isinstance(self.settings.output, dict):
                return False
            
            if not isinstance(self.settings.gui, dict):
                return False
            
            return True
            
        except Exception:
            return False
