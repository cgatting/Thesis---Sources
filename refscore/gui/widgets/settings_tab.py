"""
Settings tab widget for RefScore academic application.

This module provides the settings configuration interface where users can
adjust application preferences, scoring weights, and processing options.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTabWidget, QComboBox, QTextEdit, QFrame, QMessageBox, QGridLayout,
    QProgressDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

import logging
from pathlib import Path
from typing import List

from ...utils.config import Config
from ...utils.validators import InputValidator
from ...utils.exceptions import ValidationError
from ...training.weights_trainer import WeightsTrainer


log = logging.getLogger(__name__)


class SettingsTab(QWidget):
    """Settings tab widget for application configuration."""
    
    settings_changed = pyqtSignal()
    
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Settings tabs
        self.settings_tabs = QTabWidget()
        layout.addWidget(self.settings_tabs)
        
        # Scoring settings tab
        scoring_tab = self.create_scoring_tab()
        self.settings_tabs.addTab(scoring_tab, "Scoring")
        
        # Processing settings tab
        processing_tab = self.create_processing_tab()
        self.settings_tabs.addTab(processing_tab, "Processing")
        
        # Output settings tab
        output_tab = self.create_output_tab()
        self.settings_tabs.addTab(output_tab, "Output")
        
        # GUI settings tab
        gui_tab = self.create_gui_tab()
        self.settings_tabs.addTab(gui_tab, "Interface")
        
        # Advanced settings tab
        advanced_tab = self.create_advanced_tab()
        self.settings_tabs.addTab(advanced_tab, "Advanced")

        evaluation_tab = self.create_evaluation_tab()
        self.settings_tabs.addTab(evaluation_tab, "Evaluation")
        
        # Control buttons
        control_layout = self.create_control_layout()
        layout.addLayout(control_layout)
    
    def create_scoring_tab(self) -> QWidget:
        """Create scoring settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scoring weights group
        weights_group = QGroupBox("Scoring Weights")
        weights_layout = QGridLayout(weights_group)
        
        weights = self.config.get_scoring_weights()
        self.weight_spinboxes = {}
        
        row = 0
        for key, label in [
            ("alignment", "Semantic Alignment:"),
            ("entities", "Entity Overlap:"),
            ("number_unit", "Number/Unit Match:"),
            ("method_metric", "Method/Metric Overlap:"),
            ("recency", "Publication Recency:"),
            ("authority", "Venue Authority:")
        ]:
            weights_layout.addWidget(QLabel(label), row, 0)
            
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.0, 1.0)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(3)
            spinbox.setValue(weights.get(key, 0.0))
            spinbox.setToolTip(f"Weight for {key.replace('_', ' ')} scoring")
            
            self.weight_spinboxes[key] = spinbox
            weights_layout.addWidget(spinbox, row, 1)
            
            row += 1
        
        # Weight validation
        self.weight_warning_label = QLabel()
        self.weight_warning_label.setStyleSheet("color: red; font-style: italic;")
        weights_layout.addWidget(self.weight_warning_label, row, 0, 1, 2)
        
        layout.addWidget(weights_group)
        
        # Presets
        presets_group = QGroupBox("Weight Presets")
        presets_layout = QHBoxLayout(presets_group)
        
        balanced_btn = QPushButton("Balanced")
        balanced_btn.clicked.connect(lambda: self.apply_preset("balanced"))
        presets_layout.addWidget(balanced_btn)
        
        semantic_btn = QPushButton("Semantic Focus")
        semantic_btn.clicked.connect(lambda: self.apply_preset("semantic"))
        presets_layout.addWidget(semantic_btn)
        
        technical_btn = QPushButton("Technical Focus")
        technical_btn.clicked.connect(lambda: self.apply_preset("technical"))
        presets_layout.addWidget(technical_btn)
        
        presets_layout.addStretch()
        layout.addWidget(presets_group)

        train_group = QGroupBox("Train & Presets")
        train_layout = QGridLayout(train_group)

        train_layout.addWidget(QLabel("Preset Name"), 0, 0)
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("best")
        train_layout.addWidget(self.preset_name_edit, 0, 1)

        train_layout.addWidget(QLabel("Search Space"), 1, 0)
        self.search_space_edit = QLineEdit()
        self.search_space_edit.setPlaceholderText("alignment=0.4:0.6:0.05 entities=0.1:0.3:0.05 number_unit=0.15:0.25:0.05")
        train_layout.addWidget(self.search_space_edit, 1, 1)

        self.strict_sources_checkbox = QCheckBox("Strict source validation")
        self.strict_sources_checkbox.setChecked(True)
        train_layout.addWidget(self.strict_sources_checkbox, 2, 0, 1, 2)

        self.train_btn = QPushButton("Train Weights")
        self.train_btn.clicked.connect(self.train_weights)
        train_layout.addWidget(self.train_btn, 3, 0, 1, 2)

        train_layout.addWidget(QLabel("Saved Presets"), 4, 0)
        self.presets_combo = QComboBox()
        train_layout.addWidget(self.presets_combo, 4, 1)

        self.refresh_presets_btn = QPushButton("Refresh")
        self.refresh_presets_btn.clicked.connect(self.refresh_presets)
        train_layout.addWidget(self.refresh_presets_btn, 5, 0)

        self.apply_preset_btn = QPushButton("Apply")
        self.apply_preset_btn.clicked.connect(self.apply_saved_preset)
        train_layout.addWidget(self.apply_preset_btn, 5, 1)

        self.rollback_btn = QPushButton("Rollback")
        self.rollback_btn.clicked.connect(self.rollback_preset)
        train_layout.addWidget(self.rollback_btn, 6, 0, 1, 2)

        self.training_status_label = QLabel()
        train_layout.addWidget(self.training_status_label, 7, 0, 1, 2)

        layout.addWidget(train_group)
        self.refresh_presets()
        
        layout.addStretch()
        return widget
    
    def create_processing_tab(self) -> QWidget:
        """Create processing settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # NLP settings group
        nlp_group = QGroupBox("NLP Processing")
        nlp_layout = QGridLayout(nlp_group)
        
        nlp_config = self.config.get_nlp_model_config()
        
        # Sentence transformer model
        nlp_layout.addWidget(QLabel("Sentence Transformer:"), 0, 0)
        self.sentence_transformer_edit = QLineEdit()
        self.sentence_transformer_edit.setText(nlp_config.get("sentence_transformer", "all-MiniLM-L6-v2"))
        self.sentence_transformer_edit.setToolTip("Model name for sentence embeddings")
        nlp_layout.addWidget(self.sentence_transformer_edit, 0, 1)
        
        # SpaCy model
        nlp_layout.addWidget(QLabel("spaCy Model:"), 1, 0)
        self.spacy_model_edit = QLineEdit()
        self.spacy_model_edit.setText(nlp_config.get("spacy_model", "en_core_web_sm"))
        self.spacy_model_edit.setToolTip("spaCy model for NER and tokenization")
        nlp_layout.addWidget(self.spacy_model_edit, 1, 1)
        
        # Fallback options
        self.fallback_blank_checkbox = QCheckBox("Fallback to blank spaCy model")
        self.fallback_blank_checkbox.setChecked(nlp_config.get("fallback_to_blank_spacy", True))
        nlp_layout.addWidget(self.fallback_blank_checkbox, 2, 0, 1, 2)
        
        self.tfidf_fallback_checkbox = QCheckBox("Use TF-IDF fallback")
        self.tfidf_fallback_checkbox.setChecked(nlp_config.get("use_tfidf_fallback", True))
        nlp_layout.addWidget(self.tfidf_fallback_checkbox, 3, 0, 1, 2)
        
        self.jaccard_fallback_checkbox = QCheckBox("Use Jaccard fallback")
        self.jaccard_fallback_checkbox.setChecked(nlp_config.get("use_jaccard_fallback", True))
        nlp_layout.addWidget(self.jaccard_fallback_checkbox, 4, 0, 1, 2)
        
        layout.addWidget(nlp_group)
        
        # Processing limits group
        limits_group = QGroupBox("Processing Limits")
        limits_layout = QGridLayout(limits_group)
        
        processing_config = self.config.get_processing_config()
        
        # Min sentence length
        limits_layout.addWidget(QLabel("Min Sentence Length:"), 0, 0)
        self.min_sentence_spinbox = QSpinBox()
        self.min_sentence_spinbox.setRange(1, 50)
        self.min_sentence_spinbox.setValue(processing_config.get("min_sentence_length", 6))
        limits_layout.addWidget(self.min_sentence_spinbox, 0, 1)
        
        # Max sentence length
        limits_layout.addWidget(QLabel("Max Sentence Length:"), 1, 0)
        self.max_sentence_spinbox = QSpinBox()
        self.max_sentence_spinbox.setRange(100, 10000)
        self.max_sentence_spinbox.setValue(processing_config.get("max_sentence_length", 1000))
        limits_layout.addWidget(self.max_sentence_spinbox, 1, 1)
        
        # Timeout
        limits_layout.addWidget(QLabel("Timeout (seconds):"), 2, 0)
        self.timeout_spinbox = QSpinBox()
        self.timeout_spinbox.setRange(10, 3600)
        self.timeout_spinbox.setValue(processing_config.get("timeout_seconds", 300))
        limits_layout.addWidget(self.timeout_spinbox, 2, 1)
        
        layout.addWidget(limits_group)
        
        layout.addStretch()
        return widget
    
    def create_output_tab(self) -> QWidget:
        """Create output settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Output format group
        format_group = QGroupBox("Output Formats")
        format_layout = QVBoxLayout(format_group)
        
        output_config = self.config.get_output_config()
        
        self.save_json_checkbox = QCheckBox("Save JSON results")
        self.save_json_checkbox.setChecked(output_config.get("save_json", True))
        format_layout.addWidget(self.save_json_checkbox)
        
        self.save_csv_checkbox = QCheckBox("Save CSV results")
        self.save_csv_checkbox.setChecked(output_config.get("save_csv", True))
        format_layout.addWidget(self.save_csv_checkbox)
        
        self.save_visualizations_checkbox = QCheckBox("Save visualizations")
        self.save_visualizations_checkbox.setChecked(output_config.get("save_visualizations", True))
        format_layout.addWidget(self.save_visualizations_checkbox)
        
        self.generate_report_checkbox = QCheckBox("Generate comprehensive report")
        self.generate_report_checkbox.setChecked(output_config.get("generate_report", True))
        format_layout.addWidget(self.generate_report_checkbox)
        
        layout.addWidget(format_group)
        
        # Default output directory
        dir_group = QGroupBox("Default Output Directory")
        dir_layout = QHBoxLayout(dir_group)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(output_config.get("default_output_dir", "refscore_results"))
        dir_layout.addWidget(self.output_dir_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output_directory)
        dir_layout.addWidget(browse_btn)
        
        layout.addWidget(dir_group)
        
        layout.addStretch()
        return widget
    
    def create_gui_tab(self) -> QWidget:
        """Create GUI settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QGridLayout(appearance_group)
        
        gui_config = self.config.get_gui_config()
        
        # Theme
        appearance_layout.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combobox = QComboBox()
        self.theme_combobox.addItems(["Light", "Dark", "System"])
        current_theme = gui_config.get("theme", "light").capitalize()
        if current_theme in ["Light", "Dark", "System"]:
            self.theme_combobox.setCurrentText(current_theme)
        appearance_layout.addWidget(self.theme_combobox, 0, 1)
        
        # Font size
        appearance_layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(8, 16)
        self.font_size_spinbox.setValue(gui_config.get("font_size", 10))
        appearance_layout.addWidget(self.font_size_spinbox, 1, 1)
        
        # Show tooltips
        self.show_tooltips_checkbox = QCheckBox("Show tooltips")
        self.show_tooltips_checkbox.setChecked(gui_config.get("show_tooltips", True))
        appearance_layout.addWidget(self.show_tooltips_checkbox, 2, 0, 1, 2)
        
        layout.addWidget(appearance_group)
        
        # Behavior group
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QVBoxLayout(behavior_group)
        
        self.auto_save_checkbox = QCheckBox("Auto-save settings")
        self.auto_save_checkbox.setChecked(gui_config.get("auto_save_settings", True))
        behavior_layout.addWidget(self.auto_save_checkbox)
        
        layout.addWidget(behavior_group)
        
        layout.addStretch()
        return widget
    
    def create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Multiprocessing group
        mp_group = QGroupBox("Multiprocessing")
        mp_layout = QVBoxLayout(mp_group)
        
        processing_config = self.config.get_processing_config()
        
        self.enable_multiprocessing_checkbox = QCheckBox("Enable multiprocessing")
        self.enable_multiprocessing_checkbox.setChecked(processing_config.get("enable_multiprocessing", True))
        mp_layout.addWidget(self.enable_multiprocessing_checkbox)
        
        # Max workers
        mp_layout.addWidget(QLabel("Max Workers:"))
        self.max_workers_spinbox = QSpinBox()
        self.max_workers_spinbox.setRange(1, 16)
        self.max_workers_spinbox.setValue(processing_config.get("max_workers", 4))
        mp_layout.addWidget(self.max_workers_spinbox)
        
        layout.addWidget(mp_group)
        
        # Configuration info
        config_group = QGroupBox("Configuration Info")
        config_layout = QVBoxLayout(config_group)
        
        self.config_info_text = QTextEdit()
        self.config_info_text.setReadOnly(True)
        self.config_info_text.setMaximumHeight(150)
        self.config_info_text.setPlainText(self.config.get_config_summary())
        config_layout.addWidget(self.config_info_text)
        
        layout.addWidget(config_group)
        
        layout.addStretch()
        return widget

    def create_evaluation_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        group = QGroupBox("Quality Evaluation")
        gl = QGridLayout(group)
        qc = self.config.get_quality_config()
        gl.addWidget(QLabel("Document Type:"), 0, 0)
        self.doc_type_combo = QComboBox()
        self.doc_type_combo.addItems(["research_paper", "thesis", "proposal", "report"])
        self.doc_type_combo.setCurrentText(qc.get("document_type", "research_paper"))
        gl.addWidget(self.doc_type_combo, 0, 1)
        gl.addWidget(QLabel("Language:"), 1, 0)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["auto", "en", "es"])
        self.lang_combo.setCurrentText(qc.get("language", "auto"))
        gl.addWidget(self.lang_combo, 1, 1)
        gl.addWidget(QLabel("High Coverage ≥"), 2, 0)
        self.cov_hi_spin = QDoubleSpinBox()
        self.cov_hi_spin.setRange(0.0, 1.0)
        self.cov_hi_spin.setSingleStep(0.01)
        self.cov_hi_spin.setDecimals(2)
        hi = float(qc.get("criteria", {}).get("coverage_thresholds", [0.6, 0.3])[0])
        self.cov_hi_spin.setValue(hi)
        gl.addWidget(self.cov_hi_spin, 2, 1)
        gl.addWidget(QLabel("Mid Coverage ≥"), 3, 0)
        self.cov_mid_spin = QDoubleSpinBox()
        self.cov_mid_spin.setRange(0.0, 1.0)
        self.cov_mid_spin.setSingleStep(0.01)
        self.cov_mid_spin.setDecimals(2)
        mid = float(qc.get("criteria", {}).get("coverage_thresholds", [0.6, 0.3])[1])
        self.cov_mid_spin.setValue(mid)
        gl.addWidget(self.cov_mid_spin, 3, 1)
        gl.addWidget(QLabel("Strong Support ≥"), 4, 0)
        self.str_thr_spin = QDoubleSpinBox()
        self.str_thr_spin.setRange(0.0, 1.0)
        self.str_thr_spin.setSingleStep(0.01)
        self.str_thr_spin.setDecimals(2)
        self.str_thr_spin.setValue(float(qc.get("criteria", {}).get("strength_support_threshold", 0.65)))
        gl.addWidget(self.str_thr_spin, 4, 1)
        gl.addWidget(QLabel("Weak Support <"), 5, 0)
        self.weak_thr_spin = QDoubleSpinBox()
        self.weak_thr_spin.setRange(0.0, 1.0)
        self.weak_thr_spin.setSingleStep(0.01)
        self.weak_thr_spin.setDecimals(2)
        self.weak_thr_spin.setValue(float(qc.get("criteria", {}).get("weak_support_threshold", 0.35)))
        gl.addWidget(self.weak_thr_spin, 5, 1)
        layout.addWidget(group)
        layout.addStretch()
        return widget
    
    def create_control_layout(self) -> QHBoxLayout:
        """Create control buttons layout."""
        layout = QHBoxLayout()
        
        layout.addStretch()
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        layout.addWidget(reset_btn)
        
        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        return layout
    
    def load_settings(self):
        """Load current settings into UI."""
        # Scoring weights are loaded in create_scoring_tab
        # Other settings are loaded in their respective tabs
        pass
    
    def apply_preset(self, preset: str):
        """Apply a scoring weight preset."""
        presets = {
            "balanced": {
                "alignment": 0.45,
                "entities": 0.15,
                "number_unit": 0.20,
                "method_metric": 0.10,
                "recency": 0.07,
                "authority": 0.03,
            },
            "semantic": {
                "alignment": 0.60,
                "entities": 0.20,
                "number_unit": 0.10,
                "method_metric": 0.05,
                "recency": 0.03,
                "authority": 0.02,
            },
            "technical": {
                "alignment": 0.30,
                "entities": 0.10,
                "number_unit": 0.30,
                "method_metric": 0.20,
                "recency": 0.07,
                "authority": 0.03,
            }
        }
        
        if preset in presets:
            weights = presets[preset]
            for key, value in weights.items():
                if key in self.weight_spinboxes:
                    self.weight_spinboxes[key].setValue(value)
            
            self.validate_weights()

    def _parse_search_space(self) -> dict:
        text = self.search_space_edit.text().strip()
        if not text:
            weights = {}
            for k, sb in self.weight_spinboxes.items():
                v = float(sb.value())
                lo = max(0.0, round(v - 0.1, 2))
                hi = min(1.0, round(v + 0.1, 2))
                step = 0.05
                weights[k] = (lo, hi, step)
            return weights
        parts = text.split()
        space = {}
        for p in parts:
            if "=" not in p or ":" not in p:
                continue
            k, rng = p.split("=", 1)
            try:
                lo_s, hi_s, step_s = rng.split(":")
                lo = float(lo_s)
                hi = float(hi_s)
                step = float(step_s)
                space[k] = (lo, hi, step)
            except Exception:
                pass
        return space

    def _current_paths(self):
        parent = self.window()
        doc_path = ""
        src_paths = []
        try:
            if hasattr(parent, "analysis_tab"):
                doc_path = parent.analysis_tab.get_document_path()
            if hasattr(parent, "sources_tab"):
                src_paths = parent.sources_tab.get_source_paths()
            if not src_paths and hasattr(parent, "analysis_tab"):
                src_paths = parent.analysis_tab.get_source_paths()
            if not doc_path and hasattr(parent, "relevant_tab"):
                try:
                    files = getattr(parent.relevant_tab, "uploaded_files", [])
                    if files:
                        doc_path = files[-1].get("path", "")
                except Exception:
                    pass
        except Exception:
            doc_path = ""
            src_paths = []
        return doc_path, src_paths

    def _validate_training_files(self, doc_path: str, src_paths: list) -> bool:
        try:
            validator = InputValidator()
            validator.validate_document_file(doc_path)
            if not src_paths:
                raise ValidationError("At least one source file must be provided")
            for p in src_paths:
                validator.validate_source_file(p)
            if self.strict_sources_checkbox.isChecked():
                missing = [p for p in src_paths if not Path(p).exists()]
                if missing:
                    QMessageBox.warning(self, "Missing Sources", f"These files are missing:\n" + "\n".join(missing))
                    return False
            return True
        except ValidationError as e:
            QMessageBox.critical(self, "Invalid File", str(e))
            return False

    def train_weights(self):
        doc_path, src_paths = self._current_paths()
        if not doc_path:
            QMessageBox.warning(self, "Missing Files", "Document not selected.")
            return
        if not src_paths:
            QMessageBox.warning(self, "Missing Files", "No source files selected.")
            return
        if not self._validate_training_files(doc_path, src_paths):
            return
        progress = QProgressDialog("Training weights...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()
        try:
            trainer = WeightsTrainer(self.config)
            space = self._parse_search_space()
            jobs = [(doc_path, src_paths)]
            best_w, metrics = trainer.train(jobs, space)
            self.config.set_scoring_weights(best_w)
            for k, v in best_w.items():
                if k in self.weight_spinboxes:
                    self.weight_spinboxes[k].setValue(v)
            preset_name = self.preset_name_edit.text().strip() or "best"
            preset_id = trainer.save_preset(preset_name, best_w, metrics, {"search_space": space})
            self.config.save_configuration()
            self.config_info_text.setPlainText(self.config.get_config_summary())
            doc_name = Path(doc_path).name if doc_path else ""
            self.training_status_label.setText(f"Doc={doc_name} sources={len(src_paths)} avg_top={metrics.get('avg_top_score', 0.0):.3f} coverage={metrics.get('overall_coverage', 0.0):.3f} saved={preset_id}")
            self.refresh_presets()
            QMessageBox.information(self, "Training Complete", "Optimized weights trained and saved.")
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))
        finally:
            progress.close()

    def refresh_presets(self):
        try:
            trainer = WeightsTrainer(self.config)
            items = trainer.list_presets()
            self.presets_combo.clear()
            for e in items:
                name = e.get("name") or ""
                pid = e.get("id") or ""
                label = f"{name} ({pid})" if name else pid
                self.presets_combo.addItem(label, pid)
        except Exception:
            self.presets_combo.clear()

    def apply_saved_preset(self):
        idx = self.presets_combo.currentIndex()
        if idx < 0:
            return
        pid = self.presets_combo.itemData(idx)
        try:
            trainer = WeightsTrainer(self.config)
            ok = trainer.apply_preset(pid)
            if ok:
                weights = self.config.get_scoring_weights()
                for k, v in weights.items():
                    if k in self.weight_spinboxes:
                        self.weight_spinboxes[k].setValue(v)
                self.config.save_configuration()
                self.config_info_text.setPlainText(self.config.get_config_summary())
                QMessageBox.information(self, "Preset Applied", "Weights preset applied.")
            else:
                QMessageBox.warning(self, "Apply Failed", "Failed to apply preset.")
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", str(e))

    def rollback_preset(self):
        try:
            trainer = WeightsTrainer(self.config)
            last = trainer.rollback()
            if last:
                weights = self.config.get_scoring_weights()
                for k, v in weights.items():
                    if k in self.weight_spinboxes:
                        self.weight_spinboxes[k].setValue(v)
                self.config.save_configuration()
                self.config_info_text.setPlainText(self.config.get_config_summary())
                QMessageBox.information(self, "Rolled Back", "Rolled back to previous preset.")
            else:
                QMessageBox.information(self, "No History", "No previous preset to rollback.")
        except Exception as e:
            QMessageBox.critical(self, "Rollback Error", str(e))
    
    def validate_weights(self) -> bool:
        """Validate scoring weights."""
        try:
            weights = {}
            for key, spinbox in self.weight_spinboxes.items():
                weights[key] = spinbox.value()
            
            validator = InputValidator()
            validator.validate_scoring_weights(weights)
            
            self.weight_warning_label.setText("")
            return True
            
        except ValidationError as e:
            self.weight_warning_label.setText(str(e))
            return False
    
    def save_settings(self):
        """Save current settings."""
        try:
            # Validate weights first
            if not self.validate_weights():
                QMessageBox.warning(self, "Validation Error", "Please fix scoring weight errors.")
                return
            
            # Save scoring weights
            weights = {}
            for key, spinbox in self.weight_spinboxes.items():
                weights[key] = spinbox.value()
            self.config.set_scoring_weights(weights)
            
            # Save NLP settings
            nlp_config = {
                "sentence_transformer": self.sentence_transformer_edit.text(),
                "spacy_model": self.spacy_model_edit.text(),
                "fallback_to_blank_spacy": self.fallback_blank_checkbox.isChecked(),
                "use_tfidf_fallback": self.tfidf_fallback_checkbox.isChecked(),
                "use_jaccard_fallback": self.jaccard_fallback_checkbox.isChecked(),
            }
            
            # Save processing settings
            processing_config = {
                "min_sentence_length": self.min_sentence_spinbox.value(),
                "max_sentence_length": self.max_sentence_spinbox.value(),
                "timeout_seconds": self.timeout_spinbox.value(),
                "enable_multiprocessing": self.enable_multiprocessing_checkbox.isChecked(),
                "max_workers": self.max_workers_spinbox.value(),
            }
            
            # Save output settings
            output_config = {
                "default_output_dir": self.output_dir_edit.text(),
                "save_json": self.save_json_checkbox.isChecked(),
                "save_csv": self.save_csv_checkbox.isChecked(),
                "save_visualizations": self.save_visualizations_checkbox.isChecked(),
                "generate_report": self.generate_report_checkbox.isChecked(),
            }
            
            # Save GUI settings
            gui_config = {
                "theme": self.theme_combobox.currentText().lower(),
                "font_size": self.font_size_spinbox.value(),
                "show_tooltips": self.show_tooltips_checkbox.isChecked(),
                "auto_save_settings": self.auto_save_checkbox.isChecked(),
            }

            quality_config = {
                "document_type": self.doc_type_combo.currentText(),
                "language": self.lang_combo.currentText(),
                "criteria": {
                    "coverage_thresholds": [self.cov_hi_spin.value(), self.cov_mid_spin.value()],
                    "strength_support_threshold": self.str_thr_spin.value(),
                    "weak_support_threshold": self.weak_thr_spin.value(),
                    "min_examples_per_section": self.config.settings.quality.get("criteria", {}).get("min_examples_per_section", 2)
                }
            }
            
            # Apply all settings
            self.config.settings.nlp_models = nlp_config
            self.config.settings.processing = processing_config
            self.config.settings.output = output_config
            self.config.settings.gui = gui_config
            self.config.settings.quality = quality_config
            
            # Save to file
            self.config.save_configuration()
            
            # Update info display
            self.config_info_text.setPlainText(self.config.get_config_summary())
            
            # Notify that settings changed
            self.settings_changed.emit()
            
            QMessageBox.information(self, "Settings Saved", "Settings saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")
    
    def reset_to_defaults(self):
        """Reset settings to defaults."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config.reset_to_defaults()
            self.load_settings()
            self.config_info_text.setPlainText(self.config.get_config_summary())
            QMessageBox.information(self, "Settings Reset", "Settings reset to defaults.")
    
    def browse_output_directory(self):
        """Browse for output directory."""
        from PyQt5.QtWidgets import QFileDialog
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Default Output Directory"
        )
        
        if directory:
            self.output_dir_edit.setText(directory)
    
    def get_source_paths(self) -> List[str]:
        """Get source file paths from other tabs if needed."""
        return []
