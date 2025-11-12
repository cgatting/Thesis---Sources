"""
Analysis tab widget for RefScore academic application.

This module provides the main analysis interface where users can
select documents and sources, configure analysis parameters, and
initiate the analysis process.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QProgressBar, QGridLayout
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

import logging
from pathlib import Path
from typing import List

from ...utils.validators import InputValidator
from ...utils.exceptions import ValidationError


log = logging.getLogger(__name__)


class AnalysisTab(QWidget):
    """Analysis tab widget for document and source selection."""
    
    analysis_requested = pyqtSignal(str, list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.document_path = ""
        self.source_paths = []
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Document selection group
        doc_group = self.create_document_group()
        layout.addWidget(doc_group)
        
        # Sources selection group
        sources_group = self.create_sources_group()
        layout.addWidget(sources_group)
        
        # Analysis settings group
        settings_group = self.create_settings_group()
        layout.addWidget(settings_group)
        
        # Control buttons
        control_layout = self.create_control_layout()
        layout.addLayout(control_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
    
    def create_document_group(self) -> QGroupBox:
        """Create document selection group."""
        group = QGroupBox("Document Selection")
        layout = QGridLayout(group)
        
        # Document path
        layout.addWidget(QLabel("Document File:"), 0, 0)
        
        self.doc_path_edit = QLineEdit()
        self.doc_path_edit.setPlaceholderText("Select a LaTeX (.tex) or PDF (.pdf) document")
        layout.addWidget(self.doc_path_edit, 0, 1)
        
        browse_doc_btn = QPushButton("Browse...")
        browse_doc_btn.clicked.connect(self.browse_document)
        layout.addWidget(browse_doc_btn, 0, 2)
        
        # Document info
        self.doc_info_label = QLabel("No document selected")
        self.doc_info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.doc_info_label, 1, 1, 1, 2)
        
        return group
    
    def create_sources_group(self) -> QGroupBox:
        """Create sources selection group."""
        group = QGroupBox("Source Files Selection")
        layout = QVBoxLayout(group)
        
        # Sources list
        sources_layout = QHBoxLayout()
        
        self.sources_text = QTextEdit()
        self.sources_text.setPlaceholderText(
            "Add source files (.bib, .json, .csv) or DOI lists (.txt)\n"
            "One file path per line or use the Browse button"
        )
        self.sources_text.setMaximumHeight(100)
        sources_layout.addWidget(self.sources_text)
        
        # Buttons
        buttons_layout = QVBoxLayout()
        
        browse_sources_btn = QPushButton("Browse...")
        browse_sources_btn.clicked.connect(self.browse_sources)
        buttons_layout.addWidget(browse_sources_btn)
        
        clear_sources_btn = QPushButton("Clear")
        clear_sources_btn.clicked.connect(self.clear_sources)
        buttons_layout.addWidget(clear_sources_btn)
        
        buttons_layout.addStretch()
        sources_layout.addLayout(buttons_layout)
        
        layout.addLayout(sources_layout)
        
        # Sources info
        self.sources_info_label = QLabel("No sources selected")
        self.sources_info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.sources_info_label)
        
        return group
    
    def create_settings_group(self) -> QGroupBox:
        """Create analysis settings group."""
        group = QGroupBox("Analysis Settings")
        layout = QGridLayout(group)
        
        # Top-k sentences
        layout.addWidget(QLabel("Top-K Weak Sentences:"), 0, 0)
        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setRange(1, 100)
        self.top_k_spinbox.setValue(10)
        self.top_k_spinbox.setToolTip("Number of weakest sentences to identify")
        layout.addWidget(self.top_k_spinbox, 0, 1)
        
        # Enable multiprocessing
        self.multiprocessing_checkbox = QCheckBox("Enable Multiprocessing")
        self.multiprocessing_checkbox.setChecked(True)
        self.multiprocessing_checkbox.setToolTip("Use multiple CPU cores for faster analysis")
        layout.addWidget(self.multiprocessing_checkbox, 1, 0)
        
        # Save results
        self.save_results_checkbox = QCheckBox("Auto-save Results")
        self.save_results_checkbox.setChecked(True)
        self.save_results_checkbox.setToolTip("Automatically save results after analysis")
        layout.addWidget(self.save_results_checkbox, 1, 1)
        
        return group
    
    def create_control_layout(self) -> QHBoxLayout:
        """Create control buttons layout."""
        layout = QHBoxLayout()
        
        # Validate files button
        validate_btn = QPushButton("Validate Files")
        validate_btn.clicked.connect(self.validate_files)
        layout.addWidget(validate_btn)
        
        layout.addStretch()
        
        # Run analysis button
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setEnabled(False)
        layout.addWidget(self.run_analysis_btn)
        
        return layout
    
    def browse_document(self):
        """Browse for document file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            "LaTeX Files (*.tex);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            self.doc_path_edit.setText(file_path)
            self.update_document_info()
            self.update_run_button_state()
    
    def browse_sources(self):
        """Browse for source files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Source Files",
            "",
            "BibTeX Files (*.bib);;JSON Files (*.json);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_paths:
            # Add to existing sources
            current_text = self.sources_text.toPlainText().strip()
            if current_text:
                current_text += "\n"
            
            new_sources = "\n".join(file_paths)
            self.sources_text.setPlainText(current_text + new_sources)
            
            self.update_sources_info()
            self.update_run_button_state()
    
    def clear_sources(self):
        """Clear all source files."""
        self.sources_text.clear()
        self.source_paths = []
        self.update_sources_info()
        self.update_run_button_state()
    
    def update_document_info(self):
        """Update document information display."""
        path = self.doc_path_edit.text().strip()
        if not path:
            self.doc_info_label.setText("No document selected")
            return
        
        try:
            # Validate file
            validator = InputValidator()
            validated_path = validator.validate_document_file(path)
            
            # Get file info
            file_size = validated_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            self.doc_info_label.setText(
                f"Selected: {validated_path.name} ({file_size_mb:.1f} MB)"
            )
            self.document_path = str(validated_path)
            
        except ValidationError as e:
            self.doc_info_label.setText(f"Invalid document: {e}")
            self.document_path = ""
    
    def update_sources_info(self):
        """Update sources information display."""
        text = self.sources_text.toPlainText().strip()
        if not text:
            self.sources_info_label.setText("No sources selected")
            self.source_paths = []
            return
        
        # Parse source paths
        paths = [line.strip() for line in text.split('\n') if line.strip()]
        valid_paths = []
        
        validator = InputValidator()
        for path in paths:
            try:
                validated_path = validator.validate_source_file(path)
                valid_paths.append(str(validated_path))
            except ValidationError:
                continue
        
        self.source_paths = valid_paths
        
        if valid_paths:
            self.sources_info_label.setText(f"Selected: {len(valid_paths)} valid source files")
        else:
            self.sources_info_label.setText("No valid source files found")
    
    def update_run_button_state(self):
        """Update the state of the run analysis button."""
        has_document = bool(self.document_path)
        has_sources = bool(self.source_paths)
        
        self.run_analysis_btn.setEnabled(has_document and has_sources)
        
        if not has_document and not has_sources:
            self.run_analysis_btn.setToolTip("Select a document and source files")
        elif not has_document:
            self.run_analysis_btn.setToolTip("Select a document file")
        elif not has_sources:
            self.run_analysis_btn.setToolTip("Select source files")
        else:
            self.run_analysis_btn.setToolTip("Run the analysis")
    
    def validate_files(self):
        """Validate selected files."""
        errors = []
        
        # Validate document
        if self.document_path:
            try:
                validator = InputValidator()
                validator.validate_document_file(self.document_path)
            except ValidationError as e:
                errors.append(f"Document: {e}")
        else:
            errors.append("Document: No document selected")
        
        # Validate sources
        if self.source_paths:
            validator = InputValidator()
            for path in self.source_paths:
                try:
                    validator.validate_source_file(path)
                except ValidationError as e:
                    errors.append(f"Source {Path(path).name}: {e}")
        else:
            errors.append("Sources: No source files selected")
        
        if errors:
            error_message = "File validation failed:\n\n" + "\n".join(errors)
            QMessageBox.warning(self, "Validation Errors", error_message)
        else:
            QMessageBox.information(self, "Validation Successful", "All files are valid!")
    
    def run_analysis(self):
        """Run the analysis."""
        if not self.document_path or not self.source_paths:
            QMessageBox.warning(self, "Missing Files", "Please select both a document and source files.")
            return
        
        # Emit signal to main window
        self.analysis_requested.emit(self.document_path, self.source_paths)
    
    def show_progress(self, visible: bool = True):
        """Show or hide progress bar."""
        self.progress_bar.setVisible(visible)
        if visible:
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
    
    def update_progress(self, value: int, message: str = ""):
        """Update progress bar."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)
        if message:
            self.progress_bar.setFormat(f"{message} (%p%)")
    
    def get_document_path(self) -> str:
        """Get the selected document path."""
        return self.document_path
    
    def get_source_paths(self) -> List[str]:
        """Get the selected source paths."""
        return self.source_paths
    
    def load_files(self, files: List[str]):
        """Load files into the interface."""
        documents = []
        sources = []
        
        for file_path in files:
            path = Path(file_path)
            if path.suffix.lower() in ['.tex', '.pdf']:
                documents.append(file_path)
            elif path.suffix.lower() in ['.bib', '.json', '.csv', '.txt']:
                sources.append(file_path)
        
        # Set first document if found
        if documents:
            self.doc_path_edit.setText(documents[0])
            self.update_document_info()
        
        # Add all sources
        if sources:
            current_text = self.sources_text.toPlainText().strip()
            if current_text:
                current_text += "\n"
            self.sources_text.setPlainText(current_text + "\n".join(sources))
            self.update_sources_info()
        
        self.update_run_button_state()
    
    def reset(self):
        """Reset the tab to initial state."""
        self.doc_path_edit.clear()
        self.sources_text.clear()
        self.doc_info_label.setText("No document selected")
        self.sources_info_label.setText("No sources selected")
        self.document_path = ""
        self.source_paths = []
        self.progress_bar.setVisible(False)
        self.update_run_button_state()