"""
Main window for RefScore academic application GUI.

This module provides the main application window with tabbed interface
for document analysis, source management, and results visualization.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QMenuBar, QStatusBar, QToolBar, QAction, QMessageBox,
    QFileDialog, QProgressDialog, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QFont

import logging
from pathlib import Path
from typing import List, Optional

from ..core.analyzer import RefScoreAnalyzer
from ..utils.config import Config
from ..utils.exceptions import RefScoreError, ValidationError, ProcessingError
from .widgets.analysis_tab import AnalysisTab
from .widgets.sources_tab import SourcesTab
from .widgets.results_tab import ResultsTab
from .widgets.settings_tab import SettingsTab
from .widgets.relevant_sources_tab import RelevantSourcesTab
from .dialogs.about_dialog import AboutDialog


log = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Worker thread for running analysis."""
    
    progress_updated = pyqtSignal(int, str)
    analysis_completed = pyqtSignal(object, list, list)
    analysis_failed = pyqtSignal(str)
    
    def __init__(self, analyzer: RefScoreAnalyzer, document_path: str, source_paths: List[str]):
        super().__init__()
        self.analyzer = analyzer
        self.document_path = document_path
        self.source_paths = source_paths
    
    def run(self):
        """Run the analysis in the worker thread."""
        try:
            self.progress_updated.emit(10, "Loading document...")
            document = self.analyzer.load_document(self.document_path)
            
            self.progress_updated.emit(30, "Loading sources...")
            sources = self.analyzer.load_sources(self.source_paths)
            
            self.progress_updated.emit(50, "Computing scores...")
            scores = self.analyzer.compute_scores(document, sources)
            
            self.progress_updated.emit(90, "Generating reports...")
            
            self.analysis_completed.emit(document, sources, scores)
            
        except Exception as e:
            self.analysis_failed.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.analyzer = RefScoreAnalyzer(config)
        self.current_document = None
        self.current_sources = []
        self.current_scores = []
        
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_status_bar()
        
        self.setWindowTitle("RefScore Academic - Source Document Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        log.info("Main window initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.analysis_tab = AnalysisTab(self)
        self.sources_tab = SourcesTab(self)
        self.results_tab = ResultsTab(self)
        self.settings_tab = SettingsTab(self.config, self)
        self.relevant_tab = RelevantSourcesTab(self)
        
        # Add tabs
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        self.tab_widget.addTab(self.sources_tab, "Sources")
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.relevant_tab, "Relevant Sources")
        
        # Connect tab signals
        self.analysis_tab.analysis_requested.connect(self.run_analysis)
        self.settings_tab.settings_changed.connect(self.update_settings)
    
    def setup_menu(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # New analysis action
        new_action = QAction("New Analysis", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_analysis)
        file_menu.addAction(new_action)
        
        # Open action
        open_action = QAction("Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Save results action
        save_action = QAction("Save Results", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        # Validate files action
        validate_action = QAction("Validate Files", self)
        validate_action.triggered.connect(self.validate_files)
        tools_menu.addAction(validate_action)
        
        # Settings action
        settings_action = QAction("Preferences", self)
        settings_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.settings_tab))
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Documentation action
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
    
    def setup_toolbar(self):
        """Set up the toolbar."""
        toolbar = self.addToolBar("Main")
        
        # New analysis button
        new_action = QAction("New", self)
        new_action.triggered.connect(self.new_analysis)
        toolbar.addAction(new_action)
        
        # Run analysis button
        run_action = QAction("Run Analysis", self)
        run_action.triggered.connect(lambda: self.run_analysis())
        toolbar.addAction(run_action)
        
        toolbar.addSeparator()
        
        # Save results button
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)
    
    def setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Add permanent widgets to status bar
        self.progress_label = self.status_bar.addPermanentWidget(
            self.create_progress_label()
        )
    
    def create_progress_label(self):
        """Create a progress label for the status bar."""
        from PyQt5.QtWidgets import QLabel
        label = QLabel("Ready")
        label.setMinimumWidth(200)
        return label
    
    def new_analysis(self):
        """Start a new analysis."""
        self.current_document = None
        self.current_sources = []
        self.current_scores = []
        
        self.analysis_tab.reset()
        self.sources_tab.reset()
        self.results_tab.reset()
        
        self.status_bar.showMessage("Ready for new analysis")
        self.tab_widget.setCurrentWidget(self.analysis_tab)
    
    def open_file(self):
        """Open a file dialog to load analysis files."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("All Supported (*.tex *.pdf *.bib *.json *.csv);;Documents (*.tex *.pdf);;Sources (*.bib *.json *.csv)")
        
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            self.analysis_tab.load_files(files)
    
    def run_analysis(self, document_path: str = None, source_paths: List[str] = None):
        """Run the RefScore analysis."""
        if not document_path:
            document_path = self.analysis_tab.get_document_path()
        
        if not source_paths:
            source_paths = self.sources_tab.get_source_paths()
        
        if not document_path or not source_paths:
            QMessageBox.warning(self, "Missing Files", "Please select both a document and source files.")
            return
        
        # Create progress dialog
        progress = QProgressDialog("Running analysis...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setValue(0)
        
        # Create and start worker thread
        self.worker = AnalysisWorker(self.analyzer, document_path, source_paths)
        
        def update_progress(value, message):
            progress.setValue(value)
            progress.setLabelText(message)
        
        def analysis_completed(document, sources, scores):
            progress.setValue(100)
            self.current_document = document
            self.current_sources = sources
            self.current_scores = scores
            
            # Update tabs
            self.results_tab.display_results(document, sources, scores)
            self.tab_widget.setCurrentWidget(self.results_tab)
            
            self.status_bar.showMessage(f"Analysis completed: {len(scores)} sources scored")
        
        def analysis_failed(error_message):
            progress.cancel()
            QMessageBox.critical(self, "Analysis Failed", f"Analysis failed: {error_message}")
            self.status_bar.showMessage("Analysis failed")
        
        self.worker.progress_updated.connect(update_progress)
        self.worker.analysis_completed.connect(analysis_completed)
        self.worker.analysis_failed.connect(analysis_failed)
        
        self.worker.start()
    
    def save_results(self):
        """Save analysis results."""
        if not self.current_scores:
            QMessageBox.warning(self, "No Results", "No analysis results to save.")
            return
        
        directory = QFileDialog.getExistingDirectory(self, "Save Results")
        if directory:
            try:
                reports = self.analyzer.generate_report(
                    self.current_document,
                    self.current_sources,
                    self.current_scores,
                    directory
                )
                
                QMessageBox.information(
                    self, "Results Saved", 
                    f"Results saved successfully to:\n{directory}\n\n"
                    f"Files created:\n" + "\n".join(f"- {name}: {path}" 
                    for name, path in reports.items())
                )
                
                self.status_bar.showMessage(f"Results saved to {directory}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Failed to save results: {e}")
    
    def validate_files(self):
        """Validate selected files."""
        document_path = self.analysis_tab.get_document_path()
        source_paths = self.sources_tab.get_source_paths()
        
        if not document_path and not source_paths:
            QMessageBox.warning(self, "No Files", "No files selected for validation.")
            return
        
        # Validate document
        document_valid = True
        if document_path:
            try:
                document = self.analyzer.load_document(document_path)
                QMessageBox.information(
                    self, "Document Valid", 
                    f"Document is valid:\n{document_path}\n\n"
                    f"Sentences: {len(document.sentences)}\n"
                    f"Sections: {len(document.sections)}"
                )
            except Exception as e:
                document_valid = False
                QMessageBox.critical(self, "Document Invalid", f"Document validation failed: {e}")
        
        # Validate sources
        if source_paths:
            try:
                sources = self.analyzer.load_sources(source_paths)
                QMessageBox.information(
                    self, "Sources Valid",
                    f"Sources are valid:\n{len(source_paths)} files\n\n"
                    f"Total sources: {len(sources)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Sources Invalid", f"Source validation failed: {e}")
    
    def update_settings(self):
        """Update application settings."""
        # Recreate analyzer with new settings
        self.analyzer = RefScoreAnalyzer(self.config)
        self.status_bar.showMessage("Settings updated")
    
    def show_about(self):
        """Show about dialog."""
        about_dialog = AboutDialog(self)
        about_dialog.exec_()
    
    def show_documentation(self):
        """Show documentation."""
        import webbrowser
        webbrowser.open("https://github.com/yourusername/refscore-academic")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Save settings before closing
        try:
            self.config.save_configuration()
        except Exception as e:
            log.warning(f"Failed to save configuration on exit: {e}")
        
        event.accept()
