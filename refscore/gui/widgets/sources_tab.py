"""
Sources tab widget for RefScore academic application.

This module provides the sources management interface where users can
view, edit, and manage academic source files and their metadata.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSplitter, QFrame, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

import logging
from pathlib import Path
from typing import List, Dict, Any

from ...models.source import Source
from ...utils.validators import InputValidator
from ...utils.exceptions import ValidationError


log = logging.getLogger(__name__)


class SourcesTab(QWidget):
    """Sources tab widget for managing academic sources."""
    
    sources_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sources = []
        self.source_files = []
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Source files
        left_panel = self.create_files_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Source details
        right_panel = self.create_details_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 500])
    
    def create_files_panel(self) -> QGroupBox:
        """Create source files panel."""
        group = QGroupBox("Source Files")
        layout = QVBoxLayout(group)
        
        # File list
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(3)
        self.files_table.setHorizontalHeaderLabels(["File", "Sources", "Status"])
        self.files_table.horizontalHeader().setStretchLastSection(True)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.files_table.itemSelectionChanged.connect(self.on_file_selected)
        layout.addWidget(self.files_table)
        
        # File controls
        controls_layout = QHBoxLayout()
        
        add_file_btn = QPushButton("Add File...")
        add_file_btn.clicked.connect(self.add_source_file)
        controls_layout.addWidget(add_file_btn)
        
        remove_file_btn = QPushButton("Remove File")
        remove_file_btn.clicked.connect(self.remove_source_file)
        controls_layout.addWidget(remove_file_btn)
        
        reload_btn = QPushButton("Reload All")
        reload_btn.clicked.connect(self.reload_sources)
        controls_layout.addWidget(reload_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        return group
    
    def create_details_panel(self) -> QGroupBox:
        """Create source details panel."""
        group = QGroupBox("Source Details")
        layout = QVBoxLayout(group)
        
        # Source table
        self.sources_table = QTableWidget()
        self.sources_table.setColumnCount(6)
        self.sources_table.setHorizontalHeaderLabels([
            "ID", "Title", "Authors", "Year", "Venue", "DOI"
        ])
        self.sources_table.horizontalHeader().setStretchLastSection(True)
        self.sources_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sources_table.setAlternatingRowColors(True)
        self.sources_table.itemSelectionChanged.connect(self.on_source_selected)
        layout.addWidget(self.sources_table)
        
        # Source details
        details_frame = self.create_source_details_frame()
        layout.addWidget(details_frame)
        
        return group
    
    def create_source_details_frame(self) -> QFrame:
        """Create detailed source information frame."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        
        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit()
        self.title_edit.setReadOnly(True)
        title_layout.addWidget(self.title_edit)
        layout.addLayout(title_layout)
        
        # Authors
        authors_layout = QHBoxLayout()
        authors_layout.addWidget(QLabel("Authors:"))
        self.authors_edit = QLineEdit()
        self.authors_edit.setReadOnly(True)
        authors_layout.addWidget(self.authors_edit)
        layout.addLayout(authors_layout)
        
        # Year and Venue
        year_venue_layout = QHBoxLayout()
        
        year_layout = QHBoxLayout()
        year_layout.addWidget(QLabel("Year:"))
        self.year_edit = QLineEdit()
        self.year_edit.setReadOnly(True)
        year_layout.addWidget(self.year_edit)
        year_layout.addStretch()
        
        venue_layout = QHBoxLayout()
        venue_layout.addWidget(QLabel("Venue:"))
        self.venue_edit = QLineEdit()
        self.venue_edit.setReadOnly(True)
        venue_layout.addWidget(self.venue_edit)
        venue_layout.addStretch()
        
        year_venue_layout.addLayout(year_layout)
        year_venue_layout.addLayout(venue_layout)
        layout.addLayout(year_venue_layout)
        
        # DOI
        doi_layout = QHBoxLayout()
        doi_layout.addWidget(QLabel("DOI:"))
        self.doi_edit = QLineEdit()
        self.doi_edit.setReadOnly(True)
        doi_layout.addWidget(self.doi_edit)
        layout.addLayout(doi_layout)
        
        # Abstract
        layout.addWidget(QLabel("Abstract:"))
        self.abstract_edit = QTextEdit()
        self.abstract_edit.setReadOnly(True)
        self.abstract_edit.setMaximumHeight(100)
        layout.addWidget(self.abstract_edit)
        
        return frame
    
    def add_source_file(self):
        """Add source files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Source Files",
            "",
            "BibTeX Files (*.bib);;JSON Files (*.json);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_paths:
            # Append new files to existing ones, avoiding duplicates
            current_set = set(self.source_files)
            new_files = [f for f in file_paths if f not in current_set]
            if new_files:
                self.load_source_files(self.source_files + new_files)
    
    def remove_source_file(self):
        """Remove selected source file."""
        selected_rows = self.files_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Remove selected files (in reverse order to maintain indices)
        for row in sorted([r.row() for r in selected_rows], reverse=True):
            if row < len(self.source_files):
                self.source_files.pop(row)
        
        self.update_files_table()
        self.reload_sources()
    
    def load_source_files(self, file_paths: List[str]):
        """Load source files and extract sources."""
        self.source_files = file_paths
        self.update_files_table()
        self.reload_sources()
    
    def reload_sources(self):
        """Reload all sources from files."""
        if not self.source_files:
            self.sources = []
            self.update_sources_table()
            return
        
        try:
            from ...core.source_loader import SourceLoader
            
            loader = SourceLoader()
            all_sources = []
            
            for file_path in self.source_files:
                try:
                    sources = loader.load_sources(file_path)
                    all_sources.extend(sources)
                except Exception as e:
                    log.warning(f"Failed to load sources from {file_path}: {e}")
            
            # Remove duplicates
            seen = set()
            unique_sources = []
            for source in all_sources:
                key = source.doi.lower() if source.doi else source.title.lower()
                if key and key not in seen:
                    seen.add(key)
                    unique_sources.append(source)
            
            self.sources = unique_sources
            self.update_sources_table()
            
            log.info(f"Loaded {len(unique_sources)} unique sources from {len(self.source_files)} files")
            
        except Exception as e:
            log.error(f"Failed to reload sources: {e}")
            self.sources = []
            self.update_sources_table()
    
    def update_files_table(self):
        """Update the files table."""
        self.files_table.setRowCount(len(self.source_files))
        
        for i, file_path in enumerate(self.source_files):
            path = Path(file_path)
            
            # File name
            self.files_table.setItem(i, 0, QTableWidgetItem(path.name))
            
            # File size
            try:
                file_size = path.stat().st_size
                size_str = f"{file_size / 1024:.1f} KB"
            except:
                size_str = "Unknown"
            
            self.files_table.setItem(i, 1, QTableWidgetItem(size_str))
            
            # Status
            try:
                validator = InputValidator()
                validator.validate_source_file(file_path)
                status = "Valid"
            except ValidationError:
                status = "Invalid"
            
            self.files_table.setItem(i, 2, QTableWidgetItem(status))
            
            # Store full path in user data
            self.files_table.item(i, 0).setData(Qt.UserRole, file_path)
    
    def update_sources_table(self):
        """Update the sources table."""
        self.sources_table.setRowCount(len(self.sources))
        
        for i, source in enumerate(self.sources):
            # Source ID
            self.sources_table.setItem(i, 0, QTableWidgetItem(source.source_id))
            
            # Title
            title_item = QTableWidgetItem(source.title[:80])
            title_item.setToolTip(source.title)
            self.sources_table.setItem(i, 1, title_item)
            
            # Authors
            authors_str = source.author_string
            authors_item = QTableWidgetItem(authors_str[:50])
            authors_item.setToolTip(authors_str)
            self.sources_table.setItem(i, 2, authors_item)
            
            # Year
            year_str = str(source.year) if source.year else "Unknown"
            self.sources_table.setItem(i, 3, QTableWidgetItem(year_str))
            
            # Venue
            venue_str = source.venue[:40] if source.venue else "Unknown"
            venue_item = QTableWidgetItem(venue_str)
            venue_item.setToolTip(source.venue)
            self.sources_table.setItem(i, 4, venue_item)
            
            # DOI
            doi_str = source.doi[:30] if source.doi else ""
            doi_item = QTableWidgetItem(doi_str)
            doi_item.setToolTip(source.doi)
            self.sources_table.setItem(i, 5, doi_item)
            
            # Store source object in user data
            self.sources_table.item(i, 0).setData(Qt.UserRole, source)
    
    def on_file_selected(self):
        """Handle file selection."""
        selected_rows = self.files_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Could implement filtering of sources by file if needed
        pass
    
    def on_source_selected(self):
        """Handle source selection."""
        selected_rows = self.sources_table.selectionModel().selectedRows()
        if not selected_rows:
            self.clear_source_details()
            return
        
        row = selected_rows[0].row()
        if row < len(self.sources):
            source = self.sources[row]
            self.display_source_details(source)
    
    def display_source_details(self, source: Source):
        """Display detailed source information."""
        self.title_edit.setText(source.title)
        self.authors_edit.setText(source.author_string)
        self.year_edit.setText(str(source.year) if source.year else "Unknown")
        self.venue_edit.setText(source.venue or "Unknown")
        self.doi_edit.setText(source.doi or "")
        self.abstract_edit.setPlainText(source.abstract or "No abstract available")
    
    def clear_source_details(self):
        """Clear source details display."""
        self.title_edit.clear()
        self.authors_edit.clear()
        self.year_edit.clear()
        self.venue_edit.clear()
        self.doi_edit.clear()
        self.abstract_edit.clear()
    
    def get_source_paths(self) -> List[str]:
        """Get the list of source file paths."""
        return self.source_files.copy()
    
    def reset(self):
        """Reset the tab to initial state."""
        self.source_files = []
        self.sources = []
        self.update_files_table()
        self.update_sources_table()
        self.clear_source_details()