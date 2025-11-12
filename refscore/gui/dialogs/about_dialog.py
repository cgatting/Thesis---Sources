"""
About dialog for RefScore academic application.

This module provides the about dialog showing application information,
version details, and credits.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QTabWidget, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon

import logging
import datetime


log = logging.getLogger(__name__)


class AboutDialog(QDialog):
    """About dialog for RefScore application."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("About RefScore Academic")
        self.setFixedSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Logo placeholder
        logo_label = QLabel("📚")
        logo_label.setFont(QFont("Arial", 48))
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedSize(80, 80)
        logo_label.setStyleSheet("""
            QLabel {
                background-color: #4CAF50;
                color: white;
                border-radius: 40px;
            }
        """)
        header_layout.addWidget(logo_label)
        
        # Title and version
        title_layout = QVBoxLayout()
        
        title_label = QLabel("RefScore Academic")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #2E7D32;")
        title_layout.addWidget(title_label)
        
        version_label = QLabel("Version 1.0.0")
        version_label.setFont(QFont("Arial", 12))
        version_label.setStyleSheet("color: #666;")
        title_layout.addWidget(version_label)
        
        subtitle_label = QLabel("Academic Reference Analysis Tool")
        subtitle_label.setFont(QFont("Arial", 10))
        subtitle_label.setStyleSheet("color: #888;")
        title_layout.addWidget(subtitle_label)
        
        title_layout.addStretch()
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #ccc;")
        layout.addWidget(separator)
        
        # Tab widget for detailed information
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Description tab
        desc_widget = self.create_description_tab()
        tab_widget.addTab(desc_widget, "Description")
        
        # Features tab
        features_widget = self.create_features_tab()
        tab_widget.addTab(features_widget, "Features")
        
        # Credits tab
        credits_widget = self.create_credits_tab()
        tab_widget.addTab(credits_widget, "Credits")
        
        # License tab
        license_widget = self.create_license_tab()
        tab_widget.addTab(license_widget, "License")
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_description_tab(self) -> QWidget:
        """Create description tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        description_text = QTextEdit()
        description_text.setReadOnly(True)
        description_text.setPlainText(
            "RefScore Academic is a comprehensive tool for analyzing academic "
            "references against research documents. It provides sophisticated "
            "scoring algorithms to evaluate the relevance and coverage of "
            "cited sources within academic papers.\n\n"
            
            "The application supports multiple document formats including "
            "LaTeX (.tex) and PDF files, and can process various reference "
            "formats such as BibTeX, Zotero JSON, and CSV files.\n\n"
            
            "Key features include semantic analysis, entity recognition, "
            "number and unit matching, method identification, and "
            "comprehensive reporting with visualizations.\n\n"
            
            "This application was developed as part of a Bachelor of Science "
            "academic project, demonstrating modern software engineering "
            "principles and best practices."
        )
        
        layout.addWidget(description_text)
        return widget
    
    def create_features_tab(self) -> QWidget:
        """Create features tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        features_text = QTextEdit()
        features_text.setReadOnly(True)
        features_text.setPlainText(
            "Core Features:\n"
            "• Multi-format document support (LaTeX, PDF)\n"
            "• Multiple reference formats (BibTeX, JSON, CSV, DOI lists)\n"
            "• Six-dimensional scoring system\n"
            "• Semantic alignment using transformer models\n"
            "• Named entity recognition and matching\n"
            "• Number and unit extraction and matching\n"
            "• Academic method and metric identification\n"
            "• Publication recency scoring\n"
            "• Venue authority assessment\n\n"
            
            "User Interface:\n"
            "• Modern PyQt5-based graphical interface\n"
            "• Tabbed navigation for easy access\n"
            "• Real-time progress indicators\n"
            "• Interactive result tables and visualizations\n"
            "• Comprehensive export options (JSON, CSV, reports)\n"
            "• Configurable settings and preferences\n\n"
            
            "Technical Features:\n"
            "• Modular architecture with separation of concerns\n"
            "• Comprehensive error handling and validation\n"
            "• Extensive logging and debugging support\n"
            "• Cross-platform compatibility\n"
            "• Extensible plugin architecture\n"
            "• Comprehensive test suite\n\n"
            
            "Academic Features:\n"
            "• Section-wise coverage analysis\n"
            "• Weak sentence identification\n"
            "• Evidence-based scoring with explanations\n"
            "• Publication metadata extraction\n"
            "• Crossref API integration\n"
            "• Comprehensive reporting"
        )
        
        layout.addWidget(features_text)
        return widget
    
    def create_credits_tab(self) -> QWidget:
        """Create credits tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        credits_text = QTextEdit()
        credits_text.setReadOnly(True)
        credits_text.setPlainText(
            "Development Team:\n"
            "• Lead Developer: Academic Project Team\n"
            "• GUI Design: PyQt5 Development Team\n"
            "• NLP Integration: Hugging Face Transformers\n"
            "• Academic Consultation: University Faculty\n\n"
            
            "Third-Party Libraries:\n"
            "• PyQt5 - GUI Framework\n"
            "• sentence-transformers - Semantic Embeddings\n"
            "• spaCy - Named Entity Recognition\n"
            "• scikit-learn - Machine Learning\n"
            "• PyMuPDF - PDF Processing\n"
            "• bibtexparser - BibTeX Processing\n"
            "• requests - HTTP Client\n"
            "• matplotlib - Data Visualization\n"
            "• pandas - Data Processing\n\n"
            
            "Academic Acknowledgments:\n"
            "• Crossref API for publication metadata\n"
            "• Hugging Face for transformer models\n"
            "• Academic community for testing and feedback\n"
            "• Open source contributors\n\n"
            
            "Special Thanks:\n"
            "• University Computer Science Department\n"
            "• Academic Advisors and Mentors\n"
            "• Beta Testers and Users\n"
            "• Open Source Community"
        )
        
        layout.addWidget(credits_text)
        return widget
    
    def create_license_tab(self) -> QWidget:
        """Create license tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        license_text = QTextEdit()
        license_text.setReadOnly(True)
        license_text.setPlainText(
            "MIT License\n\n"
            
            f"Copyright (c) {datetime.datetime.now().year} RefScore Academic Project\n\n"
            
            "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            "of this software and associated documentation files (the \"Software\"), to deal\n"
            "in the Software without restriction, including without limitation the rights\n"
            "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
            "copies of the Software, and to permit persons to whom the Software is\n"
            "furnished to do so, subject to the following conditions:\n\n"
            
            "The above copyright notice and this permission notice shall be included in all\n"
            "copies or substantial portions of the Software.\n\n"
            
            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
            "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
            "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
            "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
            "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
            "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
            "SOFTWARE."
        )
        
        layout.addWidget(license_text)
        return widget