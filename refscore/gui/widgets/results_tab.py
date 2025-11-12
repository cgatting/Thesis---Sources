"""
Results tab widget for RefScore academic application.

This module provides the results visualization interface where users can
view analysis results, scores, coverage reports, and weak sentences.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QTextEdit, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView, QAbstractItemView, QSplitter, QFrame, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

import logging
from pathlib import Path
from typing import List, Dict, Any

from ...models.document import Document
from ...models.source import Source
from ...models.scoring import SourceScore


log = logging.getLogger(__name__)


class ResultsTab(QWidget):
    """Results tab widget for displaying analysis results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.document = None
        self.sources = []
        self.scores = []
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Summary section
        summary_group = self.create_summary_group()
        layout.addWidget(summary_group)
        
        # Results tabs
        self.results_tab_widget = QTabWidget()
        layout.addWidget(self.results_tab_widget)
        
        # Create result tabs
        self.sources_tab = self.create_sources_results_tab()
        self.coverage_tab = self.create_coverage_tab()
        self.weak_sentences_tab = self.create_weak_sentences_tab()
        self.quality_tab = self.create_quality_tab()
        
        self.results_tab_widget.addTab(self.sources_tab, "Source Rankings")
        self.results_tab_widget.addTab(self.coverage_tab, "Section Coverage")
        self.results_tab_widget.addTab(self.weak_sentences_tab, "Weak Sentences")
        self.results_tab_widget.addTab(self.quality_tab, "Quality Assessment")
        
        # Export controls
        export_layout = self.create_export_layout()
        layout.addLayout(export_layout)
    
    def create_summary_group(self) -> QGroupBox:
        """Create summary information group."""
        group = QGroupBox("Analysis Summary")
        layout = QHBoxLayout(group)
        
        # Document info
        doc_frame = QFrame()
        doc_frame.setFrameStyle(QFrame.StyledPanel)
        doc_layout = QVBoxLayout(doc_frame)
        
        doc_layout.addWidget(QLabel("<b>Document:</b>"))
        self.doc_label = QLabel("No document loaded")
        self.doc_label.setWordWrap(True)
        doc_layout.addWidget(self.doc_label)
        
        self.doc_stats_label = QLabel("No statistics available")
        doc_layout.addWidget(self.doc_stats_label)
        doc_layout.addStretch()
        
        layout.addWidget(doc_frame)
        
        # Analysis info
        analysis_frame = QFrame()
        analysis_frame.setFrameStyle(QFrame.StyledPanel)
        analysis_layout = QVBoxLayout(analysis_frame)
        
        analysis_layout.addWidget(QLabel("<b>Analysis:</b>"))
        self.analysis_stats_label = QLabel("No analysis performed")
        analysis_layout.addWidget(self.analysis_stats_label)
        
        self.top_score_label = QLabel("No scores computed")
        analysis_layout.addWidget(self.top_score_label)
        analysis_layout.addStretch()
        
        layout.addWidget(analysis_frame)
        
        # Progress indicator
        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)
        
        return group
    
    def create_sources_results_tab(self) -> QWidget:
        """Create sources results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sources table
        self.sources_table = QTableWidget()
        self.sources_table.setColumnCount(7)
        self.sources_table.setHorizontalHeaderLabels([
            "Rank", "Score", "Title", "Authors", "Year", "Venue", "DOI"
        ])
        self.sources_table.horizontalHeader().setStretchLastSection(True)
        self.sources_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sources_table.setAlternatingRowColors(True)
        self.sources_table.itemSelectionChanged.connect(self.on_source_selected)
        layout.addWidget(self.sources_table)
        
        # Source details
        details_group = QGroupBox("Source Details")
        details_layout = QVBoxLayout(details_group)
        
        self.source_details_text = QTextEdit()
        self.source_details_text.setReadOnly(True)
        self.source_details_text.setMaximumHeight(150)
        details_layout.addWidget(self.source_details_text)
        
        layout.addWidget(details_group)
        
        return widget
    
    def create_coverage_tab(self) -> QWidget:
        """Create coverage analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Coverage overview
        overview_group = QGroupBox("Coverage Overview")
        overview_layout = QVBoxLayout(overview_group)
        
        self.coverage_overview_text = QTextEdit()
        self.coverage_overview_text.setReadOnly(True)
        self.coverage_overview_text.setMaximumHeight(100)
        overview_layout.addWidget(self.coverage_overview_text)
        
        layout.addWidget(overview_group)
        
        # Section coverage table
        self.coverage_table = QTableWidget()
        self.coverage_table.setColumnCount(5)
        self.coverage_table.setHorizontalHeaderLabels([
            "Section", "Sentences", "Avg Support", "Min Support", "Max Support"
        ])
        self.coverage_table.horizontalHeader().setStretchLastSection(True)
        self.coverage_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.coverage_table.setAlternatingRowColors(True)
        layout.addWidget(self.coverage_table)
        
        return widget
    
    def create_weak_sentences_tab(self) -> QWidget:
        """Create weak sentences tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Weak sentences table
        self.weak_sentences_table = QTableWidget()
        self.weak_sentences_table.setColumnCount(5)
        self.weak_sentences_table.setHorizontalHeaderLabels([
            "Rank", "Section", "Support", "Sentence", "Best Source"
        ])
        self.weak_sentences_table.horizontalHeader().setStretchLastSection(True)
        self.weak_sentences_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.weak_sentences_table.setAlternatingRowColors(True)
        self.weak_sentences_table.itemSelectionChanged.connect(self.on_weak_sentence_selected)
        layout.addWidget(self.weak_sentences_table)
        
        # Sentence details
        details_group = QGroupBox("Sentence Details")
        details_layout = QVBoxLayout(details_group)
        
        self.sentence_details_text = QTextEdit()
        self.sentence_details_text.setReadOnly(True)
        self.sentence_details_text.setMaximumHeight(100)
        details_layout.addWidget(self.sentence_details_text)
        
        layout.addWidget(details_group)
        
        return widget

    def create_quality_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.quality_overview_text = QTextEdit()
        self.quality_overview_text.setReadOnly(True)
        self.quality_overview_text.setMaximumHeight(150)
        layout.addWidget(self.quality_overview_text)
        self.quality_table = QTableWidget()
        self.quality_table.setColumnCount(3)
        self.quality_table.setHorizontalHeaderLabels(["Section", "Rating", "Summary"])
        self.quality_table.horizontalHeader().setStretchLastSection(True)
        self.quality_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.quality_table.setAlternatingRowColors(True)
        layout.addWidget(self.quality_table)
        return widget
    
    def create_export_layout(self) -> QHBoxLayout:
        """Create export controls layout."""
        layout = QHBoxLayout()
        
        layout.addWidget(QLabel("Export Results:"))
        
        export_json_btn = QPushButton("JSON")
        export_json_btn.clicked.connect(self.export_json)
        layout.addWidget(export_json_btn)
        
        export_csv_btn = QPushButton("CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        layout.addWidget(export_csv_btn)
        
        export_report_btn = QPushButton("Report")
        export_report_btn.clicked.connect(self.export_report)
        layout.addWidget(export_report_btn)
        
        layout.addStretch()
        
        return layout
    
    def display_results(self, document: Document, sources: List[Source], scores: List[SourceScore]):
        """Display analysis results."""
        self.document = document
        self.sources = sources
        self.scores = scores
        
        self.update_summary()
        self.update_sources_table()
        self.update_coverage_table()
        self.update_weak_sentences_table()
        self.update_quality_assessment()
    
    def update_summary(self):
        """Update summary information."""
        if not self.document or not self.scores:
            self.doc_label.setText("No document loaded")
            self.doc_stats_label.setText("No statistics available")
            self.analysis_stats_label.setText("No analysis performed")
            self.top_score_label.setText("No scores computed")
            return
        
        # Document info
        doc_name = Path(self.document.meta.get("path", "Unknown")).name
        self.doc_label.setText(f"<b>{doc_name}</b>")
        
        doc_stats = self.document.get_section_stats()
        self.doc_stats_label.setText(
            f"Sentences: {len(self.document.sentences)} | "
            f"Sections: {len(self.document.sections)}"
        )
        
        # Analysis info
        self.analysis_stats_label.setText(
            f"Sources analyzed: {len(self.sources)} | "
            f"Sources scored: {len(self.scores)}"
        )
        
        if self.scores:
            top_score = self.scores[0].refscore
            avg_score = sum(score.refscore for score in self.scores) / len(self.scores)
            self.top_score_label.setText(
                f"Top score: {top_score:.4f} | "
                f"Average score: {avg_score:.4f}"
            )
    
    def update_sources_table(self):
        """Update sources results table."""
        self.sources_table.setRowCount(len(self.scores))
        
        for i, score in enumerate(self.scores):
            source = score.source
            
            # Rank
            rank_item = QTableWidgetItem(str(i + 1))
            rank_item.setTextAlignment(Qt.AlignCenter)
            self.sources_table.setItem(i, 0, rank_item)
            
            # Score
            score_item = QTableWidgetItem(f"{score.refscore:.4f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            
            # Color code based on score
            if score.refscore >= 0.8:
                score_item.setBackground(QColor(144, 238, 144))  # Light green
            elif score.refscore >= 0.5:
                score_item.setBackground(QColor(255, 255, 224))  # Light yellow
            else:
                score_item.setBackground(QColor(255, 182, 193))  # Light pink
            
            self.sources_table.setItem(i, 1, score_item)
            
            # Title
            title_item = QTableWidgetItem(source.title[:60])
            title_item.setToolTip(source.title)
            self.sources_table.setItem(i, 2, title_item)
            
            # Authors
            authors_item = QTableWidgetItem(source.author_string[:30])
            authors_item.setToolTip(source.author_string)
            self.sources_table.setItem(i, 3, authors_item)
            
            # Year
            year_str = str(source.year) if source.year else "Unknown"
            year_item = QTableWidgetItem(year_str)
            year_item.setTextAlignment(Qt.AlignCenter)
            self.sources_table.setItem(i, 4, year_item)
            
            # Venue
            venue_item = QTableWidgetItem(source.venue[:20] if source.venue else "Unknown")
            venue_item.setToolTip(source.venue)
            self.sources_table.setItem(i, 5, venue_item)
            
            # DOI
            doi_item = QTableWidgetItem(source.doi[:20] if source.doi else "")
            doi_item.setToolTip(source.doi)
            self.sources_table.setItem(i, 6, doi_item)
            
            # Store score object for later use
            rank_item.setData(Qt.UserRole, score)
    
    def update_coverage_table(self):
        """Update coverage analysis table."""
        if not self.document or not self.scores:
            self.coverage_overview_text.setPlainText("No coverage data available")
            self.coverage_table.setRowCount(0)
            return
        
        from ...core.scoring import ScoringEngine
        
        engine = ScoringEngine()
        coverage = engine.section_coverage_report(self.document, self.scores)
        
        # Overview
        overall_coverage = coverage.get("overall_coverage", 0.0)
        self.coverage_overview_text.setPlainText(
            f"Overall Coverage Score: {overall_coverage:.3f}\n"
            f"Total Sections: {len(coverage.get('sections', []))}\n"
            f"Sections with Sources: {len(coverage.get('section_stats', {}))}"
        )
        
        # Coverage table
        section_stats = coverage.get("section_stats", {})
        self.coverage_table.setRowCount(len(section_stats))
        
        for i, (section, stats) in enumerate(section_stats.items()):
            # Section name
            section_item = QTableWidgetItem(section)
            self.coverage_table.setItem(i, 0, section_item)
            
            # Sentence count
            count_item = QTableWidgetItem(str(stats["count"]))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.coverage_table.setItem(i, 1, count_item)
            
            # Average support
            avg_item = QTableWidgetItem(f"{stats['avg_support']:.3f}")
            avg_item.setTextAlignment(Qt.AlignCenter)
            
            # Color code based on support
            avg_support = stats["avg_support"]
            if avg_support >= 0.6:
                avg_item.setBackground(QColor(144, 238, 144))  # Light green
            elif avg_support >= 0.3:
                avg_item.setBackground(QColor(255, 255, 224))  # Light yellow
            else:
                avg_item.setBackground(QColor(255, 182, 193))  # Light pink
            
            self.coverage_table.setItem(i, 2, avg_item)
            
            # Min support
            min_item = QTableWidgetItem(f"{stats['min_support']:.3f}")
            min_item.setTextAlignment(Qt.AlignCenter)
            self.coverage_table.setItem(i, 3, min_item)
            
            # Max support
            max_item = QTableWidgetItem(f"{stats['max_support']:.3f}")
            max_item.setTextAlignment(Qt.AlignCenter)
            self.coverage_table.setItem(i, 4, max_item)
    
    def update_weak_sentences_table(self):
        """Update weak sentences table."""
        if not self.document or not self.scores:
            self.weak_sentences_table.setRowCount(0)
            self.sentence_details_text.setPlainText("No weak sentences identified")
        return

    def update_quality_assessment(self):
        if not self.document or not self.scores:
            self.quality_overview_text.setPlainText("No assessment available")
            self.quality_table.setRowCount(0)
            return
        from ...core.analyzer import RefScoreAnalyzer
        analyzer = RefScoreAnalyzer()
        assessment = analyzer.get_quality_assessment(self.document, self.scores)
        overall = assessment.get("overall", {})
        text = assessment.get("text", "")
        self.quality_overview_text.setPlainText(text)
        sections = assessment.get("sections", [])
        self.quality_table.setRowCount(len(sections))
        from PyQt5.QtGui import QColor
        for i, s in enumerate(sections):
            sec_item = QTableWidgetItem(s.get("section", ""))
            self.quality_table.setItem(i, 0, sec_item)
            rating = s.get("rating", "")
            rate_item = QTableWidgetItem(rating)
            rate_item.setTextAlignment(Qt.AlignCenter)
            avg = float(s.get("avg_support", 0.0))
            if avg >= 0.6:
                rate_item.setBackground(QColor(144, 238, 144))
            elif avg >= 0.3:
                rate_item.setBackground(QColor(255, 255, 224))
            else:
                rate_item.setBackground(QColor(255, 182, 193))
            self.quality_table.setItem(i, 1, rate_item)
            summary = []
            if s.get("strengths"):
                ex = s["strengths"][0]
                summary.append(f"Strong: {ex['text'][:80]}")
            if s.get("weaknesses"):
                exw = s["weaknesses"][0]
                summary.append(f"Weak: {exw['text'][:80]}")
            sum_item = QTableWidgetItem("; ".join(summary))
            self.quality_table.setItem(i, 2, sum_item)
        
        from ...core.scoring import ScoringEngine
        
        engine = ScoringEngine()
        weak_sentences = engine.weakest_sentences(self.document, self.scores, top_k=20)
        
        self.weak_sentences_table.setRowCount(len(weak_sentences))
        
        for i, weak_sentence in enumerate(weak_sentences):
            # Rank
            rank_item = QTableWidgetItem(str(i + 1))
            rank_item.setTextAlignment(Qt.AlignCenter)
            self.weak_sentences_table.setItem(i, 0, rank_item)
            
            # Section
            section_item = QTableWidgetItem(weak_sentence["section"])
            self.weak_sentences_table.setItem(i, 1, section_item)
            
            # Support score
            support_item = QTableWidgetItem(f"{weak_sentence['support']:.3f}")
            support_item.setTextAlignment(Qt.AlignCenter)
            
            # Color code based on support
            support = weak_sentence["support"]
            if support >= 0.5:
                support_item.setBackground(QColor(255, 255, 224))  # Light yellow
            else:
                support_item.setBackground(QColor(255, 182, 193))  # Light pink
            
            self.weak_sentences_table.setItem(i, 2, support_item)
            
            # Sentence text
            sentence_text = weak_sentence["sentence"][:80] + "..." if len(weak_sentence["sentence"]) > 80 else weak_sentence["sentence"]
            sentence_item = QTableWidgetItem(sentence_text)
            sentence_item.setToolTip(weak_sentence["sentence"])
            self.weak_sentences_table.setItem(i, 3, sentence_item)
            
            # Best source
            best_source = weak_sentence.get("best_source", "Unknown")
            source_item = QTableWidgetItem(best_source)
            self.weak_sentences_table.setItem(i, 4, source_item)
            
            # Store weak sentence data
            rank_item.setData(Qt.UserRole, weak_sentence)
    
    def on_source_selected(self):
        """Handle source selection."""
        selected_rows = self.sources_table.selectionModel().selectedRows()
        if not selected_rows:
            self.source_details_text.setPlainText("Select a source to view details")
            return
        
        row = selected_rows[0].row()
        if row < len(self.scores):
            score = self.scores[row]
            source = score.source
            
            details = f"Score: {score.refscore:.4f}\n"
            details += f"Evidence Count: {score.evidence_count}\n"
            details += f"Average Evidence Score: {score.average_evidence_score:.4f}\n"
            
            top_reasons = score.get_top_reasons(3)
            if top_reasons:
                details += f"Top Reasons: {', '.join(top_reasons)}"
            
            self.source_details_text.setPlainText(details)
    
    def on_weak_sentence_selected(self):
        """Handle weak sentence selection."""
        selected_rows = self.weak_sentences_table.selectionModel().selectedRows()
        if not selected_rows:
            self.sentence_details_text.setPlainText("Select a sentence to view details")
            return
        
        row = selected_rows[0].row()
        weak_sentence_item = self.weak_sentences_table.item(row, 0)
        if weak_sentence_item:
            weak_sentence = weak_sentence_item.data(Qt.UserRole)
            
            details = f"Support Score: {weak_sentence['support']:.3f}\n"
            details += f"Section: {weak_sentence['section']}\n"
            
            reasons = weak_sentence.get("reasons", [])
            if reasons:
                details += f"Reasons: {', '.join(reasons)}"
            else:
                details += "No specific reasons identified"
            
            self.sentence_details_text.setPlainText(details)
    
    def export_json(self):
        """Export results as JSON."""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", "refscore_results.json", "JSON Files (*.json)"
        )
        
        if file_path and self.scores:
            try:
                import json
                
                results = {
                    "document": {
                        "path": self.document.meta.get("path", "unknown"),
                        "sentences": len(self.document.sentences),
                        "sections": len(self.document.sections)
                    },
                    "sources": [score.to_dict() for score in self.scores]
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                self.progress_label.setText(f"Results exported to {file_path}")
                
            except Exception as e:
                self.progress_label.setText(f"Export failed: {e}")
    
    def export_csv(self):
        """Export results as CSV."""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "refscore_results.csv", "CSV Files (*.csv)"
        )
        
        if file_path and self.scores:
            try:
                import csv
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        "Rank", "Score", "Title", "Authors", "Year", 
                        "Venue", "DOI", "Evidence Count", "Avg Evidence Score"
                    ])
                    
                    # Data rows
                    for i, score in enumerate(self.scores):
                        source = score.source
                        writer.writerow([
                            i + 1,
                            f"{score.refscore:.4f}",
                            source.title,
                            source.author_string,
                            source.year or "Unknown",
                            source.venue or "Unknown",
                            source.doi or "",
                            score.evidence_count,
                            f"{score.average_evidence_score:.4f}"
                        ])
                
                self.progress_label.setText(f"Results exported to {file_path}")
                
            except Exception as e:
                self.progress_label.setText(f"Export failed: {e}")
    
    def export_report(self):
        """Export comprehensive report."""
        from PyQt5.QtWidgets import QFileDialog
        
        directory = QFileDialog.getExistingDirectory(self, "Export Report")
        
        if directory and self.scores:
            try:
                from ...core.analyzer import RefScoreAnalyzer
                
                analyzer = RefScoreAnalyzer()
                reports = analyzer.generate_report(
                    self.document, self.sources, self.scores, directory
                )
                
                self.progress_label.setText(f"Report exported to {directory}")
                
            except Exception as e:
                self.progress_label.setText(f"Export failed: {e}")
    
    def reset(self):
        """Reset the tab to initial state."""
        self.document = None
        self.sources = []
        self.scores = []
        
        self.update_summary()
        self.sources_table.setRowCount(0)
        self.coverage_table.setRowCount(0)
        self.weak_sentences_table.setRowCount(0)
        
        self.source_details_text.clear()
        self.sentence_details_text.clear()
        self.coverage_overview_text.clear()
        
        self.progress_label.setText("Ready")
