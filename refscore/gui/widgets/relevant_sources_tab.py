from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QProgressBar, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QEvent, QCoreApplication, QTimer
from PyQt5.QtGui import QPalette, QColor
import webbrowser
from ...core.source_ranking import SourceRankingEngine
from ...core.document_parser import DocumentParser
import time

class SourceSearchWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(list)
    failed = pyqtSignal(str)
    def __init__(self, engine: SourceRankingEngine, text: str, rows: int, thr: float, use_refine: bool):
        super().__init__()
        self.engine = engine
        self.text = text
        self.rows = rows
        self.thr = thr
        self.use_refine = use_refine
    def run(self):
        try:
            self.progress.emit(10, "Extracting terms")
            self.progress.emit(30, "Searching sources")
            results = self.engine.rank(self.text, rows=self.rows, threshold=self.thr, use_refine=self.use_refine)
            self.progress.emit(90, "Finalizing")
            self.completed.emit(results)
        except Exception as e:
            self.failed.emit(str(e))

class FileUploadWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(dict, str)
    failed = pyqtSignal(str)
    def __init__(self, path: str, size_limit_mb: int = 50):
        super().__init__()
        self.path = path
        self.size_limit_mb = size_limit_mb
        self.parser = DocumentParser()
    def run(self):
        try:
            self.progress.emit(10, "Validating file")
            p = self.path
            from pathlib import Path
            path = Path(p)
            if not path.exists():
                raise RuntimeError("File not found")
            ext = path.suffix.lower()
            if ext not in [".pdf", ".tex", ".docx"]:
                raise RuntimeError("Unsupported file type")
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.size_limit_mb:
                raise RuntimeError("File too large")
            self.progress.emit(30, "Reading file")
            if ext == ".docx":
                doc = self.parser._parse_docx(str(path))
            else:
                doc = self.parser.parse_document(str(path))
            self.progress.emit(70, "Preparing content")
            text = "\n".join(s.text for s in doc.sentences)
            info = {"name": path.name, "path": str(path), "type": ext[1:], "size_mb": round(size_mb, 2)}
            self.progress.emit(95, "Ready")
            self.completed.emit(info, text)
        except Exception as e:
            self.failed.emit(str(e))

class SearchProgressWorker(QThread):
    progress = pyqtSignal(int)
    phase = pyqtSignal(str)
    count = pyqtSignal(int, int)
    eta = pyqtSignal(str)
    completed = pyqtSignal(list)
    failed = pyqtSignal(str)
    def __init__(self, engine: SourceRankingEngine, text: str, rows: int, thr: float, use_refine: bool):
        super().__init__()
        self.engine = engine
        self.text = text
        self.rows = rows
        self.thr = thr
        self.use_refine = use_refine
        self.cancelled = False
    def run(self):
        try:
            t0 = time.time()
            self.phase.emit(self._tr("Indexing documents"))
            self.progress.emit(5)
            terms = self.engine.extract_terms(self.text)
            query = self.engine.build_query(self.text, self.use_refine)
            self.phase.emit(self._tr("Searching sources"))
            self.progress.emit(15)
            items = self.engine.search(query, self.rows)
            total = len(items)
            if total == 0:
                self.progress.emit(100)
                self.completed.emit([])
                return
            self.count.emit(0, total)
            self.phase.emit(self._tr("Analyzing content"))
            self.progress.emit(25)
            doc_vec = self.engine._embed(self.text)
            self.phase.emit(self._tr("Matching results"))
            start_match = time.time()
            results = []
            processed = 0
            for it in items:
                if self.cancelled:
                    self.failed.emit(self._tr("Cancelled"))
                    return
                title = (it.get('title') or [''])[0]
                abstract = it.get('abstract', '')
                if not it.get('container-title') or not title:
                    processed += 1
                    self.count.emit(processed, total)
                    self._update_progress(processed, total)
                    continue
                combined = f"{title} {abstract}".strip()
                vec = self.engine._embed(combined)
                sim = self.engine._similarity(doc_vec, vec)
                lc = combined.lower()
                overlap = sum(1 for t in terms if t in lc)
                score = 0.85 * sim + 0.15 * (overlap / max(1, len(terms)))
                if score >= self.thr:
                    authors = []
                    for a in it.get('author', []) or []:
                        fam = a.get('family', '')
                        giv = a.get('given', '')
                        s = fam
                        if giv:
                            s += ", " + giv
                        if s:
                            authors.append(s)
                    year = None
                    try:
                        year = it.get('published-print', {}).get('date-parts', [[None]])[0][0] or it.get('published-online', {}).get('date-parts', [[None]])[0][0]
                    except Exception:
                        year = None
                    link = it.get('URL') or (f"https://doi.org/{it.get('DOI')}" if it.get('DOI') else "")
                    summary = abstract[:300] if abstract else title[:120]
                    results.append({
                        "title": title,
                        "authors": "; ".join(authors),
                        "year": str(year or ''),
                        "journal": (it.get('container-title') or [''])[0],
                        "doi": it.get('DOI', ''),
                        "score": float(round(score, 4)),
                        "summary": summary,
                        "link": link
                    })
                processed += 1
                self.count.emit(processed, total)
                self._update_progress(processed, total)
                elapsed = time.time() - start_match
                per_item = elapsed / max(1, processed)
                remaining = (total - processed) * per_item
                self.eta.emit(self._format_eta(remaining))
            results.sort(key=lambda x: -x['score'])
            top = results[:10]
            self.progress.emit(100)
            self.phase.emit(self._tr("Completed"))
            self.completed.emit(top)
        except Exception as e:
            self.failed.emit(str(e))
    def _update_progress(self, processed: int, total: int):
        base = 40  # after indexing/search/analyzing
        span = 60  # matching phase share
        pct = base + int((processed / max(1, total)) * span)
        self.progress.emit(min(99, pct))
    def _format_eta(self, seconds: float) -> str:
        if seconds <= 0:
            return self._tr("< 1s")
        m, s = divmod(int(seconds), 60)
        if m == 0:
            return f"{s}s"
        return f"{m}m {s}s"
    def _tr(self, text: str) -> str:
        return QCoreApplication.translate("RelevantSourcesTab", text)

class RelevantSourcesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = SourceRankingEngine()
        self.uploaded_files = []
        self.current_text = ""
        self.setup_ui()
    def setup_ui(self):
        layout = QVBoxLayout(self)
        upload_group = QGroupBox("Upload Document")
        ul = QVBoxLayout(upload_group)
        self.drop_zone = QFrame()
        self.drop_zone.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.drop_zone.setMinimumHeight(120)
        self.drop_zone.setAcceptDrops(True)
        pal = self.drop_zone.palette()
        pal.setColor(QPalette.Window, QColor(245, 245, 245))
        self.drop_zone.setPalette(pal)
        self.drop_zone.setAutoFillBackground(True)
        self.drop_label = QLabel("Drop PDF, TEX, or DOCX here")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setAccessibleName("drop-zone-label")
        dzl = QVBoxLayout(self.drop_zone)
        dzl.addWidget(self.drop_label)
        self.select_btn = QPushButton("&Select File")
        self.select_btn.setAccessibleName("select-file-button")
        self.select_btn.clicked.connect(self.select_file)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAccessibleName("upload-progress")
        self.upload_status = QLabel("Ready")
        self.upload_status.setAccessibleName("upload-status")
        ul.addWidget(self.drop_zone)
        ul.addWidget(self.select_btn)
        ul.addWidget(self.progress_bar)
        ul.addWidget(self.upload_status)
        layout.addWidget(upload_group)
        params_group = QGroupBox(self.tr("Search Parameters"))
        pl = QHBoxLayout(params_group)
        pl.addWidget(QLabel(self.tr("Max Results")))
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(10, 500); self.rows_spin.setValue(100)
        pl.addWidget(self.rows_spin)
        pl.addWidget(QLabel(self.tr("Similarity Threshold")))
        self.thr_spin = QDoubleSpinBox(); self.thr_spin.setRange(0.0, 1.0); self.thr_spin.setSingleStep(0.05); self.thr_spin.setValue(0.3)
        pl.addWidget(self.thr_spin)
        self.refine_check = QCheckBox(self.tr("Use refined query"))
        self.refine_check.setChecked(True)
        pl.addWidget(self.refine_check)
        layout.addWidget(params_group)
        run_layout = QHBoxLayout()
        self.run_btn = QPushButton(self.tr("&Find Top Sources"))
        self.run_btn.clicked.connect(self.start_search)
        run_layout.addWidget(self.run_btn)
        self.progress_label = QLabel(self.tr("Ready"))
        run_layout.addWidget(self.progress_label)
        self.cancel_btn = QPushButton(self.tr("&Cancel"))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_search)
        run_layout.addWidget(self.cancel_btn)
        layout.addLayout(run_layout)
        # Search progress indicators
        self.search_progress = QProgressBar(); self.search_progress.setValue(0)
        self.search_progress.setAccessibleName("search-progress")
        layout.addWidget(self.search_progress)
        ind_layout = QHBoxLayout()
        self.phase_label = QLabel(self.tr("Idle")); self.phase_label.setAccessibleName("phase-label"); ind_layout.addWidget(self.phase_label)
        self.count_label = QLabel("0/0"); self.count_label.setAccessibleName("count-label"); ind_layout.addWidget(self.count_label)
        self.eta_label = QLabel("--"); self.eta_label.setAccessibleName("eta-label"); ind_layout.addWidget(self.eta_label)
        layout.addLayout(ind_layout)
        self.loading_anim_timer = QTimer(self); self.loading_anim_timer.setInterval(400); self.loading_anim_timer.timeout.connect(self._tick_loading)
        self._loading_dots = 0
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["Title","Authors","Year","Journal","DOI","Score","Summary","Link"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.itemDoubleClicked.connect(self.open_link)
        layout.addWidget(self.table)
        self.setAcceptDrops(True)
        self.drop_zone.installEventFilter(self)
        self.run_btn.setEnabled(False)
    def tr(self, text: str) -> str:
        return QCoreApplication.translate("RelevantSourcesTab", text)
    def eventFilter(self, obj, event):
        if obj is self.drop_zone:
            if event.type() == QEvent.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    self.drop_zone.setStyleSheet("QFrame{border:2px dashed #0078d4;}")
                else:
                    event.ignore()
                return True
            if event.type() == QEvent.DragLeave:
                self.drop_zone.setStyleSheet("")
                return True
            if event.type() == QEvent.Drop:
                self.drop_zone.setStyleSheet("")
                urls = event.mimeData().urls()
                if urls:
                    p = urls[0].toLocalFile()
                    self.handle_file(p)
                return True
        return super().eventFilter(obj, event)
    def select_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select Document", "", "Documents (*.pdf *.tex *.docx)")
        if fp:
            self.handle_file(fp)
    def handle_file(self, path: str):
        ext = path.split('.')[-1].lower()
        if ext not in ["pdf", "tex", "docx"]:
            self.upload_status.setText("Invalid file type")
            return
        self.progress_bar.setValue(0)
        self.upload_status.setText("Uploading...")
        self.run_btn.setEnabled(False)
        self.file_worker = FileUploadWorker(path)
        def on_progress(v, m):
            self.progress_bar.setValue(v)
            self.upload_status.setText(m)
        def on_completed(info, text):
            self.uploaded_files.append(info)
            self.current_text = text
            self.upload_status.setText(f"Uploaded: {info['name']} ({info['type']}, {info['size_mb']} MB)")
            self.progress_bar.setValue(100)
            self.run_btn.setEnabled(True)
        def on_failed(msg):
            self.upload_status.setText(f"Error: {msg}")
            self.progress_bar.setValue(0)
            self.run_btn.setEnabled(False)
        self.file_worker.progress.connect(on_progress)
        self.file_worker.completed.connect(on_completed)
        self.file_worker.failed.connect(on_failed)
        self.file_worker.start()
    def start_search(self):
        text = self.current_text.strip()
        if not text:
            self.progress_label.setText(self.tr("Upload a document"))
            return
        rows = int(self.rows_spin.value())
        thr = float(self.thr_spin.value())
        use_refine = bool(self.refine_check.isChecked())
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_label.setText(self.tr("Running"))
        self.search_progress.setRange(0, 100)
        self.search_progress.setValue(0)
        self.phase_label.setText(self.tr("Indexing documents"))
        self.count_label.setText("0/0")
        self.eta_label.setText("--")
        self.loading_anim_timer.start()
        self.worker = SearchProgressWorker(self.engine, text, rows, thr, use_refine)
        def on_progress(v):
            self.search_progress.setValue(v)
        def on_phase(p):
            self.phase_label.setText(p + self._loading_suffix())
        def on_count(done, total):
            self.count_label.setText(f"{done}/{total}")
            if total > 0:
                self.search_progress.setRange(0, 100)
        def on_eta(t):
            self.eta_label.setText(self.tr("ETA") + f": {t}")
        def on_completed(results):
            self.loading_anim_timer.stop()
            self.phase_label.setText(self.tr("Completed"))
            self.search_progress.setValue(100)
            self.populate_table(results)
            self.progress_label.setText(self.tr("Completed"))
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
        def on_failed(msg):
            self.loading_anim_timer.stop()
            self.search_progress.setValue(0)
            self.phase_label.setText(self.tr("Error"))
            self.progress_label.setText(self.tr("Error") + f": {msg}")
            QMessageBox.warning(self, self.tr("Search Error"), msg)
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
        self.worker.progress.connect(on_progress)
        self.worker.phase.connect(on_phase)
        self.worker.count.connect(on_count)
        self.worker.eta.connect(on_eta)
        self.worker.completed.connect(on_completed)
        self.worker.failed.connect(on_failed)
        self.worker.start()
    def cancel_search(self):
        try:
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.cancelled = True
        except Exception:
            pass
    def _tick_loading(self):
        self._loading_dots = (self._loading_dots + 1) % 4
        self.phase_label.setText(self.phase_label.text().split('.')[0] + self._loading_suffix())
    def _loading_suffix(self) -> str:
        return '.' * self._loading_dots
    def populate_table(self, results):
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(r.get('title','')))
            self.table.setItem(i, 1, QTableWidgetItem(r.get('authors','')))
            self.table.setItem(i, 2, QTableWidgetItem(r.get('year','')))
            self.table.setItem(i, 3, QTableWidgetItem(r.get('journal','')))
            self.table.setItem(i, 4, QTableWidgetItem(r.get('doi','')))
            self.table.setItem(i, 5, QTableWidgetItem(f"{float(r.get('score',0.0)):.3f}"))
            self.table.setItem(i, 6, QTableWidgetItem(r.get('summary','')))
            self.table.setItem(i, 7, QTableWidgetItem(r.get('link','')))
    def open_link(self, item):
        row = item.row()
        link_item = self.table.item(row, 7)
        if link_item:
            url = link_item.text()
            if url:
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
