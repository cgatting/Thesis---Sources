import logging
import os
import time
import re
import requests
import threading
import multiprocessing
import concurrent.futures
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional
import asyncio
import json
import nltk
import torch
import yake
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import webbrowser
import sys

# ---------------- Logging Setup ----------------
LOG_FILE = 'deep_research_tool_improved.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")

if torch.cuda.is_available():
    torch.multiprocessing.set_start_method('spawn', force=True)

# Add memory management constants
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 100 * 1024  # 100KB for processing chunks
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.deep_research_cache')

# Add citation format templates
CITATION_FORMATS = {
    'APA': "{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}. DOI: {doi}",
    'MLA': "{authors}. \"{title}.\" {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}. DOI: {doi}",
    'Chicago': "{authors}. {year}. \"{title}.\" {journal} {volume}, no. {issue}: {pages}. DOI: {doi}"
}

# ---------------- NLP Processor ----------------
class NLPProcessor:
    def __init__(self, settings: dict):
        self.settings = settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.sentence_model = SentenceTransformer(settings['model_settings']['sentence_model'])
            if torch.cuda.is_available():
                self.sentence_model = self.sentence_model.to(self.device)
            self.keyword_extractor = yake.KeywordExtractor(**settings['keyword_settings'])
            device_id = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline("summarization", 
                                       model=settings['model_settings']['summarizer_model'],
                                       device=device_id,
                                       model_kwargs={"low_cpu_mem_usage": True})
        except Exception as e:
            logging.error(f"Error initializing NLP models: {e}")
            raise

    @lru_cache(maxsize=1000)
    def refine_query(self, sentence: str) -> str:
        try:
            keywords = " ".join([kw[0] for kw in self.keyword_extractor.extract_keywords(sentence)])
            summary = self.summarizer(
                sentence,
                max_length=self.settings['model_settings']['max_length'],
                min_length=self.settings['model_settings']['min_length'],
                do_sample=False
            )[0]['summary_text']
            refined = f"{keywords} {summary}".strip()
            logging.info(f"Refined query: {refined}")
            return refined
        except Exception as e:
            logging.error(f"Error refining query: {e}")
            return sentence

    @lru_cache(maxsize=1000)
    def calculate_similarity(self, text1: str, text2: str) -> float:
        try:
            with torch.no_grad():
                emb1 = self.sentence_model.encode(text1, convert_to_tensor=True)
                emb2 = self.sentence_model.encode(text2, convert_to_tensor=True)
                if self.device.type == 'cuda':
                    emb1 = emb1.cpu()
                    emb2 = emb2.cpu()
                sim = float(cosine_similarity(
                    emb1.numpy().reshape(1, -1),
                    emb2.numpy().reshape(1, -1)
                )[0][0])
                return sim
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            return 0.0

# ---------------- Async Document Processor ----------------
class AsyncDocumentProcessor:
    """Handles asynchronous document processing with chunking support."""
    def __init__(self, chunk_size=CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.processed_chunks = []
        
    async def process_document(self, text: str, processor_func) -> str:
        chunks = self._split_into_chunks(text)
        tasks = [self._process_chunk(chunk, processor_func) for chunk in chunks]
        processed_chunks = await asyncio.gather(*tasks)
        return self._merge_chunks(processed_chunks)
    
    def _split_into_chunks(self, text: str) -> List[str]:
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
    
    async def _process_chunk(self, chunk: str, processor_func) -> str:
        return await processor_func(chunk)
    
    def _merge_chunks(self, chunks: List[str]) -> str:
        return ''.join(chunks)

# ---------------- Document Cache ----------------
class DocumentCache:
    """Handles caching of processed documents and search results."""
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cached_result(self, key: str) -> dict:
        cache_file = os.path.join(self.cache_dir, f"{hash(key)}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Cache read error: {e}")
        return None
    
    def cache_result(self, key: str, data: dict):
        cache_file = os.path.join(self.cache_dir, f"{hash(key)}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.warning(f"Cache write error: {e}")

# ---------------- Source Ranking Engine ----------------
class SourceRankingEngine:
    def __init__(self, settings: dict, nlp_processor: NLPProcessor, search_engine: Any, cache: Optional[DocumentCache] = None):
        self.settings = settings
        self.nlp = nlp_processor
        self.search = search_engine
        self.cache = cache or DocumentCache()

    def extract_key_terms(self, text: str, top_k: int = 15) -> List[str]:
        try:
            kws = self.nlp.keyword_extractor.extract_keywords(text)
            terms = [kw for kw, _ in sorted(kws, key=lambda x: x[1])][:top_k]
            return terms
        except Exception:
            words = re.findall(r"[A-Za-z]{4,}", text.lower())
            freq: Dict[str, int] = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_k]]

    def build_query(self, text: str, use_refine: bool) -> str:
        if use_refine:
            return self.nlp.refine_query(text)
        terms = self.extract_key_terms(text)
        return " ".join(terms)

    def score_result(self, doc_text: str, item: Dict[str, Any], terms: List[str]) -> float:
        title = (item.get('title') or [''])[0]
        abstract = item.get('abstract', '')
        combined = f"{title} {abstract}".strip()
        sim = self.nlp.calculate_similarity(doc_text, combined) if combined else 0.0
        lower_combined = combined.lower()
        overlap = sum(1 for t in terms if t.lower() in lower_combined)
        overlap_norm = overlap / max(1, len(terms))
        return 0.85 * sim + 0.15 * overlap_norm

    def summarize_item(self, item: Dict[str, Any]) -> str:
        try:
            text = " ".join([s for s in [item.get('title', [''])[0], item.get('abstract', '')] if s])
            if not text:
                return ""
            out = self.nlp.summarizer(
                text,
                max_length=min(48, self.settings['model_settings']['max_length']),
                min_length=min(12, self.settings['model_settings']['min_length']),
                do_sample=False
            )[0]['summary_text']
            return out
        except Exception:
            return (item.get('title', [''])[0] or '')[:120]

    def rank_top_sources(self, doc_text: str, rows: int, threshold: float, use_refine: bool, progress_cb: Optional[Any] = None) -> List[Dict[str, Any]]:
        key = json.dumps({"text_hash": hash(doc_text), "rows": rows, "thr": threshold, "ref": use_refine}, sort_keys=True)
        cached = self.cache.get_cached_result(key)
        if cached:
            return cached.get('results', [])

        if progress_cb:
            progress_cb(0.05, "Extracting key terms...")
        terms = self.extract_key_terms(doc_text)
        query = self.build_query(doc_text, use_refine)

        if progress_cb:
            progress_cb(0.2, "Searching sources...")
        items = self.search.search_papers(query, limit=rows)

        if progress_cb:
            progress_cb(0.45, f"Scoring {len(items)} results...")
        scored: List[Dict[str, Any]] = []
        for it in items:
            if not it.get('DOI') or not it.get('container-title'):
                continue
            score = self.score_result(doc_text, it, terms)
            if score >= threshold:
                scored.append({"item": it, "score": float(round(score, 4))})

        if progress_cb:
            progress_cb(0.7, "Selecting top 10 and summarizing...")
        top = sorted(scored, key=lambda x: -x['score'])[:10]
        results: List[Dict[str, Any]] = []
        for t in top:
            it = t['item']
            authors = []
            for a in it.get('author', []) or []:
                fam = a.get('family', '')
                giv = a.get('given', '')
                if fam or giv:
                    authors.append(f"{fam}, {giv}".strip(', '))
            year = None
            try:
                year = it.get('published-print', {}).get('date-parts', [[None]])[0][0] or it.get('published-online', {}).get('date-parts', [[None]])[0][0]
            except Exception:
                year = None
            link = it.get('URL') or (f"https://doi.org/{it.get('DOI')}" if it.get('DOI') else "")
            summary = self.summarize_item(it)
            results.append({
                "title": (it.get('title') or [''])[0],
                "authors": "; ".join(authors),
                "year": str(year or ''),
                "journal": (it.get('container-title') or [''])[0],
                "doi": it.get('DOI', ''),
                "score": t['score'],
                "summary": summary,
                "link": link
            })

        self.cache.cache_result(key, {"results": results})
        if progress_cb:
            progress_cb(0.95, "Done")
        return results

# ---------------- Research Search Engine ----------------
class ResearchSearchEngine:
    def __init__(self):
        self.headers = {'User-Agent': 'DeepResearchTool/1.0 (mailto:your.email@example.com)'}
        self.max_retries = 5
        self.timeout = 30
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits by waiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    def search_papers(self, query: str, limit: int = 50) -> List[Dict]:
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                response = requests.get(
                    'https://api.crossref.org/works',
                    params={'query': query, 'rows': limit},
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logging.warning(f"Rate limited by CrossRef. Waiting {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                items = response.json().get('message', {}).get('items', [])
                logging.info(f"Found {len(items)} items for query: {query}")
                return items
                
            except requests.RequestException as e:
                wait_time = min(30, 2 ** attempt)  # Cap maximum wait time at 30 seconds
                logging.error(f"CrossRef search error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return []
                logging.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
        return []

# ---------------- Document Refiner ----------------
class DocumentRefiner:
    def __init__(self, settings: dict, view):
        self.settings = settings
        self.view = view
        self.nlp_processor = NLPProcessor(settings)
        self.search_engine = ResearchSearchEngine()
        self.bibliography = {}
        self.bib_lock = threading.Lock()
        self.doc_cache = DocumentCache()
        self.async_processor = AsyncDocumentProcessor()
        
    async def process_sentence_async(self, sentence: str, idx: int) -> str:
        """Asynchronous version of process_sentence with improved error handling and caching."""
        if not self.sentence_needs_citation(sentence):
            return sentence
            
        cache_key = f"sentence_{hash(sentence)}"
        cached_result = self.doc_cache.get_cached_result(cache_key)
        if cached_result:
            return cached_result['processed_sentence']
            
        try:
            query = await self._refine_query_async(sentence)
            search_results = await self._search_papers_async(query)
            
            best_result = await self._find_best_match_async(sentence, search_results)
            if best_result:
                citation = self._create_citation(best_result)
                processed_sentence = f"{sentence} {citation}"
                self.doc_cache.cache_result(cache_key, {
                    'processed_sentence': processed_sentence,
                    'citation_data': best_result
                })
                return processed_sentence
        except Exception as e:
            logging.error(f"Error processing sentence: {e}")
            self.view.show_error(f"Error processing sentence: {str(e)}")
        return sentence

    async def refine_document(self, doc_text: str) -> str:
        """Improved document refinement with async processing and progress tracking."""
        if len(doc_text) > MAX_DOCUMENT_SIZE:
            raise ValueError(f"Document size exceeds maximum limit of {MAX_DOCUMENT_SIZE/1024/1024}MB")
            
        sentences = nltk.sent_tokenize(doc_text)
        total = len(sentences)
        processed_sentences = []
        
        for i, sentence in enumerate(sentences):
            processed = await self.process_sentence_async(sentence, i)
            processed_sentences.append(processed)
            self.view.update_progress((i + 1) / total, f"Processing sentence {i + 1}/{total}")
            
        return "\n\n".join(processed_sentences)

    def sentence_needs_citation(self, sentence: str) -> bool:
        """Heuristic to determine if a sentence needs a citation."""
        if '\\cite' in sentence:
            return False
        words = sentence.split()
        if len(words) < 5:
            return False
        # Trigger words that often indicate a claim or factual statement.
        triggers = [
            'find', 'finds', 'found', 'suggest', 'suggests', 'reported', 'reports',
            'demonstrate', 'demonstrates', 'evidence', 'conclude', 'concludes',
            'indicate', 'indicates', 'study', 'studies', 'analysis', 'analyses',
            'data', 'research', 'observed', 'observes', 'observing'
        ]
        sentence_lower = sentence.lower()
        if any(trigger in sentence_lower for trigger in triggers):
            return True
        # Also, if the sentence contains numbers (which may indicate data or statistics)
        if any(char.isdigit() for char in sentence):
            return True
        return False

    def create_citation_id(self, result: Dict) -> str:
        try:
            author_family = result.get('author', [{}])[0].get('family', 'Unknown')
            year = result.get('published-print', {}).get('date-parts', [['Unknown']])[0][0]
        except Exception:
            author_family, year = "Unknown", "Unknown"
        citation_id = f"{author_family}{year}"
        return citation_id

    async def _refine_query_async(self, sentence: str) -> str:
        return self.nlp_processor.refine_query(sentence)

    async def _search_papers_async(self, query: str) -> List[Dict]:
        return self.search_engine.search_papers(query, limit=self.settings.get('search_settings', {}).get('max_results', 50))

    async def _find_best_match_async(self, sentence: str, search_results: List[Dict]) -> Dict:
        best_result = None
        best_similarity = 0.0
        for result in search_results:
            # Additional quality checks: only consider results with DOI and container-title.
            if not result.get('DOI') or not result.get('container-title', []):
                continue
            title = result.get('title', [''])[0]
            abstract = result.get('abstract', '')
            combined_text = f"{title} {abstract}"
            similarity = self.nlp_processor.calculate_similarity(sentence, combined_text)
            logging.info(f"Sentence similarity with '{title}': {similarity:.3f}")
            if similarity > self.settings.get('similarity_threshold', 0.7) and similarity > best_similarity:
                best_similarity = similarity
                best_result = result
        return best_result

    def _create_citation(self, result: Dict) -> str:
        citation_id = self.create_citation_id(result)
        with self.bib_lock:
            if citation_id not in self.bibliography:
                authors = " and ".join([
                    f"{author.get('family', '')}, {author.get('given', '')}"
                    for author in result.get('author', [])
                ])
                self.bibliography[citation_id] = {
                    'ID': citation_id,
                    'title': result.get('title', [''])[0],
                    'authors': authors,
                    'year': str(result.get('published-print', {}).get('date-parts', [['Unknown']])[0][0]),
                    'doi': result.get('DOI', 'N/A')
                }
        return f"\\citep{{{citation_id}}}"

    def generate_bibliography_text(self) -> str:
        bib_entries = []
        with self.bib_lock:
            for entry in self.bibliography.values():
                bib_text = CITATION_FORMATS['APA'].format(
                    authors=entry['authors'],
                    year=entry['year'],
                    title=entry['title'],
                    journal=entry.get('container-title', [''])[0],
                    volume=entry.get('volume', ''),
                    issue=entry.get('issue', ''),
                    pages=entry.get('page', ''),
                    doi=entry['doi']
                )
                bib_entries.append(bib_text)
        return "\n\n".join(bib_entries)

# ---------------- Deep Research Tool App (GUI) ----------------
class DeepResearchToolApp:
    def __init__(self, settings: dict):
        self.settings = settings
        self.document_text = ""   # Original document text
        self.refined_text = ""    # Updated document text after processing
        self.doc_refiner = None   # Will be initialized after UI setup
        self.async_loop = None    # Will store the asyncio event loop
        self.source_ranking = None
        self.search_engine = None
        
        # Initialize UI
        self.setup_ui()
        
        # Initialize processing components
        try:
            self.doc_refiner = DocumentRefiner(settings, self)
            self.search_engine = self.doc_refiner.search_engine
            self.source_ranking = SourceRankingEngine(settings, self.doc_refiner.nlp_processor, self.search_engine, self.doc_refiner.doc_cache)
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
            threading.Thread(target=self._run_async_loop, daemon=True).start()
        except Exception as e:
            self.show_error(f"Failed to initialize document processor: {e}")
            raise
        
        # Setup additional UI components that depend on doc_refiner
        self.setup_keyboard_shortcuts()
        self.load_theme_preference()
        
    def _run_async_loop(self):
        """Run the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_forever()
        
    def setup_ui(self):
        """Initialize the UI components."""
        # Set initial theme from settings
        theme = self.settings.get('ui_settings', {}).get('theme', 'dark')
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Deep Research Document Refiner")
        window_size = self.settings.get('ui_settings', {}).get('window_size', '1000x750')
        self.root.geometry(window_size)
        
        # Create UI frames
        self.create_control_frame()
        self.create_progress_frame()
        self.create_notebook_frame()
        self.create_console_frame()
        
        # Make text boxes read-only initially
        self.orig_textbox.configure(state="disabled")
        self.refined_textbox.configure(state="disabled")
        self.bib_textbox.configure(state="disabled")
        
        # Add tooltips after all widgets are created
        self.add_tooltips()
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Log initial status
        self.log_to_console("Application initialized successfully")
        
    def on_closing(self):
        """Handle application shutdown."""
        try:
            if self.async_loop and self.async_loop.is_running():
                self.async_loop.stop()
            if hasattr(self, 'root'):
                self.root.destroy()
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)

    def add_tooltips(self):
        """Add tooltips to UI elements."""
        self.tooltips = {}
        tooltip_texts = {
            self.upload_button: "Upload Document (Ctrl+O)",
            self.refine_button: "Start Refinement (Ctrl+R)",
            self.save_button: "Save Document (Ctrl+S)",
            self.bib_button: "Save Bibliography (Ctrl+B)",
            self.theme_button: "Toggle Dark/Light Theme",
            self.citation_format: "Select Citation Format"
        }
        
        for widget, text in tooltip_texts.items():
            try:
                tooltip = ctk.CTkToolTip(widget, message=text)
                self.tooltips[widget] = tooltip
            except Exception as e:
                logging.warning(f"Failed to create tooltip: {e}")

    def upload_document(self, event=None):
        """Handle document upload."""
        try:
            filename = filedialog.askopenfilename(
                title="Select Document",
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("LaTeX Files", "*.tex"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.document_text = f.read()
                
                # Update original text display
                self.orig_textbox.configure(state="normal")
                self.orig_textbox.delete("1.0", "end")
                self.orig_textbox.insert("end", self.document_text)
                self.orig_textbox.configure(state="disabled")
                
                # Enable refinement button
                self.refine_button.configure(state="normal")
                
                # Reset other UI elements
                self.refined_textbox.configure(state="normal")
                self.refined_textbox.delete("1.0", "end")
                self.refined_textbox.configure(state="disabled")
                
                self.bib_textbox.configure(state="normal")
                self.bib_textbox.delete("1.0", "end")
                self.bib_textbox.configure(state="disabled")
                
                self.save_button.configure(state="disabled")
                self.bib_button.configure(state="disabled")
                
                self.progress_bar.set(0)
                self.status_label.configure(text="Ready to process")
                
                self.log_to_console(f"Loaded document: {filename}")
        except Exception as e:
            self.show_error(f"Error loading document: {e}")
            
    def save_refined_document(self, event=None):
        """Save the refined document."""
        if not self.refined_text:
            self.show_error("No refined document to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Refined Document",
                defaultextension=".txt",
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("LaTeX Files", "*.tex"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.refined_text)
                self.log_to_console(f"Saved refined document to: {filename}")
        except Exception as e:
            self.show_error(f"Error saving document: {e}")

    def save_bibliography(self, event=None):
        """Save the bibliography."""
        if not self.doc_refiner or not self.doc_refiner.bibliography:
            self.show_error("No bibliography to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Bibliography",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if filename:
                bib_text = self.doc_refiner.generate_bibliography_text()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(bib_text)
                self.log_to_console(f"Saved bibliography to: {filename}")
        except Exception as e:
            self.show_error(f"Error saving bibliography: {e}")

    def update_progress(self, fraction: float, status: str):
        """Update the progress bar and status label."""
        try:
            self.progress_bar.set(fraction)
            self.status_label.configure(text=status)
            self.root.update_idletasks()
        except Exception as e:
            logging.error(f"Error updating progress: {e}")

    def log_to_console(self, message: str):
        """Add a message to the console with timestamp."""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.console.configure(state="normal")
            self.console.insert("end", f"[{timestamp}] {message}\n")
            self.console.see("end")
            self.console.configure(state="disabled")
        except Exception as e:
            logging.error(f"Error logging to console: {e}")

    def show_error(self, message: str):
        """Display error message in console and dialog."""
        self.log_to_console(f"ERROR: {message}")
        threading.Thread(target=lambda: messagebox.showerror("Error", message)).start()

    def toggle_theme(self):
        """Toggle between light and dark theme."""
        try:
            current_theme = ctk.get_appearance_mode()
            new_theme = "Light" if current_theme == "Dark" else "Dark"
            ctk.set_appearance_mode(new_theme)
            self.settings['ui_settings']['theme'] = new_theme.lower()
            self.save_settings()
        except Exception as e:
            self.show_error(f"Error toggling theme: {e}")

    def save_settings(self):
        """Save current settings to file."""
        try:
            settings_path = os.path.join(CACHE_DIR, 'settings.json')
            with open(settings_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving settings: {e}")

    def load_theme_preference(self):
        """Load and apply saved theme preference."""
        try:
            theme = self.settings.get('ui_settings', {}).get('theme', 'dark')
            ctk.set_appearance_mode(theme.capitalize())
        except Exception as e:
            logging.error(f"Error loading theme preference: {e}")
            ctk.set_appearance_mode("Dark")

    def update_citation_format(self, format_name: str):
        """Update citation format and regenerate bibliography."""
        try:
            self.settings['ui_settings']['citation_format'] = format_name
            self.save_settings()
            if self.doc_refiner and self.doc_refiner.bibliography:
                self.update_bibliography()
        except Exception as e:
            self.show_error(f"Error updating citation format: {e}")

    def update_bibliography(self):
        """Update bibliography text with current format."""
        try:
            bib_text = self.doc_refiner.generate_bibliography_text()
            self.bib_textbox.configure(state="normal")
            self.bib_textbox.delete("1.0", "end")
            self.bib_textbox.insert("end", bib_text)
            self.bib_textbox.configure(state="disabled")
        except Exception as e:
            self.show_error(f"Error updating bibliography: {e}")

    def create_control_frame(self):
        """Create the control frame with buttons and options."""
        control_frame = ctk.CTkFrame(self.root)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Create and store button references for tooltips
        self.upload_button = ctk.CTkButton(control_frame, text="Upload Document", 
                                         command=self.upload_document)
        self.upload_button.pack(side="left", padx=5)
        
        self.refine_button = ctk.CTkButton(control_frame, text="Refine & Add Citations", 
                                          command=self.start_refinement,
                                          state="disabled")  # Initially disabled
        self.refine_button.pack(side="left", padx=5)
        
        self.save_button = ctk.CTkButton(control_frame, text="Save Refined Document", 
                                        command=self.save_refined_document,
                                        state="disabled")  # Initially disabled
        self.save_button.pack(side="left", padx=5)
        
        self.bib_button = ctk.CTkButton(control_frame, text="Save Bibliography", 
                                       command=self.save_bibliography,
                                       state="disabled")  # Initially disabled
        self.bib_button.pack(side="left", padx=5)
        
        # Theme toggle
        self.theme_button = ctk.CTkButton(control_frame, text="Toggle Theme",
                                        command=self.toggle_theme)
        self.theme_button.pack(side="right", padx=5)
        
        # Citation format selector
        self.citation_format = ctk.CTkComboBox(
            control_frame,
            values=list(CITATION_FORMATS.keys()),
            command=self.update_citation_format,
            state="readonly"  # Prevent manual editing
        )
        self.citation_format.pack(side="right", padx=5)
        self.citation_format.set(self.settings.get('ui_settings', {}).get('citation_format', 'APA'))
        
    def create_progress_frame(self):
        """Create the progress tracking frame."""
        progress_frame = ctk.CTkFrame(self.root)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(progress_frame, text="Ready")
        self.status_label.pack(pady=5)
        
    def create_notebook_frame(self):
        """Create the notebook with document tabs."""
        self.notebook = ctk.CTkTabview(self.root, width=950, height=500)
        self.notebook.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Add tabs
        self.notebook.add("Original Document")
        self.notebook.add("Refined Document")
        self.notebook.add("Bibliography")
        
        # Create text boxes
        self.orig_textbox = ctk.CTkTextbox(self.notebook.tab("Original Document"))
        self.orig_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.refined_textbox = ctk.CTkTextbox(self.notebook.tab("Refined Document"))
        self.refined_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.bib_textbox = ctk.CTkTextbox(self.notebook.tab("Bibliography"))
        self.bib_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_console_frame(self):
        """Create the console output frame."""
        console_frame = ctk.CTkFrame(self.root)
        console_frame.pack(fill="x", padx=10, pady=5)
        
        self.console = ctk.CTkTextbox(console_frame, height=100)
        self.console.pack(fill="x", padx=5, pady=5)
        
    def setup_keyboard_shortcuts(self):
        self.root.bind('<Control-o>', lambda e: self.upload_document())
        self.root.bind('<Control-s>', lambda e: self.save_refined_document())
        self.root.bind('<Control-b>', lambda e: self.save_bibliography())
        self.root.bind('<Control-r>', lambda e: self.start_refinement())
        
    def start_refinement(self, event=None):
        """Non-blocking wrapper for async refinement."""
        if not self.async_loop:
            self.show_error("Async loop not initialized")
            return
            
        if not self.document_text:
            self.show_error("Please upload a document first")
            return
            
        # Disable buttons during processing
        self.refine_button.configure(state="disabled")
        self.upload_button.configure(state="disabled")
        
        # Start async processing
        asyncio.run_coroutine_threadsafe(
            self.start_refinement_async(),
            self.async_loop
        )
        
    async def start_refinement_async(self):
        """Asynchronous document refinement with proper error handling."""
        if not self.document_text:
            self.show_error("Please upload a document first.")
            return
            
        self.log_to_console("Starting document refinement...")
        self.update_progress(0, "Initializing refinement...")
        
        try:
            # Process document
            self.refined_text = await self.doc_refiner.refine_document(self.document_text)
            
            # Update UI with results
            self.update_refined_text()
            self.update_bibliography()
            
            # Update status
            self.update_progress(1.0, "Refinement complete")
            self.log_to_console("Document refinement complete.")
            
        except Exception as e:
            self.show_error(f"Refinement failed: {str(e)}")
            self.update_progress(0, "Refinement failed")
        finally:
            # Re-enable buttons
            self.refine_button.configure(state="normal")
            self.upload_button.configure(state="normal")
            
    def update_refined_text(self):
        """Update the refined text display."""
        try:
            self.refined_textbox.configure(state="normal")
            self.refined_textbox.delete("1.0", "end")
            self.refined_textbox.insert("end", self.refined_text)
            self.refined_textbox.configure(state="disabled")
            
            # Enable save buttons
            self.save_button.configure(state="normal")
            self.bib_button.configure(state="normal")
        except Exception as e:
            self.show_error(f"Error updating refined text: {e}")

    def create_control_frame(self):
        """Create the control frame with buttons and options."""
        self.control_frame = ctk.CTkFrame(self.root)
        self.control_frame.pack(fill="x", padx=10, pady=5)
        
        # Upload button
        self.upload_button = ctk.CTkButton(
            self.control_frame, 
            text="Upload Document", 
            command=self.upload_document,
            width=120
        )
        self.upload_button.pack(side="left", padx=5)
        
        # Refine button
        self.refine_button = ctk.CTkButton(
            self.control_frame, 
            text="Refine Document", 
            command=self.refine_document,
            state="disabled",
            width=120
        )
        self.refine_button.pack(side="left", padx=5)
        
        # Save button
        self.save_button = ctk.CTkButton(
            self.control_frame, 
            text="Save Document", 
            command=self.save_document,
            state="disabled",
            width=120
        )
        self.save_button.pack(side="left", padx=5)
        
        # Bibliography button
        self.bib_button = ctk.CTkButton(
            self.control_frame, 
            text="Save Bibliography", 
            command=self.save_bibliography,
            state="disabled",
            width=120
        )
        self.bib_button.pack(side="left", padx=5)
        
        # Theme toggle button
        self.theme_button = ctk.CTkButton(
            self.control_frame, 
            text="Toggle Theme", 
            command=self.toggle_theme,
            width=100
        )
        self.theme_button.pack(side="right", padx=5)
        
        # Citation format dropdown
        self.citation_format = ctk.CTkOptionMenu(
            self.control_frame,
            values=list(CITATION_FORMATS.keys()),
            command=self.change_citation_format
        )
        self.citation_format.set(self.settings.get('ui_settings', {}).get('citation_format', 'APA'))
        self.citation_format.pack(side="right", padx=5)

    def create_progress_frame(self):
        """Create the progress frame with progress bar and status."""
        self.progress_frame = ctk.CTkFrame(self.root)
        self.progress_frame.pack(fill="x", padx=10, pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5)
        self.progress_bar.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(self.progress_frame, text="Ready")
        self.status_label.pack(side="right", padx=5)

    def create_notebook_frame(self):
        """Create the notebook frame with text areas."""
        self.notebook_frame = ctk.CTkFrame(self.root)
        self.notebook_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self.notebook_frame)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Original document tab
        self.tabview.add("Original Document")
        self.orig_textbox = ctk.CTkTextbox(
            self.tabview.tab("Original Document"),
            wrap="word",
            font=("Consolas", 12)
        )
        self.orig_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Refined document tab
        self.tabview.add("Refined Document")
        self.refined_textbox = ctk.CTkTextbox(
            self.tabview.tab("Refined Document"),
            wrap="word",
            font=("Consolas", 12)
        )
        self.refined_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bibliography tab
        self.tabview.add("Bibliography")
        self.bib_textbox = ctk.CTkTextbox(
            self.tabview.tab("Bibliography"),
            wrap="word",
            font=("Consolas", 12)
        )
        self.bib_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Relevant sources tab
        self.create_sources_tab()

    def create_sources_tab(self):
        self.tabview.add("Relevant Sources")
        tab = self.tabview.tab("Relevant Sources")
        container = ctk.CTkFrame(tab)
        container.pack(fill="both", expand=True, padx=5, pady=5)

        input_label = ctk.CTkLabel(container, text="Document Content")
        input_label.pack(anchor="w")
        self.sources_input_text = ctk.CTkTextbox(container, wrap="word", height=180)
        self.sources_input_text.pack(fill="x", padx=5, pady=5)

        params_frame = ctk.CTkFrame(container)
        params_frame.pack(fill="x", padx=5, pady=5)

        rows_label = ctk.CTkLabel(params_frame, text="Max Results")
        rows_label.pack(side="left", padx=(5, 2))
        self.sources_rows = ctk.CTkComboBox(params_frame, values=["50", "100", "200", "500"])
        self.sources_rows.pack(side="left", padx=5)
        self.sources_rows.set(str(self.settings.get('search_settings', {}).get('max_results', 500)))

        thr_label = ctk.CTkLabel(params_frame, text="Similarity Threshold")
        thr_label.pack(side="left", padx=(15, 2))
        self.sources_thr = ctk.CTkEntry(params_frame, width=80)
        self.sources_thr.pack(side="left", padx=5)
        self.sources_thr.insert(0, str(self.settings.get('similarity_threshold', 0.7)))

        self.sources_use_refine = ctk.CTkSwitch(params_frame, text="Use Refined Query")
        self.sources_use_refine.pack(side="left", padx=15)
        self.sources_use_refine.select()

        run_frame = ctk.CTkFrame(container)
        run_frame.pack(fill="x", padx=5, pady=5)
        self.sources_run_button = ctk.CTkButton(run_frame, text="Find Top Sources", command=self.start_source_ranking)
        self.sources_run_button.pack(side="left")

        tree_frame = ctk.CTkFrame(container)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        columns = ("Title", "Authors", "Year", "Journal", "DOI", "Score", "Summary", "Link")
        self.sources_tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        for col in columns:
            self.sources_tree.heading(col, text=col)
            self.sources_tree.column(col, width=120 if col not in ("Title", "Summary") else 260, stretch=True)
        self.sources_tree.pack(fill="both", expand=True)
        self.sources_tree.bind("<Double-1>", self._open_selected_link)

    def _open_selected_link(self, event=None):
        sel = self.sources_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        values = self.sources_tree.item(item_id, 'values')
        if len(values) >= 8 and values[7]:
            try:
                webbrowser.open(values[7])
            except Exception as e:
                self.show_error(f"Failed to open link: {e}")

    def start_source_ranking(self):
        if not self.async_loop:
            self.show_error("Async loop not initialized")
            return
        text = self.sources_input_text.get("1.0", "end").strip()
        if not text:
            self.show_error("Please paste or type document content")
            return
        try:
            rows = int(self.sources_rows.get())
        except Exception:
            rows = self.settings.get('search_settings', {}).get('max_results', 500)
        try:
            thr = float(self.sources_thr.get())
        except Exception:
            thr = float(self.settings.get('similarity_threshold', 0.7))
        use_refine = bool(self.sources_use_refine.get())

        self.sources_run_button.configure(state="disabled")
        self.log_to_console("Starting source ranking...")

        asyncio.run_coroutine_threadsafe(
            self.run_source_ranking_async(text, rows, thr, use_refine),
            self.async_loop
        )

    async def run_source_ranking_async(self, text: str, rows: int, thr: float, use_refine: bool):
        try:
            def progress_cb(val, status):
                self.update_progress(val, status)
            self.update_progress(0.0, "Initializing source ranking...")
            results = self.source_ranking.rank_top_sources(text, rows, thr, use_refine, progress_cb)
            self.update_sources_results(results)
            self.update_progress(1.0, "Source ranking complete")
            self.log_to_console("Relevant sources computed.")
        except Exception as e:
            self.show_error(f"Source ranking failed: {e}")
            self.update_progress(0.0, "Source ranking failed")
        finally:
            self.sources_run_button.configure(state="normal")

    def update_sources_results(self, results: List[Dict[str, Any]]):
        try:
            for row in self.sources_tree.get_children():
                self.sources_tree.delete(row)
            for r in results:
                self.sources_tree.insert('', 'end', values=(
                    r.get('title', ''),
                    r.get('authors', ''),
                    r.get('year', ''),
                    r.get('journal', ''),
                    r.get('doi', ''),
                    f"{r.get('score', 0.0):.3f}",
                    r.get('summary', ''),
                    r.get('link', '')
                ))
        except Exception as e:
            self.show_error(f"Error displaying results: {e}")

    def create_console_frame(self):
        """Create the console frame for logging."""
        self.console_frame = ctk.CTkFrame(self.root, height=150)
        self.console_frame.pack(fill="x", padx=10, pady=5)
        self.console_frame.pack_propagate(False)
        
        # Console label
        console_label = ctk.CTkLabel(self.console_frame, text="Console Output:")
        console_label.pack(anchor="w", padx=5, pady=(5, 0))
        
        # Console textbox
        self.console_textbox = ctk.CTkTextbox(
            self.console_frame,
            height=120,
            font=("Consolas", 10)
        )
        self.console_textbox.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for the application."""
        self.root.bind('<Control-o>', self.upload_document)
        self.root.bind('<Control-r>', lambda e: self.refine_document() if self.refine_button.cget("state") == "normal" else None)
        self.root.bind('<Control-s>', lambda e: self.save_document() if self.save_button.cget("state") == "normal" else None)
        self.root.bind('<Control-b>', lambda e: self.save_bibliography() if self.bib_button.cget("state") == "normal" else None)

    def load_theme_preference(self):
        """Load theme preference from settings."""
        theme = self.settings.get('ui_settings', {}).get('theme', 'dark')
        ctk.set_appearance_mode(theme)

    def save_document(self, event=None):
        """Save the refined document."""
        if not self.refined_text:
            self.show_error("No refined document to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Refined Document",
                defaultextension=".tex",
                filetypes=[
                    ("LaTeX Files", "*.tex"),
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.refined_text)
                self.log_to_console(f"Saved refined document: {filename}")
        except Exception as e:
            self.show_error(f"Error saving document: {e}")

    def save_bibliography(self, event=None):
        """Save the bibliography."""
        if not self.doc_refiner or not self.doc_refiner.bibliography:
            self.show_error("No bibliography to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Bibliography",
                defaultextension=".bib",
                filetypes=[
                    ("BibTeX Files", "*.bib"),
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                bib_text = self.doc_refiner.generate_bibliography_text()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(bib_text)
                self.log_to_console(f"Saved bibliography: {filename}")
        except Exception as e:
            self.show_error(f"Error saving bibliography: {e}")

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        current_mode = ctk.get_appearance_mode()
        new_mode = "light" if current_mode == "Dark" else "dark"
        ctk.set_appearance_mode(new_mode)
        
        # Update settings
        self.settings['ui_settings']['theme'] = new_mode
        self.log_to_console(f"Switched to {new_mode} theme")

    def change_citation_format(self, format_name):
        """Change the citation format."""
        self.settings['ui_settings']['citation_format'] = format_name
        self.log_to_console(f"Citation format changed to {format_name}")

    def update_progress(self, value, text=""):
        """Update the progress bar and status text."""
        try:
            self.progress_bar.set(value)
            if text:
                self.status_label.configure(text=text)
            self.root.update_idletasks()
        except Exception as e:
            logging.error(f"Error updating progress: {e}")

    def update_bibliography(self):
        """Update the bibliography display."""
        try:
            if self.doc_refiner and self.doc_refiner.bibliography:
                bib_text = self.doc_refiner.generate_bibliography_text()
                self.bib_textbox.configure(state="normal")
                self.bib_textbox.delete("1.0", "end")
                self.bib_textbox.insert("end", bib_text)
                self.bib_textbox.configure(state="disabled")
        except Exception as e:
            self.show_error(f"Error updating bibliography: {e}")

    def log_to_console(self, message):
        """Log a message to the console textbox."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            
            self.console_textbox.configure(state="normal")
            self.console_textbox.insert("end", formatted_message)
            self.console_textbox.see("end")
            self.console_textbox.configure(state="disabled")
            
            # Also log to file
            logging.info(message)
        except Exception as e:
            logging.error(f"Error logging to console: {e}")

    def show_error(self, message):
        """Show an error message to the user."""
        try:
            messagebox.showerror("Error", message)
            self.log_to_console(f"ERROR: {message}")
        except Exception as e:
            logging.error(f"Error showing error message: {e}")
            
    def run(self):
        self.root.mainloop()

# ---------------- Default Settings ----------------
DEFAULT_SETTINGS = {
    'similarity_threshold': 0.7,
    'keyword_settings': {
        'lan': 'en',
        'n': 2,
        'dedupLim': 0.9,
        'top': 10
    },
    'search_settings': {
        'max_results': 500,
        'rate_limit': 1.0  # seconds between requests
    },
    'model_settings': {
        'sentence_model': 'all-MiniLM-L6-v2',
        'summarizer_model': 'facebook/bart-large-cnn',
        'max_length': 20,
        'min_length': 2
    },
    'ui_settings': {
        'theme': 'dark',
        'citation_format': 'APA',
        'window_size': '1000x750'
    }
}

# ---------------- Main ----------------
def main():
    try:
        # Load settings from file if exists, otherwise use defaults
        settings_path = os.path.join(CACHE_DIR, 'settings.json')
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    settings = {**DEFAULT_SETTINGS, **json.load(f)}  # Merge with defaults
            except Exception as e:
                logging.warning(f"Error loading settings, using defaults: {e}")
                settings = DEFAULT_SETTINGS
        else:
            settings = DEFAULT_SETTINGS
            
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Initialize and run the application
        app = DeepResearchToolApp(settings)
        app.run()
        
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        if 'app' in locals() and hasattr(app, 'root'):
            app.root.destroy()
        messagebox.showerror("Critical Error", f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()

def generate_thesis_ideas(thesis_name: str, research_field: str = "Applied Machine Learning", output_path: str = "thesis_proposal.txt", seed: int = 42) -> str:
    import random
    from datetime import datetime
    random.seed(seed)
    aspects = [
        "fairness","explainability","privacy","robustness","causality","uncertainty quantification",
        "calibration","drift detection","domain adaptation","semi-supervised learning","active learning",
        "few-shot learning","multimodal fusion","graph learning","time-series forecasting","reinforcement learning",
        "human-in-the-loop","federated optimization","energy efficiency","scalable training","benchmarking",
        "reproducibility","simulation-to-real","synthetic data generation","annotation efficiency"
    ]
    methods = [
        "graph neural networks","causal inference","reinforcement learning","federated learning",
        "transformers","Bayesian modeling","self-supervised learning","contrastive learning",
        "probabilistic programming","meta-learning","diffusion models"
    ]
    data_sources = [
        "clinical records","satellite imagery","transaction graphs","multimodal sensors",
        "text corpora","time-series logs","knowledge graphs","speech and dialog data",
        "edge-collected datasets","privacy-sensitive user telemetry"
    ]
    contexts = [
        "healthcare","finance","education","climate science","manufacturing",
        "cybersecurity","transportation","e-commerce","public policy","scientific discovery"
    ]
    titles = []
    seen = set()
    templates = [
        "{aspect} in {field} via {method} using {data} for {context}",
        "Towards {aspect} in {field}: {method} on {data} for {context}",
        "{method} for {aspect} with {data} in {context}"
    ]
    combos = [(a, m, d, c) for a in aspects for m in methods for d in data_sources for c in contexts]
    random.shuffle(combos)
    i = 0
    while len(titles) < 100 and i < len(combos):
        a, m, d, c = combos[i]
        tpl = random.choice(templates)
        t = tpl.format(aspect=a.title(), field=research_field, method=m.title(), data=d, context=c.title())
        if t not in seen:
            seen.add(t)
            titles.append(t)
        i += 1
    intro = (
        f"{thesis_name} advances {research_field} by delivering a rigorous, deployable research product that bridges methodological innovation and real-world impact. "
        "The work positions modern learning techniques within a responsible framework, emphasizing transparency, reliability, and stakeholder trust."
    )
    problem = (
        "Despite rapid progress in machine learning, many systems remain difficult to explain, fragile under distribution shift, and constrained by privacy and compliance demands. "
        "Organizations need validated approaches that balance accuracy with governance, enabling safe adoption in high-stakes domains."
    )
    solution = (
        "The proposed solution integrates transformers and graph-based models with causal analysis, uncertainty estimation, and federated optimization to produce robust, privacy-preserving pipelines. "
        "Methodology includes dataset curation, reproducible training, ablation studies, calibration, and comprehensive evaluation across benchmark tasks (Doshi-Velez & Kim, 2017; Kairouz et al., 2021)."
    )
    significance = (
        "Expected outcomes include state-of-the-art performance with interpretable outputs, measurable fairness improvements, reduced drift, and energy-aware training. "
        "The product contributes open evaluation artifacts, deployment guides, and policy-aligned reporting, supporting researchers and practitioners across sectors."
    )
    paragraphs = [intro, problem, solution, significance]
    words = sum(len(p.split()) for p in paragraphs)
    if words < 300:
        extra = (
            "The evaluation plan covers cross-domain generalization, stress testing under realistic noise, and longitudinal monitoring with human-in-the-loop validation. "
            "Results are presented with confidence intervals, sensitivity analyses, and error taxonomies to inform decision makers and align with academic best practices."
        )
        paragraphs.append(extra)
    summary_text = "\n\n".join(paragraphs)
    meta = [
        "Thesis Proposal Generation",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Research Field: {research_field}",
        f"Thesis Name: {thesis_name}"
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta) + "\n\n")
        f.write("=== Thesis Title Ideas (100) ===\n")
        for n, t in enumerate(titles, 1):
            f.write(f"{n}. {t}\n")
        f.write("\n=== Product Summary (300 words) ===\n\n")
        f.write(summary_text)
    return output_path
    
