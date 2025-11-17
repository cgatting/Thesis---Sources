"""
Scoring engine for RefScore academic application.

This module implements the core scoring algorithms used to evaluate
the relevance and coverage of academic sources against documents.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set

from ..models.document import Document, Sentence
from ..models.source import Source
from ..models.scoring import SourceScore, RefEvidence
from ..utils.config import Config


log = logging.getLogger(__name__)


class ScoringEngine:
    """
    Scoring engine for computing RefScores between documents and sources.
    
    This class implements the core scoring algorithms including:
    - Semantic alignment using embeddings or TF-IDF
    - Entity overlap detection
    - Number and unit matching
    - Method and metric identification
    - Recency scoring
    - Authority scoring
    """
    
    # Default scoring weights
    DEFAULT_WEIGHTS = {
        "alignment": 0.45,
        "number_unit": 0.20,
        "entities": 0.15,
        "method_metric": 0.10,
        "recency": 0.07,
        "authority": 0.03,
    }
    
    # Academic method/metric tokens
    METHOD_TOKENS: Set[str] = {
        "svm", "hmm", "hmm-based", "bert", "roberta", "xgboost", "random forest", "shap", "lime",
        "ablation", "bayesian", "rl", "reinforcement", "regression", "classification", "segmentation",
        "cosine", "euclidean", "kmeans", "dbscan", "lstm", "gru", "transformer", "attention",
        "precision", "recall", "f1", "accuracy", "auc", "roc", "mae", "rmse", "mape", "cross-validation",
        "benchmark", "dataset", "corpus", "experiment", "controlled study", "user study",
        "svd", "pca", "anova", "t-test", "manova", "bayes", "cnn", "gan", "resnet", "vgg",
        "sentence transformer", "word2vec", "fasttext", "topic model", "lda"
    }
    
    # Scientific units
    UNITS: List[str] = [
        "%", "percent", "ms", "s", "sec", "seconds", "minutes", "hours", "hz", "khz", "mhz", "ghz",
        "v", "mv", "ma", "a", "w", "kw", "db", "°c", "c", "kelvin", "k", "gb", "mb", "kb",
        "mm", "cm", "m", "km", "mg", "g", "kg", "°f", "mph", "km/h", "n", "nm", "µm", "mmhg",
        "mbps", "kbps", "fps"
    ]
    
    # High-authority venues
    HIGH_AUTHORITY_VENUES: Set[str] = {
        "nature", "science", "neurips", "icml", "acl", "emnlp", "cvpr", "iclr"
    }
    
    # Low-authority venues
    LOW_AUTHORITY_VENUES: Set[str] = {
        "arxiv", "workshop", "preprint"
    }
    
    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the scoring engine.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.weights = self.DEFAULT_WEIGHTS.copy()
        try:
            if self.config is not None:
                self.weights = self.config.get_scoring_weights()
        except Exception:
            pass
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        log.info("Scoring engine initialized")
    
    def _initialize_nlp_components(self) -> None:
        """Initialize optional NLP components with fallbacks."""
        # Embeddings
        self.embedder = None
        self.tfidf_vectorizer = None
        self.cosine_similarity = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("Loaded SentenceTransformer all-MiniLM-L6-v2")
        except Exception:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                self.cosine_similarity = cosine_similarity
                log.info("Using TF-IDF fallback for embeddings (1-2 grams)")
            except Exception:
                log.warning("No embedding library available; using Jaccard fallback")
        
        # NER
        self.nlp = None
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
                log.info("Loaded spaCy en_core_web_sm for NER")
            except Exception:
                self.nlp = spacy.blank("en")
                log.info("Using spaCy blank model for tokenization")
        except Exception:
            log.info("spaCy not available; using regex fallback for NER")
        
        # Number parsing
        self.quant_parser = None
        try:
            from quantulum3 import parser as quant_parser
            self.quant_parser = quant_parser
            log.info("Loaded quantulum3 for number/unit parsing")
        except Exception:
            log.info("quantulum3 not available; using regex fallback")
    
    def compute_refscores(self, document: Document, sources: List[Source], 
                         weights: Optional[Dict[str, float]] = None) -> List[SourceScore]:
        """
        Compute RefScores for all sources against the document.
        
        Args:
            document: Parsed academic document
            sources: List of academic sources
            weights: Optional custom scoring weights
            
        Returns:
            List of SourceScore objects sorted by score (descending)
        """
        if weights is None:
            weights = self.weights
        
        log.info(f"Computing RefScores for {len(sources)} sources with {len(document.sentences)} sentences")
        
        scores: List[SourceScore] = []
        for source in sources:
            per_sentence: Dict[int, RefEvidence] = {}
            
            for sentence in document.sentences:
                evidence = self._score_pair(sentence.text, source, weights)
                per_sentence[sentence.idx] = evidence
            
            # Calculate overall score as average of sentence scores
            sentence_scores = [self._weighted_score(ev, weights) for ev in per_sentence.values()]
            refscore = sum(sentence_scores) / max(1, len(sentence_scores))
            
            scores.append(SourceScore(source, round(refscore, 4), per_sentence))
        
        # Sort by score in descending order
        scores.sort(key=lambda s: s.refscore, reverse=True)
        
        log.info(f"Scoring completed. Top score: {scores[0].refscore:.4f}")
        return scores
    
    def _score_pair(self, sentence: str, source: Source, weights: Dict[str, float]) -> RefEvidence:
        """
        Score a single sentence against a source.
        
        Args:
            sentence: Document sentence text
            source: Academic source
            weights: Scoring weights
            
        Returns:
            RefEvidence object with scoring results
        """
        source_text = f"{source.title}. {source.abstract or ''}"
        
        # Compute individual scores
        alignment = self._alignment_score(sentence, source_text)
        entities = self._entity_overlap(sentence, source_text)
        number_unit, nreasons = self._number_unit_match(sentence, source_text)
        method_metric = self._method_metric_overlap(sentence, source_text)
        recency = self._recency_score(source.year)
        authority = self._authority_score(source.venue)
        
        # Build reasons list
        reasons = []
        if alignment >= 0.6:
            reasons.append(f"semantic match {alignment:.2f}")
        if entities >= 0.3:
            reasons.append(f"entity overlap {entities:.2f}")
        reasons.extend(nreasons)
        if method_metric >= 0.2:
            reasons.append("method/metric overlap")
        if recency >= 0.5:
            reasons.append("recent source")
        if authority >= 0.8:
            reasons.append("high-authority venue")
        
        return RefEvidence(alignment, entities, number_unit, method_metric, recency, authority, reasons)
    
    def _weighted_score(self, evidence: RefEvidence, weights: Dict[str, float]) -> float:
        """Calculate weighted score from evidence."""
        return (
            weights["alignment"] * evidence.alignment +
            weights["entities"] * evidence.entities +
            weights["number_unit"] * evidence.number_unit +
            weights["method_metric"] * evidence.method_metric +
            weights["recency"] * evidence.recency +
            weights["authority"] * evidence.authority
        )
    
    def _alignment_score(self, text1: str, text2: str) -> float:
        """Compute semantic alignment score between two texts."""
        # Normalize common acronyms/phrases to improve lexical similarity
        def _normalize(t: str) -> str:
            t = t.replace("NLP", "natural language processing").replace("nlp", "natural language processing")
            return t
        text1 = _normalize(text1)
        text2 = _normalize(text2)
        try:
            if self.embedder is not None:
                # Use sentence transformers
                embeddings = self.embedder.encode([text1, text2], normalize_embeddings=True)
                similarity = float(max(0.0, min(1.0, (embeddings[0] @ embeddings[1]))))
                return similarity
        except Exception:
            pass
        
        try:
            if self.tfidf_vectorizer is not None and self.cosine_similarity is not None:
                # Use TF-IDF
                vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
                similarity = self.cosine_similarity(vectors[0:1], vectors[1:2])[0, 0]
                return float(max(0.0, min(1.0, similarity)))
        except Exception:
            pass
        
        # Fallback to Jaccard similarity
        return self._jaccard_similarity(text1, text2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for similarity computation."""
        import re
        
        # Convert to lowercase and remove non-alphanumeric characters
        text = re.sub(r'[^a-z0-9%\. ]+', ' ', text.lower())
        tokens = text.split()
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with',
            'from', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'being', 'been',
            'this', 'that', 'these', 'those', 'it', 'its', 'into', 'through',
            'over', 'under', 'above', 'below', 'up', 'down', 'about', 'between',
            'during', 'before', 'after', 'while', 'again', 'further', 'then', 'once',
            'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        return [token for token in tokens if token and token not in stop_words]
    
    def _entity_overlap(self, text1: str, text2: str) -> float:
        """Compute entity overlap score between two texts."""
        entities1 = self._extract_entities(text1)
        entities2 = self._extract_entities(text2)
        
        if not entities1 or not entities2:
            return 0.0
        
        return self._jaccard_similarity(' '.join(entities1), ' '.join(entities2))
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        
        try:
            if self.nlp is not None:
                # Use spaCy NER
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART", "NORP", "GPE", "PERSON", "EVENT"}:
                        entities.append(ent.text.lower())
        except Exception:
            pass
        
        # Fallback to regex-based entity detection
        # Find capitalized words (proper nouns)
        capitalized = re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,})\b", text)
        # Find acronyms
        acronyms = re.findall(r"\b([A-Z]{2,})\b", text)
        
        entities.extend([entity.lower() for entity in capitalized + acronyms])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _number_unit_match(self, sentence: str, source: str, rel_tol: float = 0.07) -> Tuple[float, List[str]]:
        """Match numbers and units between sentence and source."""
        try:
            if self.config is not None:
                proc = self.config.get_processing_config()
                rel_tol = float(proc.get("numeric_rel_tol", rel_tol))
        except Exception:
            pass
        sentence_pairs = self._extract_numbers_units(sentence)
        source_pairs = self._extract_numbers_units(source)
        
        if not sentence_pairs or not source_pairs:
            return 0.0, []
        
        best_score = 0.0
        reasons = []
        
        for s_val, s_unit in sentence_pairs:
            for r_val, r_unit in source_pairs:
                # Check unit compatibility
                if s_unit and r_unit and s_unit != r_unit:
                    continue
                
                # Calculate relative difference
                if r_val == 0:
                    continue
                
                rel_diff = abs(s_val - r_val) / abs(r_val)
                score = 1.0 if rel_diff <= rel_tol else max(0.0, 1.0 - rel_diff)
                
                if score > best_score:
                    best_score = score
                    unit_str = f" {s_unit}" if s_unit else ""
                    reasons = [f"numeric match {s_val}≈{r_val}{unit_str} (rel={rel_diff:.2f})"]
        
        return best_score, reasons
    
    def _extract_numbers_units(self, text: str) -> List[Tuple[float, Optional[str]]]:
        """Extract numbers and units from text."""
        pairs = []
        
        try:
            if self.quant_parser is not None:
                # Use quantulum3
                quantities = self.quant_parser.parse(text)
                for quant in quantities:
                    try:
                        value = float(quant.value)
                    except Exception:
                        continue
                    
                    unit = None
                    if hasattr(quant, "unit") and quant.unit:
                        unit = getattr(quant.unit, "name", None) or getattr(quant.unit, "entity", None)
                    
                    pairs.append((value, unit.lower() if unit else None))
                
                if pairs:
                    return pairs
        except Exception:
            pass
        
        # Fallback to regex
        numbers = re.findall(r"(?<!\w)(\d+(?:\.\d+)?)", text)
        units_pattern = r"\b(" + "|".join(re.escape(unit) for unit in self.UNITS) + r")\b"
        units = re.findall(units_pattern, text.lower())
        
        # Naive pairing: use first unit for all numbers if available
        unit = units[0].lower() if units else None
        for num_str in numbers:
            try:
                value = float(num_str)
                pairs.append((value, unit))
            except ValueError:
                continue
        
        return pairs
    
    def _method_metric_overlap(self, text1: str, text2: str) -> float:
        """Compute method and metric overlap score."""
        tokens1 = set(self._tokenize(text1.lower()))
        tokens2 = set(self._tokenize(text2.lower()))
        
        hits = 0
        for method_token in self.METHOD_TOKENS:
            parts = method_token.split()
            if len(parts) == 1:
                # Single token match
                if method_token in tokens1 and method_token in tokens2:
                    hits += 1
            else:
                # Phrase match
                if method_token in text1.lower() and method_token in text2.lower():
                    hits += 1
        
        # Normalize to [0, 1] range
        return min(1.0, hits / 6.0)
    
    def _recency_score(self, year: Optional[int], half_life: int = 6) -> float:
        """Compute recency score based on publication year."""
        if not year:
            return 0.0
        
        age = max(0, datetime.now().year - year)
        # Exponential decay: score = exp(-ln(2) * age / half_life)
        return math.exp(-math.log(2) * (age / max(1e-6, half_life)))
    
    def _authority_score(self, venue: str) -> float:
        """Compute authority score based on publication venue."""
        if not venue:
            return 0.0
        
        venue_lower = venue.lower().strip()
        
        # Check for high-authority venues
        if any(keyword in venue_lower for keyword in self.HIGH_AUTHORITY_VENUES):
            return 1.0
        
        # Check for low-authority venues
        if any(keyword in venue_lower for keyword in self.LOW_AUTHORITY_VENUES):
            return 0.35
        
        # Default medium authority
        return 0.6
    
    def section_coverage_report(self, document: Document, source_scores: List[SourceScore]) -> Dict[str, Any]:
        """
        Generate section-wise coverage report.
        
        Args:
            document: Parsed academic document
            source_scores: List of source scores
            
        Returns:
            Dictionary containing coverage statistics
        """
        # Find best evidence score for each sentence
        best_per_sentence: Dict[int, float] = {}
        
        for score in source_scores:
            for sentence_idx, evidence in score.per_sentence.items():
                weighted_score = self._weighted_score(evidence, self.weights)
                if (sentence_idx not in best_per_sentence or 
                    weighted_score > best_per_sentence[sentence_idx]):
                    best_per_sentence[sentence_idx] = weighted_score
        
        # Group by section
        section_scores: Dict[str, List[float]] = defaultdict(list)
        for sentence in document.sentences:
            score = best_per_sentence.get(sentence.idx, 0.0)
            section_scores[sentence.section].append(score)
        
        # Calculate statistics
        section_stats = {
            section: {
                "avg_support": round(sum(scores) / max(1, len(scores)), 3),
                "count": len(scores),
                "min_support": round(min(scores), 3) if scores else 0.0,
                "max_support": round(max(scores), 3) if scores else 0.0,
            }
            for section, scores in section_scores.items()
        }
        
        return {
            "sections": document.sections,
            "section_stats": section_stats,
            "overall_coverage": round(
                sum(sum(scores) for scores in section_scores.values()) / 
                max(1, sum(len(scores) for scores in section_scores.values())), 3
            )
        }
    
    def weakest_sentences(self, document: Document, source_scores: List[SourceScore], 
                         top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Identify sentences with weakest source support.
        
        Args:
            document: Parsed academic document
            source_scores: List of source scores
            top_k: Number of weakest sentences to return
            
        Returns:
            List of dictionaries containing weak sentence information
        """
        # Find best evidence for each sentence
        best_per_sentence: Dict[int, Tuple[float, Optional[Source], Optional[RefEvidence]]] = {}
        
        for score in source_scores:
            for sentence_idx, evidence in score.per_sentence.items():
                weighted_score = self._weighted_score(evidence, self.weights)
                if (sentence_idx not in best_per_sentence or 
                    weighted_score > best_per_sentence[sentence_idx][0]):
                    best_per_sentence[sentence_idx] = (weighted_score, score.source, evidence)
        
        # Sort sentences by support score (ascending)
        sentence_items: List[Tuple[float, Sentence]] = []
        for sentence in document.sentences:
            score = best_per_sentence.get(sentence.idx, (0.0, None, None))[0]
            sentence_items.append((score, sentence))
        
        sentence_items.sort(key=lambda x: x[0])
        
        # Format results
        results = []
        for score, sentence in sentence_items[:top_k]:
            _, best_source, best_evidence = best_per_sentence.get(sentence.idx, (0.0, None, None))
            
            results.append({
                "idx": sentence.idx,
                "section": sentence.section,
                "support": round(score, 3),
                "sentence": sentence.text,
                "reasons": best_evidence.reasons if best_evidence else [],
                "best_source": best_source.source_id if best_source else None,
            })
        
        return results
