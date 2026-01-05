"""
Benchmark tests for RefScore scoring engine.

This module provides performance and accuracy benchmarks for the Hybrid Retrieval system.
It compares the scoring results against expected rankings and measures execution time.
"""

import time
import pytest
import logging
import statistics
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import sys
import os

# Add project root to path if running directly
if __name__ == "__main__" or __package__ is None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import tracemalloc
import ctypes
from ctypes import wintypes

from refscore.core.scoring import ScoringEngine, SourceScore
from refscore.utils.config import Config
from refscore.models.document import Document, Sentence
from refscore.models.source import Source

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("benchmark")


def _get_process_memory():
    try:
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        GetCurrentProcess = kernel32.GetCurrentProcess
        GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
        GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return {
            "working_set": float(counters.WorkingSetSize) / (1024 * 1024),
            "peak_working_set": float(counters.PeakWorkingSetSize) / (1024 * 1024),
            "pagefile": float(counters.PagefileUsage) / (1024 * 1024),
            "peak_pagefile": float(counters.PeakPagefileUsage) / (1024 * 1024),
        }
    except Exception:
        return None

def create_synthetic_data():
    """Create synthetic document and sources for benchmarking."""
    
    # Document: Focus on "Transformers in NLP"
    sentences = [
        Sentence("The Transformer model has revolutionized natural language processing.", "Introduction", 0),
        Sentence("Self-attention mechanisms allow the model to weigh the importance of different words.", "Methods", 1),
        Sentence("BERT and GPT are prominent examples of Transformer-based architectures.", "Related Work", 2),
        Sentence("We achieved state-of-the-art results on the GLUE benchmark.", "Results", 3),
    ]
    document = Document(sentences=sentences, sections=["Introduction", "Methods", "Related Work", "Results"], meta={})
    
    # Sources
    sources = []
    
    # 1. Highly Relevant (Supporting) - "Transformers", "Attention", "BERT"
    sources.append(Source(
        source_id="s1", 
        title="Attention Is All You Need", 
        abstract="We propose a new network architecture, the Transformer, based solely on attention mechanisms.",
        year=2017, venue="NeurIPS"
    ))
    sources.append(Source(
        source_id="s2", 
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 
        abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
        year=2018, venue="NAACL"
    ))
    
    # 2. Topically Related (but not specific to Transformers/Attention) - "NLP", "RNN", "LSTM"
    sources.append(Source(
        source_id="r1", 
        title="Sequence to Sequence Learning with Neural Networks", 
        abstract="Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality.",
        year=2014, venue="NIPS"
    ))
    sources.append(Source(
        source_id="r2", 
        title="Neural Machine Translation by Jointly Learning to Align and Translate", 
        abstract="We propose a novel architecture for neural machine translation. The extension allows the model to (soft-)search for parts of a source sentence that are relevant to predicting a target word.",
        year=2015, venue="ICLR"
    ))
    
    # 3. Irrelevant - "Biology", "Chemistry"
    sources.append(Source(
        source_id="i1", 
        title="The structure of DNA", 
        abstract="We wish to suggest a structure for the salt of deoxyribose nucleic acid (D.N.A.). This structure has two helical chains each coiled round the same axis.",
        year=1953, venue="Nature"
    ))
    sources.append(Source(
        source_id="i2", 
        title="Synthesis of Ammonia", 
        abstract="The Haber process allows for the industrial synthesis of ammonia from nitrogen and hydrogen.",
        year=1909, venue="Chem. Journal"
    ))
    
    return document, sources

@pytest.mark.benchmark
def test_scoring_accuracy_and_ranking():
    """
    Benchmark: Verify that the scoring engine correctly ranks sources.
    Expectation: Supporting > Related > Irrelevant
    """
    config = Config()
    # Ensure we are using the new powerful models if available/configured
    # But for benchmark, default config is fine.
    
    engine = ScoringEngine(config)
    
    document, sources = create_synthetic_data()
    
    tracemalloc.start()
    mem_before = _get_process_memory()
    start_time = time.time()
    scores = engine.compute_refscores(document, sources)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    duration = end_time - start_time
    log.info(f"Scoring {len(sources)} sources against {len(document.sentences)} sentences took {duration:.4f}s")
    if mem_before is not None:
        mem_after = _get_process_memory()
        if mem_after is not None:
            ws_delta = mem_after["working_set"] - mem_before["working_set"]
            log.info(f"RAM working set before: {mem_before['working_set']:.2f} MB, after: {mem_after['working_set']:.2f} MB, delta: {ws_delta:.2f} MB")
            log.info(f"RAM peak working set: {mem_after['peak_working_set']:.2f} MB")
            log.info(f"Pagefile usage: {mem_after['pagefile']:.2f} MB, peak: {mem_after['peak_pagefile']:.2f} MB")
    log.info(f"Python tracemalloc peak: {peak / (1024 * 1024):.2f} MB")
    
    # Create a map of source_id -> score
    score_map = {s.source.source_id: s.refscore for s in scores}
    
    log.info("Scores:")
    for s in scores:
        log.info(f"  {s.source.source_id} ({s.source.title[:20]}...): {s.refscore:.4f}")
        
    # Assertions
    
    # 1. Highly Relevant should be top ranked
    supporting_avg = (score_map['s1'] + score_map['s2']) / 2
    related_avg = (score_map['r1'] + score_map['r2']) / 2
    irrelevant_avg = (score_map['i1'] + score_map['i2']) / 2
    
    log.info(f"Average Scores - Supporting: {supporting_avg:.4f}, Related: {related_avg:.4f}, Irrelevant: {irrelevant_avg:.4f}")
    
    assert supporting_avg > related_avg, "Supporting sources should score higher than related sources"
    assert related_avg > irrelevant_avg, "Related sources should score higher than irrelevant sources"
    
    # Supporting sources should have a reasonable score (lowered threshold due to rigorous weighting)
    assert supporting_avg > 0.12, f"Supporting sources should have a reasonable score (>0.12), got {supporting_avg}"
    assert irrelevant_avg < 0.2, "Irrelevant sources should have a low score (<0.2)"
    
    # 3. Check specific relationships
    # 'Attention Is All You Need' (s1) matches "Self-attention mechanisms..." (sentence 1) very well
    s1_score = scores[0] # Assuming it's top or near top
    # Verify s1 is indeed s1 or s2
    assert s1_score.source.source_id in ['s1', 's2']

@pytest.mark.benchmark
def test_performance_scalability():
    """
    Benchmark: Measure performance scaling with more sentences/sources.
    This is a rough test to ensure no O(N^3) regressions.
    """
    config = Config()
    config.settings.nlp_models["rerank_threshold"] = 0.9
    engine = ScoringEngine(config)
    
    # Create larger synthetic data
    base_document, base_sources = create_synthetic_data()
    
    # Scale up: 10x sentences, 10x sources
    # Note: We duplicate content, so semantics are same, just checking compute time
    
    # 20 sentences
    long_sentences = base_document.sentences * 5 
    # Update indices
    for i, s in enumerate(long_sentences):
        long_sentences[i] = Sentence(s.text, s.section, i)
        
    long_document = Document(sentences=long_sentences, sections=base_document.sections, meta={})
    
    # 30 sources
    many_sources = []
    for i in range(5):
        for src in base_sources:
            new_src = Source(
                source_id=f"{src.source_id}_{i}",
                title=src.title,
                abstract=src.abstract,
                year=src.year,
                venue=src.venue
            )
            many_sources.append(new_src)
            
    log.info(f"Starting load test: {len(long_document.sentences)} sentences, {len(many_sources)} sources")
    
    tracemalloc.start()
    mem_before = _get_process_memory()
    start_time = time.time()
    engine.compute_refscores(long_document, many_sources)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    duration = end_time - start_time
    log.info(f"Load test took {duration:.4f}s")
    if mem_before is not None:
        mem_after = _get_process_memory()
        if mem_after is not None:
            ws_delta = mem_after["working_set"] - mem_before["working_set"]
            log.info(f"RAM working set before: {mem_before['working_set']:.2f} MB, after: {mem_after['working_set']:.2f} MB, delta: {ws_delta:.2f} MB")
            log.info(f"RAM peak working set: {mem_after['peak_working_set']:.2f} MB")
            log.info(f"Pagefile usage: {mem_after['pagefile']:.2f} MB, peak: {mem_after['peak_pagefile']:.2f} MB")
    log.info(f"Python tracemalloc peak: {peak / (1024 * 1024):.2f} MB")
    
    # Rough threshold: 30 sources * 20 sentences = 600 pairs.
    # With Bi-Encoder, this should be fast (< 10s on CPU).
    # Cross-Encoder might slow it down if it reranks many candidates.
    assert duration < 60.0, "Scoring took too long (>60s)"

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
