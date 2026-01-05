"""
Command line interface for RefScore academic application.

This module provides the command line interface for running RefScore
analysis from the terminal.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .core.analyzer import RefScoreAnalyzer
from .utils.config import Config
from .utils.exceptions import RefScoreError, ValidationError, ProcessingError


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging for CLI.
    
    Args:
        level: Logging level
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='[%(levelname)s] %(message)s',
        stream=sys.stdout,
        force=True
    )
    logging.getLogger("quantulum3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    try:
        import warnings
        if numeric_level >= logging.ERROR:
            warnings.filterwarnings("ignore")
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            except Exception:
                pass
    except Exception:
        pass


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="RefScore Academic - Analyze academic references against documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with LaTeX document and BibTeX sources
  python -m refscore.cli --doc paper.tex --sources refs.bib

  # Multiple source files with custom output directory
  python -m refscore.cli --doc paper.pdf --sources refs.bib zotero.json --out results/

  # Custom scoring weights
  python -m refscore.cli --doc paper.tex --sources refs.bib --weights alignment=0.5 entities=0.3

  # Verbose output
  python -m refscore.cli --doc paper.tex --sources refs.bib --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--doc", "-d",
        required=True,
        help="Path to academic document (.tex or .pdf)"
    )
    
    parser.add_argument(
        "--sources", "-s",
        nargs='+',
        required=True,
        help="Paths to source files (.bib, .json, .csv, .txt)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--out", "-o",
        default="refscore_results",
        help="Output directory for results (default: refscore_results)"
    )
    
    parser.add_argument(
        "--weights",
        nargs='*',
        help="Custom scoring weights as key=value pairs (e.g., alignment=0.5 entities=0.3)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of weakest sentences to report (default: 10)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    parser.add_argument(
        "--train-weights",
        action="store_true",
        help="Train optimized scoring weights using provided doc and sources"
    )
    parser.add_argument(
        "--apply-preset",
        help="Apply a saved weights preset by id or name"
    )
    parser.add_argument(
        "--preset-name",
        help="Preset name for saving trained weights"
    )
    parser.add_argument(
        "--search-space",
        nargs='*',
        help="Weight ranges as key=min:max:step (e.g., alignment=0.5:0.8:0.1)"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List saved weight presets"
    )
    parser.add_argument(
        "--rollback-preset",
        action="store_true",
        help="Rollback to previous applied preset"
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        default=True,
        help="Minimal output: suppress logs and show only progress bar/errors (Default: True)"
    )
    
    return parser.parse_args()


def parse_weights(weights_list: Optional[List[str]]) -> Optional[dict]:
    """
    Parse weight arguments into dictionary.
    
    Args:
        weights_list: List of weight strings (key=value)
        
    Returns:
        Dictionary of weights or None
    """
    if not weights_list:
        return None
    
    weights = {}
    for weight_str in weights_list:
        try:
            key, value = weight_str.split('=')
            key = key.strip()
            value = float(value.strip())
            
            valid_keys = ["alignment", "entities", "number_unit", "method_metric", "recency", "authority"]
            if key not in valid_keys:
                raise ValidationError(f"Invalid weight key: {key}. Valid keys: {valid_keys}")
            
            if not (0.0 <= value <= 1.0):
                raise ValidationError(f"Weight value must be between 0.0 and 1.0: {value}")
            
            weights[key] = value
            
        except ValueError:
            raise ValidationError(f"Invalid weight format: {weight_str}. Use key=value format.")
    
    return weights


def run_analysis(args: argparse.Namespace) -> None:
    """
    Run the RefScore analysis.
    
    Args:
        args: Parsed command line arguments
    """
    logger = logging.getLogger(__name__)
    
    try:
        if args.apply_preset:
            from .training.weights_trainer import WeightsTrainer
            wt = WeightsTrainer()
            applied = wt.apply_preset(args.apply_preset)
            if not applied:
                logger.warning(f"Failed to apply preset: {args.apply_preset}")
            else:
                logger.info(f"Applied preset: {args.apply_preset}")
        if args.list_presets:
            from .training.weights_trainer import WeightsTrainer
            wt = WeightsTrainer()
            presets = wt.list_presets()
            print(json.dumps(presets, indent=2))
            return
        if args.rollback_preset:
            from .training.weights_trainer import WeightsTrainer
            wt = WeightsTrainer()
            prev = wt.rollback()
            logger.info(f"Rolled back to: {prev}")
            return
        if args.train_weights:
            from .training.weights_trainer import WeightsTrainer
            def parse_space(items: Optional[List[str]]):
                space = {}
                if not items:
                    return space
                for it in items:
                    k, rng = it.split('=')
                    lo, hi, st = rng.split(':')
                    space[k.strip()] = (float(lo), float(hi), float(st))
                return space
            wt = WeightsTrainer()
            space = parse_space(args.search_space)
            jobs = [(args.doc, args.sources)]
            best_w, metrics = wt.train(jobs, space or {
                "alignment": (0.4, 0.8, 0.1),
                "entities": (0.05, 0.2, 0.05),
                "number_unit": (0.05, 0.2, 0.05),
                "method_metric": (0.05, 0.15, 0.05),
                "recency": (0.02, 0.1, 0.02),
                "authority": (0.02, 0.1, 0.02),
            })
            logger.info(f"Trained weights: {best_w}")
            logger.info(f"Training metrics: {metrics}")
            if args.preset_name:
                pid = wt.save_preset(args.preset_name, best_w, metrics, {
                    "search_space": space or "default",
                    "doc": args.doc,
                    "sources": args.sources,
                })
                logger.info(f"Saved preset: {pid}")
            return
        # Load configuration
        config = Config(args.config) if args.config else Config()
        
        # Apply custom weights if provided
        if args.weights:
            weights = parse_weights(args.weights)
            if weights:
                config.set_scoring_weights(weights)
                logger.info(f"Using custom scoring weights: {weights}")
        
        # Initialize analyzer
        analyzer = RefScoreAnalyzer(config)
        
        # Load document
        logger.info(f"Loading document: {args.doc}")
        document = analyzer.load_document(args.doc)
        logger.info(f"Document loaded: {len(document.sentences)} sentences, {len(document.sections)} sections")
        
        # Load sources
        logger.info(f"Loading sources from {len(args.sources)} files")
        sources = analyzer.load_sources(args.sources)
        logger.info(f"Loaded {len(sources)} unique sources")
        
        # Compute scores
        logger.info("Computing RefScores...")
        scores = analyzer.compute_scores(document, sources)
        logger.info(f"Scoring completed. Top score: {scores[0].refscore:.4f}")
        
        # Generate reports
        logger.info(f"Generating reports in: {args.out}")
        reports = analyzer.generate_report(document, sources, scores, args.out)
        
        # Display results
        if not args.quiet:
            print_results(scores, document, args.top_k, args.out)
        
        logger.info(f"Analysis completed successfully. Results saved to {args.out}")
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        sys.exit(1)
    except RefScoreError as e:
        logger.error(f"RefScore error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)


def print_results(scores: List, document, top_k: int, out_dir: str) -> None:
    """
    Print analysis results to console.
    
    Args:
        scores: List of source scores
        document: Parsed document
        top_k: Number of top results to show
        out_dir: Output directory path
    """
    print("\n" + "="*80)
    print("REFSCORE ANALYSIS RESULTS")
    print("="*80)
    
    # Top sources
    print(f"\nTop {min(5, len(scores))} Sources:")
    print("-" * 40)
    for i, score in enumerate(scores[:5], 1):
        source = score.source
        print(f"{i}. {source.title[:60]}...")
        print(f"   Score: {score.refscore:.4f}")
        print(f"   Authors: {source.author_string}")
        print(f"   Year: {source.year or 'Unknown'}")
        print(f"   Venue: {source.venue or 'Unknown'}")
        print()
    
    # Document statistics
    print(f"\nDocument Statistics:")
    print("-" * 40)
    print(f"Total sentences: {len(document.sentences)}")
    print(f"Total sections: {len(document.sections)}")
    print(f"Sections: {', '.join(document.sections[:5])}")
    if len(document.sections) > 5:
        print(f"  ... and {len(document.sections) - 5} more")
    
    # Coverage summary
    section_stats = document.get_section_stats()
    print(f"\nSection Coverage:")
    print("-" * 40)
    for section, count in list(section_stats.items())[:5]:
        print(f"{section}: {count} sentences")
    
    print(f"\nResults saved to:")
    print("-" * 40)
    print(f"Sources ranking: {out_dir}/sources_ranked.json")
    print(f"Section coverage: {out_dir}/section_coverage.json")
    print(f"Weak sentences: {out_dir}/gaps.json")
    print("="*80)


def main() -> None:
    """
    Main CLI entry point.
    """
    args = parse_arguments()
    
    # Set up logging
    # Prioritize verbose/quiet flags over minimal default
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    elif getattr(args, "minimal", True):
        log_level = "ERROR"
    else:
        log_level = "INFO"
    
    setup_logging(log_level)
    
    # Run analysis
    run_analysis(args)


if __name__ == "__main__":
    main()