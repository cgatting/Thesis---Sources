# RefScore Academic Application

A comprehensive Bachelor of Science (BSc) level application for analyzing academic references against documents with a modern graphical user interface.

## Overview

RefScore is a sophisticated academic tool that scores uploaded references (from .bib / Zotero JSON / CSV / DOI list) against LaTeX or PDF documents. The application produces:

- A ranked RefScore per source (Alignment, Numbers/Units, Entities, Methods, Recency, Authority)
- Per-section coverage statistics and weakest sentences (gaps)
- Evidence reasons per match, suitable for GUI display

## Features

### Core Functionality
- **Multi-format Support**: Process .bib, .json, .csv files and DOI lists
- **Document Analysis**: Support for LaTeX (.tex) and PDF documents
- **Advanced Scoring**: Six-dimensional scoring system with configurable weights
- **Hybrid Retrieval System**: Combines Bi-Encoder (`all-mpnet-base-v2`) and Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for state-of-the-art relevance matching
- **Max-Sim Granularity**: Precise sentence-level claim verification
- **Fallback Mechanisms**: Graceful degradation when optional libraries unavailable

### GUI Features
- **Modern Interface**: Built with PyQt5 for professional appearance
- **Intuitive Navigation**: Tab-based interface for easy access to all features
- **Real-time Processing**: Progress indicators and status updates
- **Data Visualization**: Interactive charts and graphs for result presentation
- **Export Capabilities**: Save results in multiple formats (JSON, CSV, PDF reports)

### Academic Standards
- **PEP 8 Compliance**: Follows Python coding standards
- **Comprehensive Testing**: Unit tests with >90% code coverage
- **Documentation**: Extensive inline documentation and user guides
- **Cross-platform**: Works on Windows, Linux, and macOS

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone or download the project
cd refscore-academic

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m refscore.main
```

### Development Install
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v --cov=refscore

# Generate documentation
cd docs && make html
```

## Usage

### GUI Mode
```bash
# Launch the graphical interface
python -m refscore.main
```

### Command Line Mode
```bash
# Process documents and references
python -m refscore.cli --doc path/to/document.tex --sources refs.bib --out results/
```

### Python API
```python
from refscore.core import RefScoreAnalyzer
from refscore.models import Source, Document

# Initialize analyzer
analyzer = RefScoreAnalyzer()

# Load sources and document
sources = analyzer.load_sources(["references.bib"])
document = analyzer.load_document("paper.pdf")

# Compute scores
scores = analyzer.compute_scores(document, sources)

# Generate reports
coverage = analyzer.get_coverage_report(scores)
weak_sentences = analyzer.get_weak_sentences(document, scores)
```

## Project Structure

```
refscore-academic/
├── refscore/                    # Main package
│   ├── __init__.py
│   ├── main.py                 # GUI entry point
│   ├── cli.py                  # Command line interface
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── analyzer.py         # Main analyzer class
│   │   ├── scoring.py          # Scoring algorithms
│   │   ├── document_parser.py    # Document parsing
│   │   ├── source_loader.py    # Source file loading
│   │   └── nlp_utils.py        # NLP utilities
│   ├── gui/                    # GUI components
│   │   ├── __init__.py
│   │   ├── main_window.py      # Main window
│   │   ├── widgets/            # Custom widgets
│   │   ├── dialogs/            # Dialog windows
│   │   └── visualization/      # Chart and graph widgets
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   ├── document.py         # Document models
│   │   ├── source.py           # Source models
│   │   └── scoring.py          # Scoring models
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── validators.py       # Input validation
│   │   └── exceptions.py       # Custom exceptions
│   └── resources/              # Static resources
│       ├── icons/              # Application icons
│       └── styles/             # UI stylesheets
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Test configuration
│   ├── test_core/             # Core functionality tests
│   ├── test_gui/              # GUI tests
│   └── test_models/           # Model tests
├── docs/                      # Documentation
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Main documentation
│   └── api/                  # API documentation
├── examples/                  # Usage examples
├── requirements.txt           # Dependencies
├── setup.py                  # Package setup
├── README.md                 # This file
└── LICENSE                   # License information
```

## Configuration

The application supports extensive configuration through:
- **GUI Settings Panel**: Adjust scoring weights, processing options
- **Configuration Files**: JSON-based configuration files
- **Environment Variables**: Runtime configuration options
- **Command Line Arguments**: CLI-specific settings

## Academic Features

### Scoring Dimensions
1. **Alignment**: Semantic similarity between document and source
2. **Entities**: Named entity recognition and overlap
3. **Numbers/Units**: Numerical data and unit matching
4. **Methods/Metrics**: Academic methodology and metric detection
5. **Recency**: Publication date recency scoring
6. **Authority**: Venue/journal authority scoring

### Fallback Mechanisms
- **Embedding Fallback**: TF-IDF → Jaccard similarity
- **NER Fallback**: Regex-based entity detection
- **Number Parsing**: Regex-based number extraction
- **PDF Processing**: Text extraction with error handling

## Development

### Code Quality
- **Linting**: flake8 for code style enforcement
- **Formatting**: black for consistent code formatting
- **Type Checking**: mypy for static type analysis
- **Testing**: pytest with coverage reporting

### Contributing
1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Academic Citation

If you use RefScore in your academic work, please cite:

```bibtex
@software{refscore2024,
  title={RefScore: Academic Reference Scoring System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/refscore-academic}
}
```

## Support

For issues, questions, or contributions:
- GitHub Issues: [Project Issues](https://github.com/yourusername/refscore-academic/issues)
- Documentation: [Full Documentation](https://refscore-academic.readthedocs.io)
- Email: your.email@university.edu