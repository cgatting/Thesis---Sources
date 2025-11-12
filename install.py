#!/usr/bin/env python3
"""
Installation script for RefScore academic application.

This script handles the installation of dependencies and setup of the application.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description="Running command"):
    """Run a shell command and handle errors."""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        result = run_command(
            f"{sys.executable} -m pip install -r {requirements_file}",
            "Installing dependencies from requirements.txt"
        )
        if result is None:
            return False
    else:
        print("✗ requirements.txt not found")
        return False
    
    return True


def install_optional_dependencies():
    """Install optional dependencies for enhanced functionality."""
    print("\nInstalling optional dependencies...")
    
    optional_deps = [
        "sentence-transformers",
        "spacy",
        "bibtexparser",
        "PyMuPDF",
        "scikit-learn",
    ]
    
    for dep in optional_deps:
        result = run_command(
            f"{sys.executable} -m pip install {dep}",
            f"Installing {dep}"
        )
        if result is None:
            print(f"⚠ Optional dependency {dep} failed to install (application will still work with fallbacks)")
    
    return True


def setup_spacy():
    """Setup spaCy with English model."""
    print("\nSetting up spaCy...")
    
    # Try to install spaCy English model
    result = run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    )
    
    if result is None:
        print("⚠ spaCy English model download failed (NER functionality may be limited)")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    # Create user config directory
    config_dir = Path.home() / ".config" / "refscore"
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created config directory: {config_dir}")
    except Exception as e:
        print(f"⚠ Failed to create config directory: {e}")
    
    # Create default output directory
    output_dir = Path.home() / "Documents" / "RefScoreResults"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created default output directory: {output_dir}")
    except Exception as e:
        print(f"⚠ Failed to create output directory: {e}")
    
    return True


def create_sample_files():
    """Create sample files for testing."""
    print("\nCreating sample files...")
    
    sample_dir = Path.home() / "Documents" / "RefScoreSamples"
    try:
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample LaTeX file
        latex_content = r"""
\documentclass{article}
\title{Sample Research Paper}
\author{Test Author}
\date{2024}

\begin{document}
\maketitle

\section{Introduction}
Machine learning has become increasingly important in modern research.
Deep learning techniques have shown remarkable success in various domains.

\section{Methods}
We employed convolutional neural networks with attention mechanisms.
The training process utilized stochastic gradient descent optimization.

\section{Results}
Our model achieved 95\% accuracy on the test dataset.
The F1 score improved by 15\% compared to baseline methods.

\section{Conclusion}
This research demonstrates the effectiveness of deep learning approaches.
Future work should explore additional applications and extensions.

\end{document}
"""
        
        latex_file = sample_dir / "sample_paper.tex"
        latex_file.write_text(latex_content.strip())
        print(f"✓ Created sample LaTeX file: {latex_file}")
        
        # Sample BibTeX file
        bibtex_content = """
@article{smith2023,
  title={Deep Learning for Computer Vision},
  author={Smith, John and Doe, Jane},
  journal={Journal of Machine Learning},
  volume={25},
  number={3},
  pages={123--145},
  year={2023},
  doi={10.1234/jml.2023.123}
}

@inproceedings{jones2022,
  title={Attention Mechanisms in Neural Networks},
  author={Jones, Alice and Brown, Bob},
  booktitle={Proceedings of the International Conference on AI},
  pages={456--478},
  year={2022},
  doi={10.5678/ica.2022.456}
}

@book{wilson2021,
  title={Machine Learning Fundamentals},
  author={Wilson, Charlie},
  year={2021},
  publisher={Academic Press}
}
"""
        
        bibtex_file = sample_dir / "sample_references.bib"
        bibtex_file.write_text(bibtex_content.strip())
        print(f"✓ Created sample BibTeX file: {bibtex_file}")
        
        # Sample JSON file
        json_content = """
[
  {
    "title": "Transformer Models for Sequence Processing",
    "creators": [
      {"firstName": "Emily", "lastName": "Davis"},
      {"firstName": "Frank", "lastName": "Miller"}
    ],
    "date": "2023-08-15",
    "publicationTitle": "Neural Computation",
    "DOI": "10.9876/nc.2023.789",
    "abstractNote": "This paper explores transformer models for sequence processing tasks."
  },
  {
    "title": "Convolutional Neural Networks: A Comprehensive Review",
    "creators": [
      {"firstName": "Grace", "lastName": "Lee"}
    ],
    "date": "2022-12-01",
    "publicationTitle": "AI Review",
    "DOI": "10.5432/air.2022.321",
    "abstractNote": "A comprehensive review of CNN architectures and applications."
  }
]
"""
        
        json_file = sample_dir / "sample_zotero.json"
        json_file.write_text(json_content.strip())
        print(f"✓ Created sample JSON file: {json_file}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Failed to create sample files: {e}")
        return False


def test_installation():
    """Test the installation by importing key modules."""
    print("\nTesting installation...")
    
    try:
        # Test basic imports
        import refscore
        print(f"✓ RefScore package imported successfully (version {refscore.__version__})")
        
        # Test core functionality
        from refscore.core.analyzer import RefScoreAnalyzer
        from refscore.utils.config import Config
        
        # Create a simple test
        config = Config()
        analyzer = RefScoreAnalyzer(config)
        
        print("✓ Core components imported and initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False


def main():
    """Main installation function."""
    print("="*60)
    print("RefScore Academic Application - Installation Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        print("\n✗ Installation failed: Python version incompatible")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Installation failed: Dependencies could not be installed")
        return 1
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Setup spaCy
    setup_spacy()
    
    # Create directories
    create_directories()
    
    # Create sample files
    create_sample_files()
    
    # Test installation
    if not test_installation():
        print("\n⚠ Installation completed but some tests failed")
        print("The application may still work, but some features might be limited")
    
    print("\n" + "="*60)
    print("Installation completed successfully!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Run the GUI application: python -m refscore.main")
    print("2. Run the CLI: python -m refscore.cli --help")
    print("3. Check sample files in: ~/Documents/RefScoreSamples/")
    print("4. Results will be saved to: ~/Documents/RefScoreResults/")
    
    print("\nFor more information, see the README.md file.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())