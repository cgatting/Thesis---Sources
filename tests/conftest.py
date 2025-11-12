"""
Test suite initialization for RefScore academic application.

This module provides the main test configuration and fixtures
for the RefScore test suite.
"""

import pytest
import tempfile
import logging
from pathlib import Path

from refscore.utils.config import Config


# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='[TEST] %(levelname)s - %(name)s - %(message)s'
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = Config()
    # Override with test-specific settings
    config.settings.processing["timeout_seconds"] = 30
    config.settings.processing["max_workers"] = 2
    return config


@pytest.fixture
def sample_latex_content():
    """Sample LaTeX content for testing."""
    return r"""
\documentclass{article}
\title{Sample Research Paper}
\author{John Doe}
\date{2024}

\begin{document}
\maketitle

\section{Introduction}
This is the introduction section. We discuss the background and motivation for our research.
Machine learning has become increasingly important in recent years.

\section{Related Work}
Previous work has explored various approaches to this problem.
Smith et al. (2023) proposed a novel method using neural networks.
Their approach achieved 95\% accuracy on the benchmark dataset.

\section{Methodology}
Our methodology involves several key steps. First, we preprocess the data using standard techniques.
Then we apply our machine learning algorithm with specific parameters.
The training process takes approximately 2 hours on a standard GPU.

\section{Results}
We achieved significant improvements over baseline methods.
Our model obtained an F1 score of 0.89, compared to 0.75 for the previous approach.
The results demonstrate the effectiveness of our proposed method.

\section{Conclusion}
In conclusion, our research contributes to the field by providing a new perspective on this problem.
Future work should explore additional applications and extensions of our approach.

\end{document}
"""


@pytest.fixture
def sample_bibtex_content():
    """Sample BibTeX content for testing."""
    return """
@article{smith2023,
  title={A Novel Approach to Machine Learning},
  author={Smith, John and Doe, Jane},
  journal={Journal of Machine Learning},
  volume={25},
  number={3},
  pages={123--145},
  year={2023},
  publisher={Academic Press},
  doi={10.1234/jml.2023.123}
}

@inproceedings{jones2022,
  title={Deep Learning for Natural Language Processing},
  author={Jones, Alice and Brown, Bob},
  booktitle={Proceedings of the International Conference on AI},
  pages={456--478},
  year={2022},
  organization={AI Society},
  doi={10.5678/ica.2022.456}
}

@book{wilson2021,
  title={Machine Learning Fundamentals},
  author={Wilson, Charlie},
  year={2021},
  publisher={Academic Press},
  isbn={978-0-123-45678-9}
}
"""


@pytest.fixture
def sample_json_content():
    """Sample JSON content for testing."""
    return """
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
    "abstractNote": "This paper explores the application of transformer models to various sequence processing tasks."
  },
  {
    "title": "Attention Mechanisms in Neural Networks",
    "creators": [
      {"firstName": "Grace", "lastName": "Lee"}
    ],
    "date": "2022-12-01",
    "publicationTitle": "AI Review",
    "DOI": "10.5432/air.2022.321",
    "abstractNote": "A comprehensive review of attention mechanisms in modern neural networks."
  }
]
"""


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """
title,authors,year,journal,doi,abstract
"Computer Vision Applications","Harris, Tom; White, Sarah",2024,"Computer Vision Journal","10.1111/cvj.2024.111","This paper presents new applications of computer vision in medical imaging."
"Reinforcement Learning in Robotics","Clark, David; Lewis, Maria",2023,"Robotics Today","10.2222/rt.2023.222","Exploring the use of reinforcement learning algorithms in robotic control systems."
"Natural Language Understanding","Walker, James; Hall, Patricia",2022,"Language Processing","10.3333/lp.2022.333","Advances in natural language understanding using deep learning techniques."
"""


@pytest.fixture
def create_test_file(temp_dir):
    """Factory fixture to create test files."""
    def _create_file(filename: str, content: str) -> Path:
        file_path = temp_dir / filename
        file_path.write_text(content.strip())
        return file_path
    
    return _create_file


@pytest.fixture
def sample_latex_file(create_test_file, sample_latex_content):
    """Create a sample LaTeX file."""
    return create_test_file("sample.tex", sample_latex_content)


@pytest.fixture
def sample_bibtex_file(create_test_file, sample_bibtex_content):
    """Create a sample BibTeX file."""
    return create_test_file("sample.bib", sample_bibtex_content)


@pytest.fixture
def sample_json_file(create_test_file, sample_json_content):
    """Create a sample JSON file."""
    return create_test_file("sample.json", sample_json_content)


@pytest.fixture
def sample_csv_file(create_test_file, sample_csv_content):
    """Create a sample CSV file."""
    return create_test_file("sample.csv", sample_csv_content)