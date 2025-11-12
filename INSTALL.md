# Installation Instructions

## Quick Start

### Windows
```bash
# Clone or download the repository
cd refscore-academic

# Run the installation script
python install.py

# Launch the GUI application
python -m refscore.main
```

### Linux/macOS
```bash
# Clone or download the repository
cd refscore-academic

# Make the installation script executable
chmod +x install.py

# Run the installation script
python3 install.py

# Launch the GUI application
python3 -m refscore.main
```

## Manual Installation

### 1. Install Python
Ensure you have Python 3.8 or higher installed:
```bash
python --version
# or
python3 --version
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Optional Dependencies (Recommended)
For full functionality, install these optional packages:
```bash
# For semantic embeddings
pip install sentence-transformers

# For named entity recognition
pip install spacy
python -m spacy download en_core_web_sm

# For BibTeX parsing
pip install bibtexparser

# For PDF processing
pip install PyMuPDF

# For machine learning
pip install scikit-learn
```

### 4. Run the Application
```bash
# GUI mode
python -m refscore.main

# Command line mode
python -m refscore.cli --help
```

## Development Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/refscore-academic.git
cd refscore-academic
```

### 2. Install in Development Mode
```bash
pip install -e .
```

### 3. Install Development Dependencies
```bash
pip install -r requirements.txt
pip install -e .[dev]
```

### 4. Run Tests
```bash
pytest tests/ -v --cov=refscore
```

### 5. Generate Documentation
```bash
cd docs
make html
```

## Troubleshooting

### Common Issues

#### 1. PyQt5 Installation Issues
```bash
# On Ubuntu/Debian
sudo apt-get install python3-pyqt5

# On macOS
brew install pyqt5

# On Windows, try using conda
conda install pyqt
```

#### 2. spaCy Model Download Issues
```bash
# Try downloading the model manually
python -m spacy download en_core_web_sm

# If that fails, try the small model
python -m spacy download en_core_web_sm-3.4.0 --direct
```

#### 3. Permission Issues on Linux/macOS
```bash
# Make scripts executable
chmod +x install.py
chmod +x run_tests.py
```

#### 4. Memory Issues with Large Documents
```bash
# Increase Python memory limit
export PYTHONHASHSEED=0
python -m refscore.main
```

### Performance Optimization

#### 1. Enable GPU Acceleration (if available)
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Use Multiprocessing
Enable multiprocessing in the settings for faster processing of large documents.

#### 3. Optimize Memory Usage
- Close other applications while processing large documents
- Process documents in batches if memory is limited
- Use the fallback algorithms when full NLP libraries aren't needed

## Getting Help

### Documentation
- Full documentation: [docs/](docs/)
- API reference: [docs/api/](docs/api/)
- Examples: [examples/](examples/)

### Support
- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/refscore-academic/issues)
- Email: academic@university.edu
- Academic support: Contact your university's computer science department

### Community
- Join our academic community discussions
- Contribute to the project development
- Share your research applications and results

## Next Steps

After installation:

1. **Try the Sample Files**: Check the sample files in `~/Documents/RefScoreSamples/`
2. **Run Your First Analysis**: Use the GUI to load a document and references
3. **Explore Settings**: Customize scoring weights and processing options
4. **Generate Reports**: Export results in various formats
5. **Read the Documentation**: Learn about advanced features and customization

## Academic Use

This software is designed for academic research and education. When using RefScore in your research:

1. **Cite the Software**: Use the citation format provided in the README
2. **Validate Results**: Always manually verify critical results
3. **Follow Academic Standards**: Ensure your use complies with your institution's guidelines
4. **Share Feedback**: Help improve the software for the academic community

Happy analyzing! 📚🔍