# Greek Bible Text Analysis

This is a comprehensive Python toolkit for parsing, analyzing, and exploring a Greek Bible text with powerful APIs and an interactive web interface.

Note that this project is an evening's worth of work with Claude, as a hobby project.
Feel free to fork and make a PR if you want more functionality!

## Features

- **Text Parsing**: Parse RTF/DOCX Greek Bible files into structured data
- **Powerful Search**: Word, phrase, and regex search with accent-insensitive matching
- **Statistical Analysis**: Concordances, word frequencies, co-occurrence analysis
- **Interactive Viewer**: Web-based interface for browsing and analysis
- **ML-Ready**: Structured data export for machine learning applications
- **Unicode Handling**: Proper Greek text normalization and diacritic processing

## Quick Start

### Installation

```bash
git clone https://github.com/shininglegend/greek-bible-analysis
cd git
pip install -r requirements.txt
```

### Downloading the bible

This project is designed to read the greek bible provided for free online at https://berean.bible/downloads.htm
In order to avoid infringing on their copyright, please download the docx version of the greek bible yourself and put it this folder.
It must be named `bgb.docx`. The first few pages will be ignored when processing.
Many thanks to them for providing it! Please support them if you are able :heart:

### Basic Usage

```python
from greek_bible_parser import GreekBibleParser
from greek_bible_api import GreekTextAnalyzer

# Parse Bible text
parser = GreekBibleParser()
parser.load_from_docx("bgb.docx")

# Initialize analyzer
analyzer = GreekTextAnalyzer(parser)

# Search for words
results = analyzer.search_word("Ἰησοῦ")
print(f"Found {len(results)} occurrences")

# Get statistics
stats = analyzer.get_vocabulary_stats()
print(f"Total words: {stats['total_words']}")
```

### Web Interface

```bash
streamlit run greek_bible_viewer.py
```

## Project Structure

```
├── greek_bible_parser.py      # Core parsing engine
├── greek_bible_api.py         # Analysis API layer
├── greek_bible_viewer.py      # Interactive web interface
├── api_documentation.md       # Comprehensive API docs
├── requirements.txt           # Python dependencies
├── bgb.docx                   # Your bible (see above)
└── README.md                  # This file
```

## Core Components

### Parser (`greek_bible_parser.py`)

- `GreekBibleParser`: Main parsing class
- `BibleBook`: Book-level data structure
- `Verse`: Individual verse with metadata
- `Word`: Word-level objects with location data

### Analysis API (`greek_bible_api.py`)

- `GreekTextAnalyzer`: Primary analysis interface
- Search functions (word, phrase, regex, co-occurrence)
- Statistical analysis (concordance, frequencies, vocabulary stats)
- Text retrieval methods (verses, chapters, ranges)

### Web Viewer (`greek_bible_viewer.py`)

- Interactive text reader with customizable display
- Advanced search interface with multiple modes
- Statistical dashboards and visualizations
- Export capabilities for analysis results

## API Overview

### Search Operations

```python
# Word search with options
results = analyzer.search_word("λόγος", exact_match=False, include_accents=False)

# Phrase search
phrase_results = analyzer.search_phrase("ἐν ἀρχῇ")

# Regular expression search
regex_results = analyzer.search_regex(r"Ἰησοῦ.*Χριστοῦ")

# Co-occurrence analysis
co_occur = analyzer.find_co_occurrences("Ἰησοῦ", "λόγος", window=5)
```

### Statistical Analysis

```python
# Word statistics
word_stats = analyzer.get_word_statistics("Ἰησοῦ")
print(f"Frequency: {word_stats.frequency}")

# Vocabulary overview
vocab_stats = analyzer.get_vocabulary_stats()
print(f"Unique words: {vocab_stats['unique_words']}")

# Generate concordance
concordance = analyzer.get_concordance(min_frequency=5)
```

### Text Retrieval

```python
# Single verse
verse = analyzer.get_verse_text("Matthew", 1, 1)

# Full chapter
chapter = analyzer.get_chapter_text("Matthew", 1)

# Verse range
passage = analyzer.get_verse_range("Matthew", 1, 1, "Matthew", 1, 10)
```

## Data Structures

### SearchResult

Contains matched words with full context and location metadata.

### WordStats

Comprehensive statistics for individual words including frequency, distribution, and occurrence patterns.

### ConcordanceEntry

Complete concordance entries with all occurrences and contexts.

## Text Processing Features

- **Diacritic Normalization**: Automatic handling of Greek accent marks
- **Reference Extraction**: Separates cross-references from main text
- **Format Cleaning**: Removes English headers and formatting markers
- **Unicode Normalization**: Handles various Greek text encodings
- **Position Tracking**: Precise word location within verses

## Performance Notes

- Pre-built indexes for fast search operations
- Memory-optimized for large texts
- Caching enabled for web interface
- Suitable for corpora up to several million words
- Will take a while to launch initially while loading the text.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

See `LICENSE.md` for license information.

## Support

For issues and questions, please use the GitHub issue tracker.
