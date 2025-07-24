# Greek Bible Analysis API Documentation

## Overview
The `GreekTextAnalyzer` class provides comprehensive analysis tools for parsed Greek Bible text. Initialize with a parser instance containing structured Bible data.

## Data Structures

### SearchResult
```python
@dataclass
class SearchResult:
    word: str           # Matched word/phrase
    book: str          # Book name
    chapter: int       # Chapter number
    verse: int         # Verse number
    position: int      # Word position in verse
    context: str       # Surrounding context
    verse_text: str    # Full verse text
```

### ConcordanceEntry
```python
@dataclass
class ConcordanceEntry:
    word: str                    # Word form
    frequency: int              # Total occurrences
    occurrences: List[SearchResult]  # All instances
```

### WordStats
```python
@dataclass
class WordStats:
    word: str                           # Word
    frequency: int                      # Total count
    books: Set[str]                    # Books containing word
    first_occurrence: Tuple[str, int, int]  # (book, chapter, verse)
    last_occurrence: Tuple[str, int, int]   # (book, chapter, verse)
    avg_verse_position: float          # Average position within verses
```

## Core Methods

### Search Functions

#### `search_word(word: str, exact_match: bool = False, include_accents: bool = False) -> List[SearchResult]`
Search for a word with flexible matching options.
- `exact_match`: Match only exact forms vs. substring matching
- `include_accents`: Consider diacritical marks in matching

#### `search_phrase(phrase: str, window: int = 0) -> List[SearchResult]`
Search for multi-word phrases.
- `window`: Maximum word distance between phrase components (0 = exact sequence)

#### `search_regex(pattern: str, books: List[str] = None) -> List[SearchResult]`
Regular expression search across specified books or entire corpus.

#### `find_co_occurrences(word1: str, word2: str, window: int = 5) -> List[SearchResult]`
Find verses containing both words within specified proximity.

### Statistical Analysis

#### `get_concordance(min_frequency: int = 1) -> Dict[str, ConcordanceEntry]`
Generate complete word concordance with frequency filtering.

#### `get_word_statistics(word: str) -> Optional[WordStats]`
Comprehensive statistics for individual words.

#### `get_vocabulary_stats() -> Dict`
Corpus-wide vocabulary analysis including:
- `total_words`: Total word count
- `unique_words`: Unique word forms
- `type_token_ratio`: Lexical diversity measure
- `frequency_distribution`: Word frequency histogram
- `most_common_words`: Top 50 most frequent words
- `hapax_legomena`: Words appearing only once

#### `get_book_statistics(book_name: str) -> Optional[Dict]`
Per-book analysis including word counts, chapters, verses, and frequency data.

#### `get_word_cloud_data(min_frequency: int = 5) -> Dict[str, int]`
Word frequency data optimized for visualization tools.

### Text Retrieval

#### `get_verse_text(book: str, chapter: int, verse: int) -> Optional[str]`
Retrieve single verse text.

#### `get_chapter_text(book: str, chapter: int) -> Optional[List[str]]`
Retrieve all verses in a chapter as list.

#### `get_verse_range(book1: str, chapter1: int, verse1: int, book2: str = None, chapter2: int = None, verse2: int = None) -> List[str]`
Retrieve verse ranges. Defaults to single verse if end parameters omitted.

### Utility Functions

#### `get_books_list() -> List[str]`
Return available book names.

#### `get_chapter_count(book: str) -> int`
Count chapters in specified book.

#### `get_verse_count(book: str, chapter: int = None) -> int`
Count verses in book or specific chapter.

## Text Normalization

The analyzer automatically handles Greek text normalization:
- **Diacritic Removal**: Strips accent marks for broader matching
- **Case Normalization**: Converts to lowercase
- **Unicode Normalization**: Handles various Unicode representations

Search functions use normalized forms by default unless `include_accents=True`.

## Performance Notes

- **Indexing**: Pre-built indexes enable fast lookups
- **Memory**: All data loaded in memory for optimal performance
- **Initialization**: Index building occurs during analyzer creation
- **Search Complexity**: Most operations are O(1) or O(log n) after indexing

## Integration Patterns

### Basic Usage
```python
analyzer = GreekTextAnalyzer(parser_instance)
results = analyzer.search_word("λόγος")
stats = analyzer.get_vocabulary_stats()
```

### Batch Processing
Use `get_concordance()` for comprehensive analysis, then filter results rather than multiple individual searches.

### Memory Management
For large corpora, consider processing books individually using `get_book_statistics()` rather than corpus-wide operations.

## Error Handling

Methods return `None` or empty collections for invalid inputs:
- Invalid book/chapter/verse references return `None`
- Empty search results return `[]`
- Missing data returns appropriate empty containers

No exceptions thrown for normal usage patterns.