from greek_bible_processor import GreekBibleParser

import numpy as np
import re
from typing import List, Dict, Set, Tuple, Optional, Union, Callable
from collections import Counter, defaultdict
from dataclasses import dataclass
import unicodedata


@dataclass
class SearchResult:
    word: str
    book: str
    chapter: int
    verse: int
    position: int
    context: str
    verse_text: str


@dataclass
class ConcordanceEntry:
    word: str
    frequency: int
    occurrences: List[SearchResult]


@dataclass
class WordStats:
    word: str
    frequency: int
    books: Set[str]
    first_occurrence: Tuple[str, int, int]
    last_occurrence: Tuple[str, int, int]
    avg_verse_position: float


class GreekTextAnalyzer:
    """Powerful API for analyzing Greek Bible text"""

    def __init__(self, parser: GreekBibleParser):
        self.parser = parser
        if len(parser.books) == 0:
            raise Exception("Parser is not initialized.")
        self.books = parser.get_all_books()
        self._word_index = None
        self._lemma_cache = {}
        self._build_indexes()

    def _build_indexes(self):
        """Build internal indexes for fast searching"""
        self._word_index = defaultdict(list)
        self._verse_index = {}

        for book_name, book in self.books.items():
            for chapter_num in book.chapters:
                for verse_num, verse in book.chapters[chapter_num].items():
                    verse_key = (book_name, chapter_num, verse_num)
                    self._verse_index[verse_key] = verse

                    for word_obj in verse.words:
                        normalized_word = self._normalize_word(word_obj.text)
                        self._word_index[normalized_word].append(word_obj)

    def _normalize_word(self, word: str) -> str:
        """Normalize Greek word for searching"""
        # Remove diacritics for broader matching
        normalized = unicodedata.normalize("NFD", word)
        without_accents = "".join(
            c for c in normalized if unicodedata.category(c) != "Mn"
        )
        return without_accents.lower()

    def search_word(
        self, word: str, exact_match: bool = False, include_accents: bool = False
    ) -> List[SearchResult]:
        """Search for a word with various matching options"""
        if include_accents:
            search_term = word.lower()
        else:
            search_term = self._normalize_word(word)

        results = []

        for normalized_word, word_objects in self._word_index.items():
            match = False

            if exact_match:
                match = normalized_word == search_term
            else:
                match = search_term in normalized_word

            if match:
                for word_obj in word_objects:
                    verse_key = (word_obj.book, word_obj.chapter, word_obj.verse)
                    verse = self._verse_index[verse_key]

                    # Create context (5 words before and after)
                    verse_words = verse.get_words()
                    start_idx = max(0, word_obj.position - 5)
                    end_idx = min(len(verse_words), word_obj.position + 6)
                    context = " ".join(verse_words[start_idx:end_idx])

                    results.append(
                        SearchResult(
                            word=word_obj.text,
                            book=word_obj.book,
                            chapter=word_obj.chapter,
                            verse=word_obj.verse,
                            position=word_obj.position,
                            context=context,
                            verse_text=verse.text,
                        )
                    )

        return sorted(results, key=lambda x: (x.book, x.chapter, x.verse, x.position))

    def search_phrase(self, phrase: str, window: int = 0) -> List[SearchResult]:
        """Search for a phrase (sequence of words)"""
        phrase_words = [self._normalize_word(w) for w in phrase.split()]
        results = []

        for book_name, book in self.books.items():
            for chapter_num in book.chapters:
                for verse_num, verse in book.chapters[chapter_num].items():
                    verse_words = [self._normalize_word(w) for w in verse.get_words()]

                    # Find phrase matches
                    for i in range(len(verse_words) - len(phrase_words) + 1):
                        if verse_words[i : i + len(phrase_words)] == phrase_words:
                            # Check window constraint
                            if window == 0 or i + len(phrase_words) - 1 - i <= window:
                                original_words = verse.get_words()
                                matched_phrase = " ".join(
                                    original_words[i : i + len(phrase_words)]
                                )

                                start_idx = max(0, i - 5)
                                end_idx = min(
                                    len(original_words), i + len(phrase_words) + 5
                                )
                                context = " ".join(original_words[start_idx:end_idx])

                                results.append(
                                    SearchResult(
                                        word=matched_phrase,
                                        book=book_name,
                                        chapter=chapter_num,
                                        verse=verse_num,
                                        position=i,
                                        context=context,
                                        verse_text=verse.text,
                                    )
                                )

        return results

    def get_concordance(self, min_frequency: int = 1) -> Dict[str, ConcordanceEntry]:
        """Generate concordance of all words"""
        concordance = {}

        for normalized_word, word_objects in self._word_index.items():
            if len(word_objects) >= min_frequency:
                # Get the most common original form
                original_forms = [w.text for w in word_objects]
                most_common_form = Counter(original_forms).most_common(1)[0][0]

                occurrences = []
                for word_obj in word_objects:
                    verse_key = (word_obj.book, word_obj.chapter, word_obj.verse)
                    verse = self._verse_index[verse_key]

                    verse_words = verse.get_words()
                    start_idx = max(0, word_obj.position - 3)
                    end_idx = min(len(verse_words), word_obj.position + 4)
                    context = " ".join(verse_words[start_idx:end_idx])

                    occurrences.append(
                        SearchResult(
                            word=word_obj.text,
                            book=word_obj.book,
                            chapter=word_obj.chapter,
                            verse=word_obj.verse,
                            position=word_obj.position,
                            context=context,
                            verse_text=verse.text,
                        )
                    )

                concordance[most_common_form] = ConcordanceEntry(
                    word=most_common_form,
                    frequency=len(word_objects),
                    occurrences=sorted(
                        occurrences, key=lambda x: (x.book, x.chapter, x.verse)
                    ),
                )

        return concordance

    def get_word_statistics(self, word: str) -> Optional[WordStats]:
        """Get detailed statistics for a specific word"""
        search_results = self.search_word(word, exact_match=False)

        if not search_results:
            return None

        books = set(r.book for r in search_results)
        positions = [r.position for r in search_results]

        first_occ = search_results[0]
        last_occ = search_results[-1]

        return WordStats(
            word=word,
            frequency=len(search_results),
            books=books,
            first_occurrence=(first_occ.book, first_occ.chapter, first_occ.verse),
            last_occurrence=(last_occ.book, last_occ.chapter, last_occ.verse),
            avg_verse_position=np.mean(positions) if positions else 0,
        )

    def get_vocabulary_stats(self) -> Dict:
        """Get overall vocabulary statistics"""
        total_words = sum(len(words) for words in self._word_index.values())
        unique_words = len(self._word_index)

        # Frequency distribution
        frequencies = [len(words) for words in self._word_index.values()]
        freq_counter = Counter(frequencies)

        # Most common words
        word_freq = [
            (word, len(occurrences)) for word, occurrences in self._word_index.items()
        ]
        most_common = sorted(word_freq, key=lambda x: x[1], reverse=True)[:50]

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": unique_words / total_words if total_words > 0 else 0,
            "frequency_distribution": dict(freq_counter),
            "most_common_words": most_common,
            "hapax_legomena": sum(
                1 for freq in frequencies if freq == 1
            ),  # Words appearing only once
        }

    def get_book_statistics(self, book_name: str) -> Optional[Dict]:
        """Get statistics for a specific book"""
        if book_name not in self.books:
            return None

        book = self.books[book_name]
        all_words = book.get_word_objects()
        word_texts = [self._normalize_word(w.text) for w in all_words]

        return {
            "total_words": len(all_words),
            "unique_words": len(set(word_texts)),
            "chapters": len(book.chapters),
            "verses": sum(len(chapter) for chapter in book.chapters.values()),
            "avg_words_per_verse": len(all_words)
            / sum(len(chapter) for chapter in book.chapters.values()),
            "most_common_words": Counter(word_texts).most_common(20),
        }

    def find_co_occurrences(
        self, word1: str, word2: str, window: int = 5
    ) -> List[SearchResult]:
        """Find verses where two words co-occur within a window"""
        results = []

        for book_name, book in self.books.items():
            for chapter_num in book.chapters:
                for verse_num, verse in book.chapters[chapter_num].items():
                    verse_words = [self._normalize_word(w) for w in verse.get_words()]

                    word1_normalized = self._normalize_word(word1)
                    word2_normalized = self._normalize_word(word2)

                    # Find positions of both words
                    pos1 = [
                        i for i, w in enumerate(verse_words) if word1_normalized in w
                    ]
                    pos2 = [
                        i for i, w in enumerate(verse_words) if word2_normalized in w
                    ]

                    # Check if they co-occur within window
                    for p1 in pos1:
                        for p2 in pos2:
                            if abs(p1 - p2) <= window:
                                original_words = verse.get_words()
                                context_start = max(0, min(p1, p2) - 3)
                                context_end = min(len(original_words), max(p1, p2) + 4)
                                context = " ".join(
                                    original_words[context_start:context_end]
                                )

                                results.append(
                                    SearchResult(
                                        word=f"{original_words[p1]} ... {original_words[p2]}",
                                        book=book_name,
                                        chapter=chapter_num,
                                        verse=verse_num,
                                        position=min(p1, p2),
                                        context=context,
                                        verse_text=verse.text,
                                    )
                                )
                                break
                        else:
                            continue
                        break

        return results

    def get_verse_text(self, book: str, chapter: int, verse: int) -> Optional[str]:
        """Get the text of a specific verse"""
        if book not in self.books:
            return None

        book_obj = self.books[book]
        if chapter not in book_obj.chapters:
            return None

        if verse not in book_obj.chapters[chapter]:
            return None

        return book_obj.chapters[chapter][verse].text

    def get_chapter_text(self, book: str, chapter: int) -> Optional[List[str]]:
        """Get all verses in a chapter"""
        if book not in self.books:
            return None

        book_obj = self.books[book]
        if chapter not in book_obj.chapters:
            return None

        return [
            verse.text
            for verse_num, verse in sorted(book_obj.chapters[chapter].items())
        ]

    def get_verse_range(
        self,
        book1: str,
        chapter1: int,
        verse1: int,
        book2: str = None,
        chapter2: int = None,
        verse2: int = None,
    ) -> List[str]:
        """Get a range of verses"""
        if book2 is None:
            book2 = book1
        if chapter2 is None:
            chapter2 = chapter1
        if verse2 is None:
            verse2 = verse1

        verses = []

        # Handle single book case
        if book1 == book2:
            book_obj = self.books.get(book1)
            if not book_obj:
                return []

            # Handle single chapter case
            if chapter1 == chapter2:
                chapter_verses = book_obj.chapters.get(chapter1, {})
                for v in range(verse1, verse2 + 1):
                    if v in chapter_verses:
                        verses.append(chapter_verses[v].text)
            else:
                # Multiple chapters
                for c in range(chapter1, chapter2 + 1):
                    chapter_verses = book_obj.chapters.get(c, {})
                    if c == chapter1:
                        start_verse = verse1
                    else:
                        start_verse = (
                            min(chapter_verses.keys()) if chapter_verses else 1
                        )

                    if c == chapter2:
                        end_verse = verse2
                    else:
                        end_verse = max(chapter_verses.keys()) if chapter_verses else 1

                    for v in range(start_verse, end_verse + 1):
                        if v in chapter_verses:
                            verses.append(chapter_verses[v].text)

        return verses

    def search_regex(self, pattern: str, books: List[str] = None) -> List[SearchResult]:
        """Search using regular expressions"""
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        results = []

        target_books = books if books else list(self.books.keys())

        for book_name in target_books:
            if book_name not in self.books:
                continue

            book = self.books[book_name]
            for chapter_num in book.chapters:
                for verse_num, verse in book.chapters[chapter_num].items():
                    matches = compiled_pattern.finditer(verse.text)
                    for match in matches:
                        context_start = max(0, match.start() - 50)
                        context_end = min(len(verse.text), match.end() + 50)
                        context = verse.text[context_start:context_end]

                        results.append(
                            SearchResult(
                                word=match.group(),
                                book=book_name,
                                chapter=chapter_num,
                                verse=verse_num,
                                position=match.start(),
                                context=context,
                                verse_text=verse.text,
                            )
                        )

        return results

    def get_word_cloud_data(self, min_frequency: int = 5) -> Dict[str, int]:
        """Get word frequency data suitable for word clouds"""
        word_freq = {}

        for normalized_word, word_objects in self._word_index.items():
            if len(word_objects) >= min_frequency:
                # Use most common original form
                original_forms = [w.text for w in word_objects]
                most_common_form = Counter(original_forms).most_common(1)[0][0]
                word_freq[most_common_form] = len(word_objects)

        return word_freq

    def get_books_list(self) -> List[str]:
        """Get list of available books"""
        return list(self.books.keys())

    def get_chapter_count(self, book: str) -> int:
        """Get number of chapters in a book"""
        if book not in self.books:
            return 0
        return len(self.books[book].chapters)

    def get_verse_count(self, book: str, chapter: int = None) -> int:
        """Get number of verses in a book or chapter"""
        if book not in self.books:
            return 0

        book_obj = self.books[book]

        if chapter is None:
            return sum(len(ch) for ch in book_obj.chapters.values())
        else:
            return len(book_obj.chapters.get(chapter, {}))


# Usage example and testing
def test_api(analyzer):
    """Test the API functionality"""
    print("=== Greek Bible Analysis API Test ===\n")

    # Test basic search
    print("1. Searching for 'Ἰησοῦ':")
    jesus_results = analyzer.search_word("Ἰησοῦ")
    print(f"Found {len(jesus_results)} occurrences")
    if jesus_results:
        print(
            f"First occurrence: {jesus_results[0].book} {jesus_results[0].chapter}:{jesus_results[0].verse}"
        )
        print(f"Context: {jesus_results[0].context}\n")

    # Test word statistics
    print("2. Word statistics for 'Ἰησοῦ':")
    stats = analyzer.get_word_statistics("Ἰησοῦ")
    if stats:
        print(f"Frequency: {stats.frequency}")
        print(f"Books: {len(stats.books)}")
        print(f"First occurrence: {stats.first_occurrence}")
        print(f"Average position in verse: {stats.avg_verse_position:.2f}\n")

    # Test vocabulary stats
    print("3. Overall vocabulary statistics:")
    vocab_stats = analyzer.get_vocabulary_stats()
    print(f"Total words: {vocab_stats['total_words']}")
    print(f"Unique words: {vocab_stats['unique_words']}")
    print(f"Type-token ratio: {vocab_stats['type_token_ratio']:.4f}")
    print(f"Most common words (top 5):")
    for word, freq in vocab_stats["most_common_words"][:5]:
        print(f"  {word}: {freq}")
    print()

    # Test book statistics
    books = analyzer.get_books_list()
    if books:
        print(f"4. Statistics for {books[0]}:")
        book_stats = analyzer.get_book_statistics(books[0])
        if book_stats:
            print(f"Total words: {book_stats['total_words']}")
            print(f"Chapters: {book_stats['chapters']}")
            print(f"Verses: {book_stats['verses']}")
            print(f"Average words per verse: {book_stats['avg_words_per_verse']:.2f}")


if __name__ == "__main__":
    # This would be used with your parser
    print(
        "Import this module and initialize with: analyzer = GreekTextAnalyzer(your_parser)"
    )
    print("Then call: test_api(analyzer)")

    analyzer = GreekTextAnalyzer(GreekBibleParser(), "bgb.docx")
    test_api(analyzer)
