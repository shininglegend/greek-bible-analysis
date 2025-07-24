import re
from typing import List, Dict, Tuple, Optional
try:
    import mammoth
except ImportError:
    mammoth = None

# Comprehensive Greek Unicode pattern
GREEK_PATTERN = r'[\u0370-\u03FF\u1F00-\u1FFF]+'

class Word:
    def __init__(self, text: str, book: str, chapter: int, verse: int, position: int):
        self.text = text
        self.book = book
        self.chapter = chapter
        self.verse = verse
        self.position = position  # position within verse

    def __str__(self):
        return f"{self.text} ({self.book} {self.chapter}:{self.verse}:{self.position})"

    def __repr__(self):
        return self.__str__()

class Verse:
    def __init__(self, text: str, book: str, chapter: int, verse: int, references: List[str] = None):
        self.text = text.strip()
        self.book = book
        self.chapter = chapter
        self.verse = verse
        self.references = references or []
        self.words = self._parse_words()

    def _parse_words(self) -> List[Word]:
        # Split on whitespace and punctuation, keeping Greek text
        words = []
        word_pattern = GREEK_PATTERN
        matches = re.finditer(word_pattern, self.text)

        for i, match in enumerate(matches):
            word_text = match.group()
            words.append(Word(word_text, self.book, self.chapter, self.verse, i))

        return words

    def get_words(self) -> List[str]:
        return [word.text for word in self.words]

    def __str__(self):
        return f"{self.book} {self.chapter}:{self.verse} - {self.text}"

class BibleBook:
    def __init__(self, name: str):
        self.name = name
        self.chapters: Dict[int, Dict[int, Verse]] = {}
        self.metadata = {}

    def add_verse(self, verse: Verse):
        if verse.chapter not in self.chapters:
            self.chapters[verse.chapter] = {}
        self.chapters[verse.chapter][verse.verse] = verse

    def get_chapter_array(self, chapter: int) -> List[str]:
        """Returns array of verse texts for a chapter"""
        if chapter not in self.chapters:
            return []
        return [self.chapters[chapter][v].text for v in sorted(self.chapters[chapter].keys())]

    def get_chapters_array(self) -> List[List[str]]:
        """Returns array of arrays: [chapter][verse]"""
        result = []
        for chapter_num in sorted(self.chapters.keys()):
            chapter_verses = []
            for verse_num in sorted(self.chapters[chapter_num].keys()):
                chapter_verses.append(self.chapters[chapter_num][verse_num].text)
            result.append(chapter_verses)
        return result

    def get_flat_verses_array(self) -> List[str]:
        """Returns flat array of all verses"""
        verses = []
        for chapter_num in sorted(self.chapters.keys()):
            for verse_num in sorted(self.chapters[chapter_num].keys()):
                verses.append(self.chapters[chapter_num][verse_num].text)
        return verses

    def get_flat_words_array(self) -> List[str]:
        """Returns flat array of all words"""
        words = []
        for chapter_num in sorted(self.chapters.keys()):
            for verse_num in sorted(self.chapters[chapter_num].keys()):
                verse = self.chapters[chapter_num][verse_num]
                words.extend(verse.get_words())
        return words

    def get_word_objects(self) -> List[Word]:
        """Returns all Word objects"""
        words = []
        for chapter_num in sorted(self.chapters.keys()):
            for verse_num in sorted(self.chapters[chapter_num].keys()):
                verse = self.chapters[chapter_num][verse_num]
                words.extend(verse.words)
        return words

class GreekBibleParser:
    def __init__(self):
        self.books: Dict[str, BibleBook] = {}
        self.current_book = None
        self.current_chapter = None

    def load_from_docx(self, filepath: str):
        """Load and parse from DOCX file"""
        if mammoth is None:
            print("mammoth not installed - cannot read DOCX files")
            return
        try:
            with open(filepath, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text = result.value
                self._parse_text(text)
        except Exception as e:
            print(f"Failed to read: {e}")

    def load_from_text(self, text: str):
        """Load and parse from raw text"""
        self._parse_text(text)

    def _parse_text(self, text: str):
        """Parse the raw text into structured data"""
        lines = text.split('\n')
        current_verse_text = ""
        current_verse_num = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a book title (simple heuristic)
            if self._is_book_title(line):
                # Finalize any pending verse
                if current_verse_text and current_verse_num is not None:
                    self._finalize_verse(current_verse_num, current_verse_text)
                    current_verse_text = ""
                    current_verse_num = None
                self._handle_book_title(line)
            # Check if this is a chapter header
            elif self._is_chapter_header(line):
                # Finalize any pending verse
                if current_verse_text and current_verse_num is not None:
                    self._finalize_verse(current_verse_num, current_verse_text)
                    current_verse_text = ""
                    current_verse_num = None
                self._handle_chapter_header(line)
            # Check if this is a section header
            elif self._is_section_header(line):
                # Finalize any pending verse
                if current_verse_text and current_verse_num is not None:
                    self._finalize_verse(current_verse_num, current_verse_text)
                    current_verse_text = ""
                    current_verse_num = None
                # Skip section headers
                continue
            # Check if this is a verse
            elif self._is_verse(line):
                # Finalize any pending verse
                if current_verse_text and current_verse_num is not None:
                    self._finalize_verse(current_verse_num, current_verse_text)
                # Start new verse
                verse_match = re.match(r'^(\d+)\s*[^\w]*(.+)$', line)
                verse_num = int(verse_match.group(1))
                verse_text = verse_match.group(2).strip()

                # Check for embedded verses in this line
                embedded_verses = self._split_embedded_verses(verse_text)
                if len(embedded_verses) > 1:
                    # Process all verses found
                    for i, text in enumerate(embedded_verses):
                        if text.strip():
                            self._finalize_verse(verse_num + i, text)
                    # Reset tracking
                    current_verse_text = ""
                    current_verse_num = None
                else:
                    # Single verse
                    current_verse_num = verse_num
                    current_verse_text = verse_text
            # Check if this is a verse continuation (Greek text without verse number)
            elif self._is_verse_continuation(line):
                if current_verse_text:
                    # Check for embedded verse numbers in continuation
                    embedded_verses = self._split_embedded_verses(current_verse_text + " " + line)
                    if len(embedded_verses) > 1:
                        # Process the first complete verse
                        self._finalize_verse(current_verse_num, embedded_verses[0])
                        # Start processing subsequent verses
                        for i, verse_text in enumerate(embedded_verses[1:], 1):
                            if verse_text.strip():
                                self._finalize_verse(current_verse_num + i, verse_text)
                        # Reset current verse tracking
                        current_verse_text = ""
                        current_verse_num = None
                    else:
                        current_verse_text += " " + line
                else:
                    print("Did not recognize:", line)
            else:
                print("Did not recognize:", line)

        # Finalize any remaining verse
        if current_verse_text and current_verse_num is not None:
            self._finalize_verse(current_verse_num, current_verse_text)

    def _is_book_title(self, line: str) -> bool:
        # Simple check for book titles - they're usually single words or short phrases
        # and don't contain verse numbers or Greek text
        if not line:
            return False

        # Common book names
        book_names = ['Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans', 'Corinthians',
                     'Galatians', 'Ephesians', 'Philippians', 'Colossians', 'Thessalonians',
                     'Timothy', 'Titus', 'Philemon', 'Hebrews', 'James', 'Peter', 'Jude',
                     'Revelation', 'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']

        return any(book == line for book in book_names) and len(line.split()) <= 3

    def _is_chapter_header(self, line: str) -> bool:
        # Pattern like "Matthew 1" or section headers
        return bool(re.match(r'^[A-Za-z]+ \d+$', line)) or \
               ('Chapter' in line and any(char.isdigit() for char in line))

    def _is_section_header(self, line: str) -> bool:
        # Section headers typically contain English text and references in parentheses
        # But exclude lines that are primarily Greek text or have actual quotation marks
        has_english = bool(re.search(r'[A-Za-z]', line))
        is_chapter_header = bool(re.search(r'^[A-Za-z]+ \d+$', line))
        has_greek = bool(re.search(GREEK_PATTERN, line))
        has_quotes = '"' in line or '"' in line or '"' in line

        # Section headers have English but no Greek and no quotes (apostrophes are OK)
        return has_english and not is_chapter_header and not has_greek and not has_quotes

    def _is_verse(self, line: str) -> bool:
        # Check if line starts with verse number and contains Greek text
        greek_pattern = GREEK_PATTERN
        verse_number_pattern = r'^\d+\s*[^\w]*'

        return bool(re.search(greek_pattern, line)) and bool(re.match(verse_number_pattern, line))

    def _is_verse_continuation(self, line: str) -> bool:
        # Greek text without verse number at start
        greek_pattern = GREEK_PATTERN
        verse_number_pattern = r'^\d+\s*[^\w]*'

        # Check for Greek characters (including uppercase)
        has_greek = bool(re.search(greek_pattern, line))

        # Check for quotation marks which often indicate verse continuations
        has_quotes = '"' in line or "'" in line or '"' in line or '"' in line

        # Check for asterisks (formatting markers)
        has_formatting = '*' in line

        # Not starting with verse number
        not_verse_start = not bool(re.match(verse_number_pattern, line))

        return (has_greek or has_quotes or has_formatting) and not_verse_start

    def _split_embedded_verses(self, text: str) -> List[str]:
        """Split text that contains embedded verse numbers"""
        # Look for patterns like '" 51 καὶ' where verse number is followed by Greek
        # Use lookahead to ensure number is followed by Greek text
        verse_pattern = r'"\s+(\d+)\s+(?=' + GREEK_PATTERN.replace('+', '') + r')'

        matches = list(re.finditer(verse_pattern, text))

        if not matches:
            return [text]

        verses = []
        last_end = 0

        for match in matches:
            # Add text before this split point (include the quote)
            verse_text = text[last_end:match.start() + 1].strip()
            if verse_text:
                verses.append(verse_text)

            # Next verse starts after the verse number and space (skip the number)
            last_end = match.end()

        # Add remaining text
        remaining = text[last_end:].strip()
        if remaining:
            verses.append(remaining)

        return verses

    def _handle_book_title(self, line: str):
        book_name = line.strip()
        self.current_book = BibleBook(book_name)
        self.books[book_name] = self.current_book
        self.current_chapter = None

    def _handle_chapter_header(self, line: str):
        # Extract chapter number
        chapter_match = re.search(r'\d+', line)
        if chapter_match and self.current_book:
            self.current_chapter = int(chapter_match.group())

    def _finalize_verse(self, verse_num: int, verse_text: str):
        """Create and add a complete verse"""
        if not self.current_book or self.current_chapter is None:
            return

        # Clean the text - remove references and English annotations
        cleaned_text = self._clean_verse_text(verse_text)

        # Extract references (basic implementation)
        references = self._extract_references(verse_text)

        verse = Verse(cleaned_text, self.current_book.name, self.current_chapter, verse_num, references)
        self.current_book.add_verse(verse)

    def _handle_verse(self, line: str):
        if not self.current_book or self.current_chapter is None:
            return

        # Extract verse number and text
        verse_match = re.match(r'^(\d+)\s*[^\w]*(.+)$', line)
        if not verse_match:
            return

        verse_num = int(verse_match.group(1))
        verse_text = verse_match.group(2).strip()

        # Clean the text - remove references and English annotations
        cleaned_text = self._clean_verse_text(verse_text)

        # Extract references (basic implementation)
        references = self._extract_references(verse_text)

        verse = Verse(cleaned_text, self.current_book.name, self.current_chapter, verse_num, references)
        self.current_book.add_verse(verse)

    def _clean_verse_text(self, text: str) -> str:
        """Remove English text, references, and formatting"""
        # Remove text in parentheses (usually references)
        text = re.sub(r'\([^)]*\)', '', text)

        # Remove text in square brackets
        text = re.sub(r'\[[^\]]*\]', '', text)

        # Remove reference markers like 'a', 'b', 'c' at the end
        text = re.sub(r'\s*[a-z]\s*$', '', text)

        # Remove multiple reference markers
        text = re.sub(r'\*\*\*[a-z]\*\*\*', '', text)

        # Remove verse number at start if present
        text = re.sub(r'^\d+\s+', '', text)

        # Remove English annotations like "f a 5 Or comprehended"
        text = re.sub(r'[a-z]\s+[a-z]*\s*\d+\s+[A-Z][a-z]+[^Α-Ωα-ω]*', '', text)

        # Remove manuscript variations like "18 BYZ and TR"
        text = re.sub(r'\d+\s+[A-Z]{2,}[^Α-Ωα-ω]*', '', text)

        # Remove "See" references like "51 See Genesis 28:12"
        text = re.sub(r'\d+\s+See\s+[^Α-Ωα-ω]*', '', text)

        # Remove single letters followed by annotations (like "f a")
        text = re.sub(r'[a-z]\s+[a-z]\s+\d+.*$', '', text)

        # Remove trailing single letters and annotations
        text = re.sub(r'[a-z](\s+[a-zA-Z0-9\s]+)*$', '', text)

        # Remove non-breaking spaces
        text = text.replace('\xa0', ' ')

        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _extract_references(self, text: str) -> List[str]:
        """Extract biblical references from verse text"""
        references = []

        # Find parenthetical references
        paren_refs = re.findall(r'\(([^)]*)\)', text)
        references.extend(paren_refs)

        # Find "See" references like "See Genesis 28:12"
        see_refs = re.findall(r'See\s+([A-Za-z]+\s+\d+:\d+)', text)
        references.extend(see_refs)

        # Find "Or" alternative translations
        or_refs = re.findall(r'Or\s+([^;]+)', text)
        references.extend(or_refs)

        return references

    def get_book(self, name: str) -> Optional[BibleBook]:
        """Get a specific book"""
        return self.books.get(name)

    def get_all_books(self) -> Dict[str, BibleBook]:
        """Get all books"""
        return self.books

    def get_stats(self) -> Dict:
        """Get parsing statistics"""
        total_verses = 0
        total_words = 0
        total_chapters = 0

        for book in self.books.values():
            for chapter in book.chapters.values():
                total_chapters += 1
                for verse in chapter.values():
                    total_verses += 1
                    total_words += len(verse.words)

        return {
            'books': len(self.books),
            'chapters': total_chapters,
            'verses': total_verses,
            'words': total_words
        }

# Example usage and testing
def main():
    parser = GreekBibleParser()

    # Load from DOCX file
    try:
        parser.load_from_docx("bgb.docx")
        print("Loaded from DOCX file")
    except:
        print("Failed to load.")
        import os
        os.exit(1)

    # Display statistics
    stats = parser.get_stats()
    print(f"Parsing Statistics: {stats}")

    # Print out specific verses for debugging
    VERSE = [
        ["John", 1, 50],
        # ["Matthew", 10, 20]
    ]

    for book_name, chapter_num, verse_num in VERSE:
        book = parser.get_book(book_name)
        if book and chapter_num in book.chapters and verse_num in book.chapters[chapter_num]:
            verse = book.chapters[chapter_num][verse_num]
            print(f"\nDEBUG: {book_name} {chapter_num}:{verse_num}")
            print(f"  Raw text: '{verse.text}'")
            print(f"  Word count: {len(verse.words)}")
            print(f"  Words: {[w.text for w in verse.words]}")
            print(f"  References: {verse.references}")
        else:
            print(f"\nDEBUG: {book_name} {chapter_num}:{verse_num} - NOT FOUND")


    # Test the APIs
    matthew = parser.get_book("Matthew")
    if matthew:
        print(f"\nMatthew chapters array shape: {len(matthew.get_chapters_array())}")
        print(f"Matthew flat verses count: {len(matthew.get_flat_verses_array())}")
        print(f"Matthew flat words count: {len(matthew.get_flat_words_array())}")

        # Show first few words with their locations
        words = matthew.get_word_objects()[:10]
        print("\nFirst 10 words with locations:")
        for word in words:
            print(word)

if __name__ == "__main__":
    main()
