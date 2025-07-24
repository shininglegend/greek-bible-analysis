#!/usr/bin/env python3
"""
Mark-Matthew Comparison Analyzer

Analyzes phrase overlaps between Mark and Matthew with similarity scoring.
Shows synchronized view with phrases having >60% similarity highlighted.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

from greek_bible_processor import GreekBibleParser
from greek_bible_api import GreekTextAnalyzer


@dataclass
class PhraseMatch:
    mark_text: str
    matthew_text: str
    similarity: float
    mark_verse: Tuple[int, int]  # (chapter, verse)
    matthew_verse: Tuple[int, int]  # (chapter, verse)
    mark_position: int  # word position in verse
    matthew_position: int
    phrase_length: int


class MarkMatthewAnalyzer:
    def __init__(self, analyzer: GreekTextAnalyzer):
        self.analyzer = analyzer
        self.mark_verses = self._get_book_verses("Mark")
        self.matthew_verses = self._get_book_verses("Matthew")
        self.similarity_threshold = 0.6
        self.mark_ngrams = {}
        self.matthew_ngrams = {}
        self._build_ngram_indexes()

    def _get_book_verses(self, book_name: str) -> Dict[Tuple[int, int], List[str]]:
        """Extract all verses from a book as normalized word lists"""
        verses = {}
        book_stats = self.analyzer.get_book_statistics(book_name)
        if not book_stats:
            return verses

        for chapter in range(1, book_stats['chapters'] + 1):
            chapter_text = self.analyzer.get_chapter_text(book_name, chapter)
            if chapter_text:
                for verse_num, verse_text in enumerate(chapter_text, 1):
                    if verse_text:
                        # Normalize and split into words
                        words = self._normalize_verse(verse_text)
                        verses[(chapter, verse_num)] = words
        return verses

    def _normalize_verse(self, verse_text: str) -> List[str]:
        """Normalize verse text to word list"""
        # Remove punctuation and normalize
        cleaned = re.sub(r'[^\w\s]', ' ', verse_text.lower())
        words = [self.analyzer._normalize_word(w) for w in cleaned.split() if w.strip()]
        return [w for w in words if w]  # Remove empty strings

    def _build_ngram_indexes(self):
        """Build n-gram indexes for faster matching"""
        print("Building n-gram indexes...")

        # Build Mark n-grams
        for verse_key, words in self.mark_verses.items():
            self.mark_ngrams[verse_key] = {}
            for length in range(3, min(len(words) + 1, 8)):  # Limit to reasonable lengths
                for i in range(len(words) - length + 1):
                    ngram = tuple(words[i:i + length])
                    ngram_hash = hash(ngram)
                    if ngram_hash not in self.mark_ngrams[verse_key]:
                        self.mark_ngrams[verse_key][ngram_hash] = []
                    self.mark_ngrams[verse_key][ngram_hash].append((ngram, i))

        # Build Matthew n-grams
        for verse_key, words in self.matthew_verses.items():
            self.matthew_ngrams[verse_key] = {}
            for length in range(3, min(len(words) + 1, 8)):
                for i in range(len(words) - length + 1):
                    ngram = tuple(words[i:i + length])
                    ngram_hash = hash(ngram)
                    if ngram_hash not in self.matthew_ngrams[verse_key]:
                        self.matthew_ngrams[verse_key][ngram_hash] = []
                    self.matthew_ngrams[verse_key][ngram_hash].append((ngram, i))

    def _calculate_similarity(self, phrase1: List[str], phrase2: List[str]) -> float:
        """Calculate similarity between two phrases using faster word overlap"""
        if not phrase1 or not phrase2:
            return 0.0

        # Fast exact match check
        if phrase1 == phrase2:
            return 1.0

        # Use word overlap ratio for speed
        set1, set2 = set(phrase1), set(phrase2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _extract_phrases(self, words: List[str], min_length: int = 3, max_length: int = 12) -> List[Tuple[List[str], int]]:
        """Extract all phrases of given lengths with their positions"""
        phrases = []
        for length in range(min_length, min(len(words) + 1, max_length + 1)):
            for i in range(len(words) - length + 1):
                phrase = words[i:i + length]
                phrases.append((phrase, i))
        return phrases

    def find_phrase_matches(self, min_phrase_length: int = 3) -> List[PhraseMatch]:
        """Find phrase matches using n-gram indexes for speed"""
        matches = []
        processed_pairs = set()

        print(f"Scanning {len(self.mark_verses)} Mark verses against {len(self.matthew_verses)} Matthew verses...")

        for mark_verse_key, mark_ngrams in self.mark_ngrams.items():
            for matthew_verse_key, matthew_ngrams in self.matthew_ngrams.items():

                # Find common n-gram hashes
                common_hashes = set(mark_ngrams.keys()) & set(matthew_ngrams.keys())

                for ngram_hash in common_hashes:
                    mark_instances = mark_ngrams[ngram_hash]
                    matthew_instances = matthew_ngrams[ngram_hash]

                    for mark_ngram, mark_pos in mark_instances:
                        for matthew_ngram, matthew_pos in matthew_instances:

                            # Create unique pair identifier
                            pair_id = (mark_verse_key, matthew_verse_key, mark_pos, matthew_pos, len(mark_ngram))
                            if pair_id in processed_pairs:
                                continue
                            processed_pairs.add(pair_id)

                            similarity = self._calculate_similarity(list(mark_ngram), list(matthew_ngram))

                            if similarity >= self.similarity_threshold:
                                match = PhraseMatch(
                                    mark_text=" ".join(mark_ngram),
                                    matthew_text=" ".join(matthew_ngram),
                                    similarity=similarity,
                                    mark_verse=mark_verse_key,
                                    matthew_verse=matthew_verse_key,
                                    mark_position=mark_pos,
                                    matthew_position=matthew_pos,
                                    phrase_length=len(mark_ngram)
                                )
                                matches.append(match)

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches

    def get_verse_coverage_stats(self) -> Dict[str, Dict]:
        """Calculate how much of Mark appears in Matthew by verse"""
        matches = self.find_phrase_matches()

        mark_coverage = defaultdict(set)  # verse -> set of covered word positions
        matthew_usage = defaultdict(set)  # verse -> set of used word positions

        for match in matches:
            # Mark coverage
            mark_verse_key = match.mark_verse
            for i in range(match.phrase_length):
                mark_coverage[mark_verse_key].add(match.mark_position + i)

            # Matthew usage
            matthew_verse_key = match.matthew_verse
            for i in range(match.phrase_length):
                matthew_usage[matthew_verse_key].add(match.matthew_position + i)

        # Calculate coverage percentages
        mark_stats = {}
        matthew_stats = {}

        for verse_key, words in self.mark_verses.items():
            covered_words = len(mark_coverage[verse_key])
            total_words = len(words)
            coverage_pct = (covered_words / total_words * 100) if total_words > 0 else 0
            mark_stats[verse_key] = {
                'covered_words': covered_words,
                'total_words': total_words,
                'coverage_percentage': coverage_pct
            }

        for verse_key, words in self.matthew_verses.items():
            used_words = len(matthew_usage[verse_key])
            total_words = len(words)
            usage_pct = (used_words / total_words * 100) if total_words > 0 else 0
            matthew_stats[verse_key] = {
                'used_words': used_words,
                'total_words': total_words,
                'usage_percentage': usage_pct
            }

        return {
            'mark_coverage': mark_stats,
            'matthew_usage': matthew_stats,
            'total_matches': len(matches)
        }

    def generate_synchronized_view(self, max_mark_chapter: int = None) -> str:
        """Generate a synchronized view showing overlaps"""
        matches = self.find_phrase_matches()
        coverage_stats = self.get_verse_coverage_stats()

        # Group matches by Mark verse for display
        matches_by_mark_verse = defaultdict(list)
        for match in matches:
            matches_by_mark_verse[match.mark_verse].append(match)

        output = []
        output.append("=" * 80)
        output.append("MARK-MATTHEW SYNCHRONIZED COMPARISON")
        output.append("Phrases with >60% similarity marked with [***]")
        output.append("=" * 80)
        output.append("")

        # Process Mark verses in order
        mark_chapters = sorted(set(key[0] for key in self.mark_verses.keys()))
        if max_mark_chapter:
            mark_chapters = [c for c in mark_chapters if c <= max_mark_chapter]

        for chapter in mark_chapters:
            output.append(f"MARK CHAPTER {chapter}")
            output.append("-" * 40)

            mark_verses_in_chapter = [(ch, v) for (ch, v) in self.mark_verses.keys() if ch == chapter]
            mark_verses_in_chapter.sort(key=lambda x: x[1])

            for verse_key in mark_verses_in_chapter:
                ch, v = verse_key
                mark_words = self.mark_verses[verse_key]
                coverage = coverage_stats['mark_coverage'].get(verse_key, {})

                output.append(f"\nMark {ch}:{v} (Coverage: {coverage.get('coverage_percentage', 0):.1f}%)")

                # Show original verse with highlighted matches
                verse_display = self._highlight_verse_matches(mark_words, matches_by_mark_verse[verse_key], is_mark=True)
                output.append(f"  {verse_display}")

                # Show corresponding Matthew matches
                if verse_key in matches_by_mark_verse:
                    output.append("  MATTHEW PARALLELS:")
                    for match in matches_by_mark_verse[verse_key][:3]:  # Limit to top 3 matches
                        matt_ch, matt_v = match.matthew_verse
                        output.append(f"    Matt {matt_ch}:{matt_v} ({match.similarity:.1%}) - {match.matthew_text}")

                output.append("")

        # Summary statistics
        total_mark_words = sum(len(words) for words in self.mark_verses.values())
        total_covered_words = sum(stats.get('covered_words', 0) for stats in coverage_stats['mark_coverage'].values())
        overall_coverage = (total_covered_words / total_mark_words * 100) if total_mark_words > 0 else 0

        output.append("=" * 80)
        output.append("SUMMARY STATISTICS")
        output.append("=" * 80)
        output.append(f"Total Mark words: {total_mark_words:,}")
        output.append(f"Words with Matthew parallels: {total_covered_words:,}")
        output.append(f"Overall coverage: {overall_coverage:.1f}%")
        output.append(f"Total phrase matches found: {len(matches):,}")
        output.append(f"Similarity threshold: {self.similarity_threshold:.0%}")

        return "\n".join(output)

    def _highlight_verse_matches(self, words: List[str], matches: List[PhraseMatch], is_mark: bool = True) -> str:
        """Highlight matched phrases in verse text"""
        if not matches:
            return " ".join(words)

        highlighted = words.copy()
        match_positions = set()

        # Collect all match positions
        for match in matches:
            pos = match.mark_position if is_mark else match.matthew_position
            length = match.phrase_length
            for i in range(pos, min(pos + length, len(words))):
                match_positions.add(i)

        # Add highlighting markers
        result = []
        in_match = False
        for i, word in enumerate(highlighted):
            if i in match_positions and not in_match:
                result.append("[***")
                in_match = True
            elif i not in match_positions and in_match:
                result.append("***]")
                in_match = False
            result.append(word)

        if in_match:
            result.append("***]")

        return " ".join(result)

    def export_detailed_matches(self, filename: str = "mark_matthew_matches.txt"):
        """Export detailed match analysis to file"""
        matches = self.find_phrase_matches()

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("DETAILED MARK-MATTHEW PHRASE MATCHES\n")
            f.write("=" * 50 + "\n\n")

            for i, match in enumerate(matches, 1):
                f.write(f"Match #{i} (Similarity: {match.similarity:.1%})\n")
                f.write(f"Mark {match.mark_verse[0]}:{match.mark_verse[1]}: {match.mark_text}\n")
                f.write(f"Matt {match.matthew_verse[0]}:{match.matthew_verse[1]}: {match.matthew_text}\n")
                f.write(f"Phrase length: {match.phrase_length} words\n")
                f.write("-" * 40 + "\n\n")

        print(f"Detailed matches exported to {filename}")


def main():
    """Main analysis function"""
    print("Loading Greek Bible data...")
    parser = GreekBibleParser()
    parser.load_from_docx("bgb.docx")
    analyzer = GreekTextAnalyzer(parser)

    print("Initializing Mark-Matthew analyzer...")
    mark_matthew = MarkMatthewAnalyzer(analyzer)

    print("Finding phrase matches...")
    matches = mark_matthew.find_phrase_matches()
    print(f"Found {len(matches)} phrase matches with >60% similarity")

    print("\nGenerating synchronized view...")
    view = mark_matthew.generate_synchronized_view(max_mark_chapter=2)  # Limit to first 2 chapters for demo

    # Save to file
    with open("mark_matthew_analysis.txt", "w", encoding="utf-8") as f:
        f.write(view)

    print("Analysis complete. Results saved to mark_matthew_analysis.txt")

    # Export detailed matches
    mark_matthew.export_detailed_matches()

    # Print summary to console
    print("\n" + "="*50)
    print("QUICK SUMMARY")
    print("="*50)
    coverage_stats = mark_matthew.get_verse_coverage_stats()

    high_coverage_verses = []
    for verse_key, stats in coverage_stats['mark_coverage'].items():
        if stats['coverage_percentage'] > 80:
            high_coverage_verses.append((verse_key, stats['coverage_percentage']))

    high_coverage_verses.sort(key=lambda x: x[1], reverse=True)

    print(f"Mark verses with >80% Matthew coverage: {len(high_coverage_verses)}")
    for verse_key, pct in high_coverage_verses[:10]:
        print(f"  Mark {verse_key[0]}:{verse_key[1]} - {pct:.1f}%")


if __name__ == "__main__":
    main()
