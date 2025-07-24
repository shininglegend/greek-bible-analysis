#!/usr/bin/env python3
"""
Visual Mark-Matthew Comparison Tool

Creates a visual comparison showing phrase overlaps with progress indicators.
"""

import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import math

from greek_bible_processor import GreekBibleParser
from greek_bible_api import GreekTextAnalyzer
from mark_matthew_analyzer import MarkMatthewAnalyzer


class VisualComparison:
    def __init__(self, analyzer: MarkMatthewAnalyzer):
        self.analyzer = analyzer
        self.terminal_width = 80

    def _create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """Create a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {percentage:5.1f}%"

    def _wrap_text(self, text: str, width: int, indent: str = "") -> List[str]:
        """Wrap text to specified width with indentation"""
        words = text.split()
        lines = []
        current_line = indent

        for word in words:
            if len(current_line + word) + 1 <= width:
                if current_line != indent:
                    current_line += " "
                current_line += word
            else:
                if current_line != indent:
                    lines.append(current_line)
                current_line = indent + word

        if current_line != indent:
            lines.append(current_line)

        return lines

    def generate_visual_report(self, max_chapters: int = None) -> str:
        """Generate a visual report with progress bars and formatting"""
        matches = self.analyzer.find_phrase_matches()
        coverage_stats = self.analyzer.get_verse_coverage_stats()

        output = []

        # Header
        output.append("┌" + "─" * (self.terminal_width - 2) + "┐")
        output.append("│" + "MARK-MATTHEW VISUAL COMPARISON".center(self.terminal_width - 2) + "│")
        output.append("│" + f"Found {len(matches):,} phrase matches (≥60% similarity)".center(self.terminal_width - 2) + "│")
        output.append("└" + "─" * (self.terminal_width - 2) + "┘")
        output.append("")

        # Overall statistics
        total_mark_words = sum(len(words) for words in self.analyzer.mark_verses.values())
        total_covered_words = sum(stats.get('covered_words', 0) for stats in coverage_stats['mark_coverage'].values())
        overall_coverage = (total_covered_words / total_mark_words * 100) if total_mark_words > 0 else 0

        output.append("📊 OVERALL STATISTICS")
        output.append("─" * 20)
        output.append(f"Mark total words: {total_mark_words:,}")
        output.append(f"Coverage: {self._create_progress_bar(overall_coverage)}")
        output.append("")

        # Group matches by Mark verse
        matches_by_mark_verse = defaultdict(list)
        for match in matches:
            matches_by_mark_verse[match.mark_verse].append(match)

        # Process chapters
        mark_chapters = sorted(set(key[0] for key in self.analyzer.mark_verses.keys()))
        if max_chapters:
            mark_chapters = mark_chapters[:max_chapters]

        for chapter in mark_chapters:
            output.append(f"📖 MARK CHAPTER {chapter}")
            output.append("═" * 50)
            output.append("")

            # Get verses for this chapter
            chapter_verses = [(ch, v) for (ch, v) in self.analyzer.mark_verses.keys() if ch == chapter]
            chapter_verses.sort(key=lambda x: x[1])

            # Chapter summary
            chapter_words = sum(len(self.analyzer.mark_verses[v]) for v in chapter_verses)
            chapter_covered = sum(coverage_stats['mark_coverage'].get(v, {}).get('covered_words', 0) for v in chapter_verses)
            chapter_coverage = (chapter_covered / chapter_words * 100) if chapter_words > 0 else 0

            output.append(f"Chapter {chapter} Overview:")
            output.append(f"  Verses: {len(chapter_verses)} | Words: {chapter_words:,}")
            output.append(f"  Coverage: {self._create_progress_bar(chapter_coverage, 25)}")
            output.append("")

            for verse_key in chapter_verses:
                ch, v = verse_key
                mark_words = self.analyzer.mark_verses[verse_key]
                coverage = coverage_stats['mark_coverage'].get(verse_key, {})
                verse_matches = matches_by_mark_verse.get(verse_key, [])

                # Verse header
                coverage_pct = coverage.get('coverage_percentage', 0)
                match_count = len(verse_matches)

                output.append(f"📝 Mark {ch}:{v}")
                output.append(f"   {self._create_progress_bar(coverage_pct, 20)} ({match_count} matches)")
                output.append("")

                # Show verse text with highlighting
                verse_text = self._highlight_verse_with_visual_markers(mark_words, verse_matches)
                wrapped_lines = self._wrap_text(verse_text, self.terminal_width - 6, "   ")
                output.extend(wrapped_lines)
                output.append("")

                # Show top Matthew parallels
                if verse_matches:
                    # Group by Matthew verse to avoid duplicates
                    matthew_verses = {}
                    for match in verse_matches[:5]:  # Top 5 matches
                        matt_key = match.matthew_verse
                        if matt_key not in matthew_verses:
                            matthew_verses[matt_key] = []
                        matthew_verses[matt_key].append(match)

                    output.append("   🔗 Matthew Parallels:")
                    for matt_key, matt_matches in list(matthew_verses.items())[:3]:  # Top 3 verses
                        matt_ch, matt_v = matt_key
                        best_match = max(matt_matches, key=lambda x: x.similarity)
                        sim_bar = "▓" * int(best_match.similarity * 10) + "░" * (10 - int(best_match.similarity * 10))

                        output.append(f"   │ Matt {matt_ch}:{matt_v} [{sim_bar}] {best_match.similarity:.1%}")

                        # Show the matching text
                        matt_text = best_match.matthew_text[:60] + "..." if len(best_match.matthew_text) > 60 else best_match.matthew_text
                        matt_lines = self._wrap_text(f"     \"{matt_text}\"", self.terminal_width - 6, "   │ ")
                        output.extend(matt_lines)

                    output.append("   └─────")

                output.append("")

        # High-coverage verses summary
        output.append("🎯 HIGHEST COVERAGE VERSES")
        output.append("═" * 30)

        high_coverage = []
        for verse_key, stats in coverage_stats['mark_coverage'].items():
            if stats['coverage_percentage'] > 70:
                high_coverage.append((verse_key, stats))

        high_coverage.sort(key=lambda x: x[1]['coverage_percentage'], reverse=True)

        for verse_key, stats in high_coverage[:10]:
            ch, v = verse_key
            pct = stats['coverage_percentage']
            output.append(f"Mark {ch:2}:{v:2} {self._create_progress_bar(pct, 15)}")

        output.append("")

        # Final summary box
        output.append("┌" + "─" * (self.terminal_width - 2) + "┐")
        output.append("│" + "ANALYSIS COMPLETE".center(self.terminal_width - 2) + "│")
        output.append("│" + f"{overall_coverage:.1f}% of Mark has Matthew parallels".center(self.terminal_width - 2) + "│")
        output.append("│" + f"{len([v for v in coverage_stats['mark_coverage'].values() if v.get('coverage_percentage', 0) > 80])} verses with >80% coverage".center(self.terminal_width - 2) + "│")
        output.append("└" + "─" * (self.terminal_width - 2) + "┘")

        return "\n".join(output)

    def _highlight_verse_with_visual_markers(self, words: List[str], matches: List) -> str:
        """Add visual highlighting to matched phrases"""
        if not matches:
            return " ".join(words)

        # Collect match positions
        match_positions = set()
        for match in matches:
            for i in range(match.phrase_length):
                pos = match.mark_position + i
                if pos < len(words):
                    match_positions.add(pos)

        # Build highlighted text
        result = []
        for i, word in enumerate(words):
            if i in match_positions:
                result.append(f"⟨{word}⟩")  # Use angle brackets for matches
            else:
                result.append(word)

        return " ".join(result)

    def create_side_by_side_comparison(self, mark_chapter: int, matthew_chapter: int) -> str:
        """Create side-by-side comparison of specific chapters"""
        output = []

        # Header
        output.append("┌" + "─" * 39 + "┬" + "─" * 39 + "┐")
        output.append("│" + f"MARK CHAPTER {mark_chapter}".center(39) + "│" + f"MATTHEW CHAPTER {matthew_chapter}".center(39) + "│")
        output.append("├" + "─" * 39 + "┼" + "─" * 39 + "┤")

        # Get verses
        mark_verses = [(ch, v) for (ch, v) in self.analyzer.mark_verses.keys() if ch == mark_chapter]
        matthew_verses = [(ch, v) for (ch, v) in self.analyzer.matthew_verses.keys() if ch == matthew_chapter]

        mark_verses.sort(key=lambda x: x[1])
        matthew_verses.sort(key=lambda x: x[1])

        max_verses = max(len(mark_verses), len(matthew_verses))

        for i in range(max_verses):
            # Mark side
            if i < len(mark_verses):
                verse_key = mark_verses[i]
                ch, v = verse_key
                words = self.analyzer.mark_verses[verse_key]
                text = " ".join(words)[:35] + "..." if len(" ".join(words)) > 35 else " ".join(words)
                mark_text = f"Mark {ch}:{v} {text}"[:39]
            else:
                mark_text = ""

            # Matthew side
            if i < len(matthew_verses):
                verse_key = matthew_verses[i]
                ch, v = verse_key
                words = self.analyzer.matthew_verses[verse_key]
                text = " ".join(words)[:30] + "..." if len(" ".join(words)) > 30 else " ".join(words)
                matt_text = f"Matt {ch}:{v} {text}"[:39]
            else:
                matt_text = ""

            output.append(f"│{mark_text:39}│{matt_text:39}│")

        output.append("└" + "─" * 39 + "┴" + "─" * 39 + "┘")

        return "\n".join(output)


def main():
    """Main function"""
    print("Loading Greek Bible data...")
    parser = GreekBibleParser()
    parser.load_from_docx("bgb.docx")
    analyzer = GreekTextAnalyzer(parser)

    print("Initializing analyzers...")
    mark_matthew = MarkMatthewAnalyzer(analyzer)
    visual = VisualComparison(mark_matthew)

    print("Generating visual report...")
    report = visual.generate_visual_report(max_chapters=2)  # First 2 chapters

    # Save report
    with open("visual_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("Visual report saved to visual_comparison_report.txt")

    # Create side-by-side comparison
    print("Creating side-by-side comparison...")
    side_by_side = visual.create_side_by_side_comparison(1, 3)

    with open("side_by_side_comparison.txt", "w", encoding="utf-8") as f:
        f.write(side_by_side)

    print("Side-by-side comparison saved to side_by_side_comparison.txt")

    # Print sample to console
    print("\n" + "="*60)
    print("SAMPLE OUTPUT")
    print("="*60)
    print(report[:2000] + "\n...\n[Full report saved to file]")


if __name__ == "__main__":
    main()
