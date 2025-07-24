"""
Final Analysis Summary: Mark-Matthew Comparison

Comprehensive summary of phrase overlap analysis between Mark and Matthew,
showing how much of Mark appears in Matthew with statistical insights.
"""

from greek_bible_processor import GreekBibleParser
from greek_bible_api import GreekTextAnalyzer
from mark_matthew_analyzer import MarkMatthewAnalyzer


def generate_executive_summary():
    """Generate executive summary of Mark-Matthew analysis"""

    print("Loading Greek Bible data...")
    parser = GreekBibleParser()
    parser.load_from_docx("bgb.docx")
    analyzer = GreekTextAnalyzer(parser)

    print("Performing analysis...")
    mark_matthew = MarkMatthewAnalyzer(analyzer)
    matches = mark_matthew.find_phrase_matches()
    coverage_stats = mark_matthew.get_verse_coverage_stats()

    # Calculate key statistics
    total_mark_words = sum(len(words) for words in mark_matthew.mark_verses.values())
    total_covered_words = sum(stats.get('covered_words', 0) for stats in coverage_stats['mark_coverage'].values())
    overall_coverage = (total_covered_words / total_mark_words * 100) if total_mark_words > 0 else 0

    total_mark_verses = len(mark_matthew.mark_verses)
    verses_with_coverage = len([v for v in coverage_stats['mark_coverage'].values() if v.get('coverage_percentage', 0) > 0])
    verse_coverage_rate = (verses_with_coverage / total_mark_verses * 100) if total_mark_verses > 0 else 0

    high_coverage_verses = [v for v in coverage_stats['mark_coverage'].values() if v.get('coverage_percentage', 0) > 80]
    medium_coverage_verses = [v for v in coverage_stats['mark_coverage'].values() if 50 <= v.get('coverage_percentage', 0) <= 80]

    # Chapter-by-chapter breakdown
    chapter_stats = {}
    for verse_key, words in mark_matthew.mark_verses.items():
        chapter = verse_key[0]
        if chapter not in chapter_stats:
            chapter_stats[chapter] = {'total_words': 0, 'covered_words': 0, 'verses': 0}

        chapter_stats[chapter]['total_words'] += len(words)
        chapter_stats[chapter]['verses'] += 1
        chapter_stats[chapter]['covered_words'] += coverage_stats['mark_coverage'].get(verse_key, {}).get('covered_words', 0)

    for chapter in chapter_stats:
        total = chapter_stats[chapter]['total_words']
        covered = chapter_stats[chapter]['covered_words']
        chapter_stats[chapter]['coverage_pct'] = (covered / total * 100) if total > 0 else 0

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("MARK-MATTHEW OVERLAP ANALYSIS: EXECUTIVE SUMMARY")
    report.append("=" * 80)
    report.append("")

    report.append("🎯 KEY FINDINGS")
    report.append("-" * 15)
    report.append(f"• {overall_coverage:.1f}% of Mark's text has parallels in Matthew")
    report.append(f"• {len(matches):,} phrase matches found (≥60% similarity)")
    report.append(f"• {verses_with_coverage}/{total_mark_verses} Mark verses ({verse_coverage_rate:.1f}%) have Matthew parallels")
    report.append(f"• {len(high_coverage_verses)} verses have >80% coverage")
    report.append(f"• {len(medium_coverage_verses)} verses have 50-80% coverage")
    report.append("")

    report.append("📊 STATISTICAL OVERVIEW")
    report.append("-" * 20)
    report.append(f"Mark total words:           {total_mark_words:,}")
    report.append(f"Words with Matthew parallels: {total_covered_words:,}")
    report.append(f"Mark total verses:          {total_mark_verses}")
    report.append(f"Verses with parallels:      {verses_with_coverage}")
    report.append(f"Average similarity:         {sum(m.similarity for m in matches) / len(matches):.1%}")
    report.append("")

    report.append("📖 CHAPTER-BY-CHAPTER BREAKDOWN")
    report.append("-" * 30)
    report.append("Chapter | Verses | Words | Coverage")
    report.append("--------|--------|-------|----------")

    chapters = sorted(chapter_stats.keys())
    for chapter in chapters:
        stats = chapter_stats[chapter]
        report.append(f"   {chapter:2d}   |   {stats['verses']:2d}   | {stats['total_words']:4d}  |  {stats['coverage_pct']:5.1f}%")

    report.append("")

    report.append("🏆 HIGHEST COVERAGE VERSES (>80%)")
    report.append("-" * 35)
    high_coverage_list = []
    for verse_key, stats in coverage_stats['mark_coverage'].items():
        if stats['coverage_percentage'] > 80:
            high_coverage_list.append((verse_key, stats))

    high_coverage_list.sort(key=lambda x: x[1]['coverage_percentage'], reverse=True)

    for verse_key, stats in high_coverage_list:
        ch, v = verse_key
        pct = stats['coverage_percentage']
        covered = stats['covered_words']
        total = stats['total_words']
        report.append(f"Mark {ch:2d}:{v:2d} - {pct:5.1f}% ({covered:2d}/{total:2d} words)")

    report.append("")

    report.append("🔍 ANALYSIS INSIGHTS")
    report.append("-" * 20)

    # Find chapters with highest coverage
    best_chapter = max(chapters, key=lambda c: chapter_stats[c]['coverage_pct'])
    worst_chapter = min(chapters, key=lambda c: chapter_stats[c]['coverage_pct'])

    report.append(f"• Highest coverage chapter: Mark {best_chapter} ({chapter_stats[best_chapter]['coverage_pct']:.1f}%)")
    report.append(f"• Lowest coverage chapter: Mark {worst_chapter} ({chapter_stats[worst_chapter]['coverage_pct']:.1f}%)")

    # Identify narrative vs teaching sections
    high_coverage_chapters = [c for c in chapters if chapter_stats[c]['coverage_pct'] > 40]
    low_coverage_chapters = [c for c in chapters if chapter_stats[c]['coverage_pct'] < 30]

    if high_coverage_chapters:
        report.append(f"• High overlap chapters (>40%): {', '.join(map(str, high_coverage_chapters))}")
    if low_coverage_chapters:
        report.append(f"• Low overlap chapters (<30%): {', '.join(map(str, low_coverage_chapters))}")

    report.append("")

    report.append("💡 INTERPRETATION")
    report.append("-" * 15)
    report.append("The analysis reveals that:")
    report.append("")

    if overall_coverage > 35:
        report.append("• Mark shows SUBSTANTIAL overlap with Matthew")
    elif overall_coverage > 25:
        report.append("• Mark shows MODERATE overlap with Matthew")
    else:
        report.append("• Mark shows LIMITED overlap with Matthew")

    if verse_coverage_rate > 70:
        report.append("• Most Mark verses have some Matthew parallel")
    elif verse_coverage_rate > 50:
        report.append("• Many Mark verses have Matthew parallels")
    else:
        report.append("• Few Mark verses have Matthew parallels")

    if len(high_coverage_verses) > 20:
        report.append("• Many verses show near-complete overlap (>80%)")
    elif len(high_coverage_verses) > 10:
        report.append("• Several verses show near-complete overlap (>80%)")
    else:
        report.append("• Few verses show near-complete overlap (>80%)")

    report.append("")
    report.append("=" * 80)
    report.append("ANALYSIS COMPLETE - See detailed reports for verse-level data")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Generate and save executive summary"""
    summary = generate_executive_summary()

    # Save to file
    with open("executive_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("\nExecutive summary saved to executive_summary.txt")
    print("\n" + "="*60)
    print("SUMMARY PREVIEW")
    print("="*60)
    print(summary)


if __name__ == "__main__":
    main()
