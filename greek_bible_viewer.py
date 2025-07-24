import streamlit as st
import pandas as pd
import plotly.express as px

# Import your custom modules (adjust paths as needed)
from greek_bible_processor import GreekBibleParser
from greek_bible_api import GreekTextAnalyzer

class GreekBibleViewer:
    """Interactive web viewer for Greek Bible analysis"""
    
    def __init__(self):
        self.setup_page()
        self.initialize_data()
    
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Greek Bible Viewer",
            page_icon="📖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📖 Greek Bible Interactive Viewer")
        st.markdown("---")
    
    @st.cache_data
    def load_bible_data(_self):
        """Load and cache Bible data - replace with your actual loading code"""
        parser = GreekBibleParser()
        parser.load_from_docx("bgb.docx")
        analyzer = GreekTextAnalyzer(parser)
        return analyzer
    
    def initialize_data(self):
        """Initialize analyzer and cache data"""
        if 'analyzer' not in st.session_state:
            with st.spinner("Loading Bible data..."):
                st.session_state.analyzer = self.load_bible_data()
        
        self.analyzer = st.session_state.analyzer
        
        if self.analyzer is None:
            st.error("Please load Bible data by uncommenting and configuring the load_bible_data method")
            st.info("Update the load_bible_data method with your parser and data file path")
            st.stop()
    
    def sidebar_navigation(self):
        """Create navigation sidebar"""
        st.sidebar.title("Navigation")
        
        mode = st.sidebar.radio(
            "Select Mode:",
            ["📖 Text Reader", "🔍 Word Search", "📊 Statistics", "📈 Analysis"]
        )
        
        return mode
    
    def text_reader_tab(self):
        """Text reading interface"""
        st.header("📖 Text Reader")
        
        # Book selection
        books = self.analyzer.get_books_list()
        selected_book = st.selectbox("Select Book:", books)
        
        if selected_book:
            # Chapter selection
            chapter_count = self.analyzer.get_chapter_count(selected_book)
            chapter_cols = st.columns([1, 3])
            
            with chapter_cols[0]:
                selected_chapter = st.number_input(
                    "Chapter:", 
                    min_value=1, 
                    max_value=chapter_count,
                    value=1
                )
            
            with chapter_cols[1]:
                verse_count = self.analyzer.get_verse_count(selected_book, selected_chapter)
                st.info(f"Chapter {selected_chapter} has {verse_count} verses")
            
            # Display options
            display_cols = st.columns(3)
            with display_cols[0]:
                show_verse_numbers = st.checkbox("Show verse numbers", value=True)
            with display_cols[1]:
                font_size = st.slider("Font size", 12, 24, 16)
            with display_cols[2]:
                line_spacing = st.slider("Line spacing", 1.0, 2.0, 1.4)
            
            # Display text
            st.markdown("---")
            
            chapter_text = self.analyzer.get_chapter_text(selected_book, selected_chapter)
            if chapter_text:
                for i, verse_text in enumerate(chapter_text, 1):
                    verse_display = f"**{i}.** {verse_text}" if show_verse_numbers else verse_text
                    st.markdown(
                        f'<p style="font-size: {font_size}px; line-height: {line_spacing};">{verse_display}</p>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No text found for this chapter")
    
    def word_search_tab(self):
        """Word search interface"""
        st.header("🔍 Word Search")
        
        # Search configuration
        search_cols = st.columns([2, 1, 1])
        
        with search_cols[0]:
            search_term = st.text_input("Search term:", placeholder="Enter Greek word or phrase")
        
        with search_cols[1]:
            search_type = st.selectbox("Search type:", ["Word", "Phrase", "Regex"])
        
        with search_cols[2]:
            exact_match = st.checkbox("Exact match")
            include_accents = st.checkbox("Include accents")
        
        if search_term:
            with st.spinner("Searching..."):
                # Perform search based on type
                if search_type == "Word":
                    results = self.analyzer.search_word(search_term, exact_match, include_accents)
                elif search_type == "Phrase":
                    results = self.analyzer.search_phrase(search_term)
                elif search_type == "Regex":
                    results = self.analyzer.search_regex(search_term)
                else:
                    results = []
            
            # Display results
            if results:
                st.success(f"Found {len(results)} occurrences")
                
                # Results summary
                summary_cols = st.columns(3)
                with summary_cols[0]:
                    books_found = len(set(r.book for r in results))
                    st.metric("Books", books_found)
                
                with summary_cols[1]:
                    avg_position = sum(r.position for r in results) / len(results)
                    st.metric("Avg. Position", f"{avg_position:.1f}")
                
                with summary_cols[2]:
                    if st.button("Export Results"):
                        self.export_search_results(results)
                
                st.markdown("---")
                
                # Results display
                results_per_page = st.slider("Results per page", 5, 50, 20)
                total_pages = (len(results) - 1) // results_per_page + 1
                
                if total_pages > 1:
                    page = st.number_input("Page", 1, total_pages, 1) - 1
                    start_idx = page * results_per_page
                    end_idx = start_idx + results_per_page
                    page_results = results[start_idx:end_idx]
                else:
                    page_results = results
                
                for result in page_results:
                    with st.expander(f"{result.book} {result.chapter}:{result.verse} - {result.word}"):
                        st.write(f"**Context:** {result.context}")
                        st.write(f"**Full verse:** {result.verse_text}")
                        st.write(f"**Position in verse:** {result.position}")
            else:
                st.warning("No results found")
        
        # Word statistics section
        if search_term and search_type == "Word":
            st.markdown("---")
            st.subheader("Word Statistics")
            
            stats = self.analyzer.get_word_statistics(search_term)
            if stats:
                stats_cols = st.columns(4)
                
                with stats_cols[0]:
                    st.metric("Total Frequency", stats.frequency)
                
                with stats_cols[1]:
                    st.metric("Books", len(stats.books))
                
                with stats_cols[2]:
                    first_ref = f"{stats.first_occurrence[0]} {stats.first_occurrence[1]}:{stats.first_occurrence[2]}"
                    st.metric("First Occurrence", first_ref)
                
                with stats_cols[3]:
                    st.metric("Avg. Position", f"{stats.avg_verse_position:.1f}")
                
                # Books breakdown
                if len(stats.books) > 1:
                    st.write("**Books containing this word:**")
                    books_list = ", ".join(sorted(stats.books))
                    st.write(books_list)
    
    def statistics_tab(self):
        """Statistics dashboard"""
        st.header("📊 Statistics")
        
        # Overall statistics
        vocab_stats = self.analyzer.get_vocabulary_stats()
        
        # Key metrics
        metrics_cols = st.columns(4)
        
        with metrics_cols[0]:
            st.metric("Total Words", f"{vocab_stats['total_words']:,}")
        
        with metrics_cols[1]:
            st.metric("Unique Words", f"{vocab_stats['unique_words']:,}")
        
        with metrics_cols[2]:
            st.metric("Type-Token Ratio", f"{vocab_stats['type_token_ratio']:.4f}")
        
        with metrics_cols[3]:
            st.metric("Hapax Legomena", f"{vocab_stats['hapax_legomena']:,}")
        
        st.markdown("---")
        
        # Book-specific statistics
        st.subheader("Book Statistics")
        
        books = self.analyzer.get_books_list()
        selected_books = st.multiselect("Select books for comparison:", books, default=books[:3])
        
        if selected_books:
            book_data = []
            for book in selected_books:
                stats = self.analyzer.get_book_statistics(book)
                if stats:
                    book_data.append({
                        'Book': book,
                        'Words': stats['total_words'],
                        'Unique Words': stats['unique_words'],
                        'Chapters': stats['chapters'],
                        'Verses': stats['verses'],
                        'Avg Words/Verse': round(stats['avg_words_per_verse'], 1)
                    })
            
            if book_data:
                df = pd.DataFrame(book_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                chart_cols = st.columns(2)
                
                with chart_cols[0]:
                    fig_words = px.bar(df, x='Book', y='Words', title='Total Words by Book')
                    st.plotly_chart(fig_words, use_container_width=True)
                
                with chart_cols[1]:
                    fig_verses = px.bar(df, x='Book', y='Verses', title='Verses by Book')
                    st.plotly_chart(fig_verses, use_container_width=True)
        
        # Most common words
        st.markdown("---")
        st.subheader("Most Common Words")
        
        top_n = st.slider("Number of words to show", 10, 100, 25)
        most_common = vocab_stats['most_common_words'][:top_n]
        
        if most_common:
            words_df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
            
            chart_type = st.radio("Chart type:", ["Bar", "Horizontal Bar"], horizontal=True)
            
            if chart_type == "Bar":
                fig = px.bar(words_df, x='Word', y='Frequency', title=f'Top {top_n} Most Common Words')
            else:
                fig = px.bar(words_df, x='Frequency', y='Word', orientation='h', 
                           title=f'Top {top_n} Most Common Words')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
    
    def analysis_tab(self):
        """Advanced analysis tools"""
        st.header("📈 Analysis")
        
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Concordance", "Co-occurrence", "Word Cloud Data", "Frequency Distribution"]
        )
        
        if analysis_type == "Concordance":
            st.subheader("Word Concordance")
            
            min_freq = st.number_input("Minimum frequency:", min_value=1, value=5)
            
            with st.spinner("Generating concordance..."):
                concordance = self.analyzer.get_concordance(min_freq)
            
            if concordance:
                st.success(f"Generated concordance for {len(concordance)} words")
                
                # Search within concordance
                search_concordance = st.text_input("Search concordance:")
                
                matching_words = []
                if search_concordance:
                    matching_words = [word for word in concordance.keys() 
                                    if search_concordance.lower() in word.lower()]
                else:
                    matching_words = list(concordance.keys())
                
                # Display words
                words_per_page = 10
                total_pages = (len(matching_words) - 1) // words_per_page + 1
                
                if total_pages > 1:
                    page = st.number_input("Page", 1, total_pages, 1) - 1
                    start_idx = page * words_per_page
                    end_idx = start_idx + words_per_page
                    page_words = matching_words[start_idx:end_idx]
                else:
                    page_words = matching_words
                
                for word in page_words:
                    entry = concordance[word]
                    with st.expander(f"{word} ({entry.frequency} occurrences)"):
                        for occurrence in entry.occurrences[:10]:  # Show first 10
                            st.write(f"**{occurrence.book} {occurrence.chapter}:{occurrence.verse}** - {occurrence.context}")
                        
                        if len(entry.occurrences) > 10:
                            st.write(f"... and {len(entry.occurrences) - 10} more occurrences")
        
        elif analysis_type == "Co-occurrence":
            st.subheader("Word Co-occurrence Analysis")
            
            co_cols = st.columns([1, 1, 1])
            
            with co_cols[0]:
                word1 = st.text_input("First word:")
            
            with co_cols[1]:
                word2 = st.text_input("Second word:")
            
            with co_cols[2]:
                window = st.number_input("Window size:", min_value=1, max_value=20, value=5)
            
            if word1 and word2:
                with st.spinner("Finding co-occurrences..."):
                    co_results = self.analyzer.find_co_occurrences(word1, word2, window)
                
                if co_results:
                    st.success(f"Found {len(co_results)} co-occurrences")
                    
                    for result in co_results:
                        with st.expander(f"{result.book} {result.chapter}:{result.verse}"):
                            st.write(f"**Context:** {result.context}")
                            st.write(f"**Full verse:** {result.verse_text}")
                else:
                    st.warning("No co-occurrences found")
        
        elif analysis_type == "Word Cloud Data":
            st.subheader("Word Cloud Data")
            
            min_freq = st.number_input("Minimum frequency:", min_value=1, value=10)
            
            word_cloud_data = self.analyzer.get_word_cloud_data(min_freq)
            
            if word_cloud_data:
                st.success(f"Generated data for {len(word_cloud_data)} words")
                
                # Convert to DataFrame for display
                wc_df = pd.DataFrame(list(word_cloud_data.items()), columns=['Word', 'Frequency'])
                wc_df = wc_df.sort_values('Frequency', ascending=False)
                
                st.dataframe(wc_df.head(50), use_container_width=True)
                
                # Download button
                csv = wc_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="word_cloud_data.csv",
                    mime="text/csv"
                )
        
        elif analysis_type == "Frequency Distribution":
            st.subheader("Word Frequency Distribution")
            
            vocab_stats = self.analyzer.get_vocabulary_stats()
            freq_dist = vocab_stats['frequency_distribution']
            
            if freq_dist:
                # Convert to DataFrame
                freq_df = pd.DataFrame(list(freq_dist.items()), columns=['Frequency', 'Word Count'])
                freq_df = freq_df.sort_values('Frequency')
                
                # Plot
                fig = px.bar(freq_df, x='Frequency', y='Word Count', 
                           title='Word Frequency Distribution')
                fig.update_layout(xaxis_title='Word Frequency', yaxis_title='Number of Words')
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.write("**Distribution Summary:**")
                total_words = sum(freq * count for freq, count in freq_dist.items())
                st.write(f"- Words appearing once: {freq_dist.get(1, 0)}")
                st.write(f"- Words appearing 2-5 times: {sum(freq_dist.get(i, 0) for i in range(2, 6))}")
                st.write(f"- Words appearing >10 times: {sum(count for freq, count in freq_dist.items() if freq > 10)}")
    
    def export_search_results(self, results):
        """Export search results to CSV"""
        data = []
        for result in results:
            data.append({
                'Word': result.word,
                'Book': result.book,
                'Chapter': result.chapter,
                'Verse': result.verse,
                'Position': result.position,
                'Context': result.context,
                'Verse Text': result.verse_text
            })
        
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download Search Results",
            data=csv,
            file_name="search_results.csv",
            mime="text/csv"
        )
    
    def run(self):
        """Main application runner"""
        mode = self.sidebar_navigation()
        
        if mode == "📖 Text Reader":
            self.text_reader_tab()
        elif mode == "🔍 Word Search":
            self.word_search_tab()
        elif mode == "📊 Statistics":
            self.statistics_tab()
        elif mode == "📈 Analysis":
            self.analysis_tab()

# Main execution
if __name__ == "__main__":
    viewer = GreekBibleViewer()
    viewer.run()
