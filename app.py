import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import string
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Assessment Recommendation Engine",
    layout="wide",
    page_icon="🎯"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .recommendation-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .similarity-score {
        font-weight: bold;
        color: #4CAF50;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_text(text):
    """Preprocess text by removing punctuation and converting to lowercase"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_topic_embedding(text, lda_model, vectorizer, n_topics=5):
    """Get topic distribution embedding using LDA"""
    try:
        # Transform text to bag of words
        bow = vectorizer.transform([text])

        # Get topic distribution
        topic_dist = lda_model.transform(bow)

        return topic_dist.flatten()
    except:
        return np.zeros(n_topics)

def hybrid_similarity(tfidf_sim, lda_sim, alpha=0.7):
    """Combine TF-IDF and LDA similarities using weighted average"""
    return alpha * tfidf_sim + (1 - alpha) * lda_sim

def main():
    st.title("🎯 Advanced Assessment Recommendation Engine")
    st.markdown("""
    Upload your SHL assessment dataset and get intelligent recommendations based on your hiring requirements.
    Uses multiple recommendation algorithms for accurate results.
    """)

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        recommendation_method = st.selectbox(
            "Recommendation Method",
            ["TF-IDF (Fast)", "LDA (Semantic)", "Hybrid (Best)"],
            help="Choose the recommendation algorithm to use"
        )

        top_n = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5,
            help="How many assessment recommendations to display"
        )

        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        This recommendation engine uses:
        - **TF-IDF**: Fast text-based matching
        - **LDA**: Topic modeling for semantic understanding
        - **Hybrid**: Combines both for best results
        """)

    # Main content
    st.header("📤 Upload Assessment Data")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Excel file (.xlsx)",
        type=["xlsx"],
        help="Upload your assessment dataset containing 'Query' and 'Assessment_url' columns"
    )

    if uploaded_file:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            df.fillna("", inplace=True)

            st.success("✅ File uploaded successfully!")
            st.write(f"📊 Dataset contains {len(df)} assessments")

            # Show data preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head())

            # Validate required columns
            if "Query" not in df.columns or "Assessment_url" not in df.columns:
                st.error("❌ Excel file must contain 'Query' and 'Assessment_url' columns")
                return

            # Preprocess data
            st.write("🔄 Processing data...")
            progress_bar = st.progress(0)

            # Preprocess queries
            progress_bar.progress(20)
            df['Processed_Query'] = df['Query'].apply(preprocess_text)

            # TF-IDF Vectorization
            progress_bar.progress(40)
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Query'])

            # LDA Model for semantic analysis
            progress_bar.progress(60)
            try:
                # Create bag of words representation
                count_vectorizer = CountVectorizer(max_features=1000)
                bow_matrix = count_vectorizer.fit_transform(df['Processed_Query'])

                # Train LDA model
                n_topics = min(5, len(df))  # Use fewer topics for small datasets
                lda_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=10
                )
                lda_matrix = lda_model.fit_transform(bow_matrix)

                # Get LDA embeddings for all queries
                lda_embeddings = lda_matrix
            except Exception as e:
                st.warning(f"⚠️ LDA model training failed: {str(e)}. Falling back to TF-IDF only.")
                lda_model = None
                count_vectorizer = None
                lda_embeddings = None

            progress_bar.progress(100)
            st.success("✅ Data processing complete!")

            # User input section
            st.header("🎯 Find Assessments")

            user_input = st.text_area(
                "Enter your hiring requirement or job description",
                placeholder="Example: Hiring a Python developer with machine learning and SQL experience for a data science role",
                height=150
            )

            if st.button("🔍 Get Recommendations", disabled=not user_input.strip()):
                if not user_input.strip():
                    st.warning("⚠️ Please enter a query")
                    return

                with st.spinner("🤖 Analyzing and finding best matches..."):
                    # Preprocess user input
                    processed_input = preprocess_text(user_input)

                    # TF-IDF similarity
                    user_tfidf = tfidf_vectorizer.transform([processed_input])
                    tfidf_scores = cosine_similarity(user_tfidf, tfidf_matrix)[0]

                    # LDA similarity
                    if lda_model is not None:
                        user_lda = get_topic_embedding(user_input, lda_model, count_vectorizer, n_topics)
                        lda_scores = cosine_similarity(
                            [user_lda],
                            lda_embeddings
                        )[0]
                    else:
                        lda_scores = tfidf_scores  # Fallback

                    # Combine scores based on selected method
                    if recommendation_method == "TF-IDF (Fast)":
                        final_scores = tfidf_scores
                    elif recommendation_method == "LDA (Semantic)":
                        final_scores = lda_scores
                    else:  # Hybrid
                        final_scores = hybrid_similarity(tfidf_scores, lda_scores)

                    # Get top recommendations
                    top_indices = final_scores.argsort()[::-1][:top_n]
                    results = df.iloc[top_indices].copy()
                    results["Similarity_Score"] = final_scores[top_indices]

                    # Display results
                    st.success(f"✅ Found {len(results)} recommendations!")

                    # Show recommendation method used
                    st.info(f"📊 Using: **{recommendation_method}** method")

                    # Display each recommendation
                    for idx, (_, row) in enumerate(results.iterrows(), 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h3>🎯 Recommendation #{idx}</h3>
                                <p><strong>🔗 Assessment Link:</strong> <a href="{row['Assessment_url']}" target="_blank">{row['Assessment_url']}</a></p>
                                <p><strong>📝 Original Query:</strong> {row['Query']}</p>
                                <p><strong>📊 Similarity Score:</strong> <span class="similarity-score">{row['Similarity_Score']:.3f}</span></p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Show detailed scores table
                    with st.expander("📊 View Detailed Scores"):
                        scores_df = results[['Query', 'Assessment_url', 'Similarity_Score']].copy()
                        st.dataframe(scores_df.style.format({'Similarity_Score': '{:.3f}'}))

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
