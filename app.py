import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import string
import warnings
import json
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🎯 Assessment Recommendation Engine",
    layout="wide",
    page_icon="🔒"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 28px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .recommendation-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .similarity-score {
        font-weight: bold;
        color: #4CAF50;
        font-size: 20px;
    }
    .strict-match {
        background-color: #e8f5e9;
        padding: 8px;
        border-radius: 4px;
        border-left: 4px solid #2E7D32;
        margin: 10px 0;
    }
    .rejected-match {
        background-color: #ffebee;
        padding: 8px;
        border-radius: 4px;
        border-left: 4px solid #C62828;
        margin: 10px 0;
    }
    .stSlider .stSlider > div > div > div > div {
        background-color: #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG components with caching for performance
@st.cache_resource
def load_rag_components():
    """Load and cache RAG components for optimal performance"""
    # Load advanced sentence transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize FAISS index for efficient vector search
    dimension = 384  # Dimension for all-MiniLM-L6-v2
    faiss_index = faiss.IndexFlatL2(dimension)

    return embedding_model, faiss_index

embedding_model, faiss_index = load_rag_components()

# Knowledge base management
KNOWLEDGE_BASE_FILE = "knowledge_base.json"

def load_knowledge_base():
    """Load existing knowledge base with error handling"""
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        try:
            with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Knowledge base loading error: {str(e)}")
            return []
    return []

def save_knowledge_base(knowledge_base):
    """Save knowledge base with error handling"""
    try:
        with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2)
    except Exception as e:
        st.error(f"❌ Failed to save knowledge base: {str(e)}")

def preprocess_text(text):
    """Advanced text preprocessing for better matching"""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove common stop words for better semantic matching
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]

    return ' '.join(words)

def extract_key_skills(text, min_length=3):
    """Extract key skills/technologies from text for strict matching"""
    if not isinstance(text, str):
        return set()

    # Common programming languages, frameworks, and technologies
    tech_keywords = {
        # Languages
        'java', 'python', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
        'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'sql', 'html', 'css',

        # Frameworks & Technologies
        'spring', 'django', 'flask', 'react', 'angular', 'vue', 'node', 'express',
        'laravel', 'rails', 'asp.net', 'hibernate', 'jpa', 'springboot', 'microservices',

        # Databases
        'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlserver', 'redis', 'cassandra',

        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',

        # Data Science
        'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit', 'keras', 'spark', 'hadoop',

        # Other
        'rest', 'graphql', 'api', 'git', 'linux', 'windows', 'agile', 'scrum'
    }

    # Extract words that match our tech keywords
    words = set(preprocess_text(text).split())
    return words.intersection(tech_keywords)

def strict_requirement_matching(query, document, required_skills):
    """
    Strict matching algorithm - document must contain ALL required skills
    Returns True if document matches all requirements, False otherwise
    """
    if not required_skills:
        return True  # No specific requirements, accept all

    # Extract skills from document
    doc_skills = extract_key_skills(document)

    # Check if document contains ALL required skills
    missing_skills = required_skills - doc_skills

    return len(missing_skills) == 0

def get_embeddings(texts, model):
    """Get high-quality embeddings using sentence transformer"""
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, convert_to_tensor=True)

def strict_rag_retrieval(query, knowledge_base, model, index, top_k=25):
    """Strict RAG retrieval with perfect matching - rejects mismatched results"""
    if not knowledge_base or len(knowledge_base) == 0:
        return []

    try:
        # Extract required skills from query for strict matching
        required_skills = extract_key_skills(query)

        # Show what skills we're looking for
        if required_skills:
            st.info(f"🔒 **Strict Match Requirements**: {', '.join(required_skills)}")
        else:
            st.info("🔍 **General Search**: No specific skills detected")

        # Get query embedding with preprocessing
        processed_query = preprocess_text(query)
        query_embedding = get_embeddings(processed_query, model).cpu().numpy()

        # Perform FAISS search with enhanced parameters
        search_k = min(top_k * 3, len(knowledge_base))  # Search wider for strict matching
        distances, indices = index.search(query_embedding, search_k)

        # Apply strict matching filter
        strict_matches = []
        rejected_count = 0

        for i, idx in enumerate(indices[0]):
            if idx < len(knowledge_base):
                doc = knowledge_base[idx].copy()
                doc['similarity_score'] = 1.0 - distances[0][i]  # Convert distance to similarity

                    # Apply strict requirement matching
                if strict_requirement_matching(query, doc['content'], required_skills):
                    strict_matches.append(doc)
                else:
                    rejected_count += 1
                    # Log first few rejections for debugging
                    if rejected_count <= 3:
                        doc_skills = extract_key_skills(doc['content'])
                        missing = required_skills - doc_skills
                        st.write(f"📋 Rejected assessment: Missing skills {missing}")

        # Sort strict matches by similarity score (descending)
        strict_matches.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Show rejection statistics
        if rejected_count > 0:
            st.info(f"🗑️ **Strict Filter Applied**: {rejected_count} assessments rejected for missing required skills")

        # Return top_k strict matches
        return strict_matches[:top_k]

    except Exception as e:
        st.error(f"❌ Strict RAG retrieval failed: {str(e)}")
        return []

def generate_strict_match_response(query, retrieved_docs):
    """Generate comprehensive AI response with strict match analysis"""
    if not retrieved_docs:
        required_skills = extract_key_skills(query)
        if required_skills:
            return f"❌ **NO STRICT MATCHES FOUND**: No assessments contain ALL required skills: {', '.join(required_skills)}. Try broadening your requirements."
        else:
            return "⚠️ No relevant assessments found. Try broadening your search criteria."

    # Extract required skills for explanation
    required_skills = extract_key_skills(query)

    # Analyze the quality of matches
    avg_similarity = np.mean([doc['similarity_score'] for doc in retrieved_docs])
    match_quality = "Excellent" if avg_similarity > 0.8 else "Good" if avg_similarity > 0.6 else "Fair"

    # Generate detailed response
    response = f"🎯 **STRICT MATCH ANALYSIS**\n\n"
    response += f"🔍 **Your Requirements**: {', '.join(required_skills) if required_skills else 'General search'}\n\n"
    response += f"✅ **Match Quality**: {match_quality} (Avg. Similarity: {avg_similarity:.2f})\n\n"
    response += f"📊 **Found {len(retrieved_docs)} assessments with ALL required skills**:\n\n"

    for i, doc in enumerate(retrieved_docs, 1):
        similarity_percent = doc['similarity_score'] * 100
        response += f"{i}. **{doc.get('title', 'Assessment')}**\n"
        response += f"   - 🎯 **Strict Match Confidence**: {similarity_percent:.1f}%\n"
        response += f"   - 🔗 **URL**: {doc.get('url', 'N/A')}\n"
        response += f"   - 📝 **Description**: {doc.get('description', 'N/A')[:100]}...\n"
        response += f"   - ✅ **Contains ALL required skills**: "

        # Show which required skills are present
        doc_skills = extract_key_skills(doc['content'])
        present_skills = required_skills.intersection(doc_skills)
        response += f"{', '.join(present_skills)}\n"
        response += "\n"

    response += "🤖 **STRICT MATCHING GUARANTEE**: These assessments contain ALL your required skills. "
    response += "Any assessments missing required skills have been automatically rejected. "
    response += "This ensures you only see perfect matches for your hiring requirements."

    return response

def build_enhanced_knowledge_base(df):
    """Build comprehensive knowledge base with additional metadata"""
    knowledge_base = []

    for idx, row in df.iterrows():
        # Extract and clean data
        query_text = str(row.get('Query', ''))
        assessment_url = str(row.get('Assessment_url', ''))

        # Create enhanced document with metadata
        doc = {
            'id': f"assessment_{idx}",
            'title': query_text[:50] + "..." if len(query_text) > 50 else query_text,
            'url': assessment_url,
            'description': query_text,
            'content': query_text,
            'original_query': query_text,
            'source': 'uploaded_data',
            'timestamp': datetime.now().isoformat(),
            'keywords': preprocess_text(query_text).split(),
            'length': len(query_text)
        }
        knowledge_base.append(doc)

    return knowledge_base

def update_faiss_index(knowledge_base, model, index):
    """Update FAISS index with optimized embeddings"""
    if not knowledge_base:
        return False

    try:
        # Extract content for embedding
        contents = [doc['content'] for doc in knowledge_base]

        # Generate high-quality embeddings
        embeddings = get_embeddings(contents, model).cpu().numpy()

        # Clear and update FAISS index
        index.reset()
        index.add(embeddings)

        return True
    except Exception as e:
        st.error(f"❌ Failed to update FAISS index: {str(e)}")
        return False

def display_strict_matches(retrieved_docs, use_best_match_25=False):
    """Display strict matches with enhanced visualization"""
    if not retrieved_docs:
        st.warning("⚠️ No strict matches found. Try adjusting your search criteria.")
        return

    # Show success message
    st.success(f"🎯 Found {len(retrieved_docs)} strict matches with ALL required skills!")

    # Show strategy info
    if use_best_match_25:
        st.info("🔍 Using **Best Match 25** strategy for comprehensive strict matching")
    else:
        st.info("🤖 Using **Strict RAG** for precise requirement matching")

    # Display each strict match
    for idx, doc in enumerate(retrieved_docs, 1):
        similarity_percent = doc.get('similarity_score', 0.0) * 100

        # Determine match quality color
        if similarity_percent >= 80:
            quality_color = "#2E7D32"  # Dark Green
            quality_text = "Perfect Match"
        elif similarity_percent >= 60:
            quality_color = "#689F38"  # Light Green
            quality_text = "Strong Match"
        else:
            quality_color = "#AFB42B"  # Amber
            quality_text = "Acceptable Match"

        with st.container():
            st.markdown(f"""
            <div class="recommendation-card" style="border-left: 4px solid {quality_color};">
                <h3 style="color: {quality_color};">🎯 Strict Match #{idx} - {quality_text}</h3>
                <p><strong>📊 Match Confidence:</strong> <span class="similarity-score">{similarity_percent:.1f}%</span></p>
                <p><strong>📝 Assessment Title:</strong> {doc.get('title', 'Assessment')}</p>
                <p><strong>🔗 Assessment URL:</strong> <a href="{doc.get('url', '#')}" target="_blank">{doc.get('url', 'N/A')}</a></p>
                <p><strong>💬 Description:</strong> {doc.get('description', 'N/A')}</p>
                <p><strong>✅ All Required Skills Present</strong></p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function with strict matching focus"""
    st.title("🎯 Assessment Recommendation Engine")
    st.markdown("""
    **Advanced AI-Powered Assessment Matching System**

    Upload your SHL assessment dataset and get intelligent recommendations based on your hiring requirements.
    Uses advanced RAG technology with strict matching capabilities.
    """)

    # Sidebar for focused settings
    with st.sidebar:
        st.header("⚙️ Strict Match Settings")

        st.markdown("### 🎯 Matching Strategy")
        use_best_match_25 = st.toggle(
            "🔥 Best Match 25 Mode",
            value=True,
            help="Search top 25 candidates for comprehensive strict matching (recommended)"
        )

        num_recommendations = st.slider(
            "📊 Number of Recommendations",
            min_value=1,
            max_value=25,
            value=5,
            help="How many strict matches to display"
        )

        # Auto-adjust for Best Match 25
        if use_best_match_25:
            num_recommendations = min(num_recommendations, 25)
            st.info(f"🎯 Optimized for {num_recommendations} best matches from top 25")

        st.markdown("---")
        st.markdown("### 🔒 About Strict Matching")
        st.info("""
        **How It Works:**
        - 🧠 **Semantic Understanding**: Analyzes meaning, not just keywords
        - 🔒 **Strict Requirements**: Only shows assessments with ALL required skills
        - 🗑️ **Automatic Rejection**: Filters out incomplete matches
        - 🎯 **Perfect Accuracy**: Guarantees all requirements are met
        """)

    # Main content
    st.header("📤 Upload Assessment Data")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Excel file (.xlsx)",
        type=["xlsx"],
        help="Upload your assessment dataset with 'Query' and 'Assessment_url' columns"
    )

    if uploaded_file:
        try:
            # Read and validate data
            df = pd.read_excel(uploaded_file)
            df = df.dropna(how='all')  # Remove empty rows

            if len(df) == 0:
                st.error("❌ Uploaded file is empty")
                return

            st.success(f"✅ File uploaded successfully! Found {len(df)} assessments")

            # Validate required columns
            if "Query" not in df.columns or "Assessment_url" not in df.columns:
                st.error("❌ Excel file must contain 'Query' and 'Assessment_url' columns")
                return

            # Show data preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head())

            # Process data with progress
            with st.spinner("🤖 Building strict match knowledge base..."):
                progress_bar = st.progress(0)

                # Step 1: Preprocess data
                progress_bar.progress(20)
                df['Processed_Query'] = df['Query'].apply(preprocess_text)

                # Step 2: Build enhanced knowledge base
                progress_bar.progress(50)
                knowledge_base = build_enhanced_knowledge_base(df)

                # Step 3: Update FAISS index
                progress_bar.progress(80)
                index_success = update_faiss_index(knowledge_base, embedding_model, faiss_index)

                # Step 4: Save knowledge base
                progress_bar.progress(90)
                save_knowledge_base(knowledge_base)

                progress_bar.progress(100)
                st.success("✅ Strict match knowledge base ready!")

            # Query section
            st.header("🎯 Find Strict Matches")

            user_query = st.text_area(
                "Enter your hiring requirement or job description",
                placeholder="Example: Hiring Java developers with SQL experience",
                height=150
            )

            if st.button("🔍 Get Strict Matches", disabled=not user_query.strip()):
                if not user_query.strip():
                    st.warning("⚠️ Please enter your hiring requirements")
                    return

                with st.spinner("🤖 Finding strict matches with ALL required skills..."):
                    try:
                        # Load knowledge base
                        kb = load_knowledge_base()
                        if not kb:
                            st.error("❌ Knowledge base is empty. Please upload data first.")
                            return

                        # Determine retrieval strategy
                        retrieval_k = 25 if use_best_match_25 else max(num_recommendations * 3, 15)

                        # Perform strict RAG retrieval
                        strict_matches = strict_rag_retrieval(
                            user_query, kb, embedding_model, faiss_index, top_k=retrieval_k
                        )

                        if not strict_matches:
                            # Show helpful message with required skills
                            required_skills = extract_key_skills(user_query)
                            if required_skills:
                                st.warning(f"❌ **NO STRICT MATCHES**: No assessments contain ALL required skills: {', '.join(required_skills)}")
                                st.info("💡 **Suggestion**: Try broadening your requirements or check if your skills are spelled correctly")
                            else:
                                st.warning("⚠️ No relevant assessments found. Try broadening your search criteria.")
                            return

                        # Limit to requested number of recommendations
                        final_matches = strict_matches[:num_recommendations]

                        # Generate AI response
                        ai_response = generate_strict_match_response(user_query, final_matches)

                        # Display results
                        st.markdown("### 📝 AI Strict Match Analysis")
                        st.text_area("AI Analysis:", value=ai_response, height=250, disabled=True)

                        # Display individual strict matches
                        display_strict_matches(final_matches, use_best_match_25)

                        # Show detailed metrics
                        with st.expander("📊 Advanced Match Metrics"):
                            metrics_df = pd.DataFrame([{
                                'Rank': i+1,
                                'Title': doc.get('title', 'N/A'),
                                'Similarity': f"{doc.get('similarity_score', 0.0)*100:.1f}%",
                                'URL': doc.get('url', 'N/A'),
                                'All Skills': '✅ Yes'
                            } for i, doc in enumerate(final_matches)])

                            st.dataframe(metrics_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Strict match search failed: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
