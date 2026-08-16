import streamlit as st
import pandas as pd
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config FIRST (must be the first Streamlit command)
st.set_page_config(
    page_title="🎓 Elective Compass",
    page_icon="🧭",
    layout="wide"
)

# Title
st.title("🧭 Elective Compass")
st.subheader("Find Your Perfect Elective Match")

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
@st.cache_data
def load_data():
    job_path = os.path.join(BASE_DIR, "data/raw/jobs/*.txt")
    syllabus_path = os.path.join(BASE_DIR, "data/raw/syllabus_*.txt")
    
    job_files = glob.glob(job_path)
    job_data = []
    for file in job_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            filename = os.path.basename(file).replace('.txt', '')
            job_data.append({'filename': filename, 'text': text, 'type': 'job'})
    
    syllabus_files = glob.glob(syllabus_path)
    syllabus_data = []
    for file in syllabus_files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            filename = os.path.basename(file).replace('.txt', '')
            syllabus_data.append({'filename': filename, 'text': text, 'type': 'syllabus'})
    
    return pd.DataFrame(job_data + syllabus_data)

# Load and process
all_data = load_data()

# TF-IDF
@st.cache_data
def get_tfidf_matrix(data):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['text'])
    return tfidf_matrix, vectorizer

tfidf_matrix, vectorizer = get_tfidf_matrix(all_data)

# Separate syllabi and jobs
syllabi_indices = all_data[all_data['type'] == 'syllabus'].index
job_indices = all_data[all_data['type'] == 'job'].index
syllabi_names = all_data.loc[syllabi_indices, 'filename'].values
job_names = all_data.loc[job_indices, 'filename'].values

# Calculate similarity
similarity_matrix = cosine_similarity(tfidf_matrix[syllabi_indices], tfidf_matrix[job_indices])
similarity_df = pd.DataFrame(similarity_matrix, index=syllabi_names, columns=job_names)

# --- ELECTIVE NAME MAPPING (Full Names) ---
# Map short names to full display names
ELECTIVE_NAME_MAP = {
    'cloud': '☁️ Cloud Computing',
    'mobile': '📱 Mobile Development',
    'blockchain': '⛓️ Blockchain Development',
    'cybersecurity': '🔒 Cybersecurity',
    'data_eng': '📊 Data Engineering',
    'system_integration': '🔗 System Integration',
    'qa': '✅ Quality Assurance (QA)'
}

# Reverse mapping: display name -> syllabus filename
DISPLAY_TO_FILE = {}

for name in syllabi_names:
    # Remove 'syllabus_' prefix
    clean_name = name.replace('syllabus_', '')
    # Remove numbers and underscores at the end (like _1, _2, etc.)
    base_name = clean_name.rstrip('0123456789_')
    # Get the full display name from the map
    if base_name in ELECTIVE_NAME_MAP:
        display_name = ELECTIVE_NAME_MAP[base_name]
    else:
        # Fallback: convert to title case
        display_name = base_name.replace('_', ' ').title()
    # Keep the first occurrence only
    if display_name not in DISPLAY_TO_FILE:
        DISPLAY_TO_FILE[display_name] = name

elective_display_names = list(DISPLAY_TO_FILE.keys())

# Sidebar
with st.sidebar:
    st.header("📌 Quick Info")
    st.write(f"📚 **{len(syllabi_names)}** Electives")
    st.write(f"💼 **{len(job_names)}** Job Descriptions")
    st.write("---")
    st.write("🎯 **How it works:**")
    st.write("1. Select an elective below")
    st.write("2. See top job matches")
    st.write("3. Make your decision!")
    st.write("---")
    st.write("🔢 **Match Scores:**")
    st.write("🟢 **> 30%** = Strong match")
    st.write("🟡 **15-30%** = Moderate match")
    st.write("🔴 **< 15%** = Weak match")

# Main content
st.write("### 🔍 Select an Elective")

# Dropdown with full elective names
selected_display = st.selectbox("Choose an elective:", elective_display_names)

# Get the actual syllabus filename
selected_syllabus = DISPLAY_TO_FILE[selected_display]

# Get similarity scores for the selected syllabus
scores = similarity_df.loc[selected_syllabus]
top_matches = scores.sort_values(ascending=False).head(5)

# --- Display Results ---
st.write(f"### 📊 Top 5 Job Matches for **{selected_display}**")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    for job, score in top_matches.items():
        job_name = job.replace('job_', '').replace('_', ' ').title()
        
        # Color coding based on score
        if score > 0.3:
            color = "🟢"
            label = "Strong match"
        elif score > 0.15:
            color = "🟡"
            label = "Moderate match"
        else:
            color = "🔴"
            label = "Weak match"
        
        st.write(f"{color} **{job_name}**")
        st.progress(min(score, 1.0), text=f"Score: {score:.2%} - {label}")
        st.write("---")

with col2:
    # Show the top match prominently
    top_job = top_matches.index[0]
    top_score = top_matches.iloc[0]
    job_name = top_job.replace('job_', '').replace('_', ' ').title()
    
    st.metric(
        label="🏆 Best Match",
        value=job_name,
        delta=f"{top_score:.2%}",
        delta_color="normal"
    )
    
    if top_score > 0.3:
        st.success("✅ This elective strongly aligns with this career path!")
    elif top_score > 0.15:
        st.warning("⚠️ This elective has moderate alignment with this career path.")
    else:
        st.info("ℹ️ This elective may prepare you for broader roles.")

# --- All Matches Table (Optional) ---
with st.expander("📋 See all matches"):
    # Create a DataFrame with all matches sorted
    all_scores = scores.sort_values(ascending=False)
    all_matches_df = pd.DataFrame({
        'Job Title': [job.replace('job_', '').replace('_', ' ').title() for job in all_scores.index],
        'Match Score': [f"{score:.2%}" for score in all_scores.values],
        'Similarity': [f"{score:.3f}" for score in all_scores.values]
    })
    st.dataframe(all_matches_df, use_container_width=True)

# --- Footer ---
st.write("---")
st.caption("🎓 Elective Compass • Built with ❤️ using NLP and Streamlit")
st.caption("📊 Data: 7 Electives × 5 Job Descriptions each = 35 job postings")