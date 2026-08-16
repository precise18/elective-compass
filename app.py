import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

# Page configuration
st.set_page_config(
    page_title="🎓 Elective Compass",
    page_icon="🧭",
    layout="wide"
)

# Title
st.title("🧭 Elective Compass")
st.subheader("Find Your Perfect Elective Match")

# Load data
@st.cache_data
def load_data():
    # Load all text files
    job_path = "data/raw/jobs/*.txt"
    syllabus_path = "data/raw/syllabus_*.txt"
    
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

# Main content
st.write("### 🔍 Select an Elective")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Dropdown for electives
    elective_names = [name.replace('syllabus_', '').replace('_', ' ').title() for name in syllabi_names]
    selected_elective = st.selectbox("Choose an elective:", elective_names)

# Find the selected syllabus
selected_idx = elective_names.index(selected_elective)
selected_syllabus = syllabi_names[selected_idx]

# Get similarity scores
scores = similarity_df.loc[selected_syllabus]
top_matches = scores.sort_values(ascending=False).head(5)

with col2:
    st.write(f"### 📊 Top 5 Job Matches")
    st.write(f"**{selected_elective}**")

# Display results
for idx, (job, score) in enumerate(top_matches.items()):
    job_name = job.replace('job_', '').replace('_', ' ').title()
    
    # Create a progress bar for visual similarity
    progress = int(score * 100)
    
    # Color coding based on score
    if score > 0.3:
        color = "🟢"
    elif score > 0.15:
        color = "🟡"
    else:
        color = "🔴"
    
    st.write(f"{color} **{job_name}**")
    st.progress(progress / 100, text=f"Match Score: {score:.2%}")
    st.write("---")

# Additional info
st.write("---")
st.write("### 💡 What this means")

if top_matches.iloc[0] > 0.3:
    st.success(f"✅ **Strong match!** {selected_elective} is highly aligned with {top_matches.index[0].replace('job_', '').replace('_', ' ').title()}")
elif top_matches.iloc[0] > 0.15:
    st.warning(f"⚠️ **Moderate match.** {selected_elective} has some alignment with {top_matches.index[0].replace('job_', '').replace('_', ' ').title()}")
else:
    st.info(f"ℹ️ **Explore options.** {selected_elective} may prepare you for roles beyond traditional job titles.")

# Footer
st.write("---")
st.caption("🎓 Elective Compass • Built with ❤️ using NLP")