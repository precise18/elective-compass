import streamlit as st
import pandas as pd
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config FIRST (must be the first Streamlit command)
st.set_page_config(
    page_title="Elective Compass",
    page_icon=":material/explore:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Poppins:wght@300;400;500;600;700&display=swap');

/* ---------- GLOBAL ---------- */

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #fff8fb 0%,
        #fff 45%,
        #fff5f9 100%
    );
    color: #332936;
}

/* ---------- MAIN CONTENT ---------- */

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ---------- HEADINGS ---------- */

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #3b2635 !important;
}

h1 {
    font-size: 3.2rem !important;
    letter-spacing: -1px;
}

h2 {
    font-size: 2rem !important;
}

h3 {
    font-size: 1.5rem !important;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #fff0f6 0%,
        #ffe4ef 100%
    );
    border-right: 1px solid #f4c5d8;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #4a3040 !important;
}

/* ---------- SELECT BOX ---------- */

div[data-baseweb="select"] > div {
    background-color: #fff;
    border: 1.5px solid #efb5cc;
    border-radius: 14px;
}

div[data-baseweb="select"] > div:hover {
    border-color: #d98aaa;
}

/* ---------- PROGRESS BAR ---------- */

div[data-testid="stProgress"] > div > div {
    background-color: #f3d6e1;
}

div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(
        90deg,
        #e89ab8,
        #d979a1
    );
}

/* ---------- METRIC CARD ---------- */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #f1cada;
    padding: 1.5rem;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(190, 110, 145, 0.10);
}

div[data-testid="stMetricLabel"] {
    color: #8b6075 !important;
}

div[data-testid="stMetricValue"] {
    color: #4a3040 !important;
    font-family: 'DM Serif Display', serif !important;
}

/* ---------- ALERTS ---------- */

div[data-testid="stAlert"] {
    border-radius: 16px;
}

/* ---------- EXPANDER ---------- */

div[data-testid="stExpander"] {
    border: 1px solid #f0c7d7;
    border-radius: 16px;
    background-color: #fff;
}

/* ---------- DATAFRAME ---------- */

div[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

/* ---------- DIVIDERS ---------- */

hr {
    border-color: #f1d3df !important;
}

/* ---------- CAPTIONS ---------- */

.stCaption {
    color: #967486 !important;
}

</style>
""", unsafe_allow_html=True)

# Title
# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

st.markdown("""
<div style="
    text-align: center;
    padding: 2rem 1rem 3rem 1rem;
">

    <div style="
        font-family: 'Poppins', sans-serif;
        color: #d47d9f;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 12px;
    ">
        ✦ Career • Skills • Direction ✦
    </div>

    <h1 style="
        font-family: 'DM Serif Display', serif;
        font-size: 4.2rem;
        color: #3b2635;
        margin-bottom: 10px;
    ">
        Elective Compass
    </h1>

    <p style="
        font-family: 'Poppins', sans-serif;
        color: #8b6075;
        font-size: 1.15rem;
        margin: 0 auto;
        max-width: 650px;
    ">
        Find the elective that aligns with your future.
        Explore career paths, discover job opportunities,
        and make your decision with confidence.
    </p>

    <div style="
        margin-top: 25px;
        color: #d98aaa;
        font-size: 1.3rem;
    ">
        ♡ &nbsp; Your career journey starts here &nbsp; ♡
    </div>

</div>
""", unsafe_allow_html=True)

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
    'cloud': 'Cloud Computing',
    'mobile': 'Mobile Development',
    'blockchain': 'Blockchain Development',
    'cybersecurity': 'Cybersecurity',
    'data_eng': 'Data Engineering',
    'system_integration': 'System Integration',
    'qa': 'Quality Assurance (QA)'
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

    st.markdown("""
    <div style="
        text-align: center;
        padding: 10px 0 25px 0;
    ">
        <div style="
            font-size: 2rem;
            color: #d47d9f;
        ">✦</div>

        <div style="
            font-family: 'DM Serif Display', serif;
            font-size: 1.7rem;
            color: #4a3040;
        ">
            Elective Compass
        </div>

        <div style="
            font-size: 0.8rem;
            color: #9b7185;
            margin-top: 5px;
        ">
            Find your direction ♡
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Quick Info")

    st.markdown(f"""
    <div style="
        background: white;
        padding: 14px;
        border-radius: 14px;
        margin-bottom: 10px;
        border: 1px solid #f1cada;
    ">
        <strong>{len(syllabi_names)}</strong><br>
        <span style="color:#967486;">Electives</span>
    </div>

    <div style="
        background: white;
        padding: 14px;
        border-radius: 14px;
        margin-bottom: 20px;
        border: 1px solid #f1cada;
    ">
        <strong>{len(job_names)}</strong><br>
        <span style="color:#967486;">Job Descriptions</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### How it works")

    st.markdown("""
    **01** &nbsp; Select an elective  
    **02** &nbsp; Explore job matches  
    **03** &nbsp; Make your decision ♡
    """)

    st.markdown("---")

    st.markdown("### Match Scores")

    st.markdown("""
    <div style="line-height: 2; color: #6f5262;">
        <strong style="color:#69a85a;">●</strong>
        Strong match &nbsp; <small>above 30%</small><br>

        <strong style="color:#e7ad3d;">●</strong>
        Moderate match &nbsp; <small>15–30%</small><br>

        <strong style="color:#d96b79;">●</strong>
        Weak match &nbsp; <small>below 15%</small>
    </div>
    """, unsafe_allow_html=True)
    

# Main content
st.markdown("### Choose Your Elective")
st.caption("Explore how each elective connects to potential career opportunities.")

# Dropdown with full elective names
selected_display = st.selectbox("Choose an elective:", elective_display_names)

# Get the actual syllabus filename
selected_syllabus = DISPLAY_TO_FILE[selected_display]

# Get similarity scores for the selected syllabus
scores = similarity_df.loc[selected_syllabus]
top_matches = scores.sort_values(ascending=False).head(5)

# --- Display Results ---
st.markdown(f"### Top Career Matches")
st.caption(f"Showing the strongest job-description matches for **{selected_display}**.")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    for job, score in top_matches.items():
        job_name = job.replace('job_', '').replace('_', ' ').title()
        
        # Color coding based on score
        if score > 0.3:
            label = "Strong match"
            badge_color = "#78a96b"
        elif score > 0.15:
            label = "Moderate match"
            badge_color = "#d9a441"
        else:
            label = "Weak match"
            badge_color = "#cf6d78"
        
        
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        ">
            <span style="
                background: {badge_color};
                width: 9px;
                height: 9px;
                border-radius: 50%;
                display: inline-block;
            "></span>

            <strong style="color:#4a3040;">
                {job_name}
            </strong>

            <span style="
                color: {badge_color};
                font-size: 0.8rem;
                font-weight: 600;
            ">
                {label}
            </span>
        </div>
""", unsafe_allow_html=True)
        st.progress(min(score, 1.0), text=f"Score: {score:.2%} - {label}")
        st.write("---")

with col2:
    # Show the top match prominently
    top_job = top_matches.index[0]
    top_score = top_matches.iloc[0]
    job_name = top_job.replace('job_', '').replace('_', ' ').title()
    
    st.markdown("""
    <div style="
        color: #d47d9f;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    ">
        Top Match
    </div>
    """, unsafe_allow_html=True)

    st.metric(
        label="Best Career Match",
        value=job_name,
        delta=f"{top_score:.2%} similarity",
        delta_color="normal"
    )
    
    if top_score > 0.3:
        st.success("This elective strongly aligns with this career path.")
    elif top_score > 0.15:
        st.warning("This elective has moderate alignment with this career path.")
    else:
        st.info("This elective may prepare you for broader roles.")

# --- All Matches Table (Optional) ---
with st.expander("See all career matches"):
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
st.markdown("""
<div style="
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: #967486;
">

    <div style="
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #4a3040;
    ">
        Elective Compass
    </div>

    <div style="
        margin-top: 8px;
        font-size: 0.8rem;
    ">
        Built with NLP & Streamlit
        <br>
        7 electives • 35 job descriptions
    </div>

    <div style="
        margin-top: 15px;
        color: #d47d9f;
    ">
        ✦ Find your direction ✦
    </div>

</div>
""", unsafe_allow_html=True)