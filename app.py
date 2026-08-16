import streamlit as st
import pandas as pd
import os
import glob
import textwrap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Elective Compass",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HELPER FOR CUSTOM HTML
# ============================================================

def render_html(html):
    """
    Removes indentation from multiline HTML before sending it
    to Streamlit so the HTML is rendered instead of displayed
    as a code block.
    """
    st.markdown(
        textwrap.dedent(html),
        unsafe_allow_html=True
    )


# ============================================================
# CUSTOM DESIGN / CSS
# ============================================================

render_html("""
<style>

    /* --------------------------------------------------------
       IMPORT FONTS
    -------------------------------------------------------- */

    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Poppins:wght@300;400;500;600;700&display=swap');


    /* --------------------------------------------------------
       GLOBAL APP
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 90% 5%,
                rgba(255, 226, 239, 0.55),
                transparent 25%
            ),
            radial-gradient(
                circle at 10% 90%,
                rgba(255, 239, 246, 0.65),
                transparent 30%
            ),
            #fffafd;
    }


    /* Main content */

    .main .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* --------------------------------------------------------
       STREAMLIT HEADER
    -------------------------------------------------------- */

    header[data-testid="stHeader"] {
        background: rgba(255, 250, 253, 0.90);
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #fff0f6 0%,
                #ffe5f0 45%,
                #fff8fb 100%
            );

        border-right: 1px solid #f1cada;
    }


    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }


    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'DM Serif Display', serif !important;
        color: #4a3040 !important;
    }


    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        font-family: 'Poppins', sans-serif !important;
        color: #6f5362;
    }


    /* --------------------------------------------------------
       HERO SECTION
    -------------------------------------------------------- */

    .hero {
        text-align: center;
        padding: 1rem 1rem 3rem 1rem;
    }


    .hero-sparkle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        color: #d47d9f;
        letter-spacing: 8px;
        margin-bottom: 0.7rem;
    }


    .hero-eyebrow {
        font-family: 'Poppins', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #b47791;
        margin-bottom: 0.8rem;
    }


    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 4.2rem;
        line-height: 1.05;
        color: #4a3040;
        margin: 0;
    }


    .hero-subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.05rem;
        font-weight: 400;
        color: #8b6878;
        max-width: 650px;
        margin: 1.2rem auto 0 auto;
        line-height: 1.8;
    }


    .hero-divider {
        width: 70px;
        height: 3px;
        background: #dfa0bb;
        border-radius: 50px;
        margin: 1.8rem auto 0 auto;
    }


    /* --------------------------------------------------------
       SECTION HEADINGS
    -------------------------------------------------------- */

    .section-eyebrow {
        font-family: 'Poppins', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #c27c99;
        margin-bottom: 0.35rem;
    }


    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.1rem;
        color: #4a3040;
        margin-bottom: 0.3rem;
    }


    .section-description {
        font-family: 'Poppins', sans-serif;
        color: #8b6878;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }


    /* --------------------------------------------------------
       SIDEBAR BRAND
    -------------------------------------------------------- */

    .sidebar-brand {
        text-align: center;
        padding: 0.8rem 0 2rem 0;
    }


    .sidebar-symbol {
        font-size: 2rem;
        color: #d47d9f;
        margin-bottom: 0.5rem;
    }


    .sidebar-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.7rem;
        color: #4a3040;
    }


    .sidebar-subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 0.72rem;
        color: #a47788;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }


    /* --------------------------------------------------------
       SIDEBAR INFO CARDS
    -------------------------------------------------------- */

    .info-card {
        background: rgba(255, 255, 255, 0.80);
        border: 1px solid #f1cada;
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 5px 20px rgba(180, 110, 140, 0.06);
    }


    .info-number {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #4a3040;
    }


    .info-label {
        font-family: 'Poppins', sans-serif;
        font-size: 0.75rem;
        color: #a47788;
        margin-top: 0.1rem;
    }


    .sidebar-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 1.25rem;
        color: #4a3040;
        margin: 1.8rem 0 0.8rem 0;
    }


    .sidebar-step {
        font-family: 'Poppins', sans-serif;
        font-size: 0.78rem;
        color: #76596a;
        line-height: 1.7;
        padding: 0.45rem 0;
    }


    .step-number {
        display: inline-flex;
        width: 24px;
        height: 24px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #f5c7da;
        color: #70455a;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.45rem;
    }


    /* --------------------------------------------------------
       ELECTIVE SELECTOR
    -------------------------------------------------------- */

    .selector-card {
        background: #ffffff;
        border: 1px solid #f1cada;
        border-radius: 22px;
        padding: 1.3rem 1.4rem 0.7rem 1.4rem;
        box-shadow: 0 8px 30px rgba(180, 110, 140, 0.07);
        margin-bottom: 2rem;
    }


    /* Selectbox */

    div[data-baseweb="select"] > div {
        background-color: #fff8fb !important;
        border: 1px solid #efbfd3 !important;
        border-radius: 14px !important;
    }


    div[data-baseweb="select"] span {
        font-family: 'Poppins', sans-serif !important;
        color: #4a3040 !important;
    }


    /* --------------------------------------------------------
       CAREER MATCH CARDS
    -------------------------------------------------------- */

    .match-card {
        background: #ffffff;
        border: 1px solid #f1cada;
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 7px 25px rgba(180, 110, 140, 0.06);
    }


    .match-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
    }


    .job-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.35rem;
        color: #4a3040;
    }


    .match-badge {
        font-family: 'Poppins', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.35rem 0.7rem;
        border-radius: 50px;
    }


    .strong-badge {
        background: #f3f9ef;
        color: #67934e;
    }


    .moderate-badge {
        background: #fff7e8;
        color: #b28128;
    }


    .weak-badge {
        background: #fff0f1;
        color: #b15d67;
    }


    .score-label {
        font-family: 'Poppins', sans-serif;
        font-size: 0.75rem;
        color: #8b6878;
        margin-bottom: 0.35rem;
    }


    .progress-background {
        height: 8px;
        background: #f5dce7;
        border-radius: 50px;
        overflow: hidden;
    }


    .progress-fill {
        height: 100%;
        background: linear-gradient(
            90deg,
            #d98aaa,
            #edb3c9
        );
        border-radius: 50px;
    }


    /* --------------------------------------------------------
       BEST MATCH CARD
    -------------------------------------------------------- */

    .best-match-label {
        font-family: 'Poppins', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #c27c99;
        margin-bottom: 0.7rem;
    }


    .best-match-card {
        background:
            linear-gradient(
                145deg,
                #ffffff 0%,
                #fff4f8 100%
            );
        border: 1px solid #efbfd3;
        border-radius: 24px;
        padding: 1.8rem;
        box-shadow: 0 12px 35px rgba(180, 110, 140, 0.10);
        margin-bottom: 1rem;
    }


    .best-match-small {
        font-family: 'Poppins', sans-serif;
        font-size: 0.75rem;
        color: #8b6878;
        margin-bottom: 0.5rem;
    }


    .best-match-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #4a3040;
        margin-bottom: 0.7rem;
    }


    .similarity-pill {
        display: inline-block;
        font-family: 'Poppins', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #6c9857;
        background: #eff8eb;
        border-radius: 50px;
        padding: 0.4rem 0.75rem;
    }


    .recommendation {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        font-family: 'Poppins', sans-serif;
        font-size: 0.78rem;
        line-height: 1.6;
    }


    .recommendation-strong {
        background: #eef8eb;
        color: #557a49;
        border: 1px solid #d9ebd2;
    }


    .recommendation-moderate {
        background: #fff8e9;
        color: #8c6d32;
        border: 1px solid #f4e5bd;
    }


    .recommendation-weak {
        background: #fff2f3;
        color: #98616a;
        border: 1px solid #f0d4d8;
    }


    /* --------------------------------------------------------
       EXPANDER
    -------------------------------------------------------- */

    div[data-testid="stExpander"] {
        border: 1px solid #f1cada !important;
        border-radius: 18px !important;
        background: #ffffff !important;
    }


    div[data-testid="stExpander"] summary {
        font-family: 'Poppins', sans-serif !important;
        color: #4a3040 !important;
    }


    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #f1cada;
        margin-top: 3rem;
    }


    .footer-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #4a3040;
    }


    .footer-text {
        font-family: 'Poppins', sans-serif;
        font-size: 0.7rem;
        color: #a47788;
        margin-top: 0.4rem;
    }


    /* --------------------------------------------------------
       HIDE DEFAULT STREAMLIT BRANDING
    -------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }


    footer {
        visibility: hidden;
    }


</style>
""")


# ============================================================
# GET BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    job_path = os.path.join(BASE_DIR, "data/raw/jobs/*.txt")
    syllabus_path = os.path.join(BASE_DIR, "data/raw/syllabus_*.txt")

    job_files = glob.glob(job_path)
    job_data = []

    for file in job_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(file).replace(".txt", "")
        job_data.append({
            "filename": filename,
            "text": text,
            "type": "job"
        })

    syllabus_files = glob.glob(syllabus_path)
    syllabus_data = []

    for file in syllabus_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(file).replace(".txt", "")
        syllabus_data.append({
            "filename": filename,
            "text": text,
            "type": "syllabus"
        })

    return pd.DataFrame(job_data + syllabus_data)


all_data = load_data()


# ============================================================
# TF-IDF
# ============================================================

@st.cache_data
def get_tfidf_matrix(data):
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(
        data["text"]
    )
    return tfidf_matrix, vectorizer


tfidf_matrix, vectorizer = get_tfidf_matrix(all_data)


# ============================================================
# SEPARATE SYLLABI AND JOBS
# ============================================================

syllabi_indices = all_data[
    all_data["type"] == "syllabus"
].index

job_indices = all_data[
    all_data["type"] == "job"
].index

syllabi_names = all_data.loc[
    syllabi_indices,
    "filename"
].values

job_names = all_data.loc[
    job_indices,
    "filename"
].values


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix[syllabi_indices],
    tfidf_matrix[job_indices]
)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=syllabi_names,
    columns=job_names
)


# ============================================================
# ELECTIVE NAME MAPPING
# ============================================================

ELECTIVE_NAME_MAP = {
    "cloud": "Cloud Computing",
    "mobile": "Mobile Development",
    "blockchain": "Blockchain Development",
    "cybersecurity": "Cybersecurity",
    "data_eng": "Data Engineering",
    "system_integration": "System Integration",
    "qa": "Quality Assurance (QA)"
}

# Reverse mapping: display name -> syllabus filename
DISPLAY_TO_FILE = {}

for name in syllabi_names:
    clean_name = name.replace("syllabus_", "")
    base_name = clean_name.rstrip("0123456789_")

    if base_name in ELECTIVE_NAME_MAP:
        display_name = ELECTIVE_NAME_MAP[base_name]
    else:
        display_name = base_name.replace("_", " ").title()

    if display_name not in DISPLAY_TO_FILE:
        DISPLAY_TO_FILE[display_name] = name

elective_display_names = list(DISPLAY_TO_FILE.keys())


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html("""
    <div class="sidebar-brand">
        <div class="sidebar-symbol">✦</div>
        <div class="sidebar-title">Elective Compass</div>
        <div class="sidebar-subtitle">CAREER • SKILLS • DIRECTION</div>
    </div>
    """)

    render_html(f"""
    <div class="sidebar-heading">Quick Information</div>

    <div class="info-card">
        <div class="info-number">{len(syllabi_names)}</div>
        <div class="info-label">Electives</div>
    </div>

    <div class="info-card">
        <div class="info-number">{len(job_names)}</div>
        <div class="info-label">Job descriptions</div>
    </div>
    """)

    render_html("""
    <div class="sidebar-heading">How it works</div>
    <div class="sidebar-step"><span class="step-number">1</span>Select an elective</div>
    <div class="sidebar-step"><span class="step-number">2</span>Explore career matches</div>
    <div class="sidebar-step"><span class="step-number">3</span>Compare your results</div>
    <div class="sidebar-step"><span class="step-number">4</span>Make your decision</div>
    """)

    render_html("""
    <div class="sidebar-heading">Match Scores</div>
    <div class="sidebar-step">
        <span style="color:#78a96b;font-size:1.1rem;">●</span>
        <strong>Above 30%</strong> Strong match
    </div>
    <div class="sidebar-step">
        <span style="color:#d9a441;font-size:1.1rem;">●</span>
        <strong>15–30%</strong> Moderate match
    </div>
    <div class="sidebar-step">
        <span style="color:#c66c78;font-size:1.1rem;">●</span>
        <strong>Below 15%</strong> Weak match
    </div>
    """)


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">
    <div class="hero-sparkle">✦ ♡ ✦</div>
    <div class="hero-eyebrow">Career • Skills • Direction</div>
    <h1 class="hero-title">Elective Compass</h1>
    <p class="hero-subtitle">
        Find the elective that aligns with your future.
        Explore career paths, discover job opportunities,
        and make your decision with confidence.
    </p>
    <div class="hero-divider"></div>
</div>
""")


# ============================================================
# ELECTIVE SELECTION
# ============================================================

render_html("""
<div class="section-eyebrow">Start here</div>
<div class="section-title">Choose your elective</div>
<div class="section-description">
    Select an elective to discover which career opportunities
    align most closely with its skills and content.
</div>
""")

selected_display = st.selectbox(
    "Choose an elective:",
    elective_display_names,
    label_visibility="collapsed"
)


# ============================================================
# GET SELECTED SYLLABUS
# ============================================================

selected_syllabus = DISPLAY_TO_FILE[selected_display]


# ============================================================
# GET SIMILARITY SCORES
# ============================================================

scores = similarity_df.loc[selected_syllabus]
top_matches = scores.sort_values(ascending=False).head(5)


# ============================================================
# RESULTS HEADING
# ============================================================

render_html(f"""
<div style="margin-top:2.5rem;margin-bottom:1.5rem;">
    <div class="section-eyebrow">Your results</div>
    <div class="section-title">Top Career Matches</div>
    <div class="section-description">
        Showing the strongest job-description matches
        for <strong>{selected_display}</strong>.
    </div>
</div>
""")


# ============================================================
# RESULTS COLUMNS
# ============================================================

col1, col2 = st.columns([1.65, 0.9], gap="large")


# ============================================================
# LEFT COLUMN — TOP 5 MATCHES
# ============================================================

with col1:
    for job, score in top_matches.items():
        job_name = job.replace("job_", "").replace("_", " ").title()

        if score > 0.3:
            label = "Strong match"
            badge_class = "strong-badge"
        elif score > 0.15:
            label = "Moderate match"
            badge_class = "moderate-badge"
        else:
            label = "Weak match"
            badge_class = "weak-badge"

        percentage = score * 100

        render_html(f"""
        <div class="match-card">
            <div class="match-header">
                <div class="job-title">{job_name}</div>
                <div class="match-badge {badge_class}">{label}</div>
            </div>
            <div class="score-label">Match score · {score:.2%}</div>
            <div class="progress-background">
                <div class="progress-fill" style="width:{percentage:.2f}%;"></div>
            </div>
        </div>
        """)


# ============================================================
# RIGHT COLUMN — BEST MATCH
# ============================================================

with col2:
    top_job = top_matches.index[0]
    top_score = top_matches.iloc[0]
    job_name = top_job.replace("job_", "").replace("_", " ").title()

    render_html('<div class="best-match-label">Top Match</div>')

    render_html(f"""
    <div class="best-match-card">
        <div class="best-match-small">Best Career Match</div>
        <div class="best-match-title">{job_name}</div>
        <div class="similarity-pill">↑ {top_score:.2%} similarity</div>
    </div>
    """)

    if top_score > 0.3:
        render_html("""
        <div class="recommendation recommendation-strong">
            This elective strongly aligns with this career path.
            <br><br>
            It may be a strong option if you are interested in developing skills relevant to this type of role.
        </div>
        """)
    elif top_score > 0.15:
        render_html("""
        <div class="recommendation recommendation-moderate">
            This elective has moderate alignment with this career path.
            <br><br>
            Consider exploring the job requirements alongside the elective content.
        </div>
        """)
    else:
        render_html("""
        <div class="recommendation recommendation-weak">
            This elective may prepare you for broader career opportunities.
            <br><br>
            Consider comparing it with the other available electives.
        </div>
        """)


# ============================================================
# ALL MATCHES
# ============================================================

st.write("")

with st.expander("See all career matches"):
    all_scores = scores.sort_values(ascending=False)

    all_matches_df = pd.DataFrame({
        "Job Title": [job.replace("job_", "").replace("_", " ").title() for job in all_scores.index],
        "Match Score": [f"{score:.2%}" for score in all_scores.values],
        "Similarity": [f"{score:.3f}" for score in all_scores.values]
    })

    st.dataframe(
        all_matches_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

render_html(f"""
<div class="footer">
    <div class="footer-title">Elective Compass</div>
    <div class="footer-text">Built with NLP & Streamlit</div>
    <div class="footer-text">{len(syllabi_names)} electives · {len(job_names)} job descriptions</div>
    <div class="footer-text" style="margin-top:0.8rem;">✦ Find your direction ✦</div>
</div>
""")