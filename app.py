import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import textwrap

import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Survey data loader (cached)
@st.cache_data
def load_survey():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/raw/survey.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["satisfaction"]    = pd.to_numeric(df["satisfaction"], errors="coerce")
    df["had_enough_info"] = pd.to_numeric(df["had_enough_info"], errors="coerce")
    df["would_choose_again"] = df["would_choose_again"].str.strip()
    df["elective"] = df["elective"].str.strip()
    # Normalise "Data Science" -> kept as-is (it is a real WTC elective in the survey)
    return df


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Elective Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HELPER
# ============================================================

def render_html(html):
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


# ============================================================
# DESIGN TOKENS  (professional dark-navy / slate theme)
# ============================================================
#
#  Primary navy  : #0f1f3d
#  Accent blue   : #2563eb
#  Accent teal   : #0d9488
#  Surface light : #f8fafc
#  Border        : #e2e8f0
#  Text primary  : #0f172a
#  Text muted    : #64748b
#  Green badge   : #166534 / #dcfce7
#  Amber badge   : #92400e / #fef3c7
#  Red badge     : #991b1b / #fee2e2
# ============================================================

render_html("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    header[data-testid="stHeader"] {
        background: #0f1f3d;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0f1f3d;
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        font-family: 'Inter', sans-serif !important;
        color: #94a3b8 !important;
    }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #0f1f3d 0%, #1e3a5f 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        color: #ffffff;
    }
    .hero-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-family: 'Sora', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        color: #ffffff;
        margin: 0 0 1rem 0;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 400;
        color: #cbd5e1;
        max-width: 600px;
        line-height: 1.7;
        margin: 0;
    }
    .hero-divider {
        width: 50px;
        height: 3px;
        background: #2563eb;
        border-radius: 50px;
        margin: 1.5rem 0 0 0;
    }

    /* ── Section headings ── */
    .section-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #2563eb;
        margin-bottom: 0.3rem;
    }
    .section-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }
    .section-description {
        font-family: 'Inter', sans-serif;
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }

    /* ── Sidebar brand ── */
    .sidebar-brand {
        padding: 1rem 1.2rem 2rem 1.2rem;
    }
    .sidebar-logo {
        font-family: 'Sora', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .sidebar-tagline {
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #60a5fa;
    }

    /* ── Sidebar info pills ── */
    .stat-row {
        display: flex;
        gap: 0.6rem;
        margin: 0 1.2rem 1.4rem 1.2rem;
    }
    .stat-pill {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 0.7rem 0.5rem;
        text-align: center;
    }
    .stat-num {
        font-family: 'Sora', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    .stat-lbl {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        color: #94a3b8;
        margin-top: 0.1rem;
    }

    /* ── Sidebar nav items ── */
    .sb-section {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #475569;
        padding: 0 1.2rem;
        margin: 1.6rem 0 0.6rem 0;
    }
    .sb-step {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #94a3b8;
        padding: 0.45rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        line-height: 1.5;
    }
    .sb-num {
        display: inline-flex;
        width: 20px;
        height: 20px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #1e3a5f;
        color: #60a5fa;
        font-size: 0.65rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .sb-legend {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #94a3b8;
        padding: 0.3rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Job match cards ── */
    .match-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.15rem 1.3rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(15,31,61,0.06);
        transition: box-shadow 0.15s ease;
    }
    .match-card:hover {
        box-shadow: 0 4px 16px rgba(37,99,235,0.10);
        border-color: #bfdbfe;
    }
    .match-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.65rem;
    }
    .job-title-text {
        font-family: 'Sora', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
    }
    .match-badge {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 0.28rem 0.65rem;
        border-radius: 50px;
        white-space: nowrap;
    }
    .strong-badge  { background: #dcfce7; color: #166534; }
    .moderate-badge{ background: #fef3c7; color: #92400e; }
    .weak-badge    { background: #fee2e2; color: #991b1b; }
    .score-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .progress-bg {
        height: 6px;
        background: #e2e8f0;
        border-radius: 50px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #2563eb, #0d9488);
        border-radius: 50px;
    }

    /* ── Best match panel ── */
    .best-panel {
        background: #0f1f3d;
        border-radius: 16px;
        padding: 1.6rem;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .best-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 0.6rem;
    }
    .best-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .best-score {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #34d399;
        background: rgba(52,211,153,0.12);
        border-radius: 50px;
        padding: 0.3rem 0.7rem;
        margin-bottom: 1rem;
    }
    .rec-box {
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        line-height: 1.6;
    }
    .rec-strong   { background: rgba(52,211,153,0.12); color: #a7f3d0; border: 1px solid rgba(52,211,153,0.2); }
    .rec-moderate { background: rgba(251,191,36,0.10); color: #fde68a; border: 1px solid rgba(251,191,36,0.2); }
    .rec-weak     { background: rgba(248,113,113,0.10); color: #fca5a5; border: 1px solid rgba(248,113,113,0.2); }

    /* ── Job detail panel (inside expander) ── */
    .job-detail-header {
        font-family: 'Sora', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.4rem;
    }
    .job-detail-body {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #334155;
        white-space: pre-wrap;
        line-height: 1.7;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        max-height: 420px;
        overflow-y: auto;
    }

    /* ── Keyword chips ── */
    .chip-container { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
    .chip {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 500;
        padding: 0.28rem 0.65rem;
        border-radius: 50px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
    }
    .chip-gap {
        background: #fff7ed;
        border-color: #fed7aa;
        color: #c2410c;
    }

    /* ── Quiz ── */
    .quiz-q-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem 0.4rem 1.4rem;
        margin-bottom: 0.4rem;
    }
    .quiz-q-text {
        font-family: 'Sora', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }
    .quiz-result-panel {
        background: linear-gradient(135deg, #0f1f3d 0%, #1e3a5f 100%);
        border-radius: 16px;
        padding: 2rem;
        color: #ffffff;
        margin-bottom: 1.2rem;
    }
    .qr-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 0.5rem;
    }
    .qr-name {
        font-family: 'Sora', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.6rem;
    }
    .qr-pill {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: #34d399;
        background: rgba(52,211,153,0.12);
        border-radius: 50px;
        padding: 0.35rem 0.8rem;
    }
    .qr-body {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.7;
        margin-top: 0.8rem;
    }

    /* ── Info/tip boxes ── */
    .tip-box {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #334155;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0d9488;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        line-height: 1.7;
        margin-top: 1rem;
    }
    .insight-box {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #334155;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        line-height: 1.8;
        margin-top: 1rem;
    }

    /* ── Streamlit widget overrides ── */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] span {
        font-family: 'Inter', sans-serif !important;
        color: #0f172a !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif !important;
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* Make radio options clearly visible */
    .stRadio > div { gap: 0.5rem !important; }
    .stRadio label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        color: #0f172a !important;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.55rem 0.9rem !important;
        cursor: pointer;
        transition: all 0.12s ease;
        display: block;
        width: 100%;
    }
    .stRadio label:hover {
        border-color: #2563eb;
        background: #eff6ff;
    }
    /* Selected state */
    .stRadio [data-testid="stMarkdownContainer"] p {
        font-family: 'Inter', sans-serif !important;
        color: #0f172a !important;
    }

    /* ── How-to steps (main pane) ── */
    .how-to-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0 2rem 0;
    }
    .how-to-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 3px solid #2563eb;
        border-radius: 12px;
        padding: 1.2rem 1.1rem;
        box-shadow: 0 1px 4px rgba(15,31,61,0.05);
    }
    .how-to-num {
        font-family: 'Sora', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #2563eb;
        opacity: 0.25;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .how-to-title {
        font-family: 'Sora', sans-serif;
        font-size: 0.9rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .how-to-body {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.55;
    }

    /* ── Survey page ── */
    .survey-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    .survey-stat {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(15,31,61,0.05);
    }
    .survey-stat-num {
        font-family: 'Sora', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
    }
    .survey-stat-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.3rem;
        line-height: 1.4;
    }
    .survey-stat-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        color: #94a3b8;
        margin-top: 0.15rem;
    }
    .quote-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 1px 3px rgba(15,31,61,0.04);
    }
    .quote-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.65;
        font-style: italic;
    }
    .quote-meta {
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-style: normal;
    }
    .calibration-badge {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 50px;
        margin-left: 0.4rem;
    }
    .badge-boosted  { background: #dcfce7; color: #166534; }
    .badge-reduced  { background: #fee2e2; color: #991b1b; }
    .badge-unchanged{ background: #f1f5f9; color: #475569; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    .footer-title {
        font-family: 'Sora', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
    }
    .footer-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: #94a3b8;
        margin-top: 0.35rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

</style>
""")


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ELECTIVE_NAME_MAP = {
    "cloud":              "Cloud Computing",
    "mobile":             "Mobile Development",
    "blockchain":         "Blockchain Development",
    "cybersecurity":      "Cybersecurity",
    "data_eng":           "Data Engineering",
    "system_integration": "System Integration",
    "qa":                 "Quality Assurance",
}

ELECTIVE_EMOJI = {
    "Cloud Computing":        "☁️",
    "Mobile Development":     "📱",
    "Blockchain Development": "⛓️",
    "Cybersecurity":          "🔒",
    "Data Engineering":       "📊",
    "System Integration":     "🔗",
    "Quality Assurance":      "✅",
}

# Professional, distinct per-elective colours (no pink)
ELECTIVE_COLORS = {
    "Cloud Computing":        "#2563eb",   # blue
    "Mobile Development":     "#0d9488",   # teal
    "Blockchain Development": "#7c3aed",   # violet
    "Cybersecurity":          "#dc2626",   # red
    "Data Engineering":       "#d97706",   # amber
    "System Integration":     "#0891b2",   # cyan
    "Quality Assurance":      "#16a34a",   # green
}

# ── Quiz questions ───────────────────────────────────────────────────────────
#
# Weights are calibrated using real WeThinkCode_ student survey data (n=36).
# Survey-derived satisfaction multipliers (computed in notebooks/01_data_loading.ipynb §12.6):
#
#   system_integration : ×1.0355  → +1 point added to its primary answers
#   mobile             : ×1.0332  → +1 point added to its primary answers
#   cloud              : ×1.0116  → +1 point added to two primary answers
#   cybersecurity      : ×1.0084  → +1 point added to one primary answer
#   data_eng           : ×0.9722  → -1 point removed from one primary answer
#   qa                 : ×0.9557  → -1 point removed from two primary answers
#   blockchain         : no survey data → weights unchanged
#
# The adjustment is applied as ±1 integer delta on the elective's strongest
# answer per question. Floating-point weights are deliberately avoided so the
# quiz remains readable and auditable. Maximum effect = ≈ ±4% of max score.
# NLP signal (TF-IDF cosine similarity) still dominates the final recommendation.
# ─────────────────────────────────────────────────────────────────────────────
QUIZ_QUESTIONS = [
    {
        "id": "q1",
        "question": "When you build something, what outcome excites you most?",
        "options": {
            # mobile ×1.0332 → primary weight 3 → 3+1=4
            "A beautifully designed app on someone's phone":                       {"mobile": 4, "qa": 1},
            # data_eng ×0.9722 → primary weight 3 → 3-1=2
            "An automated system that moves and transforms data at scale":          {"data_eng": 2, "cloud": 1},
            "A smart contract that executes trustlessly on a blockchain":           {"blockchain": 3},
            "A secure, battle-tested application that can't be breached":          {"cybersecurity": 3, "qa": 1},
            "A reliable cloud infrastructure running 24/7":                        {"cloud": 3, "system_integration": 1},
            # system_integration ×1.0355 → primary weight 3 → 3+1=4
            "A distributed system where many services work together seamlessly":   {"system_integration": 4, "cloud": 1},
        }
    },
    {
        "id": "q2",
        "question": "Which activity sounds most like your ideal work day?",
        "options": {
            # mobile ×1.0332 → 3 → 4
            "Writing Flutter widgets and tweaking UI animations":                  {"mobile": 4},
            # cybersecurity ×1.0084 → 3 → 4 (one question boost)
            "Investigating a suspicious network alert and tracing an attacker":    {"cybersecurity": 4},
            "Building an ETL pipeline that ingests a million records per hour":    {"data_eng": 2, "cloud": 1},
            # qa ×0.9557 → 3 → 2
            "Designing and running test plans to catch bugs before users do":      {"qa": 2},
            # cloud ×1.0116 → 3 → 4 (one question boost)
            "Deploying infrastructure on AWS and setting up auto-scaling":         {"cloud": 4},
            "Writing a DeFi smart contract with tokenomics logic":                 {"blockchain": 3},
            # system_integration ×1.0355 → 3 → 4
            "Building a message queue that connects five different microservices": {"system_integration": 4},
        }
    },
    {
        "id": "q3",
        "question": "Which tech trend excites you the most?",
        "options": {
            "Web3, NFTs, and decentralised finance (DeFi)":                        {"blockchain": 3},
            # cloud ×1.0116 → second boost applied here (Q3)
            "Cloud-native apps and serverless computing":                          {"cloud": 4},
            "Cybersecurity threats, penetration testing, and zero-trust":          {"cybersecurity": 3},
            "Big data, Apache Spark, and real-time streaming pipelines":           {"data_eng": 3},
            "Cross-platform mobile apps and push notifications":                   {"mobile": 3},
            "Microservices, event-driven architecture, and distributed systems":   {"system_integration": 3},
            # qa ×0.9557 → second reduction (Q3)
            "Test automation frameworks and software quality engineering":         {"qa": 2},
        }
    },
    {
        "id": "q4",
        "question": "What kind of problem do you most enjoy solving?",
        "options": {
            "Broken UI — something looks wrong on a specific screen size":         {"mobile": 2, "qa": 1},
            "Security vulnerability — a login endpoint leaks user data":           {"cybersecurity": 3, "qa": 1},
            "Data quality issue — a pipeline silently drops records":              {"data_eng": 3},
            "Cryptographic puzzle — how to verify ownership without an authority": {"blockchain": 3},
            "Infrastructure cost spike — a cloud service is over-provisioned":     {"cloud": 3},
            "Integration failure — two services aren't talking to each other":     {"system_integration": 3},
            "Test coverage gap — a production bug that tests missed":              {"qa": 3},
        }
    },
    {
        "id": "q5",
        "question": "Which set of tools would you most like to master?",
        "options": {
            "Flutter, Dart, Firebase, and mobile deployment":                      {"mobile": 3},
            "Solidity, Hardhat, MetaMask, and Ethereum testnets":                  {"blockchain": 3},
            "AWS, Terraform, CloudFormation, and Docker":                          {"cloud": 3},
            "Burp Suite, TryHackMe, Nmap, and Metasploit":                        {"cybersecurity": 3},
            "Apache Kafka, Airflow, Spark, and BigQuery":                          {"data_eng": 3},
            "REST APIs, JMS, message queues, and microservices":                   {"system_integration": 3},
            "Selenium, pytest, JIRA, and the Test Automation Pyramid":            {"qa": 3},
        }
    },
    {
        "id": "q6",
        "question": "How do you feel about working with financial or economic systems?",
        "options": {
            "Love it — especially decentralised finance and crypto economics":      {"blockchain": 3},
            "Interested through data — I like modelling and pipelines":            {"data_eng": 2},
            "Mostly concerned about the security implications":                    {"cybersecurity": 2},
            "I prefer building the apps people use, not the underlying systems":   {"mobile": 2},
            "I'd rather focus on infrastructure reliability than finance":         {"cloud": 2, "system_integration": 1},
            "I care more about quality than the domain — I test anything":         {"qa": 2},
        }
    },
    {
        "id": "q7",
        "question": "What is your relationship with breaking things?",
        "options": {
            "I love finding bugs — I'd break software for sport":                  {"cybersecurity": 2, "qa": 2},
            "I prefer building things so good they can't be broken":               {"qa": 3, "mobile": 1},
            "I like stress-testing infrastructure and finding its limits":         {"cloud": 2, "system_integration": 1},
            "I think about attack vectors in every contract I write":              {"blockchain": 3},
            "I focus on making systems connect correctly rather than break":       {"system_integration": 3},
            "I prefer ensuring data integrity — no corruption, no silent failures":{"data_eng": 3},
        }
    },
    {
        "id": "q8",
        "question": "Where do you see yourself in 3 years?",
        "options": {
            "Building mobile apps that millions of people use daily":              {"mobile": 3},
            "Working as a smart contract developer or Web3 engineer":              {"blockchain": 3},
            "Managing cloud infrastructure for a fast-growing startup":            {"cloud": 3},
            "Hunting vulnerabilities as a penetration tester or security analyst": {"cybersecurity": 3},
            "Building data pipelines that power ML and analytics platforms":       {"data_eng": 3},
            # system_integration final boost applied here (Q8)
            "Architecting distributed systems and microservices":                   {"system_integration": 4},
            "Leading a QA function and building automation frameworks":            {"qa": 3},
        }
    },
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    records = []
    for f in glob.glob(os.path.join(BASE_DIR, "data/raw/jobs/*.txt")):
        text = open(f, encoding="utf-8").read()
        records.append({"filename": os.path.basename(f).replace(".txt", ""), "text": text, "type": "job"})
    for f in glob.glob(os.path.join(BASE_DIR, "data/raw/syllabus_*.txt")):
        text = open(f, encoding="utf-8").read()
        records.append({"filename": os.path.basename(f).replace(".txt", ""), "text": text, "type": "syllabus"})
    return pd.DataFrame(records)


@st.cache_data
def build_similarity(data):
    vec    = TfidfVectorizer(max_features=1000, stop_words="english")
    matrix = vec.fit_transform(data["text"])
    si     = data[data["type"] == "syllabus"].index
    ji     = data[data["type"] == "job"].index
    sim    = cosine_similarity(matrix[si], matrix[ji])
    snames = data.loc[si, "filename"].values
    jnames = data.loc[ji, "filename"].values
    return pd.DataFrame(sim, index=snames, columns=jnames), vec, matrix, si, ji


all_data                           = load_data()
sim_df, vectorizer, tfidf_matrix, syl_idx, job_idx = build_similarity(all_data)
syl_names = all_data.loc[syl_idx, "filename"].values
job_names = all_data.loc[job_idx, "filename"].values

# Mappings
DISPLAY_TO_FILE = {}
FILE_TO_DISPLAY = {}
for name in syl_names:
    key     = name.replace("syllabus_", "")
    display = ELECTIVE_NAME_MAP.get(key, key.replace("_", " ").title())
    DISPLAY_TO_FILE[display] = name
    FILE_TO_DISPLAY[name]    = display

elective_display_names = sorted(DISPLAY_TO_FILE.keys())


# ============================================================
# HELPERS
# ============================================================

def get_top_terms(text, n=12):
    vec    = vectorizer.transform([text])
    scores = vec.toarray().flatten()
    feats  = vectorizer.get_feature_names_out()
    idx    = [i for i in scores.argsort()[::-1] if scores[i] > 0][:n]
    return [(feats[i], round(float(scores[i]), 4)) for i in idx]


def get_skills_gap(elective_display, top_n=16):
    syl_file = DISPLAY_TO_FILE[elective_display]
    syl_text = all_data[all_data["filename"] == syl_file]["text"].values[0].lower()
    key      = syl_file.replace("syllabus_", "")
    job_rows = all_data[(all_data["type"] == "job") &
                        (all_data["filename"].str.startswith(f"job_{key}"))]
    if job_rows.empty:
        return []
    combined = " ".join(job_rows["text"].tolist())
    cv       = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    cv.fit([combined, syl_text])
    jv    = cv.transform([combined]).toarray().flatten()
    sv    = cv.transform([syl_text]).toarray().flatten()
    terms = cv.get_feature_names_out()
    gap   = jv - sv * 2
    seen, result = set(), []
    for i in gap.argsort()[::-1]:
        if len(result) >= top_n or gap[i] <= 0:
            break
        t = terms[i]
        if not any(t in s or s in t for s in seen):
            result.append(t)
            seen.add(t)
    return result


def job_label(filename):
    """Return a clean human-readable title from a job filename."""
    return filename.replace("job_", "").replace("_", " ").title()


# ============================================================
# SESSION STATE
# ============================================================

if "page"           not in st.session_state: st.session_state.page           = "explore"
if "quiz_answers"   not in st.session_state: st.session_state.quiz_answers   = {}
if "quiz_submitted" not in st.session_state: st.session_state.quiz_submitted = False
if "selected_job"   not in st.session_state: st.session_state.selected_job   = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(f"""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🧭 Elective Compass</div>
        <div class="sidebar-tagline">Career · Skills · Direction</div>
    </div>
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-num">{len(syl_names)}</div>
            <div class="stat-lbl">Electives</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">{len(job_names)}</div>
            <div class="stat-lbl">Job Ads</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">36</div>
            <div class="stat-lbl">Surveys</div>
        </div>
    </div>
    """)

    render_html("""
    <div class="sb-section" style="margin-top:1.4rem;">Match scores</div>
    <div class="sb-legend"><span style="color:#16a34a;font-size:1rem;">●</span> Above 30% — Strong</div>
    <div class="sb-legend"><span style="color:#d97706;font-size:1rem;">●</span> 15–30% — Moderate</div>
    <div class="sb-legend"><span style="color:#dc2626;font-size:1rem;">●</span> Below 15% — Weak</div>
    """)


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">
    <div class="hero-eyebrow">WeThinkCode_ · Career Intelligence Tool</div>
    <h1 class="hero-title">Elective Compass</h1>
    <p class="hero-subtitle">
        A data science tool that uses NLP to match course syllabi against
        real job descriptions — helping you choose the specialisation that
        aligns with your actual interests and career goals.
    </p>
    <div class="hero-divider"></div>
</div>
""")


# ============================================================
# HOW TO USE  (main pane, always visible)
# ============================================================

render_html("""
<div class="how-to-grid">
    <div class="how-to-card">
        <div class="how-to-num">01</div>
        <div class="how-to-title">🎯 Take the Quiz</div>
        <div class="how-to-body">
            Answer 8 questions about your interests and working style.
            Get a personalised elective ranking based on your answers.
        </div>
    </div>
    <div class="how-to-card">
        <div class="how-to-num">02</div>
        <div class="how-to-title">🔍 Explore Matches</div>
        <div class="how-to-body">
            Select any elective to see which real-world job descriptions
            it aligns with most — scored by TF-IDF cosine similarity.
        </div>
    </div>
    <div class="how-to-card">
        <div class="how-to-num">03</div>
        <div class="how-to-title">📄 Read Job Postings</div>
        <div class="how-to-body">
            Click any job match card to expand it and read the full
            job description — see exactly what employers are looking for.
        </div>
    </div>
    <div class="how-to-card">
        <div class="how-to-num">04</div>
        <div class="how-to-title">📊 Review the Data</div>
        <div class="how-to-body">
            Check the Skills Gap to know what to self-study, and open
            the Heatmap to see how all 7 electives compare at once.
        </div>
    </div>
</div>
""")


# ============================================================
# TOP NAV
# ============================================================

nav1, nav2, nav3, nav4, _pad = st.columns([1, 1, 1, 1, 0.4])
with nav1:
    if st.button("🔍  Explore Electives", use_container_width=True,
                 type="primary" if st.session_state.page == "explore" else "secondary"):
        st.session_state.page = "explore"
        st.rerun()
with nav2:
    if st.button("🎯  Interest Quiz", use_container_width=True,
                 type="primary" if st.session_state.page == "quiz" else "secondary"):
        st.session_state.page = "quiz"
        st.rerun()
with nav3:
    if st.button("📊  Similarity Heatmap", use_container_width=True,
                 type="primary" if st.session_state.page == "heatmap" else "secondary"):
        st.session_state.page = "heatmap"
        st.rerun()
with nav4:
    if st.button("🎓  Student Survey", use_container_width=True,
                 type="primary" if st.session_state.page == "survey" else "secondary"):
        st.session_state.page = "survey"
        st.rerun()

st.write("")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE: EXPLORE ELECTIVES                                     ║
# ╚══════════════════════════════════════════════════════════════╝

if st.session_state.page == "explore":

    render_html("""
    <div class="section-eyebrow">Explore</div>
    <div class="section-title">Choose an Elective</div>
    <div class="section-description">
        Select an elective to see its top career matches, distinctive keywords,
        and the skills gap between the syllabus and real job requirements.
        Click any job card to read the full job description.
    </div>
    """)

    selected_display  = st.selectbox("Elective", elective_display_names, label_visibility="collapsed")
    selected_syllabus = DISPLAY_TO_FILE[selected_display]
    scores            = sim_df.loc[selected_syllabus]
    top_matches       = scores.sort_values(ascending=False).head(5)

    # ── Top career matches heading ───────────────────────────────────────────
    st.write("")
    render_html(f"""
    <div class="section-eyebrow">Results</div>
    <div class="section-title">Top Career Matches</div>
    <div class="section-description">
        Strongest job-description matches for <strong>{selected_display}</strong>.
        Expand any card to read the full job posting.
    </div>
    """)

    col_left, col_right = st.columns([1.7, 0.9], gap="large")

    with col_left:
        for job_file, score in top_matches.items():
            title = job_label(job_file)

            if score > 0.3:
                badge_cls, badge_txt = "strong-badge",   "Strong match"
            elif score > 0.15:
                badge_cls, badge_txt = "moderate-badge", "Moderate match"
            else:
                badge_cls, badge_txt = "weak-badge",     "Weak match"

            pct = score * 100

            # ── Clickable expander showing the full job description ──────────
            with st.expander(f"**{title}**   ·   {score:.1%}"):
                # Score bar inside the expander header area
                render_html(f"""
                <div style="margin-bottom:0.9rem;">
                    <span class="match-badge {badge_cls}" style="font-size:0.7rem;padding:0.3rem 0.65rem;">{badge_txt}</span>
                    <span style="font-family:Inter,sans-serif;font-size:0.72rem;color:#64748b;margin-left:0.6rem;">
                        Cosine similarity: {score:.4f}
                    </span>
                </div>
                <div class="progress-bg" style="margin-bottom:1.2rem;">
                    <div class="progress-bar" style="width:{pct:.2f}%;"></div>
                </div>
                """)

                # Full job description text
                job_text = all_data[all_data["filename"] == job_file]["text"].values
                if len(job_text) > 0:
                    render_html(f"""
                    <div class="job-detail-header">{title}</div>
                    <div class="job-detail-body">{job_text[0].strip()}</div>
                    """)
                else:
                    st.info("Job description text not available.")

    # ── Right column: best match panel ──────────────────────────────────────
    with col_right:
        top_file  = top_matches.index[0]
        top_score = float(top_matches.iloc[0])
        top_title = job_label(top_file)

        if top_score > 0.3:
            rc, rt = "rec-strong",   "Strong alignment — this role closely mirrors the syllabus content."
        elif top_score > 0.15:
            rc, rt = "rec-moderate", "Moderate alignment — explore the job spec alongside the syllabus."
        else:
            rc, rt = "rec-weak",     "Broad alignment — consider comparing multiple electives."

        render_html(f"""
        <div class="best-panel">
            <div class="best-label">Top Match</div>
            <div class="best-title">{top_title}</div>
            <div class="best-score">↑ {top_score:.1%} similarity</div>
            <div class="rec-box {rc}">{rt}</div>
        </div>
        """)

    st.write("")

    # ── TF-IDF keyword extraction ────────────────────────────────────────────
    render_html(f"""
    <div class="section-eyebrow">NLP Analysis</div>
    <div class="section-title">Elective Keywords</div>
    <div class="section-description">
        Most distinctive TF-IDF terms in the <strong>{selected_display}</strong> syllabus —
        these words define what this specialisation is really about.
    </div>
    """)

    syl_text  = all_data[all_data["filename"] == selected_syllabus]["text"].values[0]
    top_terms = get_top_terms(syl_text, n=14)

    if top_terms:
        chips = "".join(f'<span class="chip">{t}</span>' for t, _ in top_terms)
        render_html(f'<div class="chip-container">{chips}</div>')

        terms_df = pd.DataFrame(top_terms, columns=["term", "score"])
        color    = ELECTIVE_COLORS.get(selected_display, "#2563eb")

        fig_kw = go.Figure(go.Bar(
            x=terms_df["score"], y=terms_df["term"],
            orientation="h",
            marker=dict(color=color, opacity=0.85),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_kw.update_layout(
            xaxis_title="TF-IDF Score",
            yaxis=dict(autorange="reversed"),
            height=340,
            margin=dict(l=10, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(family="Inter, sans-serif", color="#0f172a", size=11),
        )
        st.plotly_chart(fig_kw, use_container_width=True)

    st.write("")

    # ── Skills gap ───────────────────────────────────────────────────────────
    render_html(f"""
    <div class="section-eyebrow">Skills Gap</div>
    <div class="section-title">What Industry Wants Beyond the Syllabus</div>
    <div class="section-description">
        Terms that appear frequently in <strong>{selected_display}</strong> job descriptions
        but are largely absent from the course syllabus — your self-study roadmap.
    </div>
    """)

    gap_terms = get_skills_gap(selected_display, top_n=16)
    if gap_terms:
        chips = "".join(f'<span class="chip chip-gap">{t}</span>' for t in gap_terms)
        render_html(f'<div class="chip-container">{chips}</div>')
        render_html("""
        <div class="tip-box">
            💡 <strong>What to do with this:</strong> These are skills employers expect
            that the syllabus doesn't explicitly cover. Use
            <em>freeCodeCamp, official docs, YouTube, or TryHackMe</em>
            to fill these gaps alongside your elective work.
        </div>
        """)
    else:
        st.info("No significant gap detected — the syllabus closely mirrors job requirements.")

    st.write("")

    # ── All matches (collapsible) ────────────────────────────────────────────
    with st.expander("See all 70 career matches"):
        all_scores = scores.sort_values(ascending=False)
        st.dataframe(
            pd.DataFrame({
                "Job Title":   [job_label(j) for j in all_scores.index],
                "Match Score": [f"{s:.2%}"   for s in all_scores.values],
                "Similarity":  [f"{s:.4f}"   for s in all_scores.values],
            }),
            use_container_width=True, hide_index=True
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE: INTEREST QUIZ                                         ║
# ╚══════════════════════════════════════════════════════════════╝

elif st.session_state.page == "quiz":

    render_html("""
    <div class="section-eyebrow">Find your path</div>
    <div class="section-title">Interest Quiz</div>
    <div class="section-description">
        Answer 8 questions about your interests and working style.
        Your answers are scored against all 7 electives to produce
        a personalised alignment ranking — no right or wrong answers.
    </div>
    """)

    if not st.session_state.quiz_submitted:

        # ── Questions rendered OUTSIDE st.form so radio widgets are interactive ──
        # We store answers directly into session_state via widget keys.
        # A plain submit button at the bottom triggers scoring.

        for i, q in enumerate(QUIZ_QUESTIONS):
            render_html(f"""
            <div class="quiz-q-box">
                <div class="quiz-q-text">Q{i+1}. {q["question"]}</div>
            </div>
            """)
            options = list(q["options"].keys())
            st.radio(
                label=f"q{i+1}_radio",           # internal label (hidden)
                options=options,
                key=q["id"],                      # answer stored in st.session_state[q["id"]]
                label_visibility="collapsed",
                index=None,                       # no default selection — user must pick
            )
            st.write("")

        st.write("")
        if st.button("✦  See My Recommended Electives", use_container_width=True, type="primary"):
            # Collect answers from session state
            all_answered = all(st.session_state.get(q["id"]) is not None for q in QUIZ_QUESTIONS)
            if not all_answered:
                st.warning("Please answer all 8 questions before submitting.")
            else:
                # Save answers into dedicated dict for scoring
                for q in QUIZ_QUESTIONS:
                    st.session_state.quiz_answers[q["id"]] = st.session_state[q["id"]]
                st.session_state.quiz_submitted = True
                st.rerun()

    else:
        # ── Scoring ─────────────────────────────────────────────────────────
        scores_map = {k: 0 for k in ELECTIVE_NAME_MAP}
        for q in QUIZ_QUESTIONS:
            ans = st.session_state.quiz_answers.get(q["id"])
            if ans and ans in q["options"]:
                for key, w in q["options"][ans].items():
                    scores_map[key] += w

        total  = sum(scores_map.values()) or 1
        ranked = sorted(
            [(ELECTIVE_NAME_MAP[k], v, round(v / total * 100, 1)) for k, v in scores_map.items()],
            key=lambda x: x[1], reverse=True
        )

        top_name, _, top_pct = ranked[0]
        top_emoji = ELECTIVE_EMOJI.get(top_name, "✦")
        top_color = ELECTIVE_COLORS.get(top_name, "#2563eb")

        # ── Top recommendation panel ─────────────────────────────────────────
        render_html(f"""
        <div class="quiz-result-panel">
            <div class="qr-label">Your top match</div>
            <div class="qr-name">{top_emoji} {top_name}</div>
            <div class="qr-pill">Alignment score: {top_pct}%</div>
            <div style="height:8px;background:rgba(255,255,255,0.12);border-radius:50px;
                        overflow:hidden;margin:1rem 0 0.5rem 0;">
                <div style="height:100%;width:{min(top_pct * 1.4, 100):.1f}%;
                            background:{top_color};border-radius:50px;"></div>
            </div>
            <p class="qr-body">
                Based on your answers, <strong>{top_name}</strong> aligns most closely
                with your interests, preferred tools, and the kinds of problems
                you enjoy solving.
            </p>
        </div>
        """)

        # ── Full ranking chart ───────────────────────────────────────────────
        render_html("""
        <div class="section-eyebrow" style="margin-top:1.5rem;">Full Ranking</div>
        <div class="section-title">All Electives Scored</div>
        <div class="section-description">
            How well each elective matched your quiz responses.
        </div>
        """)

        fig_quiz = go.Figure()
        for name, _, pct in ranked:
            fig_quiz.add_trace(go.Bar(
                name=name,
                x=[pct],
                y=[f"{ELECTIVE_EMOJI.get(name,'')} {name}"],
                orientation="h",
                marker=dict(color=ELECTIVE_COLORS.get(name, "#2563eb"), opacity=0.88),
                hovertemplate=f"<b>{name}</b><br>Alignment: {pct}%<extra></extra>",
                text=[f"{pct}%"],
                textposition="outside",
            ))
        fig_quiz.update_layout(
            showlegend=False,
            xaxis=dict(title="Alignment Score (%)", range=[0, 105]),
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=dict(l=10, r=60, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(family="Inter, sans-serif", color="#0f172a", size=11),
        )
        st.plotly_chart(fig_quiz, use_container_width=True)

        # ── Table ────────────────────────────────────────────────────────────
        st.dataframe(
            pd.DataFrame(
                [(ELECTIVE_EMOJI.get(n, ""), n, f"{p}%") for n, _, p in ranked],
                columns=["", "Elective", "Alignment Score"]
            ),
            use_container_width=True, hide_index=True
        )

        st.write("")
        col_a, col_b, _ = st.columns([1, 1, 2])
        with col_a:
            if st.button("↩  Retake Quiz", use_container_width=True):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers   = {}
                # Clear individual answer keys
                for q in QUIZ_QUESTIONS:
                    if q["id"] in st.session_state:
                        del st.session_state[q["id"]]
                st.rerun()
        with col_b:
            if st.button(f"🔍  Explore {top_name}", use_container_width=True, type="primary"):
                st.session_state.page           = "explore"
                st.session_state.quiz_submitted = False
                st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE: SIMILARITY HEATMAP                                    ║
# ╚══════════════════════════════════════════════════════════════╝

elif st.session_state.page == "heatmap":

    render_html("""
    <div class="section-eyebrow">Data Science View</div>
    <div class="section-title">Similarity Heatmaps</div>
    <div class="section-description">
        Visualise how each syllabus matches all 70 job descriptions,
        and how the 7 electives compare to each other — the core
        analytical output of the TF-IDF model.
    </div>
    """)

    # ── 7×70 full heatmap ────────────────────────────────────────────────────
    render_html("""
    <strong style="font-family:Inter,sans-serif;font-size:0.95rem;color:#0f172a;">
        7 Syllabi × 70 Job Descriptions
    </strong>
    <p style="font-family:Inter,sans-serif;font-size:0.82rem;color:#64748b;margin:0.3rem 0 0.8rem 0;">
        Each cell = cosine similarity between one syllabus and one job posting.
        Jobs are sorted by elective — the darker diagonal blocks confirm the model
        correctly clusters jobs to their own specialisation.
    </p>
    """)

    job_meta = all_data[all_data["type"] == "job"].copy()
    job_meta["ekey"]    = job_meta["filename"].apply(lambda f: "_".join(f.replace("job_","").split("_")[:-1]))
    job_meta["elabel"]  = job_meta["ekey"].map(ELECTIVE_NAME_MAP)
    job_meta            = job_meta.sort_values("elabel")
    ordered_jobs        = job_meta["filename"].tolist()
    heatmap_data        = sim_df[ordered_jobs].copy()
    col_labels          = [job_label(f) for f in ordered_jobs]
    row_labels          = [FILE_TO_DISPLAY.get(r, r) for r in heatmap_data.index]

    fig_full = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=col_labels,
        y=row_labels,
        colorscale=[[0.0,"#f1f5f9"],[0.3,"#93c5fd"],[0.6,"#2563eb"],[1.0,"#0f1f3d"]],
        zmin=0, zmax=heatmap_data.values.max(),
        hovertemplate="Syllabus: %{y}<br>Job: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Cosine Similarity", thickness=14, len=0.8),
    ))
    fig_full.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=10, b=160),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a", size=9),
        xaxis=dict(tickangle=-70, tickfont=dict(size=7.5)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_full, use_container_width=True)

    st.write("")

    # ── 7×7 cross-elective heatmap ───────────────────────────────────────────
    render_html("""
    <strong style="font-family:Inter,sans-serif;font-size:0.95rem;color:#0f172a;">
        Cross-Elective Similarity: 7 × 7
    </strong>
    <p style="font-family:Inter,sans-serif;font-size:0.82rem;color:#64748b;margin:0.3rem 0 0.8rem 0;">
        How similar are the electives to <em>each other</em>?
        High similarity means overlapping skills — useful if you're deciding between two paths.
    </p>
    """)

    cross_sim    = cosine_similarity(tfidf_matrix[syl_idx], tfidf_matrix[syl_idx])
    cross_labels = [FILE_TO_DISPLAY.get(n, n) for n in syl_names]
    diag_mask    = cross_sim.copy().astype(object)
    np.fill_diagonal(diag_mask, None)

    ann = [dict(x=cross_labels[j], y=cross_labels[i],
                text=f"{cross_sim[i,j]:.2f}", showarrow=False,
                font=dict(size=11, color="#0f172a"))
           for i in range(len(cross_labels))
           for j in range(len(cross_labels)) if i != j]

    fig_cross = go.Figure(go.Heatmap(
        z=diag_mask, x=cross_labels, y=cross_labels,
        colorscale=[[0.0,"#f1f5f9"],[0.5,"#93c5fd"],[1.0,"#0f1f3d"]],
        zmin=0, zmax=0.55,
        hovertemplate="%{y} × %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Cosine Similarity", thickness=14, len=0.8),
    ))
    fig_cross.update_layout(
        annotations=ann,
        height=490,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a", size=11),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_cross, use_container_width=True)

    render_html("""
    <div class="insight-box">
        <strong>How to read this:</strong><br>
        • <strong>Dark cells</strong> = high vocabulary overlap — those electives share skills and tools.<br>
        • <strong>Light cells</strong> = distinct specialisations — choosing one takes you down a unique path.<br>
        • <strong>Blockchain</strong> is consistently the lightest row/column — the most niche, self-contained track.<br>
        • <strong>Cloud &amp; Data Engineering</strong> share the highest similarity — both rely on Python, AWS, and distributed systems.
    </div>
    """)

    st.write("")

    # ── Average match score ──────────────────────────────────────────────────
    render_html("""
    <strong style="font-family:Inter,sans-serif;font-size:0.95rem;color:#0f172a;">
        Average Job-Match Score per Elective
    </strong>
    <p style="font-family:Inter,sans-serif;font-size:0.82rem;color:#64748b;margin:0.3rem 0 0.8rem 0;">
        Mean cosine similarity of each syllabus across all 70 job descriptions.
        Higher = the elective's vocabulary is more broadly applicable to the job market.
    </p>
    """)

    avg_scores = sim_df.mean(axis=1).sort_values(ascending=False)
    avg_labels = [FILE_TO_DISPLAY.get(n, n) for n in avg_scores.index]
    avg_colors = [ELECTIVE_COLORS.get(lbl, "#2563eb") for lbl in avg_labels]

    fig_avg = go.Figure(go.Bar(
        x=avg_scores.values, y=avg_labels, orientation="h",
        marker=dict(color=avg_colors, opacity=0.88),
        hovertemplate="%{y}<br>Avg: %{x:.4f}<extra></extra>",
        text=[f"{v:.4f}" for v in avg_scores.values],
        textposition="outside",
    ))
    fig_avg.update_layout(
        xaxis=dict(title="Average Cosine Similarity", range=[0, avg_scores.max() * 1.35]),
        yaxis=dict(autorange="reversed"),
        height=360,
        margin=dict(l=10, r=60, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter, sans-serif", color="#0f172a", size=11),
    )
    st.plotly_chart(fig_avg, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE: STUDENT SURVEY INSIGHTS                               ║
# ╚══════════════════════════════════════════════════════════════╝

elif st.session_state.page == "survey":

    survey_df = load_survey()
    n_total   = len(survey_df)

    # ── Pre-compute key metrics ──────────────────────────────────────────────
    avg_satisfaction  = survey_df["satisfaction"].mean()
    avg_info_score    = survey_df["had_enough_info"].mean()
    pct_yes           = (survey_df["would_choose_again"] == "Yes").sum() / n_total * 100
    pct_no            = (survey_df["would_choose_again"] == "No").sum()  / n_total * 100
    pct_unsure        = (survey_df["would_choose_again"] == "Not sure").sum() / n_total * 100

    # Satisfaction by elective
    sat_by_elective = (
        survey_df.groupby("elective")["satisfaction"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_sat", "count": "n"})
        .sort_values("avg_sat", ascending=False)
    )

    # "Would choose again" breakdown by elective
    choice_pivot = (
        survey_df.groupby(["elective", "would_choose_again"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Yes", "No", "Not sure"]:
        if col not in choice_pivot.columns:
            choice_pivot[col] = 0

    # Information confidence distribution
    info_dist = survey_df["had_enough_info"].value_counts().sort_index()

    # Most common satisfying tasks
    task_counts = survey_df["satisfying_task"].dropna().value_counts()

    # Alternative electives people would switch to
    alt_counts = (
        survey_df[survey_df["alt_elective"].notna() &
                  (survey_df["alt_elective"].str.strip() != "") &
                  (survey_df["alt_elective"].str.lower() != "n/a")]
        ["alt_elective"]
        .str.strip()
        .value_counts()
    )

    # Student comments (non-empty, non-N/A)
    comments = survey_df[
        survey_df["comments"].notna() &
        (~survey_df["comments"].str.strip().str.lower().isin(["", "n/a", "nan"]))
    ][["elective", "satisfaction", "comments"]].copy()

    # ── Page heading ─────────────────────────────────────────────────────────
    render_html("""
    <div class="section-eyebrow">Real Data</div>
    <div class="section-title">Student Survey Insights</div>
    <div class="section-description">
        Analysis of survey responses from current and former WeThinkCode_ students.
        This real-world data validates, challenges, and calibrates the model's recommendations.
    </div>
    """)

    # ── Top-line stats row ────────────────────────────────────────────────────
    render_html(f"""
    <div class="survey-stat-grid">
        <div class="survey-stat">
            <div class="survey-stat-num">{n_total}</div>
            <div class="survey-stat-label">Survey Responses</div>
            <div class="survey-stat-sub">Current &amp; former students</div>
        </div>
        <div class="survey-stat">
            <div class="survey-stat-num" style="color:#16a34a;">{avg_satisfaction:.1f}<span style="font-size:1rem;color:#64748b;">/5</span></div>
            <div class="survey-stat-label">Avg Satisfaction</div>
            <div class="survey-stat-sub">With their chosen elective</div>
        </div>
        <div class="survey-stat">
            <div class="survey-stat-num" style="color:#2563eb;">{pct_yes:.0f}%</div>
            <div class="survey-stat-label">Would Choose Again</div>
            <div class="survey-stat-sub">{pct_no:.0f}% would not · {pct_unsure:.0f}% unsure</div>
        </div>
        <div class="survey-stat">
            <div class="survey-stat-num" style="color:{'#dc2626' if avg_info_score < 3 else '#d97706' if avg_info_score < 4 else '#16a34a'};">{avg_info_score:.1f}<span style="font-size:1rem;color:#64748b;">/5</span></div>
            <div class="survey-stat-label">Avg Info Confidence</div>
            <div class="survey-stat-sub">Before choosing their elective</div>
        </div>
    </div>
    """)

    render_html("""
    <div class="insight-box" style="margin-bottom:2rem;">
        <strong>Why this matters:</strong> The average information-confidence score reveals how prepared students
        felt when making their elective choice. A score below 3 signals a clear gap — this is exactly the
        problem Elective Compass is built to solve.
    </div>
    """)

    # ── Row 1: Satisfaction by elective + Would choose again ────────────────
    col_sat, col_choice = st.columns([1.1, 1], gap="large")

    with col_sat:
        render_html("""
        <div class="section-eyebrow">Satisfaction</div>
        <div class="section-title" style="font-size:1.3rem;">Avg Satisfaction by Elective</div>
        <div class="section-description">
            Mean satisfaction score (1–5) reported by students in each track.
            Higher = students felt the elective matched their expectations.
        </div>
        """)

        colors_sat = []
        for e in sat_by_elective["elective"]:
            # Map survey elective names to ELECTIVE_COLORS keys
            mapped = {
                "Data Science":      "#d97706",  # amber (closest to Data Engineering)
                "Cloud Computing":   ELECTIVE_COLORS.get("Cloud Computing",   "#2563eb"),
                "Cybersecurity":     ELECTIVE_COLORS.get("Cybersecurity",     "#dc2626"),
                "Mobile Development":ELECTIVE_COLORS.get("Mobile Development","#0d9488"),
                "System Integration":ELECTIVE_COLORS.get("System Integration","#0891b2"),
                "Quality Assurance": ELECTIVE_COLORS.get("Quality Assurance", "#16a34a"),
                "Blockchain Development": ELECTIVE_COLORS.get("Blockchain Development","#7c3aed"),
            }
            colors_sat.append(mapped.get(e, "#2563eb"))

        fig_sat = go.Figure(go.Bar(
            x=sat_by_elective["avg_sat"],
            y=sat_by_elective["elective"],
            orientation="h",
            marker=dict(color=colors_sat, opacity=0.88),
            text=[f"{v:.2f}  (n={n})" for v, n in
                  zip(sat_by_elective["avg_sat"], sat_by_elective["n"])],
            textposition="outside",
            hovertemplate="%{y}<br>Avg satisfaction: %{x:.2f}<extra></extra>",
        ))
        fig_sat.add_vline(x=avg_satisfaction, line_dash="dot", line_color="#64748b",
                          annotation_text=f"Overall avg {avg_satisfaction:.2f}",
                          annotation_position="top right",
                          annotation_font=dict(size=10, color="#64748b"))
        fig_sat.update_layout(
            xaxis=dict(title="Average Satisfaction (1–5)", range=[0, 6.2]),
            yaxis=dict(autorange="reversed"),
            height=340,
            margin=dict(l=10, r=80, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(family="Inter, sans-serif", color="#0f172a", size=11),
        )
        st.plotly_chart(fig_sat, use_container_width=True)

    with col_choice:
        render_html("""
        <div class="section-eyebrow">Regret Analysis</div>
        <div class="section-title" style="font-size:1.3rem;">Would You Choose Again?</div>
        <div class="section-description">
            Breakdown of whether students would pick the same elective
            if they could choose again today.
        </div>
        """)

        fig_donut = go.Figure(go.Pie(
            labels=["Yes, same one", "No, different", "Not sure"],
            values=[
                (survey_df["would_choose_again"] == "Yes").sum(),
                (survey_df["would_choose_again"] == "No").sum(),
                (survey_df["would_choose_again"] == "Not sure").sum(),
            ],
            hole=0.55,
            marker=dict(colors=["#16a34a", "#dc2626", "#d97706"]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} students (%{percent})<extra></extra>",
            textfont=dict(size=11),
        ))
        fig_donut.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#0f172a"),
            annotations=[dict(text=f"<b>{pct_yes:.0f}%</b><br>Yes",
                              x=0.5, y=0.5, font_size=14, showarrow=False,
                              font=dict(color="#0f172a", family="Sora, sans-serif"))]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Would choose again breakdown by elective
        render_html("""
        <div style="font-family:Inter,sans-serif;font-size:0.78rem;color:#64748b;
                    margin-top:0.5rem;margin-bottom:0.4rem;font-weight:600;">
            Breakdown by elective
        </div>
        """)
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(
            name="Yes", x=choice_pivot["elective"], y=choice_pivot["Yes"],
            marker_color="#16a34a", opacity=0.88,
        ))
        fig_stack.add_trace(go.Bar(
            name="No", x=choice_pivot["elective"], y=choice_pivot["No"],
            marker_color="#dc2626", opacity=0.88,
        ))
        fig_stack.add_trace(go.Bar(
            name="Not sure", x=choice_pivot["elective"], y=choice_pivot["Not sure"],
            marker_color="#d97706", opacity=0.88,
        ))
        fig_stack.update_layout(
            barmode="stack",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(tickangle=-20, tickfont=dict(size=9)),
            yaxis=dict(title="Students"),
            height=280,
            margin=dict(l=10, r=10, t=30, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(family="Inter, sans-serif", color="#0f172a", size=10),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    st.write("")

    # ── Row 2: Information confidence + alternative electives ────────────────
    col_info, col_alt = st.columns([1, 1], gap="large")

    with col_info:
        render_html("""
        <div class="section-eyebrow">The Core Problem</div>
        <div class="section-title" style="font-size:1.3rem;">Information Confidence Before Choosing</div>
        <div class="section-description">
            How confident did students feel about their choice before committing?
            1 = not confident at all · 5 = very confident.
        </div>
        """)

        fig_info = go.Figure(go.Bar(
            x=info_dist.index.astype(str),
            y=info_dist.values,
            marker=dict(
                color=["#dc2626","#f97316","#d97706","#2563eb","#16a34a"],
                opacity=0.88
            ),
            text=info_dist.values,
            textposition="outside",
            hovertemplate="Confidence level %{x}: %{y} students<extra></extra>",
        ))
        fig_info.add_hline(y=info_dist.values.mean(), line_dash="dot", line_color="#64748b")
        fig_info.update_layout(
            xaxis=dict(title="Confidence Level (1 = Low, 5 = High)"),
            yaxis=dict(title="Number of Students"),
            height=300,
            margin=dict(l=10, r=20, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.8)",
            font=dict(family="Inter, sans-serif", color="#0f172a", size=11),
        )
        st.plotly_chart(fig_info, use_container_width=True)

        low_conf = (survey_df["had_enough_info"] <= 2).sum()
        render_html(f"""
        <div class="tip-box">
            <strong>{low_conf} out of {n_total} students</strong> ({low_conf/n_total*100:.0f}%)
            rated their pre-choice information confidence at 2 or below.
            This directly validates the need for Elective Compass as a decision-support tool.
        </div>
        """)

    with col_alt:
        render_html("""
        <div class="section-eyebrow">Regret Signals</div>
        <div class="section-title" style="font-size:1.3rem;">Alternative Electives Students Would Pick</div>
        <div class="section-description">
            Among students who said "No" or "Not sure", which elective would they
            have chosen instead? This reveals demand that the model can help redirect.
        </div>
        """)

        if not alt_counts.empty:
            alt_colors = []
            color_map = {
                "Data Science":       "#d97706",
                "Cloud Computing":    ELECTIVE_COLORS.get("Cloud Computing",    "#2563eb"),
                "Cybersecurity":      ELECTIVE_COLORS.get("Cybersecurity",      "#dc2626"),
                "Mobile Development": ELECTIVE_COLORS.get("Mobile Development", "#0d9488"),
                "System Integration": ELECTIVE_COLORS.get("System Integration", "#0891b2"),
                "Quality Assurance":  ELECTIVE_COLORS.get("Quality Assurance",  "#16a34a"),
                "Blockchain Development": ELECTIVE_COLORS.get("Blockchain Development","#7c3aed"),
            }
            for e in alt_counts.index:
                alt_colors.append(color_map.get(e, "#2563eb"))

            fig_alt = go.Figure(go.Bar(
                x=alt_counts.values,
                y=alt_counts.index,
                orientation="h",
                marker=dict(color=alt_colors, opacity=0.88),
                text=alt_counts.values,
                textposition="outside",
                hovertemplate="%{y}: %{x} students<extra></extra>",
            ))
            fig_alt.update_layout(
                xaxis=dict(title="Number of Students", range=[0, alt_counts.max() + 1.5]),
                yaxis=dict(autorange="reversed"),
                height=300,
                margin=dict(l=10, r=60, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(248,250,252,0.8)",
                font=dict(family="Inter, sans-serif", color="#0f172a", size=11),
            )
            st.plotly_chart(fig_alt, use_container_width=True)
        else:
            st.info("No alternative elective data available.")

    st.write("")

    # ── Row 3: What students find satisfying ─────────────────────────────────
    render_html("""
    <div class="section-eyebrow">Motivations</div>
    <div class="section-title">What Students Find Most Satisfying</div>
    <div class="section-description">
        The most common "satisfying task" responses — these map directly
        to the quiz question weightings and validate the interest categories.
    </div>
    """)

    # Shorten labels for display
    short_labels = {
        "Making sure things work correctly and don't break in production":
            "Making sure things work / don't break",
        "Making things scale reliably (infrastructure performance uptime)":
            "Making things scale reliably",
        "Getting separate systems/services to work together smoothly":
            "Getting systems to work together",
        "Making something visual and interactive that people directly use":
            "Making something visual & interactive",
        "Turning raw data into something useful or understandable":
            "Turning raw data into insights",
        "Finding and closing security gaps before someone else finds them":
            "Finding and closing security gaps",
    }
    task_labels  = [short_labels.get(t, t[:55] + "…" if len(t) > 55 else t)
                    for t in task_counts.index]
    task_colors  = ["#2563eb","#0d9488","#7c3aed","#dc2626","#d97706","#0891b2","#16a34a"]

    fig_tasks = go.Figure(go.Bar(
        x=task_counts.values,
        y=task_labels,
        orientation="h",
        marker=dict(color=task_colors[:len(task_counts)], opacity=0.88),
        text=task_counts.values,
        textposition="outside",
        hovertemplate="%{y}: %{x} students<extra></extra>",
    ))
    fig_tasks.update_layout(
        xaxis=dict(title="Number of Students", range=[0, task_counts.max() + 1.5]),
        yaxis=dict(autorange="reversed"),
        height=340,
        margin=dict(l=10, r=60, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(family="Inter, sans-serif", color="#0f172a", size=11),
    )
    st.plotly_chart(fig_tasks, use_container_width=True)

    st.write("")

    # ── Quiz calibration insight ──────────────────────────────────────────────
    render_html("""
    <div class="section-eyebrow">Model Calibration</div>
    <div class="section-title">How Real Data Calibrates the Quiz</div>
    <div class="section-description">
        The satisfaction scores from actual students are used to adjust
        the quiz's elective weights. Electives with high real-world satisfaction
        get a small boost; those with lower satisfaction get a slight reduction.
        This makes recommendations grounded in lived experience, not just text similarity.
    </div>
    """)

    # Compute calibration multipliers from survey satisfaction
    # Scale: avg_sat normalised to a multiplier around 1.0
    # sat 5.0 → 1.15,  sat 4.0 → 1.00,  sat 3.0 → 0.90,  sat 2.0 → 0.80
    overall_mean = survey_df["satisfaction"].mean()

    calibration_rows = []
    survey_elective_to_key = {
        "Cloud Computing":        "cloud",
        "Cybersecurity":          "cybersecurity",
        "Data Science":           "data_eng",   # mapped to closest model elective
        "Mobile Development":     "mobile",
        "System Integration":     "system_integration",
        "Quality Assurance":      "qa",
        "Blockchain Development": "blockchain",
    }

    for _, row in sat_by_elective.iterrows():
        e_name  = row["elective"]
        avg_s   = row["avg_sat"]
        n_resp  = int(row["n"])
        e_key   = survey_elective_to_key.get(e_name, "")
        multiplier = round(1.0 + (avg_s - overall_mean) * 0.08, 3)  # gentle slope

        if multiplier > 1.02:
            badge_cls, badge_txt = "badge-boosted",   f"↑ ×{multiplier}"
        elif multiplier < 0.98:
            badge_cls, badge_txt = "badge-reduced",   f"↓ ×{multiplier}"
        else:
            badge_cls, badge_txt = "badge-unchanged", f"= ×{multiplier}"

        pct_yes_e = 0
        subset = survey_df[survey_df["elective"] == e_name]
        if len(subset) > 0:
            pct_yes_e = (subset["would_choose_again"] == "Yes").sum() / len(subset) * 100

        calibration_rows.append({
            "Elective":          e_name,
            "n":                 n_resp,
            "Avg Satisfaction":  f"{avg_s:.2f}",
            "Would Choose Again":f"{pct_yes_e:.0f}%",
            "Weight Multiplier": multiplier,
            "badge_cls":         badge_cls,
            "badge_txt":         badge_txt,
        })

    # Render calibration table
    for row in calibration_rows:
        render_html(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:0.75rem 1.1rem;margin-bottom:0.5rem;
                    box-shadow:0 1px 3px rgba(15,31,61,0.04);">
            <div style="font-family:Sora,sans-serif;font-size:0.9rem;
                        font-weight:600;color:#0f172a;min-width:180px;">
                {row['Elective']}
            </div>
            <div style="font-family:Inter,sans-serif;font-size:0.75rem;color:#64748b;min-width:80px;">
                n = {row['n']} students
            </div>
            <div style="font-family:Inter,sans-serif;font-size:0.75rem;color:#0f172a;min-width:120px;">
                ⭐ {row['Avg Satisfaction']} / 5
            </div>
            <div style="font-family:Inter,sans-serif;font-size:0.75rem;color:#0f172a;min-width:120px;">
                ✅ {row['Would Choose Again']} yes
            </div>
            <span class="calibration-badge {row['badge_cls']}">{row['badge_txt']}</span>
        </div>
        """)

    render_html("""
    <div class="insight-box" style="margin-top:1rem;">
        <strong>How this works:</strong> Each elective's average satisfaction score is compared
        to the overall mean. Electives rated above average get a small positive multiplier applied
        to their quiz weight totals, making them slightly more likely to appear as a top recommendation.
        The effect is intentionally gentle (max ±15%) so the NLP signal still dominates —
        real student experience nudges, not overrides, the model.
    </div>
    """)

    st.write("")

    # ── Student voice — comments ──────────────────────────────────────────────
    render_html("""
    <div class="section-eyebrow">Student Voice</div>
    <div class="section-title">What Students Said</div>
    <div class="section-description">
        Direct quotes from the survey — what would have helped students
        choose their elective more confidently.
    </div>
    """)

    if not comments.empty:
        for _, row in comments.iterrows():
            sat_stars = "⭐" * int(row["satisfaction"]) if pd.notna(row["satisfaction"]) else ""
            render_html(f"""
            <div class="quote-card">
                <div class="quote-text">"{row['comments'].strip()}"</div>
                <div class="quote-meta">{row['elective']} student &nbsp;·&nbsp; {sat_stars}</div>
            </div>
            """)
    else:
        st.info("No comments available.")

    st.write("")

    # ── Elective breakdown filter ─────────────────────────────────────────────
    render_html("""
    <div class="section-eyebrow">Drill Down</div>
    <div class="section-title">Filter by Elective</div>
    <div class="section-description">
        See all survey responses for a specific elective.
    </div>
    """)

    elective_options = ["All"] + sorted(survey_df["elective"].unique().tolist())
    selected_filter  = st.selectbox("Filter by elective", elective_options,
                                    label_visibility="collapsed")

    filtered_df = survey_df if selected_filter == "All" else \
                  survey_df[survey_df["elective"] == selected_filter]

    display_cols = ["elective", "satisfaction", "would_choose_again",
                    "satisfying_task", "had_enough_info"]
    st.dataframe(
        filtered_df[display_cols].rename(columns={
            "elective":           "Elective",
            "satisfaction":       "Satisfaction (1–5)",
            "would_choose_again": "Would Choose Again",
            "satisfying_task":    "Most Satisfying Task",
            "had_enough_info":    "Info Confidence (1–5)",
        }).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

render_html(f"""
<div class="footer">
    <div class="footer-title">🧭 Elective Compass</div>
    <div class="footer-text">Built with NLP &amp; Streamlit · TF-IDF + Cosine Similarity · Real Student Survey Data</div>
    <div class="footer-text">{len(syl_names)} electives · {len(job_names)} job descriptions · 36 student responses</div>
    <div class="footer-text" style="margin-top:0.6rem;color:#cbd5e1;">WeThinkCode_ · Paballo Precision Malepa</div>
</div>
""")
