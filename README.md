# 🧭 Elective Compass

> **A Data Science-powered tool that helps students find their perfect elective by matching course syllabi to real-world job descriptions.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://precise18-elective-compass-app-yyrpmu.streamlit.app)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

**Elective Compass** is a data-driven application that helps students make informed decisions about which elective to choose. It uses Natural Language Processing (NLP) to compare university course syllabi with real job descriptions, showing students how well each elective prepares them for specific careers.

### 🎯 The Problem

Students often struggle to choose electives because:
- Course descriptions are written in academic language
- It's unclear how a course translates to real-world jobs
- There's no data-driven way to compare electives

### 💡 The Solution

Elective Compass solves this by:
- Comparing **course syllabi** to **real job descriptions**
- Using **TF-IDF** and **Cosine Similarity** to quantify matches
- Providing **visual, intuitive results** with progress bars and color coding
- Helping students make **data-driven decisions**

---

## 🚀 Live Demo

Try the app here: **[https://precise18-elective-compass-app-yyrpmu.streamlit.app](https://precise18-elective-compass-app-yyrpmu.streamlit.app)**

---

## 📊 How It Works

### Architecture
┌─────────────────────────────────────────────────────────────────┐
│ Elective Compass │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │
│ │ Data │ │ Processing │ │ Visualization │ │
│ │ Collection │───▶│ (NLP/ML) │───▶│ (Streamlit) │ │
│ └─────────────┘ └─────────────┘ └─────────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │
│ │ 7 Syllabi │ │ TF-IDF │ │ Dropdown UI │ │
│ │ 35 Jobs │ │ Cosine │ │ Progress Bars │ │
│ └─────────────┘ └─────────────┘ └─────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────┘


### Step-by-Step

| Step | Process | Description |
| :--- | :--- | :--- |
| **1** | **Data Collection** | 7 course syllabi + 35 job descriptions (5 per elective) |
| **2** | **Text Cleaning** | Remove punctuation, lowercase, remove stopwords |
| **3** | **TF-IDF Vectorization** | Convert text to numerical features (1000 dimensions) |
| **4** | **Cosine Similarity** | Calculate similarity between each syllabus and every job |
| **5** | **Results** | Top 5 job matches with visual progress bars |

### Example Output
🧭 Elective Compass

🔍 Select an Elective: [Cloud ▼]

📊 Top 5 Job Matches for Cloud

🟢 Cloud Engineer Score: 85% ████████████████████
🟡 DevOps Engineer Score: 62% ██████████████
🟡 AWS Architect Score: 58% █████████████
🔴 Sys Admin Score: 22% █████
🔴 Network Engineer Score: 15% ███


---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Main programming language |
| **Data Processing** | Pandas | Data loading and manipulation |
| **NLP** | Scikit-learn | TF-IDF vectorization, Cosine Similarity |
| **Web App** | Streamlit | Interactive user interface |
| **Deployment** | Streamlit Cloud | Hosting and sharing |
| **Version Control** | Git + GitHub | Code management |

---

## 📁 Project Structure
elective-compass/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Git ignore rules
│
├── data/
│ └── raw/
│ ├── jobs/ # 35 job description files
│ │ ├── job_cloud_1.txt
│ │ ├── job_cloud_2.txt
│ │ ├── job_mobile_1.txt
│ │ └── ...
│ └── syllabus_*.txt # 7 syllabus files
│ ├── syllabus_cloud.txt
│ ├── syllabus_mobile.txt
│ ├── syllabus_blockchain.txt
│ └── ...
│
├── notebooks/
│ └── 01_data_loading.ipynb # Jupyter notebook for exploration
│
└── venv/ # Python virtual environment


---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.12 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/precise18/elective-compass.git
cd elective-compass

Set Up Virtual Environment
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

Install Dependencies
pip install -r requirements.txt

Run Locally
streamlit run app.py

Open your browser at: http://localhost:8501

📦 Dependencies
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.2.0
numpy>=1.24.0
plotly>=5.14.0

🚀 Deployment
Deploy to Streamlit Cloud

    Push your code to GitHub

    Go to share.streamlit.io

    Sign in with GitHub

    Select your repository and branch

    Set app.py as the main file

    Click Deploy

Your app will be live at: https://your-username-app-name.streamlit.app
📊 Data Sources
Source	Type	Count
University Course Catalog	Syllabi	7
LinkedIn, Indeed	Job Descriptions	35
AWS Educate, Cisco Networking Academy	Course Requirements	Various
🎓 Electives Included
Elective	Job Matches
☁️ Cloud Computing	Cloud Engineer, DevOps, AWS Architect
📱 Mobile Development	Flutter Developer, iOS/Android Developer
⛓️ Blockchain	Smart Contract Developer, Web3 Engineer
🔒 Cybersecurity	Security Analyst, Penetration Tester
📊 Data Engineering	Data Engineer, ETL Developer
🔗 System Integration	Integration Engineer, Middleware Developer
✅ Quality Assurance	QA Engineer, SDET, Automation Tester
🤝 Contributing

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
👤 Author

Paballo Precision Malepa

    GitHub: @precise18

    Email: precisionmalepa@gmail.com

🙏 Acknowledgments

    WeThinkCode_ for the elective curriculum

    Streamlit for the amazing web framework

    Scikit-learn for the NLP capabilities

📈 Future Enhancements

    □

    Add interactive quiz to suggest electives
    □

    Include salary data for each job role
    □

    Add alumni feedback and ratings
    □

    Create comparison view for multiple electives
    □

    Add export/share functionality

📧 Contact

Questions, suggestions, or feedback? Reach out at precisionmalepa@gmail.com

Built with ❤️ by Paballo Precision Malepa