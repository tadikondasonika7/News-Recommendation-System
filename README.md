# 📰 MIND News Recommendation System

An end-to-end Machine Learning web application that delivers personalized news recommendations using the Microsoft MIND (MIcrosoft News Dataset).

Built with **Python**, **Streamlit**, **Scikit-Learn**, and **TF-IDF Vectorization**, featuring three supervised click-prediction classifiers:
1. **Logistic Regression**
2. **Random Forest**
3. **Linear Support Vector Classifier (Linear SVC)**

---

## 📁 Repository Folder Structure

```
News-Recommendation-System/
├── app.py                      # Interactive Streamlit Web Application
├── recommender.py              # ML Engine (Preprocessing, TF-IDF, Model Training & Recommendation Logic)
├── requirements.txt            # Project Python Dependencies
├── Procfile                    # Deployment Start Command (Heroku / Render / Railway)
├── render.yaml                 # Render Blueprint Infrastructure Configuration
├── .streamlit/
│   └── config.toml             # Streamlit Production Server Settings & UI Theme
├── data/
│   ├── news.tsv                # News Articles Dataset
│   └── behaviors.tsv           # User Impression & Click Behaviors Dataset
└── models/                     # Saved Scikit-Learn Trained Models (.pkl)
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    └── linear_svc.pkl
```

---

## 🚀 Quick Start & Local Execution

### 1. Clone & Setup Environment
```bash
git clone https://github.com/YOUR_USERNAME/News-Recommendation-System.git
cd News-Recommendation-System
python -m venv venv
```

Activate environment:
- **Windows**: `.\venv\Scripts\activate`
- **Linux/Mac**: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit App Locally
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment Instructions

### Option 1: Deploy on Render.com (Recommended for Free HTTPS Web Hosting)

1. **Push to GitHub**:
   Ensure all files (`app.py`, `recommender.py`, `requirements.txt`, `Procfile`, `render.yaml`) are committed and pushed to your GitHub repository.

2. **Connect to Render**:
   - Go to [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** → **Web Service**.
   - Select **Build and deploy from a Git repository** and connect your repo.

3. **Configure Build & Start Settings**:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Instance Type**: Free Plan

4. **Deploy**:
   Click **Create Web Service**. Render will install dependencies and launch your live HTTPS website URL!

---

### Option 2: Deploy on Streamlit Community Cloud (Instant 1-Click Deploy)

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
5. Click **Deploy!**

---

## 📊 Features & Architecture

- **User Click History & Candidate Scoring**: Analyzes user reading patterns against unseen candidate news articles.
- **TF-IDF + Category Feature Extraction**: Vectorizes title and abstract content combined with one-hot encoded news categories.
- **Model Evaluation Dashboard**: Real-time comparative metrics table, confusion matrix heatmaps, and ROC curves.
- **Auto Data Fallback**: Automatically initializes sample datasets if full MIND TSV dataset files are not present.

---

## 🔗 Original Google Colab Notebooks
| Model                               | Link |
| :---------------------------------- | :--- |
| **SVC (Support Vector Classifier)** | [Open in Colab](https://colab.research.google.com/drive/1ntFHBrT4Rri4vZYeCrAuPMsg5UkzY-33?usp=sharing) |
| **Logistic Regression**             | [Open in Colab](https://colab.research.google.com/drive/1amVSUSybO0dLsLfLAxoN0nqbEl7F-8rI?usp=sharing) |
| **Random Forest**                   | [Open in Colab](https://colab.research.google.com/drive/1xgXl7P8cfNA7m7y0tgYY61LBWIur4GHn?usp=sharing) |
