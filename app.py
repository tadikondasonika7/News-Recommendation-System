import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from recommender import NewsRecommenderEngine

st.set_page_config(
    page_title="News Recommendation System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #F1F5F9;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing Recommendation Engine & Training Models...")
def load_engine():
    return NewsRecommenderEngine()

engine = load_engine()

st.markdown('<div class="main-title">📰 News Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Personalized AI recommendation engine based on Microsoft MIND Dataset using TF-IDF & Supervised Classifiers</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Recommendation Control Panel")

user_list = engine.user_ids
selected_user = st.sidebar.selectbox("👤 Select User ID", user_list, index=0)

if st.sidebar.button("🎲 Pick Random User"):
    random_user = np.random.choice(user_list)
    selected_user = random_user
    st.sidebar.success(f"Selected: {selected_user}")

model_options = ['Logistic Regression', 'Random Forest', 'Linear SVC']
selected_model = st.sidebar.selectbox("🤖 Select Recommendation Algorithm", model_options, index=0)

n_recs = st.sidebar.slider("🔢 Number of Recommendations", min_value=1, max_value=20, value=5, step=1)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommendations",
    "📜 User Reading History",
    "📈 Model Evaluation",
    "📁 Dataset Overview"
])

with tab1:
    st.subheader(f"Recommendations for User `{selected_user}`")
    st.caption(f"Engine algorithm: **{selected_model}** | Top **{n_recs}** recommendations")

    with st.spinner("Generating personalized recommendations..."):
        rec_df = engine.recommend_news(selected_model, selected_user, n=n_recs)

    if rec_df.empty:
        st.warning("No new recommendations could be generated for this user.")
    else:
        # Format table output
        rec_display = rec_df.copy()
        rec_display.index = np.arange(1, len(rec_display) + 1)
        rec_display['Recommendation Score'] = rec_display['score'].apply(lambda x: f"{x:.4f}")
        
        display_cols = ['Title', 'Category', 'SubCategory', 'Recommendation Score']
        st.dataframe(
            rec_display[display_cols],
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("News Title", width="large"),
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "SubCategory": st.column_config.TextColumn("SubCategory", width="medium"),
                "Recommendation Score": st.column_config.TextColumn("Score", width="small")
            }
        )

        st.subheader("🔍 Article Details & Summaries")
        for idx, row in rec_display.iterrows():
            with st.expander(f"#{idx} | [{row['Category'].upper()}] {row['Title']} (Score: {float(row['score']):.4f})"):
                st.write(f"**Abstract:** {row['Abstract']}")
                if pd.notna(row['URL']) and str(row['URL']).startswith('http'):
                    st.markdown(f"[🔗 Read Full Article]({row['URL']})")

with tab2:
    st.subheader(f"Reading History for `{selected_user}`")
    history_df = engine.get_user_history(selected_user)
    if history_df.empty:
        st.info("No prior reading history recorded for this user in the dataset.")
    else:
        history_df.index = np.arange(1, len(history_df) + 1)
        st.dataframe(
            history_df[['NewsID', 'Title', 'Category', 'SubCategory']],
            use_container_width=True
        )

with tab3:
    st.subheader("📊 Classifier Performance & Evaluation Metrics")

    # Metrics Summary Table
    metrics_list = []
    for m_name in model_options:
        res = engine.eval_results[m_name]
        metrics_list.append({
            "Model": m_name,
            "Accuracy": f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall": f"{res['recall']:.4f}",
            "F1-Score": f"{res['f1']:.4f}",
            "ROC AUC": f"{res['roc_auc']:.4f}"
        })
    metrics_df = pd.DataFrame(metrics_list)
    st.table(metrics_df)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Confusion Matrix: {selected_model}**")
        cm = engine.eval_results[selected_model]['confusion_matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                    xticklabels=['Predicted 0 (Non-Click)', 'Predicted 1 (Click)'],
                    yticklabels=['Actual 0 (Non-Click)', 'Actual 1 (Click)'], ax=ax_cm)
        ax_cm.set_title(f'Confusion Matrix ({selected_model})')
        st.pyplot(fig_cm)

    with col2:
        st.write("**Receiver Operating Characteristic (ROC) Curves**")
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        for m_name in model_options:
            res = engine.eval_results[m_name]
            ax_roc.plot(res['fpr'], res['tpr'], label=f"{m_name} (AUC={res['roc_auc']:.2f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curves Comparison')
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)

with tab4:
    st.subheader("📁 Dataset Statistics & Information")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total News Articles", len(engine.df_news))
    col_b.metric("Total Users", len(engine.user_ids))
    col_c.metric("Total Interactions", len(engine.df_interactions))

    st.write("---")
    st.write("### 📰 Sample News Data (`news.tsv`)")
    st.dataframe(engine.df_news[['NewsID', 'Category', 'SubCategory', 'Title']].head(10), use_container_width=True)

    st.write("### 👤 Sample User Behaviors (`behaviors.tsv`)")
    st.dataframe(engine.df_behaviors[['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']].head(10), use_container_width=True)
