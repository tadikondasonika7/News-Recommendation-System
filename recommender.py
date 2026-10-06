import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib

NEWS_COLUMNS = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
BEHAVIOR_COLUMNS = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']

DATA_DIR = 'data'
NEWS_FILE = os.path.join(DATA_DIR, 'news.tsv')
BEHAVIOR_FILE = os.path.join(DATA_DIR, 'behaviors.tsv')
MODELS_DIR = 'models'

def ensure_dataset_exists():
    """Generates a realistic sample dataset if files are not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if files already exist and have valid size
    if os.path.exists(NEWS_FILE) and os.path.exists(BEHAVIOR_FILE):
        if os.path.getsize(NEWS_FILE) > 500 and os.path.getsize(BEHAVIOR_FILE) > 500:
            return

    print("Generating sample news and behavior dataset...")
    
    categories = ['sports', 'news', 'finance', 'technology', 'entertainment', 'health', 'lifestyle', 'travel']
    subcategories = ['football', 'basketball', 'ai', 'markets', 'movies', 'wellness', 'culture', 'policy']
    
    sample_news = []
    titles_pool = [
        ("N10001", "sports", "football", "Champions League Final Preview: Tactics, Players, and Predictions", "An in-depth look into the upcoming Champions League clash featuring key player matchups and managerial tactics."),
        ("N10002", "technology", "ai", "Artificial Intelligence Breakthrough in Medical Diagnostics", "Researchers introduce a novel deep learning framework capable of detecting early-stage diseases from imaging data."),
        ("N10003", "finance", "markets", "Global Stock Markets Rally Following Central Bank Interest Rate Cut", "Major indices hit record highs today after financial authorities announced a lower interest rate benchmark."),
        ("N10004", "entertainment", "movies", "Summer Blockbuster Breaks Opening Weekend Box Office Records", "The long-awaited cinematic masterpiece shattered global box office records with unanimous critical acclaim."),
        ("N10005", "health", "wellness", "10 Daily Habits for Long-Term Cardiovascular and Mental Wellness", "Health experts share actionable lifestyle choices to reduce stress, improve sleep quality, and boost immunity."),
        ("N10006", "sports", "basketball", "Superstar Drops 50 Points in Thrilling Overtime Victory", "An extraordinary individual performance sealed a dramatic game-7 playoff win in front of a sold-out arena."),
        ("N10007", "technology", "ai", "Quantum Computing Breakthrough Promises Faster Data Security Encryption", "Engineers demonstrate quantum supremacy in cryptographic calculations, revolutionizing cybersecurity standards."),
        ("N10008", "news", "policy", "Global Climate Summit Reaches Landmark Renewable Energy Agreement", "World leaders unite to sign a binding treaty accelerating clean energy investments over the next decade."),
        ("N10009", "lifestyle", "culture", "The Rise of Sustainable Fashion and Eco-Friendly Wardrobes", "How consumers and top design brands are pivoting towards zero-waste materials and ethical manufacturing."),
        ("N10010", "travel", "culture", "Top Hidden Travel Destinations to Visit Without Heavy Crowds", "Explore serene landscapes, untouched beaches, and rich cultural heritage spots off the beaten path."),
        ("N10011", "finance", "markets", "Cryptocurrency Trends and Regulatory Updates for the Next Quarter", "Market analysts evaluate decentralized finance protocols and upcoming international trading regulations."),
        ("N10012", "sports", "football", "Transfer Window Analysis: Top Club Signings and Strategic Moves", "A comprehensive review of high-profile player transfers and how teams are restructuring for the season."),
        ("N10013", "technology", "ai", "Autonomous Electric Vehicles Achieve Next Generation Driving Safety", "New sensor technology and real-time decision algorithms dramatically reduce navigation errors in urban traffic."),
        ("N10014", "health", "wellness", "The Science of Nutrition: How Balanced Diets Enhance Longevity", "Clinical studies highlight the profound influence of gut microbiome diversity on overall immune response."),
        ("N10015", "entertainment", "movies", "Behind the Scenes of Award-Winning Cinematography and Sound Design", "Directors and audio engineers discuss the craftsmanship behind this year's most visually striking films.")
    ]

    for i in range(1, 101):
        nid = f"N{10000+i}"
        base = titles_pool[i % len(titles_pool)]
        cat = base[1]
        subcat = base[2]
        title = f"{base[3]} - Edition {i}"
        abstract = base[4]
        url = f"https://www.msn.com/en-us/news/{cat}/{nid}"
        sample_news.append([nid, cat, subcat, title, abstract, url, "[]", "[]"])

    df_sample_news = pd.DataFrame(sample_news, columns=NEWS_COLUMNS)
    df_sample_news.to_csv(NEWS_FILE, sep='\t', index=False, header=False)

    sample_behaviors = []
    np.random.seed(42)
    all_news_ids = df_sample_news['NewsID'].tolist()

    for u_idx in range(1, 51):
        user_id = f"U{10000+u_idx}"
        history = " ".join(np.random.choice(all_news_ids, size=np.random.randint(2, 6), replace=False))
        
        # Create impressions (NewsID-Label pairs)
        candidates = np.random.choice(all_news_ids, size=6, replace=False)
        impressions_list = []
        for cand in candidates:
            label = np.random.choice([0, 1], p=[0.7, 0.3])
            impressions_list.append(f"{cand}-{label}")
        
        impressions_str = " ".join(impressions_list)
        time_str = "11/11/2019 9:05:58 AM"
        sample_behaviors.append([u_idx, user_id, time_str, history, impressions_str])

    df_sample_behaviors = pd.DataFrame(sample_behaviors, columns=BEHAVIOR_COLUMNS)
    df_sample_behaviors.to_csv(BEHAVIOR_FILE, sep='\t', index=False, header=False)

class NewsRecommenderEngine:
    def __init__(self):
        ensure_dataset_exists()
        self.df_news = None
        self.df_behaviors = None
        self.df_interactions = None
        self.df_final = None
        self.df_news_features = None
        self.df_news_features_cand = None
        self.tfidf = None
        self.feature_cols = []
        self.models = {}
        self.eval_results = {}
        self.user_ids = []
        self._load_and_process_data()
        self._train_or_load_models()

    def _load_and_process_data(self):
        print("Loading datasets...")
        self.df_news = pd.read_csv(NEWS_FILE, sep='\t', names=NEWS_COLUMNS, encoding='utf-8', on_bad_lines='skip')
        self.df_behaviors = pd.read_csv(BEHAVIOR_FILE, sep='\t', names=BEHAVIOR_COLUMNS, encoding='utf-8', on_bad_lines='skip')

        # Extract user interactions
        interaction_data = []
        for _, row in self.df_behaviors.iterrows():
            user_id = row['UserID']
            impressions = str(row['Impressions']).split()
            for imp in impressions:
                if '-' in imp:
                    parts = imp.split('-')
                    if len(parts) == 2:
                        news_id, label = parts[0], parts[1]
                        if label.isdigit():
                            interaction_data.append({'UserID': user_id, 'NewsID': news_id, 'Label': int(label)})

        self.df_interactions = pd.DataFrame(interaction_data)
        if self.df_interactions.empty:
            raise ValueError("No valid user interactions found in behaviors dataset.")

        self.user_ids = sorted(self.df_interactions['UserID'].unique().tolist())

        # Preprocess news text & TF-IDF
        self.df_news['content'] = self.df_news['Title'].fillna('') + ' ' + self.df_news['Abstract'].fillna('')
        valid_news_ids = self.df_interactions['NewsID'].unique()
        df_news_filtered = self.df_news[self.df_news['NewsID'].isin(valid_news_ids)].copy()
        if df_news_filtered.empty:
            df_news_filtered = self.df_news.copy()

        self.tfidf = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = self.tfidf.fit_transform(df_news_filtered['content']).toarray()
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
        
        df_news_filtered.reset_index(drop=True, inplace=True)
        self.df_news_features = pd.concat([df_news_filtered[['NewsID', 'Category']], tfidf_df], axis=1)

        # One-Hot Encoding for Category
        df_category_ohe = pd.get_dummies(self.df_news_features['Category'], prefix='cat')
        self.df_news_features = pd.concat([self.df_news_features.drop('Category', axis=1), df_category_ohe], axis=1)

        # Candidate news feature table (for scoring unseen news)
        self.df_news_features_cand = self.df_news_features.drop_duplicates(subset=['NewsID']).copy()

        # Merge interactions with features to create dataset X, y
        self.df_final = pd.merge(self.df_interactions, self.df_news_features, on='NewsID', how='inner')
        self.feature_cols = [col for col in self.df_final.columns if col not in ['UserID', 'NewsID', 'Label']]

    def _train_or_load_models(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        X = self.df_final[self.feature_cols]
        y = self.df_final['Label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        model_specs = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Linear SVC': LinearSVC(random_state=42, class_weight='balanced', dual=False, max_iter=10000)
        }

        for name, clf in model_specs.items():
            print(f"Training {name}...")
            clf.fit(X_train, y_train)
            self.models[name] = clf

            y_pred = clf.predict(X_test)
            
            # Scores / Decision function for ROC
            if hasattr(clf, "predict_proba"):
                y_scores = clf.predict_proba(X_test)[:, 1]
            else:
                y_scores = clf.decision_function(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)

            self.eval_results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'confusion_matrix': cm,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc
            }

            # Save trained model to disk
            model_file_name = name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(clf, os.path.join(MODELS_DIR, model_file_name))

    def get_user_history(self, user_id):
        """Returns news articles the user has previously interacted with."""
        user_rows = self.df_behaviors[self.df_behaviors['UserID'] == user_id]
        if user_rows.empty:
            return pd.DataFrame()
        
        history_str = user_rows.iloc[0]['History']
        if pd.isna(history_str) or not str(history_str).strip():
            return pd.DataFrame()

        history_ids = str(history_str).split()
        history_news = self.df_news[self.df_news['NewsID'].isin(history_ids)][['NewsID', 'Title', 'Category', 'SubCategory', 'URL']]
        return history_news

    def recommend_news(self, model_name, user_id, n=5):
        """Generates N news recommendations for a given user using the chosen ML model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not recognized.")

        clf = self.models[model_name]
        candidate_df = self.df_news_features_cand.copy()
        
        # Match feature columns
        X_cand = candidate_df[self.feature_cols]

        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_cand)[:, 1]
        else:
            raw_scores = clf.decision_function(X_cand)
            # Sigmoid normalization for clean display between 0 and 1
            scores = 1 / (1 + np.exp(-raw_scores))

        candidate_df['score'] = scores

        # Exclude news user has already seen
        seen_news_ids = self.df_interactions[self.df_interactions['UserID'] == user_id]['NewsID'].unique()
        unseen_candidates = candidate_df[~candidate_df['NewsID'].isin(seen_news_ids)]

        if unseen_candidates.empty:
            unseen_candidates = candidate_df

        top_recs = unseen_candidates.sort_values(by='score', ascending=False).head(n)

        # Merge with full news details
        rec_results = pd.merge(
            top_recs[['NewsID', 'score']],
            self.df_news[['NewsID', 'Title', 'Category', 'SubCategory', 'Abstract', 'URL']],
            on='NewsID',
            how='left'
        )

        return rec_results
