import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import random

@st.cache_data
def load_data():
    users = pd.read_csv('ml-1m/users.dat', sep='::', engine='python', 
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin-1')
    ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python', 
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')
    movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', 
                         names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
    occupations = {
        0: 'other', 1: 'academic/educator', 2: 'artist', 3: 'clerical/admin',
        4: 'college/grad student', 5: 'customer service', 6: 'doctor/health care',
        7: 'executive/managerial', 8: 'farmer', 9: 'homemaker', 10: 'K-12 student',
        11: 'lawyer', 12: 'programmer', 13: 'retired', 14: 'sales/marketing',
        15: 'scientist', 16: 'self-employed', 17: 'technician/engineer', 18: 'tradesman/craftsman',
        19: 'unemployed', 20: 'writer'
    }
    return users, ratings, movies, occupations

def preprocess(users, ratings, movies):
    data = ratings.merge(users, on='UserID').merge(movies, on='MovieID')
    data['Genres'] = data['Genres'].apply(lambda x: x.split('|'))
    data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})
    return data

def build_dataset(data, mlb):
    genre_encoded = mlb.fit_transform(data['Genres'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
    features = pd.concat([data[['Gender', 'Age', 'Occupation']].reset_index(drop=True), genre_df], axis=1)
    labels = (data['Rating'] >= 4).astype(int)
    return features, labels

@st.cache_resource
def train_xgb_model(data):
    mlb = MultiLabelBinarizer()
    X, y = build_dataset(data, mlb)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"🎯Accuracy: **{acc:.2%}**") 

    return model, mlb

def find_similar_user(data, age, occupation, gender, genres):
    profiles = data.groupby('UserID').agg({
        'Gender': 'first',
        'Age': 'first',
        'Occupation': 'first',
        'Genres': lambda x: list(set(g for sublist in x for g in sublist))
    }).reset_index()

    mlb_temp = MultiLabelBinarizer()
    genre_encoded = mlb_temp.fit_transform(profiles['Genres'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb_temp.classes_)

    user_features = pd.concat([profiles[['Gender', 'Age', 'Occupation']], genre_df], axis=1)
    input_vec = pd.DataFrame([[gender, age, occupation] + [1 if g in genres else 0 for g in mlb_temp.classes_]],
                             columns=['Gender', 'Age', 'Occupation'] + list(mlb_temp.classes_))

    sim_scores = cosine_similarity(input_vec, user_features)[0]
    most_similar_user_id = profiles.iloc[np.argmax(sim_scores)]['UserID']
    return most_similar_user_id

def collaborative_filtering(data, user_id):
    matrix = data.pivot_table(index='UserID', columns='MovieID', values='Rating')
    matrix.fillna(0, inplace=True)
    similarity = cosine_similarity(matrix)
    sim_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

    if user_id in sim_df:
        similar_users = sim_df[user_id].sort_values(ascending=False)[1:6].index
        recommended = data[data['UserID'].isin(similar_users)]
        avg_ratings = recommended.groupby('MovieID')['Rating'].mean()
        return avg_ratings.to_dict()
    return {}

# Load Data
users, ratings, movies, occupations = load_data()
occupation_map = {v: k for k, v in occupations.items()}
data = preprocess(users, ratings, movies)
model, mlb = train_xgb_model(data)

# Streamlit UI
st.title("🎬 Rohul.ID - Movie Recommender")

gender = st.radio("Select Gender", ["Male", "Female"])
age = st.slider("Select Age", 10, 100, 25)
occupation = st.selectbox("Select Occupation", list(occupations.values()))
genres = st.multiselect("Pick Up to 3 Favorite Genres", options=sorted(mlb.classes_), max_selections=3)

if st.button("Get Recommendations"):
    if not genres:
        st.warning("Please select at least one genre.")
    else:
        occ_id = occupation_map[occupation]
        gender_val = 1 if gender == "Male" else 0

        candidate_movies = movies[movies['Genres'].apply(lambda x: any(g in x.split('|') for g in genres))].copy()
        candidate_movies['Genres'] = candidate_movies['Genres'].apply(lambda x: x.split('|'))

        rows = []
        for _, row in candidate_movies.iterrows():
            genre_vec = [1 if g in row['Genres'] else 0 for g in mlb.classes_]
            input_vec = [gender_val, age, occ_id] + genre_vec
            rows.append((row['MovieID'], input_vec))

        content_df = pd.DataFrame([r[1] for r in rows], columns=['Gender', 'Age', 'Occupation'] + list(mlb.classes_))
        content_scores = model.predict_proba(content_df)[:, 1]

        result_df = pd.DataFrame(rows, columns=['MovieID', 'features'])
        result_df['content_score'] = content_scores

        similar_user_id = find_similar_user(data, age, occ_id, gender_val, genres)
        collab_scores = collaborative_filtering(data, similar_user_id)
        result_df['collab_score'] = result_df['MovieID'].apply(lambda mid: collab_scores.get(mid, 0))

        result_df['final_score'] = result_df[['content_score', 'collab_score']].mean(axis=1)
        top_movies = result_df.sort_values('final_score', ascending=False).head(10)

        st.subheader("🎥 Recommended Movies")
        rec_movies = movies[movies['MovieID'].isin(top_movies['MovieID'])]
        rec_movies['Genres'] = rec_movies['Genres'].apply(lambda x: x.split('|'))

        for _, row in rec_movies.iterrows():
            gstring = ", ".join(row['Genres'])
            st.write(f"- **{row['Title']}** _(Genres: {gstring})_")
