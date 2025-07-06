from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load dataset
df1 = pd.read_csv('movie_recommend/movies.csv')

# Select relevant features
select_feature = df1[["genres", "keywords", "original_title", "tagline", "cast", "director"]].copy()

# Fill missing values with empty strings
for feature in select_feature:
    select_feature[feature] = select_feature[feature].fillna("")

# Combine features
combined_features = (
    select_feature['genres'] +
    select_feature["keywords"] +
    select_feature["original_title"] +
    select_feature["tagline"] +
    select_feature["cast"] +
    select_feature["director"]
)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

# Compute similarity matrix
similarity = cosine_similarity(feature_vector)

def get_similar_movies(user_input):
    # Find closest match
    titles = df1["original_title"].tolist()
    close_matches = difflib.get_close_matches(user_input, titles)

    if not close_matches:
        return None, "No close match found for your movie. Please check the spelling or try another title."

    close_match = close_matches[0]

    # Get recommendations
    index_of_movie = df1[df1.original_title == close_match].index[0]
    similarity_scores = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie in sorted_similar_movies[1:6]:
        index = movie[0]
        title_from_index = df1.loc[index, "original_title"]
        recommendations.append(title_from_index)

    return close_match, recommendations
