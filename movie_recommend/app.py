import streamlit as st
from movies import get_similar_movies

st.title("🎬 Movie Recommender")

# Input from user
user_input = st.text_input("Enter a movie name")

# On button click
if st.button("Get Recommendations"):
    movie_name, recommendations_or_error = get_similar_movies(user_input)

    if not movie_name:
        st.error(recommendations_or_error)
    else:
        st.success(f"Movies similar to: {movie_name}")
        for idx, movie in enumerate(recommendations_or_error, 1):
            st.write(f"{idx}. {movie}")
