from IPython.display import display
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import sklearn

df_main = pd.read_csv(r"C:\Formation\projet_2\fiches_csv\movie_learn.csv", sep=",", low_memory=False)
movie_learn = df_main.copy()

# On va définir y ici comme la colonne 'title'
y = movie_learn['title']
# Sélection des colonnes numériques et pondération
scaler = StandardScaler()
weights_col = np.array([3,1,3,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])
X = movie_learn.select_dtypes('number')
X_scaled = scaler.fit_transform(X)
weighted_num_col = np.multiply(X_scaled, weights_col)
# Fit the KNN model on the scaled data
model_KNN_gen = NearestNeighbors(n_neighbors=5).fit(weighted_num_col)

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(#FFFF00, #FFFFFF);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'> Les 100 ans du cinéma! </h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>RECHERCHE TON FILM FAVORIS </h2>", unsafe_allow_html=True)

st.write("[Regardez les bandes-annonces](https://www.imdb.com/trailers/)")


film = st.text_input("Nom de film : ")
if film:
    film_search = movie_learn.loc[movie_learn['title'].str.contains(film, case=False)]
    if film_search.empty:
        st.write(f"Aucun film ne correspond à votre recherche pour '{film}'.")
    else:
        list_film = film_search['title'].tolist()
        film_list = set(list_film)
        film_choice = st.selectbox("De quel film parlez-vous ?", film_list)
        if st.button("Nos recommandations de film pour vous"):
            neighbor_gen = model_KNN_gen.kneighbors(scaler.transform(movie_learn.loc[movie_learn['title'] == film_choice, X.columns]))
            closest_fil = neighbor_gen[1][0]
            closest_film = movie_learn["title"].iloc[closest_fil]
            st.write(f"\nRecommandations pour le film :")
            for film in closest_film.tolist()[1:]:
                st.write("- " + film)


