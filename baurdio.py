import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import config
import pandas as pd
import streamlit as st
from PIL import Image

image=Image.open('bardio.jpeg')
st.markdown("### Bard ( Basic Ai Reccomending Ditties )")
st.image(image)
cid=config.API_ID
secret=config.API_KEY
username=config.USERNAME

#for avaliable scopes see https://developer.spotify.com/web-api/using-scopes/
#setting up all spotify API token stuff
scope = 'user-library-read playlist-modify-public playlist-read-private'
redirect_uri='https://developer.spotify.com/dashboard/applications/23a254a3df2244eeb46aaeb68c1bab2e'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
token = util.prompt_for_user_token(username,scope,client_id=cid,client_secret=secret,redirect_uri="https://developer.spotify.com/dashboard/applications/23a254a3df2244eeb46aaeb68c1bab2e")

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

sourcePlaylistID =st.text_input("Enter your spotify playlist here!","https://open.spotify.com/playlist/71Ri1Raa61KSH4t9qwJdM4")
sourcePlaylist = sp.user_playlist(username, sourcePlaylistID)
tracks = sourcePlaylist["tracks"]
songs = tracks["items"]

track_ids = []
track_names = []
#grab all track from spotify playlist
for i in range(0, len(songs)):
    if songs[i]['track']['id'] != None: # Removes the local tracks in your playlist if there is any
        track_ids.append(songs[i]['track']['id'])
        track_names.append(songs[i]['track']['name'])
#grab all audio features for tracks
features = []
for i in range(0,len(track_ids)):
    audio_features = sp.audio_features(track_ids[i])
    for track in audio_features:
      
      if track is None:
        print(track)
        features.append({'danceability': 0, 'energy': 0, 'key': 0, 'loudness': 0, 'mode': 0, 'speechiness': 0, 'acousticness': 0, 'instrumentalness': 0, 'liveness': 0, 'valence': 0, 'tempo': 0, 'type': 'audio_features', 'id': '00000', 'uri': 'spotify:track:0', 'track_href': 'https://api.spotify.com/', 'analysis_url': 'https://api.spotify.com/', 'duration_ms': 0, 'time_signature': 0})
      else:
        features.append(track)
      
#retain only the relevant stuff
playlist_df = pd.DataFrame(features, index = track_names)
playlist_df=playlist_df[["id", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness",  "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]]

from sklearn.feature_extraction.text import TfidfVectorizer
#feature vectors (creates a fv for each track that descrives the tracks)
v=TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
X_names_sparse = v.fit_transform(track_names)
#rate and store the ratings of the songs
import numpy as np
ratings=pd.DataFrame(track_names,columns=["tracks"])
ratings['score']=None
con=True
edited_df = st.experimental_data_editor(ratings)

if st.button("Click once you finish rating EVERYTHING"):
  if edited_df['score'].isnull().values.any():
     st.mardown("#### You missed rating a song or two, Refresh and try again")
     st.stop()
  results=edited_df['score'].astype(int).tolist()
  playlist_df['ratings']=results
  playlist_df.to_csv("Spotify_playlist_Dataset.csv")

  # Analyze feature importances
  from sklearn.ensemble._forest import RandomForestRegressor, RandomForestClassifier

  X_train = playlist_df.drop(['id', 'ratings'], axis=1)
  y_train = playlist_df['ratings']
  forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=12) # Set by GridSearchCV below
  forest.fit(X_train, y_train)
  importances = forest.feature_importances_
  indices = np.argsort(importances)[::-1]

  # Print the feature rankings
  st.write("Features ranked on how important they are to you:")
  
  for f in range(len(importances)):
      st.write("%d. %s " % (f + 1, X_train.columns[f]))
    
  from sklearn import decomposition
  from sklearn.preprocessing import StandardScaler
  X_scaled = StandardScaler().fit_transform(X_train)

  pca = decomposition.PCA().fit(X_scaled)

    # Fit your dataset to the optimal pca (dimenstionality reduction)
  pca1 = decomposition.PCA(n_components=8)
  X_pca = pca1.fit_transform(X_scaled)


  from sklearn.manifold import TSNE

  tsne = TSNE(random_state=17,perplexity=5)
  X_tsne = tsne.fit_transform(X_scaled)
  #data vis

  from scipy.sparse import csr_matrix, hstack

  X_train_last = csr_matrix(hstack([X_pca, X_names_sparse])) # Check with X_tsne + X_names_sparse also


  from sklearn.model_selection import StratifiedKFold, GridSearchCV
  import warnings
  warnings.filterwarnings('ignore')

  # Initialize a stratified split for the validation process
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

  # Decision Trees First
  from sklearn.tree import DecisionTreeClassifier

  tree = DecisionTreeClassifier()

  tree_params = {'max_depth': range(1,11), 'max_features': range(4,19)}

  tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)

  tree_grid.fit(X_train_last, y_train)

  # Random Forests second

  parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
  rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                           n_jobs=-1, oob_score=True)
  gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
  gcv1.fit(X_train_last, y_train)

  # nearest neigbors last
  from sklearn.neighbors import KNeighborsClassifier

  knn_params = {'n_neighbors': range(1, 8)}
  knn = KNeighborsClassifier(n_jobs=-1)

  knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
  knn_grid.fit(X_train_last, y_train)

  rec_tracks = []
  for i in playlist_df['id'].values.tolist():
      rec_tracks += sp.recommendations(seed_tracks=[i], limit=int(len(playlist_df)/2))['tracks']

  rec_track_ids = []
  rec_track_names = []
  for i in rec_tracks:
    rec_track_ids.append(i['id'])
    rec_track_names.append(i['name'])

  rec_features = []
  for i in range(0,len(rec_track_ids)):
      rec_audio_features = sp.audio_features(rec_track_ids[i])
      for track in rec_audio_features:
          rec_features.append(track)
        
  rec_playlist_df = pd.DataFrame(rec_features, index = rec_track_ids)


  X_test_names = v.transform(rec_track_names)

  rec_playlist_df=rec_playlist_df[["acousticness", "danceability", "duration_ms", "energy", "instrumentalness",  "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]]

  # Make predictions
  tree_grid.best_estimator_.fit(X_train_last, y_train)
  rec_playlist_df_scaled = StandardScaler().fit_transform(rec_playlist_df)
  rec_playlist_df_pca = pca1.transform(rec_playlist_df_scaled)
  X_test_last = csr_matrix(hstack([rec_playlist_df_pca, X_test_names]))
  y_pred_class = tree_grid.best_estimator_.predict(X_test_last)

  rec_playlist_df['ratings']=y_pred_class
  rec_playlist_df = rec_playlist_df.sort_values('ratings', ascending = False)
  rec_playlist_df = rec_playlist_df.reset_index()
  # Pick the top ranking tracks to add your new playlist 70% for passing grade :)
  recs_to_add = rec_playlist_df[rec_playlist_df['ratings']>=7]['index'].values.tolist()


  # Create a new playlist for tracks to add - you may also add these tracks to your source playlist and proceed
  playlist_recs = sp.user_playlist_create(username,name='Reccomended by Bard - {}'.format(sourcePlaylist['name']))
  np.array(recs_to_add)
  # Add tracks to the new playlist
  sp.user_playlist_add_tracks(user=username,playlist_id=playlist_recs['id'], tracks=recs_to_add)
  st.balloons()
  st.success("Check your playlist tab, We've created a new playlist full of reccomended songs, Happy Listening ~ Bard")
  fulllink="https://open.spotify.com/playlist/"+playlist_recs['id']
  st.write(fulllink)