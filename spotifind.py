import re
import ast
import spotipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from spotipy.oauth2 import SpotifyClientCredentials


def main():
    # Add your Spotify API details here
    SPOTIFY_API_CLIENT_ID = ""
    SPOTIFY_API_CLIENT_SECRET = ""
    # pd.set_option("display.max_rows", None)

    # My melodic playlist
    playlist_link = (
        "https://open.spotify.com/playlist/1OG2pW6wRSdAAr28WgnZWN?si=cc48d4304dcc4057"
    )

    recs = spotifind(playlist_link, SPOTIFY_API_CLIENT_ID,
                     SPOTIFY_API_CLIENT_SECRET)
    print(recs)


def add_columns(df):
    print("add_columns() started")

    df["duration_min"] = df["duration_ms"] / 60000
    df["year"] = df["year"].astype(int)
    df["decade"] = (df["year"] // 10) * 10

    print("add_columns() completed")


def string_to_list(s):
    if isinstance(s, str):
        return ast.literal_eval(s)
    elif isinstance(s, list):
        return s
    else:
        return []


def fetch_playlist_details(playlist_url, spotipyClient):
    print("fetch_playlist_details() started")

    sp = spotipyClient

    # Extract the playlist ID from the URL
    playlist_id = re.search(
        "https://open.spotify.com/playlist/([^?]*)", playlist_url
    ).group(1)

    # Fetch the playlist details
    playlist = sp.playlist(playlist_id)

    # Get the total number of tracks in the playlist
    total_tracks = playlist["tracks"]["total"]

    # Data storage
    data = []

    # Fetch tracks in batches of 100
    offset = 0
    batch_size = 100

    while offset < total_tracks:
        # Fetch a batch of tracks
        results = sp.playlist_tracks(
            playlist_id, offset=offset, limit=batch_size)

        # Get the tracks in the batch
        tracks = results["items"]

        for i, item in enumerate(tracks):
            track = item["track"]

            if track is None or track["id"] is None:
                continue

            # Fetch audio features for the track
            audio_features = sp.audio_features(track["id"])[0]

            # Prepare the track data
            track_data = {
                "id": track["id"],
                "name": track["name"],
                "album": track["album"]["name"],
                "album_id": track["album"]["id"],
                "artists": [artist["name"] for artist in track["artists"]],
                "artist_ids": [artist["id"] for artist in track["artists"]],
                "track_number": track["track_number"],
                "disc_number": track["disc_number"],
                "explicit": track["explicit"],
                "release_date": track["album"]["release_date"],
                # Assume that the year is the first 4 characters of the release date
                "year": track["album"]["release_date"][:4],
            }

            # Combine track details and audio features
            track_data.update(audio_features)

            # Add the track data to the data storage
            if track["name"] and track["artists"] != []:
                data.append(track_data)

        # Increment the offset to fetch the next batch
        offset += batch_size

    # Convert the data storage into a DataFrame
    df = pd.DataFrame(data)

    # Return the DataFrame
    print("fetch_playlist_details() complete")
    return df


def get_genres_for_artist(artist_id, spotipyClient):
    sp = spotipyClient
    artist = sp.artist(artist_id)
    return artist["genres"]


def get_popularity_for_artist(artist_id, spotipyClient):
    sp = spotipyClient
    artist = sp.artist(artist_id)
    return artist["popularity"]


def create_feature_set(df, float_cols):
    print("create_feature_set() started")
    """
    Process main df to create a final set of features that will be used to generate recommendations.
    Parameters:
        df (pandas dataframe): Main Dataframe
        float_cols (list(str)): List of float columns that will be scaled
    Returns:
        final: final set of features
    """
    # tfidf genre lists
    tfidf = TfidfVectorizer()
    genre_matrix = tfidf.fit_transform(df["genres"].apply(" ".join))

    # One-Hot Encoding (ohe) year and popularity
    year_ohe = pd.get_dummies(df["year"], prefix="year").values * 0.5
    popularity_ohe = pd.get_dummies(
        df["popularity_red"], prefix="pop").values * 0.15

    # scale float columns
    scaler = MinMaxScaler()
    floats_scaled = scaler.fit_transform(df[float_cols]) * 0.2

    # concatenate all features
    final_features = hstack(
        [
            genre_matrix,
            csr_matrix(floats_scaled),
            csr_matrix(popularity_ohe),
            csr_matrix(year_ohe),
        ]
    )

    # Create DataFrame from sparse matrix
    final_df = pd.DataFrame.sparse.from_spmatrix(final_features)

    # add song id
    final_df["id"] = df["id"].values

    print("create_feature_set() complete")
    return final_df


def create_necessary_outputs(playlist_id, df, spotifyClient):
    print("create_necessary_outputs() started")
    """
    Pull songs from a specific playlist.

    Parameters:
        playlist_id (str): ID of the playlist you'd like to pull from the Spotify API
        df (pandas dataframe): spotify dataframe

    Returns:
        playlist: all songs in the playlist THAT ARE AVAILABLE IN THE DATASET (Should be all of them)
    """

    sp = spotifyClient

    # Generate playlist dataframe
    playlist = pd.DataFrame()

    for ix, i in enumerate(sp.playlist(playlist_id)["tracks"]["items"]):
        playlist.loc[ix, "artist"] = i["track"]["artists"][0]["name"]
        playlist.loc[ix, "name"] = i["track"]["name"]
        playlist.loc[ix, "id"] = i["track"]["id"]
        playlist.loc[ix, "date_added"] = i["added_at"]

    playlist["date_added"] = pd.to_datetime(playlist["date_added"])

    playlist = playlist[playlist["id"].isin(df["id"].values)].sort_values(
        "date_added", ascending=False
    )

    print("create_necessary_outputs() complete")
    return playlist


def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    print("generate_playlist_feature() started")
    """
    Summarize a user's playlist into a single vector

    Parameters:
        complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
        playlist_df (pandas dataframe): playlist dataframe
        weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1.

    Returns:
        playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
        complete_feature_set_nonplaylist (pandas dataframe):
    """

    complete_feature_set_playlist = complete_feature_set[
        complete_feature_set["id"].isin(playlist_df["id"].values)
    ]  # .drop('id', axis = 1).mean(axis =0)
    complete_feature_set_playlist = complete_feature_set_playlist.merge(
        playlist_df[["id", "date_added"]], on="id", how="inner"
    )
    complete_feature_set_nonplaylist = complete_feature_set[
        ~complete_feature_set["id"].isin(playlist_df["id"].values)
    ]  # .drop('id', axis = 1)

    playlist_feature_set = complete_feature_set_playlist.sort_values(
        "date_added", ascending=False
    )

    most_recent_date = playlist_feature_set.iloc[0, -1]

    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix, "months_from_recent"] = int(
            (most_recent_date.to_pydatetime() -
             row.iloc[-1].to_pydatetime()).days / 30
        )

    playlist_feature_set["weight"] = playlist_feature_set["months_from_recent"].apply(
        lambda x: weight_factor ** (-x)
    )

    playlist_feature_set_weighted = playlist_feature_set.copy()
    # print(playlist_feature_set_weighted.iloc[:,:-4].columns)
    playlist_feature_set_weighted.update(
        playlist_feature_set_weighted.iloc[:, :-4].mul(
            playlist_feature_set_weighted.weight, 0
        )
    )
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    # playlist_feature_set_weighted_final['id'] = playlist_feature_set['id']

    print("generate_playlist_feature() complete")
    return (
        playlist_feature_set_weighted_final.sum(axis=0),
        complete_feature_set_nonplaylist,
    )


def generate_playlist_recos(df, features, nonplaylist_features, num):
    print("generate_playlist_recos() started")
    """
    Pull songs from a specific playlist.

    Parameters:
        df (pandas dataframe): spotify dataframe
        features (pandas series): summarized playlist feature
        nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist

    Returns:
        non_playlist_df_top: Recommendations for that playlist
    """
    non_playlist_df = df[df["id"].isin(
        nonplaylist_features["id"].values)].copy()
    non_playlist_df.loc[:, "sim"] = cosine_similarity(
        nonplaylist_features.drop(
            "id", axis=1).values, features.values.reshape(1, -1)
    )[:, 0]
    non_playlist_df_top = non_playlist_df.sort_values(
        "sim", ascending=False).head(num)

    print("generate_playlist_recos() complete")
    return non_playlist_df_top


def spotifind(
    playlistLink,
    spotifyAPIClientID,
    spotifyAPIClientSecret,
    numRecs=25,
    recencyBiasWeight=1.00,
):
    print("spotifind() started")
    playlist_link = playlistLink

    df = pd.read_csv("data/tracks_features.csv")
    artists_df = pd.read_csv("data/artists.csv")

    add_columns(df)

    df["artists"] = df["artists"].apply(string_to_list)
    df["artist_ids"] = df["artist_ids"].apply(string_to_list)
    df["time_signature"] = df["time_signature"].astype(int)

    str_cols = ["id", "name", "album", "album_id"]

    for col in str_cols:
        df[col] = df[col].astype(str)

    artists_df["genres"] = artists_df["genres"].apply(string_to_list)

    # Create a dictionary of artist ID-genres + ID-popularity mapping
    artist_genres_map = artists_df.set_index("id")["genres"].to_dict()
    artist_popularity_map = artists_df.set_index("id")["popularity"].to_dict()

    def get_genres_for_artist_df(artist_id):
        return artist_genres_map.get(artist_id, [])

    def get_popularity_for_artist_df(artist_id):
        return artist_popularity_map.get(artist_id, 0)

    # Apply the function to populate the 'genres'/'popularity' column
    df["genres"] = df["artist_ids"].apply(
        lambda ids: [
            genre for id in ids for genre in get_genres_for_artist_df(id)]
    )
    df["popularity"] = df["artist_ids"].apply(
        lambda ids: np.mean([get_popularity_for_artist_df(id) for id in ids])
    )

    client_credentials_manager = SpotifyClientCredentials(
        client_id=spotifyAPIClientID, client_secret=spotifyAPIClientSecret
    )

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    playlist_df = fetch_playlist_details(playlist_link, sp)

    playlist_df["genres"] = playlist_df["artist_ids"].apply(
        lambda ids: [
            genre for id in ids for genre in get_genres_for_artist(id, sp)]
    )

    playlist_df["popularity"] = playlist_df["artist_ids"].apply(
        lambda ids: np.mean([get_popularity_for_artist(id, sp) for id in ids])
    )

    playlist_df = playlist_df.drop(
        columns=["type", "uri", "track_href", "analysis_url"]
    )
    add_columns(playlist_df)

    float_cols = df.dtypes[df.dtypes == "float64"].index.values
    float_cols = float_cols[float_cols != "popularity"]

    # create 5 point buckets for popularity
    df["popularity_red"] = df["popularity"].apply(lambda x: int(x / 5))

    playlist_df["popularity_red"] = playlist_df["popularity"].apply(
        lambda x: int(x / 5)
    )

    column_names = [
        "id",
        "name",
        "album",
        "album_id",
        "artists",
        "artist_ids",
        "track_number",
        "disc_number",
        "explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
        "time_signature",
        "year",
        "release_date",
        "duration_min",
        "decade",
        "genres",
        "popularity",
        "popularity_red",
    ]

    playlist_df = playlist_df[column_names]

    fset = create_feature_set(df, float_cols)

    playlist_id = re.search(
        "https://open.spotify.com/playlist/([^?]*)", playlist_link
    ).group(1)
    my_playlist = create_necessary_outputs(playlist_id, df, sp)

    (
        complete_feature_set_playlist_vector,
        complete_feature_set_nonplaylist,
    ) = generate_playlist_feature(fset, my_playlist, recencyBiasWeight)

    recs = generate_playlist_recos(
        df,
        complete_feature_set_playlist_vector,
        complete_feature_set_nonplaylist,
        numRecs,
    )

    print("spotifind() complete")
    return recs


if __name__ == "__main__":
    main()
