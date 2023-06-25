import re
import spotipy
import pandas as pd
import matplotlib.pyplot as plt
from spotipy.oauth2 import SpotifyClientCredentials

# Add your Spotify API details here
SPOTIFY_API_CLIENT_ID = ""
SPOTIFY_API_CLIENT_SECRET = ""
# pd.set_option("display.max_rows", None)


def main():
    # My rap playlist
    playlist_link = (
        "https://open.spotify.com/playlist/3lGvVPZ9ZFR6UQVYuWKuLZ?si=59afe60ef3ef4dd2"
    )

    playlist_df = fetch_playlist_details(playlist_link)

    playlist_df = playlist_df.drop(
        columns=["type", "uri", "track_href", "analysis_url"]
    )

    playlist_df["duration_min"] = playlist_df["duration_ms"] / 60000
    playlist_df["year"] = playlist_df["year"].astype(int)
    playlist_df["decade"] = (playlist_df["year"] // 10) * 10

    print(playlist_df)

    visualize(playlist_df)


def fetch_playlist_details(playlist_url):
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_API_CLIENT_ID, client_secret=SPOTIFY_API_CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

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
    return df


def visualize(df):
    features = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_min",
    ]

    categorical_features = ["key", "mode", "time_signature", "decade"]

    total_rows = 3  # Change as needed
    total_cols = max(len(categorical_features), len(features) // 2)

    fig, axs = plt.subplots(total_rows, total_cols, figsize=(15, 8))

    for ax, feature in zip(axs[0], categorical_features):
        df[feature].value_counts().sort_index().plot(
            kind="bar", title=f"{feature} distribution", ax=ax
        )
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")

    for ax, feature in zip(axs[1:].flatten(), features):
        ax.hist(df[feature], bins=100)
        ax.set_title(f"Histogram of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
