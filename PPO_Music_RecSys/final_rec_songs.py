# This is the final step where the learned parameters of the PPO network are used during the inference phase (real-time recommendation).
# The program recommends a sequence of 10 songs by following the policy learned during training.
# It starts with a randomly selected song and continues to select songs based on the policy until the episode (10-song playlist) is complete.
# During playback, user interaction logs (percentage listened and liked) are recorded
# so that it can later be used to fine-tune the model through additional training.
# This prints details about the current state (current song) and the 4 possible actions,
# along with their corresponding transition probabilities to the next state (next song).
#---------------------------------------------------------------------------------------------#
# Run this program to enjoy real-time song recommendations powered by PPO :))                 #
# Make sure you have your Spotify Client ID and Client Secret ready to enable track playback. #
#---------------------------------------------------------------------------------------------#
# Reference used:
# 1. PPO Implementation from Scratch | Reinforcement Learning, YT Channel: "https://www.youtube.com/watch?v=xHf8oKd7cgU&t=72s"
# 2. Proximal Policy Optimization (PPO) is Easy With PyTorch | Full PPO Tutorial, YT Channel: "https://www.youtube.com/watch?v=hlv79rcHws0&t=11s"
# 3. Proximal Policy Optimization (PPO) for LLMs Explained Intuitively, YT Channel: "https://www.youtube.com/watch?v=8jtAzxUwDj0&t=27s"
# 4. Proximal Policy Optimization (PPO) - How to train Large Language Models, YT Channel: "https://www.youtube.com/watch?v=TjHH_--7l8g"
# 5. RL CH10 - Policy Gradient algorithms (PPO and Deep Reinforcement Learning), YT Channel: "https://www.youtube.com/watch?v=Hvau7oC8TU0&list=PLZ_sI4f41TGvthD8dA7daahlbLV0yDW0w&index=10"
# 6. The 37 Implementation Details of Proximal Policy Optimization, Link: "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"
# 7. Proximal Policy Optimization by OpenAI Spinning Up, Link: "https://spinningup.openai.com/en/latest/algorithms/ppo.html"
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Spotify details")
parser.add_argument('--client_id', type=str, required=True, help='Spotify client ID')
parser.add_argument('--client_secret', type=str, required=True, help='Spotify client secret')
args = parser.parse_args()

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation = nn.LeakyReLU(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.layer_norm1(x)
        x = self.activation(self.fc2(x))
        x = self.layer_norm2(x)
        return self.fc3(x)

df = pd.read_csv("ppo_input.csv")
latent_map = {
    row['id']: {
        'latent': np.array([row[f'latent_{i}'] for i in range(5)], dtype=np.float32),
        'reward': row['reward']
    } for _, row in df.iterrows()
}
song_ids = list(latent_map.keys())
latents = np.array([latent_map[sid]['latent'] for sid in song_ids])

policy_net = PolicyNetwork(input_dim=5, hidden_dim=128, output_dim=4)
policy_net.load_state_dict(torch.load("policy_network.pt"))
policy_net.eval()

value_net = ValueNetwork(input_dim=5, hidden_dim=128)
value_net.load_state_dict(torch.load("value_network.pt"))
value_net.eval()

knn = NearestNeighbors(n_neighbors=10)
knn.fit(latents)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=args.client_id,
    client_secret=args.client_secret,
    redirect_uri='http://127.0.0.1:8888/callback',
    scope='user-modify-playback-state user-read-playback-state user-library-read'
))

def recommend_next_song(current_song_id, history=[], step=None):
    current_latent = latent_map[current_song_id]['latent']
    current_index = song_ids.index(current_song_id)
    _, indices = knn.kneighbors(current_latent.reshape(1, -1), n_neighbors=10)
    candidates = [i for i in indices[0] if i != current_index and song_ids[i] not in history]
    if len(candidates) < 4:
        candidates += [i for i in indices[0] if i not in candidates and i != current_index]
    candidates = candidates[:4]
    # Get action probabilities
    with torch.no_grad():
        probs = policy_net(torch.FloatTensor(current_latent)).numpy()
    probs = probs / probs.sum()
    # action = np.random.choice(4, p=probs)
    action = np.argmax(probs)
    next_song_id = song_ids[candidates[action]]
    if step is not None:
        print(f"\nStep {step + 1}")
    print(f"Current song ID (state): {current_song_id}")
    print("Top 4 recommended songs (actions) with probabilities:")
    for rank, (idx, prob) in enumerate(zip(candidates, probs)):
        print(f"  {rank + 1}. {song_ids[idx]} — Prob: {prob:.3f}")
    print(f"Selected next song (next state): {next_song_id}")
    return next_song_id

def add_to_spotify_queue(song_ids):
    print("\nAdding songs to Spotify queue...")
    try:
        current_playback = sp.current_playback()
        if not current_playback or not current_playback['is_playing']:
            print("Spotify is not currently playing. Please start playback first.")
            return False
        for song_id in song_ids:
            try:
                sp.add_to_queue(f"spotify:track:{song_id}")
                print(f"Added track {song_id} to queue")
                time.sleep(0.5)  # Small delay to avoid rate limiting
            except Exception as e:
                print(f"Error adding track {song_id}: {str(e)}")
        print("\nAll songs added to queue successfully!")
        return True
    except Exception as e:
        print(f"Error connecting to Spotify: {str(e)}")
        print("Please make sure:")
        print("1. Do you have Spotify Premium (required for queue modification)")
        print("2. Are you running the Spotify app on your device")
        print("3. Your device is active in Spotify Connect")
        return False

def is_track_liked(track_uri):
    try:
        track_id = track_uri.split(":")[-1]
        return sp.current_user_saved_tracks_contains([track_id])[0]
    except Exception as e:
        print(f"Error checking liked songs: {e}")
        return False

def get_duration_ms(track_uri):
    try:
        track_id = track_uri.split(":")[-1]
        return sp.track(track_id)['duration_ms']
    except Exception as e:
        print(f"Error getting duration: {e}")
        return 1

def get_track_latent(track_id):
    try:
        return latent_map[track_id]['latent'].tolist()
    except KeyError:
        return []

def generate_and_play_playlist(start_song_id, length=10):
    history = [start_song_id]
    current_id = start_song_id
    playlist = [start_song_id]

    print("Generating Playlist")
    for step in range(length - 1):
        next_id = recommend_next_song(current_id, history, step=step)
        history.append(next_id)
        if len(history) > 3:
            history.pop(0)
        current_id = next_id
        playlist.append(next_id)

    print("\nFinal Playlist")
    for idx, song_id in enumerate(playlist):
        print(f"{idx + 1}. {song_id}")
    results = []
    track_uris = [f"spotify:track:{sid}" for sid in playlist]
    try:
        sp.start_playback(uris=[track_uris[0]])
    except Exception as e:
        print(f"Error starting playback: {e}")
        return

    last_played_uri = track_uris[0]
    start_time = time.time()
    played_count = 1
    total_tracks = len(track_uris)

    for uri in track_uris[1:]:
        try:
            sp.add_to_queue(uri)
            print(f"Queued: {uri}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Queue error: {e}")

    print("\nLogging Interactions")
    while played_count < total_tracks + 1:
        try:
            playback = sp.current_playback()
            if not playback or not playback['is_playing']:
                time.sleep(2)
                continue
            current_uri = playback['item']['uri']
            if current_uri != last_played_uri:
                time_spent = time.time() - start_time
                track_id = last_played_uri.split(":")[-1]
                duration_sec = get_duration_ms(last_played_uri) / 1000
                percentage_listened = round(min(time_spent / duration_sec, 1.0), 2)
                liked = 1 if is_track_liked(last_played_uri) else 0
                latent_vector = get_track_latent(track_id)
                results.append({
                    "track_id": track_id,
                    "latent_vector": latent_vector,
                    "percentage_listened": percentage_listened,
                    "liked": liked
                })
                print(f"Logged: {track_id} | {percentage_listened} | liked: {liked}")
                last_played_uri = current_uri
                start_time = time.time()
                played_count += 1
        except Exception as e:
            print(f"Playback error: {e}")
            break
        time.sleep(2)

    output_path = "user_song_interactions.json"
    try:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print("File is empty or corrupted. Resetting log.")
                    existing_data = []
        else:
            existing_data = []
        existing_data.extend(results)
        with open(output_path, "w") as f:
            json.dump(existing_data, f, indent=2)
        print("\nInteractions saved to user_song_interactions.json")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    # Start with a random song
    start_song_id = np.random.choice(song_ids)
    generate_and_play_playlist(start_song_id, length=10)



