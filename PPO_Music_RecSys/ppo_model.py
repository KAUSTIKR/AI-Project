# This is a PPO network built from scratch. The policy network outputs probabilities over the top 4 songs for a given input state.
# These top 4 songs represent the possible actions.
# We ensure that the same track (latent vector) is not selected again,
# and we also exclude the last 3 songs that were previously chosen to maintain diversity in the playlist.
# The PolicyNetwork consists of two hidden layers (each of size 128) with ReLU activations and an output layer followed by softmax for action probabilities over 4 songs(found using KNN).
# The ValueNetwork also has two hidden layers (size 128) with LeakyReLU activations and layer normalization, followed by a single output neuron for state value estimation.
# Both networks take a 5-dimensional input representing the latent vector of a track.
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Reference used:
# 1. PPO Implementation from Scratch | Reinforcement Learning, YT Channel: "https://www.youtube.com/watch?v=xHf8oKd7cgU&t=72s"
# 2. Proximal Policy Optimization (PPO) is Easy With PyTorch | Full PPO Tutorial, YT Channel: "https://www.youtube.com/watch?v=hlv79rcHws0&t=11s"
# 3. Proximal Policy Optimization (PPO) for LLMs Explained Intuitively, YT Channel: "https://www.youtube.com/watch?v=8jtAzxUwDj0&t=27s"
# 4. Proximal Policy Optimization (PPO) - How to train Large Language Models, YT Channel: "https://www.youtube.com/watch?v=TjHH_--7l8g"
# 5. RL CH10 - Policy Gradient algorithms (PPO and Deep Reinforcement Learning), YT Channel: "https://www.youtube.com/watch?v=Hvau7oC8TU0&list=PLZ_sI4f41TGvthD8dA7daahlbLV0yDW0w&index=10"
# 6. The 37 Implementation Details of Proximal Policy Optimization, Link: "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"
# 7. Proximal Policy Optimization by OpenAI Spinning Up, Link: "https://spinningup.openai.com/en/latest/algorithms/ppo.html"
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd

class MusicDataset:
    def __init__(self, data):
        self.data = data
        self.artist_ids = list(data.keys())
        self.latents = np.array([data[aid]['latents'] for aid in self.artist_ids])
        self.rewards = np.array([data[aid]['reward'] for aid in self.artist_ids])
        self.knn = NearestNeighbors(n_neighbors=10)
        self.knn.fit(self.latents)

    def get_knn_indices(self, state, current_idx, history_ids, k=4):
        _, indices = self.knn.kneighbors(state.reshape(1, -1), n_neighbors=10)
        indices = indices[0]
        filtered = [i for i in indices if i != current_idx and self.artist_ids[i] not in history_ids]
        if len(filtered) < k:
            extras = [i for i in indices if i not in filtered and i != current_idx]
            for i in extras:
                if i not in filtered:
                    filtered.append(i)
                if len(filtered) == k:
                    break
        return filtered[:k]

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

class PPOAgent:
    def __init__(self, dataset, state_dim=5, hidden_dim=128, lr=1e-4, gamma=0.95, clip_epsilon=0.1, ppo_epochs=100, batch_size=64):
        self.dataset = dataset
        self.state_dim = state_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, 4)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
        self.mse_loss = nn.MSELoss()
        self.policy_losses, self.value_losses, self.episode_rewards = [], [], []

    def get_action(self, state, current_idx, history_ids):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
        knn_indices = self.dataset.get_knn_indices(state, current_idx, history_ids)
        action_idx = torch.multinomial(action_probs, 1).item()
        selected_artist_idx = knn_indices[action_idx]
        return selected_artist_idx, action_probs[action_idx].item(), action_idx

    def compute_advantages(self, rewards, values, dones):
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * last_advantage
            last_advantage = advantages[t]
            next_value = values[t]
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, states, actions, old_probs, rewards, dones):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_probs = torch.FloatTensor(np.array(old_probs))
        rewards = torch.FloatTensor(np.array(rewards))
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        with torch.no_grad():
            values = self.value_net(states).numpy().flatten()
        advantages = torch.FloatTensor(self.compute_advantages(rewards.numpy(), values, dones)).unsqueeze(1)
        returns = advantages + torch.FloatTensor(values).unsqueeze(1)
        for _ in range(self.ppo_epochs):
            action_probs = self.policy_net(states)
            current_probs = action_probs.gather(1, actions.unsqueeze(1))
            current_values = self.value_net(states)
            ratios = current_probs / old_probs.unsqueeze(1)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.1 * self.mse_loss(current_values, returns)
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.optimizer.step()
        return policy_loss.item(), value_loss.item()

    def train(self, num_episodes, episode_length=10, print_interval=10):
        for episode in range(num_episodes):
            states, actions, old_probs, rewards, dones = [], [], [], [], []
            start_idx = np.random.randint(0, len(self.dataset.artist_ids))
            current_state = self.dataset.latents[start_idx]
            history_ids = [self.dataset.artist_ids[start_idx]]
            for step in range(episode_length):
                artist_idx, action_prob, action_idx = self.get_action(current_state, start_idx, history_ids)
                reward = self.dataset.rewards[artist_idx]
                states.append(current_state)
                actions.append(action_idx)
                old_probs.append(action_prob)
                rewards.append(reward)
                dones.append(step == episode_length - 1)
                current_state = self.dataset.latents[artist_idx]
                start_idx = artist_idx
                history_ids.append(self.dataset.artist_ids[artist_idx])
                if len(history_ids) > 3:
                    history_ids.pop(0)
            policy_loss, value_loss = self.update(states, actions, old_probs, rewards, dones)
            self.policy_losses.append(policy_loss)
            self.value_losses.append(value_loss)
            self.episode_rewards.append(np.mean(rewards))
            if episode % print_interval == 0:
                print(f"Episode {episode}: Policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}, Avg Reward = {np.mean(self.episode_rewards[-print_interval:]):.2f}")

    def plot_losses(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.policy_losses)
        plt.title("Policy Loss")
        plt.subplot(1, 2, 2)
        plt.plot(self.value_losses)
        plt.title("Value Loss")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_path = "ppo_input.csv"
    df = pd.read_csv(file_path)
    data = {
        row['id']: {
            "latents": [row[f'latent_{i}'] for i in range(5)],
            "reward": row['reward']
        } for _, row in df.iterrows()
    }
    dataset = MusicDataset(data)
    agent = PPOAgent(dataset)
    agent.train(num_episodes=100)
    agent.plot_losses()
    torch.save(agent.policy_net.state_dict(), 'policy_network.pt')
    torch.save(agent.value_net.state_dict(), 'value_network.pt')
    print("Parameters saved successfully.")
