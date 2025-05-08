# AI-Project
Adaptive Spotify Music Recommendation using Reinforcement Learning

## Project Description

This project implements a personalized music recommendation system using the **Proximal Policy Optimization (PPO)** algorithm with an **Actor-Critic architecture**. It recommends songs based on user preferences derived from real-time feedback (percentage listened, likes) and automatically plays them using the Spotify API. Over time, the model adapts to user behavior, improving the quality of future playlists.

In the context of music recommendation systems, the system is faced with millions of songs to choose from. This creates a vast action space, where each song represents a possible recommendation the system could make (state). Additionally, user behavior is highly uncertain. Since musical preferences are personal, dynamic, and often unpredictable, there is no fixed formula to determine whether a user will like or dislike a particular song. Moreover, the system must learn from the user’s listening history to improve future recommendations. This transforms the task from a simple, one-step prediction into a sequential decision-making problem that requires ongoing learning and adaptation. In such an environment, the agent must learn through interaction by recommending songs, observing user feedback, and refining its strategy over time. However, due to the huge size of the action space, it is infeasible to explore or store information about every possible state. To address these challenges, we apply deep reinforcement learning, where neural networks enable the agent to generalize from limited experience and make effective predictions without needing to visit or memorize every individual state or action.

---
## Related Solutions

Traditional recommendation methods, such as collaborative filtering, have been widely used to model user-item interactions. But these methods usually treat each recommendation as a single guess, without thinking about the order of what the user listened to before.  
Recent works have introduced reinforcement learning (RL) techniques to address these limitations:

1. **Deep Q-Network (DQN) with Simulated Training**  
One study proposes the use of a Deep Q-Network (DQN) trained in a simulated playlist-generation environment. This approach allows the system to handle the large action space by learning from trial-and-error interactions in a safe, offline setting *(Tomasi et al., 2023)*. But DQN faces limitations when it comes to large action space or state space.

2. **List-wise Recommendations via MDP and Online Simulation**  
Another work models the recommendation process as a Markov Decision Process (MDP), where:  
- Each moment of interaction is a state,  
- Recommending something is an action,  
- And the user’s reaction (e.g: click, skip) is the reward.  
Similar to the DQN-based method, this approach uses an online environment simulator to pre-train and evaluate the model.  
*(Zhao et al., 2018)*

3. **Continuous Action Space with DDPG (Deep Deterministic Policy Gradient)**  
A third approach leverages DDPG, a type of reinforcement learning designed for continuous action spaces. Rather than selecting songs by Id, this method represents each song using continuous features such as tempo, energy, or mood. This allows the system to handle a much larger number of song options while still providing accurate and varied recommendations.  
*(Qian, Zhao, & Wang, 2019)*
---
## State Space Representation

The state space is represented by a latent vector derived from audio features and metadata of individual tracks. These **latent vectors** are learned using a **Variational Autoencoder (VAE)**, which compresses high-dimensional audio feature data into a lower-dimensional embedding space. This latent representation captures the essential characteristics of each song, enabling compact and meaningful state descriptions. The state space **𝑆** thus consists of all such latent vectors corresponding to the available tracks, where each vector serves as a unique, continuous representation of the musical content and style of a track. This formulation allows the reinforcement learning agent to generalize across similar tracks and effectively learn user preferences, even in **cold start (no user history)** scenarios. Since we have a cold-start problem, we cannot use a user-track interaction matrix for state space representation and must instead rely on latent vector generation.

## Problem Formulation

We model the problem as a **Partially Observable Markov Decision Process (POMDP)**, where the agent does not have full access to the state of the environment, such as a user's preferences or listening context. Instead, it must make decisions based on partial observations and indirect feedback.

In reinforcement learning, an agent interacts with an environment over a sequence of time steps. At each time step 
𝑡, the agent selects an action **𝑎<sub>t</sub> ∈ 𝐴** based on its current state **𝑠<sub>t</sub> ∈ 𝑆**, following a policy **𝜋: 𝑆 → 𝐴**. After executing the action, it receives a **reward 𝑟<sub>t</sub> :𝑆 × 𝐴 → 𝑅** and transitions to the **next state 𝑠<sub>t+1</sub>**. The objective is to learn a policy that maximizes the expected cumulative reward.

To achieve this, we will be using **Proximal Policy Optimization (PPO)** a **policy-gradient algorithm** that enables stable learning in high-dimensional and continuous state spaces.

# Mathematical Description
## States Space:

Each state **𝑠 ∈ 𝑆** is a continuous-valued latent vector derived from the audio features of a song using a trained Variational Autoencoder (VAE): 

**𝑠<sub>t</sub> ∈ 𝑅<sup>d</sup>** , where **𝑠<sub>t</sub>** = VAEencoder(𝑥<sub>t</sub>)

**x<sub>t</sub> ∈ 𝑅<sup>n</sup>** : vector of audio features for track 𝑡 (e.g., danceability, energy, valence, etc.)

**𝑑 ≪ n** : dimensionality of latent space (4)

## Action:

Actions 𝑎 ∈ 𝐴 represent the track recommendations made by the agent at each time step. In our approach, actions are encoded in a continuous latent space, where each action corresponds to the latent vector of a candidate track. These vectors are learned via a Variational Autoencoder (VAE) and normalized within a bounded range (0-1). To learn the optimal policy, we use Proximal Policy Optimization (PPO), a policy-gradient method that which can take continuous action spaces as input.

In our formulation, actions represent track recommendations made by the agent at each time step. Each action corresponds to a latent vector in a continuous space:
**𝑎<sub>t</sub> ∈ 𝐴 ⊆ 𝑅<sup>d</sup>**

Where: 

**𝑎<sub>t</sub>** is the latent vector of the recommended track at time t,

**𝑑** is the dimensionality of the latent space learned by the VAE (4)

## Reward:

The reward  𝑟 represents the immediate feedback received by the agent as a consequence of executing an action **𝑎<sub>t</sub>** and transitioning from the current state **𝑠<sub>t</sub>** to the next state **𝑠<sub>t+1</sub>**. It may be positive (reward) or zero (penalty), depending on the quality of the action taken in the given context.

The reward is based on implicit user feedback (e.g., play, skip, like):

**𝑟<sub>t</sub> = 𝑅(𝑠<sub>t</sub>,𝑎<sub>t</sub>)** = +1 if user liked or completed track, or 
                                                   = 0 if user skipped or disliked track

## Transition Probability:

The transition probability function **𝑇** defines the likelihood of the agent transitioning to the next state **𝑠<sub>t+1</sub>**, given the current state **𝑠<sub>t</sub>** and action **𝑎<sub>t</sub>**. 
Formally, this is expressed as:
**𝑃(𝑠<sub>t+1</sub>∣𝑠<sub>t</sub>,𝑎<sub>t</sub>)**

In classical reinforcement learning, a transition tensor can be constructed where each element represents:

**𝑃<sub>𝑠𝑎𝑠′</sub>=𝑃(𝑠<sub>t+1</sub>=𝑠′∣𝑠<sub>t</sub>=𝑠,𝑎<sub>t</sub>=𝑎)**

These transition probabilities effectively model the environment’s dynamics, allowing an agent to anticipate future states resulting from its actions. Algorithms that rely on such knowledge are known as model-based methods.
However, in our setting, the state **𝑠<sub>t</sub>** ∈ **𝑅<sup>4</sup>** is a continuous latent vector generated by **VAE**, and the action **𝑎<sub>t</sub>** is also selected in a continuous space. This makes it infeasible to explicitly define or compute transition probabilities for the vast number of possible state action next state combinations.
As a result, we adopt a model-free reinforcement learning approach, specifically **Proximal Policy Optimization (PPO)**. PPO does not require explicit modeling of the transition probabilities. Instead, it learns the optimal policy directly from sampled experience, updating the agents behavior based on observed and received rewards.

## Observations:

In your case, since user preferences (the true state) are hidden, observations represent implicit or explicit feedback from the user in response to a recommended track.
The observation function **𝑍** defines the probability of observing **𝑜<sub>t</sub>** given that the system is in state **𝑠<sub>t</sub>** and the agent took action 
**𝑎<sub>t</sub>:**

**𝑍(𝑜<sub>t</sub>∣𝑠<sub>t</sub>,𝑎<sub>t</sub>) = 𝑃(𝑜<sub>t</sub>∣𝑠<sub>t</sub>,𝑎<sub>t</sub>)**

**𝑠<sub>t</sub>**: latent representation of the currently playing track

**𝑎<sub>t</sub>**: action (i.e., track recommendation)

**𝑜<sub>t</sub>**: observation (e.g., feedback such as play, skip, like)

## PPO
We use [PPO for Policy Update](https://spinningup.openai.com/en/latest/algorithms/ppo.html#key-equations)

---
## Solution Method

Our solution leverages Deep Reinforcement Learning using Proximal Policy Optimization (PPO), which is an algorithm that trains both a policy network and a value network at the same time, improving them together throughout the learning process.  
We adopt an Actor-Critic architecture, where:  
- The actor represents the policy (i.e., how the system chooses songs),  
- The critic estimates the value of the current state (i.e., how promising the situation is based on the user's history).  

- **State Space**:  
Each state includes the currently playing song along with its features (such as tempo, energy, or genre), all represented as a vector.  

- **Action Space**:  
The action corresponds to selecting the next song from the available pool.  

- **Reward Function**:  
The system receives a positive reward when a user likes or listens to the song fully, and a negative reward if the user skips it, especially if they skip it immediately.
---
## Solution Implementation

### 1. State Representation
- Each song is originally represented in an 11-dimensional feature space (e.g., tempo, energy, etc.).
- To make the model more efficient, we use a Variational Autoencoder (VAE) to reduce this to a 5-dimensional latent space.
- These compressed vectors become our state representations.
- All feature values are normalized to the range [0, 1].

### 2. Value Network
- **Input**: The 5D latent vector representing the current song.  
- **Output**: A single value estimating the expected future reward (i.e., how good the current situation is).  
- **Training**: The value network is trained to minimize the Mean Squared Error (MSE) between its prediction and the actual return.

### 3. Policy Network
- **Input**: Same 5D latent vector.  
- **Output**: A probability distribution over the top 4 recommended songs, filtered using k-Nearest Neighbors (kNN).  
- **Training**: Trained using PPO loss, based on advantage estimates.  
- The model either samples from or selects the most likely action (i.e., next song) to maximize user satisfaction.

### 4. Reward
It is calculated by:  
**reward = percentage_listened + λ × liked**  

Where:  
- **percentage_listened**: how much of the song was played (e.g., 0.7 if 70% was played).  
- **liked**: a binary signal (1 if the user liked the song, 0 otherwise).  
- **λ**: a weight that emphasizes the importance of “liking” (e.g., 10 or 100).

### 5. Offline Training
We use offline reinforcement learning, training both:  
- A policy network to decide which song to recommend  
- A value network to estimate how good a state is  
This is all done using pre-collected logs (records of past user interactions that have already been saved).

### 6. Inference Phase
Once trained, the model is used online as follows:

1. Getting the Current Song's Vector (State)  
→ The 5D latent representation of the current song.  

2. Using kNN to Filter Action Space  
→ Find the top 4 nearest songs as potential recommendations.  

3. Running the Value & Policy Network  
→ Generate values & probabilities over those top 4 candidates.  

4. Choosing the Next Song  
→ Either pick the most probable one or sample based on the policy.  
→ If real-time feedback is available (liked/skipped), the system can log new interactions and periodically fine-tune the models (online learning).

### 7. Reward Propagation with Clustering

#### 1. Clustering the Songs
We grouped songs into clusters based on how similar they are by using K-Means on their 2D latent vectors.

#### 2. Collecting Real Feedback for a Few Songs in Each Cluster
From each cluster, we picked around 4 songs.  
For each of those songs, we record data:  
- How much of the song was played (percentage_listened, e.g., 0.64)  
- Whether the user liked it (liked, e.g., 0.9)  
- And we compute the reward.

#### 3. Estimate Rewards for the Other Songs in the Cluster
For each untested song within a cluster, we:  
- Calculated its distance to the cluster centroid using Euclidean distance:  
  `d = ∥latent_vector − centroid∥`  
  This distance indicates how similar the song is to the center of the cluster (and by extension, to the top-4 songs used for evaluation).  
- Normalized the distances so that the maximum distance within the cluster is scaled to 1.  
- Applied a decay function based on the normalized distance. Songs closer to the centroid receive rewards similar to the top-4 songs, while those farther away receive proportionally lower rewards. This allows the estimated reward to decrease smoothly with increasing distance.

## Features

- Real-time song playback via Spotify Web API
- Actor-Critic network trained using PPO for adaptive recommendations
- Feedback loop based on percentage listened and liked status
- Top-4 song selection using KNN over latent vectors
- Logs user interactions to retrain and fine-tune the model
- Episode-based policy improvement with live policy/value loss tracking
  
---

## Spotify Premium Setup Instructions

Before running the real-time playlist recommender, you **must** have:

- A **Spotify Premium** account (required for playback control)
- Your own **Spotify API credentials** (Client ID & Client Secret)


### Step-by-Step: Get Spotify Client ID & Secret

1. Go to [Spotify Developer Console](https://developer.spotify.com/documentation/web-api)
2. **Log in** with your **Spotify Premium** account.
3. Click your **profile icon** (top-right) → Select **Dashboard**.
4. Click **"Create an App"**
   - **App Name**: any name (e.g., `Playlist Recommender`)
   - **App Description**: e.g., `Real-time Spotify playlist generator using PPO`
   - **Redirect URI**:  
     ```
     http://127.0.0.1:8888/callback
     ```
   - Click **Add**, then click **Save** the app.

5. After saving:
   - Copy your **Client ID**
   - Click **"View Client Secret"** to copy the secret

---
## Spotify API Used

```bash
user-modify-playback-state

user-read-playback-state

user-library-read
```
Ensure these scopes are enabled when authenticating via SpotifyOAuth.

---
## Installation

Install my-project with npm
1. **Clone the repository:**
```bash
git clone https://github.com/KAUSTIKR/AI-Project
cd AI-Project/PPO_Music_RecSys
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Run the Recommendation Script:** <br>
NOTE: Replace <CLIENT_ID> and <CLIENT_SECRET> with your Spotify API credentials:

```bash
python final_rec_songs.py --client_id=<CLIENT_ID> --client_secret=<CLIENT_SECRET>
```
When running the program for the first time, you’ll be prompted to grant permission for playlist and playback access. Simply click “Agree” on the authorization screen to continue and enjoy your music recommended.

**Interact with the System:**
- The script will start playing songs.
- Your interactions (% listened and liked) are recorded in user_song_interactions.json in the current directory.
  
---

# PPO Model Architecture
**State**: A 5-D latent vector representing each song, extracted from a Variational Autoencoder (VAE)

**Action**: 
The top-4 recommended songs selected using K-Nearest Neighbors (KNN) based on cosine similarity in latent space. The actor network outputs a probability distribution over these 4 songs.

**Reward**: 
Computed from user feedback as combination of: percentage_listened, liked (binary feedback)

**Policy network (Actor)** → outputs softmax distribution over top-4 actions
```bash
Input: 5D latent vector

Layer (type)              Output Shape           Param #
-----------------------------------------------------------
Linear (fc1)              (128,)                 768       # 5×128 + 128
ReLU
Linear (fc2)              (128,)                 16,512    # 128×128 + 128
ReLU
Linear (fc3)              (4,)                   516       # 128×4 + 4
Softmax
-----------------------------------------------------------
Total Parameters:                               17,796
Trainable Params:                               17,796

```
**Value network (Critic)** → estimates expected return from state
```bash
Input: 5D latent vector

Layer (type)              Output Shape           Param #
-----------------------------------------------------------
Linear (fc1)              (128,)                 768       # 5×128 + 128
LeakyReLU
LayerNorm                 (128,)                 256       # 2×128 (weight + bias)

Linear (fc2)              (128,)                 16,512    # 128×128 + 128
LeakyReLU
LayerNorm                 (128,)                 256       # 2×128

Linear (fc3)              (1,)                   129       # 128×1 + 1
-----------------------------------------------------------
Total Parameters:                               17,921
Trainable Params:                               17,921
```
---
## Dataset
Source: [Spotify Songs Dataset on Kaggle](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)

This dataset includes 11 audio features along with additional metadata. We apply an autoencoder to compress the feature space from 11 dimensions to 5, resulting in a new dataset named "filtered_artist_data", which is used for subsequent processing.

---
## Results
Looking at results:

**Current State(s) --> Actions(Top 4 songs) --> Next State(s')**
   
```bash 
Current song ID (state): 05S5yY7H0WuiQsEhrtjQj5
Top 4 recommended songs (actions) with probabilities:
  1. 72GBvm75IHjawz11FCcDma — Prob: 0.728
  2. 0PG9fbaaHFHfre2gUVo7AN — Prob: 0.239
  3. 7aBxcRw77817BrkdPChAGY — Prob: 0.017
  4. 4lJNen4SMTIJMahALc3DcB — Prob: 0.016
Selected next song (next state): 72GBvm75IHjawz11FCcDma
```
The displayed probabilities represent the policy learned by the model during the training phase.

**Generates Playlists (10 songs)**

```bash
Final Playlist
1. 05S5yY7H0WuiQsEhrtjQj5
2. 72GBvm75IHjawz11FCcDma
3. 6d1B1k4lvWNSu3LFlUw1Gj
4. 04zTmVio529e6gGh21Tcnb
5. 0ILEnJtqVXZZ5zyH12YIIc
6. 1NCSa6QvClPVe7KqsqjnMn
7. 6axMDyb9uCb30oXvVSlANp
8. 1L3NV7VrCiuE8C5QlhdeQL
9. 7GZCNHOruZsbNYIaPud5Lb
10. 2HOjSDwKRMq2NZ78aGewy2
```
The playlist generation begins with a starting track, such as **05S5yY7H0WuiQsEhrtjQj5**. Each time the program runs, a different starting track is randomly selected from the dataset, resulting in a unique playlist. All recommended tracks are then added to the Spotify app's playback queue.

**Logging Interactions**
 ```bash
Logging Interactions
Logged: 05S5yY7H0WuiQsEhrtjQj5 | 0.3 | liked: 0
Logged: 72GBvm75IHjawz11FCcDma | 1.0 | liked: 1
```
Logs for each track are stored in a JSON file and can be used later to fine-tune or update the learned policy.

 ---
## Acknowledgements

 - [Spotify](https://developer.spotify.com/documentation/web-api)
 - [OpenAI PPO Resources](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
 - [37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
 - [PPO Implementation](https://www.youtube.com/watch?app=desktop&v=hlv79rcHws0)
 - [Proximal Policy Optimization (PPO) - Explained](https://www.youtube.com/watch?v=TjHH_--7l8g&t=37s)
 - [Deep RL class by huggingface](https://huggingface.co/blog/deep-rl-ppo)


## References
- Qadeer Khan, Torsten Schön, Patrick Wenzel. *Latent Space Reinforcement Learning for Steering Angle Prediction*. [arXiv](https://arxiv.org/abs/1902.03765)
- Nick Qian - Sophie Zhao - Yizhou Wang. (n.d.). *Spotify Reinforcement Learning Recommendation System*. [Link](https://sophieyanzhao.github.io/AC297r_2019_SpotifyRL/2019-12-14-Spotify-Reinforcement-Learning-Recommendation-System/)
- Tomasi, F., Cauteruccio, J., Kanoria, S., Ciosek, K., Rinaldi, M., & Dai, Z. (2023, October 13). *Automatic Music Playlist Generation via simulation-based reinforcement learning*. [arXiv](https://arxiv.org/abs/2310.09123)
- Zhao, X., Xia, L., Zhang, L., Ding, Z., Yin, D., & Tang, J. (2018). *Deep reinforcement learning for page-wise recommendations*. [DOI](https://doi.org/10.1145/3240323.3240374)
  
---

