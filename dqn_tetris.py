import os
import random
import numpy as np
from collections import deque
from datetime import datetime
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py


class Config:
    #Hyperparameters and configuration settings
    
    # Environment
    ENV_NAME = "ALE/Tetris-v5"
    FRAME_STACK = 4
    FRAME_SIZE = 84
    
    # Network
    HIDDEN_SIZE = 512
    
    # Training
    BATCH_SIZE = 32
    GAMMA = 0.99  # Discount factor
    LEARNING_RATE = 1e-4
    TARGET_UPDATE_FREQ = 1000  # Steps between target network updates
    
    # Experience Replay
    BUFFER_SIZE = 100_000
    MIN_BUFFER_SIZE = 10_000  # Minimum experiences before training
    
    # Exploration (slow decay for Tetris)
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_STEPS = 500_000  # Slow decay over 500k steps
    
    # Reward Shaping (simplified for speed)
    REWARD_SCALE = 0.01      # Scale down raw game score
    REWARD_ALIVE = 0.1       # Small reward for staying alive each step
    REWARD_GAME_OVER = -1.0  # Penalty for game over
    
    # Logging
    LOG_FREQ = 100  # Episodes
    SAVE_FREQ = 500  # Episodes
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#neural network
class DQN(nn.Module):
#DQN w CNN
    
    def __init__(self, n_actions: int, frame_stack: int = 4):
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions
        # Input: 84x84 -> conv1: 20x20 -> conv2: 9x9 -> conv3: 7x7
        conv_out_size = 64 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, Config.HIDDEN_SIZE)
        self.fc2 = nn.Linear(Config.HIDDEN_SIZE, n_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #forward pass thru network
        if x.max() > 1.0:
            x = x / 255.0
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    #expierence replay
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        # store
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        #sample random transitions
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(Config.DEVICE),
            torch.LongTensor(actions).to(Config.DEVICE),
            torch.FloatTensor(rewards).to(Config.DEVICE),
            torch.FloatTensor(np.array(next_states)).to(Config.DEVICE),
            torch.BoolTensor(dones).to(Config.DEVICE)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)



def shape_reward(reward: float, done: bool, truncated: bool) -> float:
    #simple reward shaping
    shaped = reward * Config.REWARD_SCALE  # Scale down raw score
    shaped += Config.REWARD_ALIVE          # Small reward for surviving
    
    # Penalty for game over (not truncation)
    if done and not truncated:
        shaped += Config.REWARD_GAME_OVER
        
    return shaped


#env wrapper
def make_tetris_env(render_mode: Optional[str] = None) -> gym.Env:
    # Register ALE environments
    gym.register_envs(ale_py)
    
    # Create base environment
    env = gym.make(
        Config.ENV_NAME,
        render_mode=render_mode,
        frameskip=1  # We'll handle frame skipping in preprocessing
    )
    
    # Apply Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=Config.FRAME_SIZE,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False  # Keep uint8 for memory efficiency
    )
    
    # Stack frames
    env = FrameStackObservation(env, stack_size=Config.FRAME_STACK)
    
    return env

class DQNAgent:
    #Double DQN agent
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQN(n_actions, Config.FRAME_STACK).to(Config.DEVICE)
        self.target_net = DQN(n_actions, Config.FRAME_STACK).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(Config.BUFFER_SIZE)
        
        # Training state
        self.steps_done = 0
        self.epsilon = Config.EPSILON_START
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        #select action through greedy policy
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def update_epsilon(self):
        #update exploration w linear decay
        self.epsilon = max(
            Config.EPSILON_END,
            Config.EPSILON_START - (self.steps_done / Config.EPSILON_DECAY_STEPS) * 
            (Config.EPSILON_START - Config.EPSILON_END)
        )
        
    def train_step(self) -> Optional[float]:
        #one training step
        if len(self.replay_buffer) < Config.MIN_BUFFER_SIZE:
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(Config.BATCH_SIZE)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values (Double DQN style)
        with torch.no_grad():
            # Use policy net to select actions
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Use target net to evaluate
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + Config.GAMMA * next_q * (~dones).unsqueeze(1)
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps_done % Config.TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def save(self, path: str):
        #save
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path: str):
        #load 
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']



def train(num_episodes: int = 10000, save_dir: str = "checkpoints"):
    #training loop
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment and agent
    env = make_tetris_env()
    agent = DQNAgent(n_actions=env.action_space.n)
    
    print(f"Training on device: {Config.DEVICE}")
    print(f"Action space: {env.action_space.n} actions")
    print(f"Observation space: {env.observation_space.shape}")
    print("-" * 60)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        state, info = env.reset()
        
        episode_reward = 0
        episode_length = 0
        episode_loss = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Shape reward (simplified - no expensive board analysis)
            shaped_reward = shape_reward(reward, done, truncated)
            
            # Store transition
            agent.replay_buffer.push(state, action, shaped_reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward  # Track original reward for metrics
            episode_length += 1
            agent.steps_done += 1
            agent.update_epsilon()
            
        # Episode complete
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_loss:
            episode_losses.append(np.mean(episode_loss))
            
        # Logging
        if (episode + 1) % Config.LOG_FREQ == 0:
            avg_reward = np.mean(episode_rewards[-Config.LOG_FREQ:])
            avg_length = np.mean(episode_lengths[-Config.LOG_FREQ:])
            avg_loss = np.mean(episode_losses[-Config.LOG_FREQ:]) if episode_losses else 0
            
            print(f"Episode {episode + 1:5d} | "
                  f"Avg Reward: {avg_reward:8.1f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer):6d}")
            
        # Save checkpoints
        if (episode + 1) % Config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(save_dir, f"dqn_tetris_ep{episode+1}.pt")
            agent.save(checkpoint_path)
            
        # Save best model
        if episode_rewards[-1] > best_reward:
            best_reward = episode_rewards[-1]
            agent.save(os.path.join(save_dir, "dqn_tetris_best.pt"))
            
    # Save final model
    agent.save(os.path.join(save_dir, "dqn_tetris_final.pt"))
    env.close()
    
    return agent, episode_rewards

def evaluate(agent: DQNAgent, num_episodes: int = 10, render: bool = False) -> dict:
    #eval train agent
    env = make_tetris_env(render_mode="human" if render else None)
    
    rewards = []
    lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
        rewards.append(episode_reward)
        lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Length = {episode_length}")
        
    env.close()
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DQN for Tetris")
    parser.add_argument("--mode", choices=["train", "eval", "demo"], default="train",
                        help="Mode: train, eval, or demo")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Number of training episodes")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval/demo")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("=" * 60)
        print("DQN Training for Tetris")
        print("=" * 60)
        agent, rewards = train(num_episodes=args.episodes, save_dir=args.save_dir)
        
    elif args.mode == "eval":
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.save_dir, "dqn_tetris_best.pt")
        
        env = make_tetris_env()
        agent = DQNAgent(n_actions=env.action_space.n)
        agent.load(args.checkpoint)
        env.close()
        
        print("=" * 60)
        print("Evaluating trained agent")
        print("=" * 60)
        metrics = evaluate(agent, num_episodes=10, render=False)
        print(f"\nResults: {metrics}")
        
    elif args.mode == "demo":
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.save_dir, "dqn_tetris_best.pt")
            
        env = make_tetris_env()
        agent = DQNAgent(n_actions=env.action_space.n)
        agent.load(args.checkpoint)
        env.close()
        
        print("=" * 60)
        print("Demo mode - watching trained agent play")
        print("=" * 60)
        evaluate(agent, num_episodes=3, render=True)
