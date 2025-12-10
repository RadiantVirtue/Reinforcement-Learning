import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ale_py
from ale_py import ALEInterface
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#hyperparameters class

#proper environment wrapper for interactions
class TetrisALE(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, rom_path="./ale_py/roms/tetris.bin", frame_skip=4, render=False):
        super().__init__()

        self.ale = ALEInterface()
        self.ale.setInt("frame_skip", 1)
        self.ale.setBool("display_screen", render)
        self.ale.setBool("sound", False)
        self.ale.loadROM(rom_path)
        self.actions = self.ale.getLegalActionSet()
        self.action_space = spaces.Discrete(len(self.actions))
        self.obs_height = 84
        self.obs_width = 84
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_height, self.obs_width),
            dtype=np.uint8
        )
        self.frame_skip = frame_skip

    def _get_obs(self):
        #fetch current frame, 84x84
        frame = self.ale.getScreenRGB() 
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.ale.setInt("random_seed", seed)
        self.ale.reset_game()
        obs = self._get_obs()
        return obs, {}

    def step(self, action_index):
        #ALE actions list
        ale_action = self.actions[action_index]
        total_reward = 0.0
        done = False

        for _ in range(self.frame_skip):
            reward = self.ale.act(ale_action)
            total_reward += reward

            if self.ale.game_over():
                done = True
                break

        obs = self._get_obs()
        info = {}
        return obs, total_reward, done, False, info
    def render(self):
        pass  
    def close(self):
        pass

#CNN Actor-Critic 
class PPO(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        #Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # (84x84) --> (20x20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # --> (9x9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # --> (7x7)
            nn.ReLU()
        )
        #output size: 84x84 --> 20x20 --> 9x9 --> 7x7
        conv_output_size = 64 * 7 * 7

        #pytorch nn.Sequential feeds data in a cascading manner
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
        )

        # actor and critic heads
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        #pass data through layers
        # (batch, 84, 84) --> (batch, 1, 84, 84)
        x = x.unsqueeze(1).float() / 255.0
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        hidden = self.fc(features)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

#PPO model wrapper
class PPOAgent:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        # memory buffers:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    #given a particular state, select an action from policy
    def select_action(self, state):
        #convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        #pass state into network
        logits, value = self.model(state_tensor)
        #convert logit values to probabilities
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, log_prob, value, done):
        #after each step, store everything for ppo update
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)


env = TetrisALE()
num_actions = env.action_space.n
model = PPO(num_actions)
agent = PPOAgent(env, model)
state, _ = env.reset()
done = False
step = 0

print("=== PPO Test ===")

while not done and step < 20:   # run 20 steps for testing
    action, log_prob, value = agent.select_action(state)

    print(f"Step {step}")
    print(f"  Action:       {action}")
    print(f"  Log Prob:     {log_prob}")
    print(f"  Value:        {value}")

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    agent.store_transition(
        state, action, reward, log_prob, value, done
    )
    state = next_state
    step += 1

print("=== Test Finished ===")
print("Stored transitions:", len(agent.states))

#TO-DO:
#1) clipped updates function
#2) advantage compute function
#3) learn and demo 
