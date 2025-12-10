import numpy as np
import torch

class RolloutBuffer:
    # https://www.youtube.com/watch?v=xHf8oKd7cgU
    def __init__(self, buffer_size: int, state_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.device = device

        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0

    def store(self, state_flat, action, reward, done, log_prob, value):
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer is full. Cannot store more data.")

        self.states[self.ptr] = state_flat
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def is_full(self):
        return self.ptr >= self.buffer_size

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        advantages = np.zeros_like(self.rewards)
        last_gae = 0.0

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:self.ptr]
        return returns, advantages
    
    def get(self, last_value, gamma=0.99, lam=0.95):
        returns, advantages = self.compute_returns_and_advantages(last_value, gamma, lam)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        states_tensor = torch.from_numpy(self.states[:self.ptr]).to(self.device)
        actions_tensor = torch.from_numpy(self.actions[:self.ptr]).to(self.device)
        old_log_probs_tensor = torch.from_numpy(self.log_probs[:self.ptr]).to(self.device)
        returns_tensor = torch.from_numpy(returns[:self.ptr]).to(self.device)
        advantages_tensor = torch.from_numpy(advantages[:self.ptr]).to(self.device)

        return states_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor