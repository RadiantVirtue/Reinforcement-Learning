import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    # https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py
    def __init__(self, state_size, action_size, hidden1=128, hidden2=256):
        
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.linear1 = nn.Linear(self.state_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, self.action_size)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0) # (1, state_size)

        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)

        probs = F.softmax(output, dim=-1)
        dist = Categorical(probs)
        return dist
    
    def act(self, state):
        dist = self.forward(state)        # batch = 1
        action = dist.sample()            # (1,)
        log_prob = dist.log_prob(action)  # (1,)
        return action.item(), log_prob.squeeze(0)
    
class Critic(nn.Module):
    def __init__(self, state_size, hidden1=128, hidden2=256):
        # state_size: flattened dimension of the observation
        super(Critic, self).__init__()
        self.state_size = state_size

        self.linear1 = nn.Linear(self.state_size, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, 1)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0) # (1, state_size)

        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)  # (batch, 1)
        return value
    
    def get_value(self, state):
        value = self.forward(state)  # (1, 1)
        return value.squeeze(0).squeeze(0)