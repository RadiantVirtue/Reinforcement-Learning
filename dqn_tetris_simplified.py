"""
DQN for Simplified Tetris
Adapted from ALE Tetris DQN to use placement-based actions.

Key differences from ALE version:
1. Actions = final piece placements (column + rotation), not frame-by-frame
2. State = 4 hand-crafted features, not pixel observations
3. Network = small MLP, not CNN
4. Much faster training - hours instead of days

Based on nuno-faria/tetris-ai approach.
"""

import os
import random
import numpy as np
from collections import deque
from datetime import datetime
from typing import Tuple, Optional, List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =============================================================================
# TETRIS ENVIRONMENT (Simplified - placement based)
# =============================================================================

class Tetris:
    """
    Simplified Tetris environment where actions are final piece placements.
    Each step: select (rotation, column) -> piece drops instantly -> get reward
    """
    
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    
    # Tetromino shapes (each rotation)
    PIECES = {
        'I': [
            [[1, 1, 1, 1]],
            [[1], [1], [1], [1]]
        ],
        'O': [
            [[1, 1], [1, 1]]
        ],
        'T': [
            [[0, 1, 0], [1, 1, 1]],
            [[1, 0], [1, 1], [1, 0]],
            [[1, 1, 1], [0, 1, 0]],
            [[0, 1], [1, 1], [0, 1]]
        ],
        'S': [
            [[0, 1, 1], [1, 1, 0]],
            [[1, 0], [1, 1], [0, 1]]
        ],
        'Z': [
            [[1, 1, 0], [0, 1, 1]],
            [[0, 1], [1, 1], [1, 0]]
        ],
        'J': [
            [[1, 0, 0], [1, 1, 1]],
            [[1, 1], [1, 0], [1, 0]],
            [[1, 1, 1], [0, 0, 1]],
            [[0, 1], [0, 1], [1, 1]]
        ],
        'L': [
            [[0, 0, 1], [1, 1, 1]],
            [[1, 0], [1, 0], [1, 1]],
            [[1, 1, 1], [1, 0, 0]],
            [[1, 1], [0, 1], [0, 1]]
        ]
    }
    
    PIECE_NAMES = list(PIECES.keys())
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset game and return initial state features."""
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.current_piece = self._new_piece()
        return self._get_state_features()
    
    def _new_piece(self) -> str:
        """Get a random new piece."""
        return random.choice(self.PIECE_NAMES)
    
    def _get_piece_shape(self, piece: str, rotation: int) -> np.ndarray:
        """Get the shape array for a piece at given rotation."""
        rotations = self.PIECES[piece]
        return np.array(rotations[rotation % len(rotations)])
    
    def _get_num_rotations(self, piece: str) -> int:
        """Get number of unique rotations for a piece."""
        return len(self.PIECES[piece])
    
    def get_valid_actions(self) -> List[Tuple[int, int, np.ndarray, int]]:
        """
        Get all valid (rotation, column) placements for current piece.
        Returns list of (rotation, column, resulting_board, lines_cleared)
        """
        actions = []
        num_rotations = self._get_num_rotations(self.current_piece)
        
        for rotation in range(num_rotations):
            shape = self._get_piece_shape(self.current_piece, rotation)
            piece_width = shape.shape[1]
            
            for col in range(self.BOARD_WIDTH - piece_width + 1):
                result = self._simulate_drop(shape, col)
                if result is not None:
                    new_board, lines = result
                    actions.append((rotation, col, new_board, lines))
        
        return actions
    
    def _simulate_drop(self, shape: np.ndarray, col: int) -> Optional[Tuple[np.ndarray, int]]:
        """
        Simulate dropping a piece at given column.
        Returns (new_board, lines_cleared) or None if invalid.
        """
        board = self.board.copy()
        piece_h, piece_w = shape.shape
        
        # Find the row where the piece lands
        drop_row = 0
        for row in range(self.BOARD_HEIGHT - piece_h + 1):
            if self._check_collision(board, shape, row, col):
                break
            drop_row = row
        
        # Check if placement is valid (not above the board)
        if drop_row == 0 and self._check_collision(board, shape, 0, col):
            return None
        
        # Place the piece
        for py in range(piece_h):
            for px in range(piece_w):
                if shape[py, px]:
                    if drop_row + py < 0:
                        return None  # Piece doesn't fit
                    board[drop_row + py, col + px] = 1
        
        # Check for game over (piece placed at top)
        if drop_row == 0:
            # Check if any part of the piece is in the top rows
            for py in range(piece_h):
                for px in range(piece_w):
                    if shape[py, px] and drop_row + py < 2:
                        # Allow it but mark as risky
                        pass
        
        # Clear completed lines
        lines_cleared = 0
        new_board = []
        for row in range(self.BOARD_HEIGHT):
            if np.sum(board[row]) < self.BOARD_WIDTH:
                new_board.append(board[row])
            else:
                lines_cleared += 1
        
        # Add empty rows at top
        while len(new_board) < self.BOARD_HEIGHT:
            new_board.insert(0, np.zeros(self.BOARD_WIDTH, dtype=np.uint8))
        
        return np.array(new_board), lines_cleared
    
    def _check_collision(self, board: np.ndarray, shape: np.ndarray, row: int, col: int) -> bool:
        """Check if piece collides with board or boundaries."""
        piece_h, piece_w = shape.shape
        
        for py in range(piece_h):
            for px in range(piece_w):
                if shape[py, px]:
                    new_row = row + py
                    new_col = col + px
                    
                    # Check boundaries
                    if new_row >= self.BOARD_HEIGHT:
                        return True
                    if new_col < 0 or new_col >= self.BOARD_WIDTH:
                        return True
                    
                    # Check collision with existing pieces
                    if new_row >= 0 and board[new_row, new_col]:
                        return True
        
        return False
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action by index.
        Returns (state_features, reward, done)
        """
        actions = self.get_valid_actions()
        
        if not actions or action_idx >= len(actions):
            self.game_over = True
            return self._get_state_features(), -10.0, True
        
        rotation, col, new_board, lines = actions[action_idx]
        self.board = new_board
        self.pieces_placed += 1
        self.lines_cleared += lines
        
        # Calculate reward
        reward = self._calculate_reward(lines)
        
        # Check game over
        if np.any(self.board[0]):
            self.game_over = True
            reward -= 2.0  # Penalty for dying
        
        # Get next piece
        self.current_piece = self._new_piece()
        
        # Check if no valid moves
        if not self.game_over and not self.get_valid_actions():
            self.game_over = True
            reward -= 2.0
        
        return self._get_state_features(), reward, self.game_over
    
    def _calculate_reward(self, lines_cleared: int) -> float:
        """Calculate reward for the action taken."""
        # Reward for clearing lines (squared to incentivize tetrises)
        reward = lines_cleared ** 2 * self.BOARD_WIDTH
        
        # Small reward for placing a piece
        reward += 1
        
        self.score += reward
        return float(reward)
    
    def _get_state_features(self) -> np.ndarray:
        """
        Extract hand-crafted features from current board state.
        Features: [lines_cleared_potential, holes, bumpiness, total_height]
        """
        # Column heights
        heights = np.zeros(self.BOARD_WIDTH)
        for col in range(self.BOARD_WIDTH):
            for row in range(self.BOARD_HEIGHT):
                if self.board[row, col]:
                    heights[col] = self.BOARD_HEIGHT - row
                    break
        
        # Total height
        total_height = np.sum(heights)
        
        # Bumpiness (sum of absolute height differences between adjacent columns)
        bumpiness = np.sum(np.abs(np.diff(heights)))
        
        # Holes (empty cells with filled cells above them)
        holes = 0
        for col in range(self.BOARD_WIDTH):
            block_found = False
            for row in range(self.BOARD_HEIGHT):
                if self.board[row, col]:
                    block_found = True
                elif block_found:
                    holes += 1
        
        # Max height
        max_height = np.max(heights) if np.any(heights) else 0
        
        # Normalize features
        features = np.array([
            self.lines_cleared / 100.0,  # Normalized lines cleared so far
            holes / 40.0,                 # Normalized holes
            bumpiness / 40.0,             # Normalized bumpiness  
            total_height / 200.0          # Normalized total height
        ], dtype=np.float32)
        
        return features
    
    def get_next_states(self) -> List[Tuple[np.ndarray, int]]:
        """
        Get features for all possible next states.
        Returns list of (features, action_idx)
        """
        actions = self.get_valid_actions()
        next_states = []
        
        for idx, (rotation, col, new_board, lines) in enumerate(actions):
            # Temporarily set board to calculate features
            old_board = self.board.copy()
            old_lines = self.lines_cleared
            
            self.board = new_board
            self.lines_cleared += lines
            
            features = self._get_state_features()
            next_states.append((features, idx))
            
            # Restore
            self.board = old_board
            self.lines_cleared = old_lines
        
        return next_states


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Hyperparameters optimized for simplified Tetris."""
    
    # Network (small MLP - only 4 input features)
    INPUT_SIZE = 4
    HIDDEN_SIZES = [64, 64]
    
    # Training
    BATCH_SIZE = 512
    GAMMA = 0.99
    LEARNING_RATE = 1e-3
    
    # Experience Replay
    BUFFER_SIZE = 20_000
    MIN_BUFFER_SIZE = 1_000
    
    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_EPISODES = 1500  # Episodes to decay over
    
    # Logging
    LOG_FREQ = 50
    SAVE_FREQ = 200
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# NEURAL NETWORK (Small MLP)
# =============================================================================

class DQN(nn.Module):
    """
    Simple MLP that takes board features and outputs a single Q-value.
    We evaluate each possible next state and pick the highest.
    """
    
    def __init__(self):
        super(DQN, self).__init__()
        
        layers = []
        input_size = Config.INPUT_SIZE
        
        for hidden_size in Config.HIDDEN_SIZES:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))  # Output single Q-value
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Store transition (no action needed - we store state features directly)."""
        self.buffer.append((state, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample random batch."""
        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(Config.DEVICE),
            torch.FloatTensor(rewards).to(Config.DEVICE),
            torch.FloatTensor(np.array(next_states)).to(Config.DEVICE),
            torch.BoolTensor(dones).to(Config.DEVICE)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# DQN AGENT
# =============================================================================

class DQNAgent:
    """DQN Agent for simplified Tetris."""
    
    def __init__(self):
        self.policy_net = DQN().to(Config.DEVICE)
        self.target_net = DQN().to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(Config.BUFFER_SIZE)
        
        self.epsilon = Config.EPSILON_START
        self.episodes_done = 0
    
    def select_action(self, env: Tetris, training: bool = True) -> Tuple[int, np.ndarray]:
        """
        Select best action by evaluating all possible next states.
        Returns (action_idx, next_state_features)
        """
        next_states = env.get_next_states()
        
        if not next_states:
            return 0, env._get_state_features()  # No valid moves
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            idx = random.randrange(len(next_states))
            return next_states[idx][1], next_states[idx][0]
        
        # Evaluate all next states
        features = np.array([s[0] for s in next_states])
        features_tensor = torch.FloatTensor(features).to(Config.DEVICE)
        
        with torch.no_grad():
            q_values = self.policy_net(features_tensor)
            best_idx = q_values.argmax().item()
        
        return next_states[best_idx][1], next_states[best_idx][0]
    
    def update_epsilon(self):
        """Decay epsilon based on episodes."""
        self.epsilon = max(
            Config.EPSILON_END,
            Config.EPSILON_START - (self.episodes_done / Config.EPSILON_DECAY_EPISODES) *
            (Config.EPSILON_START - Config.EPSILON_END)
        )
    
    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < Config.MIN_BUFFER_SIZE:
            return None
        
        # Sample batch
        states, rewards, next_states, dones = self.replay_buffer.sample(Config.BATCH_SIZE)
        
        # Current Q values
        current_q = self.policy_net(states).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).squeeze()
            target_q = rewards + Config.GAMMA * next_q * (~dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_done': self.episodes_done
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episodes_done = checkpoint['episodes_done']


# =============================================================================
# TRAINING
# =============================================================================

def train(num_episodes: int = 2000, save_dir: str = "checkpoints"):
    """Main training loop."""
    os.makedirs(save_dir, exist_ok=True)
    
    env = Tetris()
    agent = DQNAgent()
    
    print(f"Training on device: {Config.DEVICE}")
    print(f"State features: {Config.INPUT_SIZE}")
    print(f"Network: {Config.HIDDEN_SIZES}")
    print("-" * 60)
    
    # Metrics
    episode_scores = []
    episode_lines = []
    episode_pieces = []
    best_score = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_loss = []
        
        while not env.game_over:
            # Select action (get best next state)
            action_idx, next_state_features = agent.select_action(env)
            
            # Execute action
            new_state, reward, done = env.step(action_idx)
            
            # Store transition
            agent.replay_buffer.push(
                next_state_features,  # State we chose
                reward,
                new_state,  # Actual resulting state
                done
            )
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
        
        # Episode complete
        agent.episodes_done += 1
        agent.update_epsilon()
        
        # Update target network every episode
        agent.update_target_network()
        
        # Track metrics
        episode_scores.append(env.score)
        episode_lines.append(env.lines_cleared)
        episode_pieces.append(env.pieces_placed)
        
        # Logging
        if (episode + 1) % Config.LOG_FREQ == 0:
            avg_score = np.mean(episode_scores[-Config.LOG_FREQ:])
            avg_lines = np.mean(episode_lines[-Config.LOG_FREQ:])
            avg_pieces = np.mean(episode_pieces[-Config.LOG_FREQ:])
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            
            print(f"Ep {episode+1:5d} | "
                  f"Score: {avg_score:8.1f} | "
                  f"Lines: {avg_lines:6.1f} | "
                  f"Pieces: {avg_pieces:6.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Eps: {agent.epsilon:.3f}")
        
        # Save checkpoints
        if (episode + 1) % Config.SAVE_FREQ == 0:
            agent.save(os.path.join(save_dir, f"tetris_ep{episode+1}.pt"))
        
        # Save best
        if env.score > best_score:
            best_score = env.score
            agent.save(os.path.join(save_dir, "tetris_best.pt"))
    
    # Final save
    agent.save(os.path.join(save_dir, "tetris_final.pt"))
    
    return agent, episode_scores, episode_lines


def evaluate(agent: DQNAgent, num_episodes: int = 10, render: bool = False) -> dict:
    """Evaluate trained agent."""
    env = Tetris()
    
    scores = []
    lines = []
    pieces = []
    
    for ep in range(num_episodes):
        state = env.reset()
        
        while not env.game_over:
            action_idx, _ = agent.select_action(env, training=False)
            state, reward, done = env.step(action_idx)
            
            if render:
                print_board(env)
        
        scores.append(env.score)
        lines.append(env.lines_cleared)
        pieces.append(env.pieces_placed)
        
        print(f"Episode {ep+1}: Score={env.score}, Lines={env.lines_cleared}, Pieces={env.pieces_placed}")
    
    return {
        'mean_score': np.mean(scores),
        'max_score': np.max(scores),
        'mean_lines': np.mean(lines),
        'max_lines': np.max(lines),
        'mean_pieces': np.mean(pieces)
    }


def print_board(env: Tetris):
    """Simple text rendering of the board."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print(f"Score: {env.score} | Lines: {env.lines_cleared} | Piece: {env.current_piece}")
    print("+" + "-" * env.BOARD_WIDTH + "+")
    for row in env.board:
        print("|" + "".join("█" if cell else " " for cell in row) + "|")
    print("+" + "-" * env.BOARD_WIDTH + "+")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DQN for Simplified Tetris")
    parser.add_argument("--mode", choices=["train", "eval", "demo"], default="train")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("=" * 60)
        print("DQN Training for Simplified Tetris")
        print("=" * 60)
        agent, scores, lines = train(num_episodes=args.episodes, save_dir=args.save_dir)
        
        # Plot results if matplotlib available
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Moving average
            window = 50
            scores_ma = np.convolve(scores, np.ones(window)/window, mode='valid')
            lines_ma = np.convolve(lines, np.ones(window)/window, mode='valid')
            
            ax1.plot(scores_ma)
            ax1.set_title('Score (Moving Average)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            
            ax2.plot(lines_ma)
            ax2.set_title('Lines Cleared (Moving Average)')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Lines')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'training_results.png'))
            print(f"Saved training plot to {args.save_dir}/training_results.png")
        except ImportError:
            print("Matplotlib not available, skipping plot")
    
    elif args.mode == "eval":
        checkpoint = args.checkpoint or os.path.join(args.save_dir, "tetris_best.pt")
        
        agent = DQNAgent()
        agent.load(checkpoint)
        
        print("=" * 60)
        print("Evaluating trained agent")
        print("=" * 60)
        
        metrics = evaluate(agent, num_episodes=10)
        print(f"\nResults: {metrics}")
    
    elif args.mode == "demo":
        checkpoint = args.checkpoint or os.path.join(args.save_dir, "tetris_best.pt")
        
        agent = DQNAgent()
        agent.load(checkpoint)
        
        print("=" * 60)
        print("Demo - watching agent play")
        print("=" * 60)
        
        import time
        env = Tetris()
        state = env.reset()
        
        while not env.game_over:
            action_idx, _ = agent.select_action(env, training=False)
            state, reward, done = env.step(action_idx)
            print_board(env)
            time.sleep(0.1)
        
        print(f"\nFinal Score: {env.score}")
        print(f"Lines Cleared: {env.lines_cleared}")
        print(f"Pieces Placed: {env.pieces_placed}")
