import gymnasium as gym
import ale_py
import numpy as np
import torch

from buffer import RolloutBuffer
from config import (
    ENV,
    SEED,
    GAMMA,
    LAMBDA,
    ACTOR_LR,
    CRITIC_LR,
    CLIP_EPSILON,
    PPO_EPOCHS,
    BATCH_SIZE,
    VALUE_COEF,
    ENTROPY_COEF,
    ROLLOUT_STEPS,
    NUM_UPDATES,
)
from networks import Actor, Critic
from ppo import ppo_update

"""
.\.venv\Scripts\Activate.ps1
"""

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)


def make_env(render_mode=None):
    """
    render mode options:
    - None: No rendering
    - "human": Render to the screen
    - "rgb_array": Return RGB array observations
    """
    env = gym.make(ENV, render_mode=render_mode)
    return env


def run_random_agent(num_episodes=3, render=True):
    """
    Simple random agent to verify environment runs correctly.
    """
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)

    for episode in range(num_episodes):
        obs, info = env.reset(seed=SEED + episode)

        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")

    env.close()


def test_actor_critic_once(render=False):
    """
    Test that:
    - we can flatten Tetris observations
    - Actor can sample an action + log-prob
    - Critic can output a value estimate
    """
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)

    # One seeded reset for reproducibility
    obs, info = env.reset(seed=SEED)

    # Flatten observation to match state_size for Actor/Critic
    obs_flat = np.array(obs, dtype=np.float32).flatten()
    state_size = obs_flat.shape[0]
    action_size = env.action_space.n

    print("Flattened state_size:", state_size)
    print("Number of actions:", action_size)

    # Create networks
    actor = Actor(state_size, action_size)
    critic = Critic(state_size)

    # Convert obs_flat to torch tensor
    state_tensor = torch.from_numpy(obs_flat)

    # Actor: sample action + log_prob
    action, log_prob = actor.act(state_tensor)
    # Critic: value estimate
    value = critic.get_value(state_tensor)

    print("Sampled action:", action)
    print("Log prob:", log_prob.item())
    print("Value estimate:", value.item())

    env.close()

def test_rollout_buffer_once():
    """
    Collect a small rollout using Actor/Critic + Buffer,
    then print the shapes of what we get out.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(render_mode=None)
    obs, info = env.reset(seed=SEED)

    # Flatten obs to 1D
    obs_flat = np.array(obs, dtype=np.float32).flatten()
    state_size = obs_flat.shape[0]
    action_size = env.action_space.n

    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size).to(device)

    buffer_size = 128  # small test rollout
    rollout = RolloutBuffer(buffer_size, state_size, device)

    state = obs_flat
    terminated = False
    truncated = False

    while not rollout.is_full():
        state_tensor = torch.from_numpy(state).to(device)

        # Actor + Critic
        action, log_prob = actor.act(state_tensor)
        value = critic.get_value(state_tensor).item()

        # Step env
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_flat = np.array(next_obs, dtype=np.float32).flatten()

        rollout.store(
            state_flat=state,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob.item(),
            value=value,
        )

        state = next_flat

        if done:
            state, info = env.reset()
            state = np.array(state, dtype=np.float32).flatten()
            terminated = truncated = False

    # Get critic value for last state
    last_state_tensor = torch.from_numpy(state).to(device)
    last_value = critic.get_value(last_state_tensor).item()

    states, actions, old_log_probs, returns, advantages = rollout.get(
        last_value, gamma=0.99, lam=0.95
    )

def test_ppo_update_once():
    """
    Collect one rollout with Actor/Critic + Buffer,
    then run a single PPO update step.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(render_mode=None)
    obs, info = env.reset(seed=SEED)

    # Flatten obs to 1D
    obs_flat = np.array(obs, dtype=np.float32).flatten()
    state_size = obs_flat.shape[0]
    action_size = env.action_space.n

    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size).to(device)

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    buffer_size = ROLLOUT_STEPS
    rollout = RolloutBuffer(buffer_size, state_size, device)

    state = obs_flat
    terminated = False
    truncated = False

    # Collect one rollout
    while not rollout.is_full():
        state_tensor = torch.from_numpy(state).to(device)

        # Actor + Critic
        action, log_prob = actor.act(state_tensor)
        value = critic.get_value(state_tensor).item()

        # Step env
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_flat = np.array(next_obs, dtype=np.float32).flatten()

        rollout.store(
            state_flat=state,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob.item(),
            value=value,
        )

        state = next_flat

        if done:
            obs, info = env.reset()
            state = np.array(obs, dtype=np.float32).flatten()
            terminated = truncated = False

    # Get critic value for last state
    last_state_tensor = torch.from_numpy(state).to(device)
    last_value = critic.get_value(last_state_tensor).item()

    # Get tensors from buffer
    states, actions, old_log_probs, returns, advantages = rollout.get(
        last_value, gamma=GAMMA, lam=LAMBDA
    )

    print("Collected rollout:")
    print("  states:", states.shape)
    print("  actions:", actions.shape)
    print("  old_log_probs:", old_log_probs.shape)
    print("  returns:", returns.shape)
    print("  advantages:", advantages.shape)

    # Run one PPO update
    policy_loss, value_loss, entropy = ppo_update(
        actor=actor,
        critic=critic,
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        returns=returns,
        advantages=advantages,
        clip_eps=CLIP_EPSILON,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    print("One PPO update done.")
    print(f"  policy_loss: {policy_loss:.4f}")
    print(f"  value_loss:  {value_loss:.4f}")
    print(f"  entropy:     {entropy:.4f}")

    env.close()


    print("states shape:", states.shape)
    print("actions shape:", actions.shape)
    print("log_probs shape:", old_log_probs.shape)
    print("returns shape:", returns.shape)
    print("advantages shape:", advantages.shape)

    env.close()

def train_ppo():
    """
    Full PPO training loop:
    - repeatedly collect rollouts
    - run PPO updates
    - track episode returns
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(render_mode=None)
    obs, info = env.reset(seed=SEED)

    # Flatten obs to 1D
    obs_flat = np.array(obs, dtype=np.float32).flatten()
    state_size = obs_flat.shape[0]
    action_size = env.action_space.n

    # Create networks and optimizers
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size).to(device)

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # Rollout buffer
    rollout = RolloutBuffer(ROLLOUT_STEPS, state_size, device)

    episode_returns = []
    ep_return = 0.0

    for update in range(1, NUM_UPDATES + 1):
        # ----- Collect one rollout -----
        state = obs_flat
        terminated = False
        truncated = False

        while not rollout.is_full():
            state_tensor = torch.from_numpy(state).to(device)

            # Actor + Critic
            action, log_prob = actor.act(state_tensor)
            value = critic.get_value(state_tensor).item()

            # Step env
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_return += reward

            next_flat = np.array(next_obs, dtype=np.float32).flatten()

            rollout.store(
                state_flat=state,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value,
            )

            state = next_flat

            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0

                obs, info = env.reset()
                state = np.array(obs, dtype=np.float32).flatten()
                terminated = truncated = False

        # ----- Compute last value for GAE -----
        last_state_tensor = torch.from_numpy(state).to(device)
        last_value = critic.get_value(last_state_tensor).item()

        states, actions, old_log_probs, returns, advantages = rollout.get(
            last_value, gamma=GAMMA, lam=LAMBDA
        )

        # ----- PPO update -----
        policy_loss, value_loss, entropy = ppo_update(
            actor=actor,
            critic=critic,
            optimizer_actor=optimizer_actor,
            optimizer_critic=optimizer_critic,
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns,
            advantages=advantages,
            clip_eps=CLIP_EPSILON,
            value_coef=VALUE_COEF,
            entropy_coef=ENTROPY_COEF,
            ppo_epochs=PPO_EPOCHS,
            batch_size=BATCH_SIZE,
        )

        # ----- Logging -----
        if episode_returns:
            recent = episode_returns[-10:]
            mean_return = float(np.mean(recent))
        else:
            mean_return = 0.0

        print(
            f"Update {update}/{NUM_UPDATES} | "
            f"Mean return (last {min(len(episode_returns), 10)} eps): {mean_return:.2f} | "
            f"Policy loss: {policy_loss:.3f} | "
            f"Value loss: {value_loss:.3f} | "
            f"Entropy: {entropy:.3f}"
        )

        # Prepare obs_flat for next update loop
        obs, info = env.reset()
        obs_flat = np.array(obs, dtype=np.float32).flatten()

    env.close()

    print("Training finished.")
    if episode_returns:
        print(f"Final mean return (last 10 eps): {np.mean(episode_returns[-10:]):.2f}")


if __name__ == "__main__":
    # Train PPO on Tetris
    train_ppo()

    # For debugging you can still use:
    # test_ppo_update_once()
    # test_rollout_buffer_once()
    # test_actor_critic_once()
    # run_random_agent(num_episodes=3, render=True)
