import torch
import torch.nn.functional as F

def ppo_update(
    actor,
    critic,
    optimizer_actor,
    optimizer_critic,
    states,
    actions,
    old_log_probs,
    returns,
    advantages,
    clip_eps=0.1,
    value_coef=0.5,
    entropy_coef=0.01,
    ppo_epochs=4,
    batch_size=64,
):
    device = next(actor.parameters()).device

    # make sure everything is on the same device
    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)

    N = states.size(0)

    for epoch in range(ppo_epochs):
        # shuffle indices for te mini batches
        indices = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]

            # ACTOR: get new log_probs + entropy
            dist = actor(mb_states)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # CRITIC: value loss
            values = critic(mb_states).squeeze(-1)

            # ppo ratio
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            # clipped ratio
            unclipped_ratio = ratio * mb_advantages
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(unclipped_ratio, clipped_ratio).mean()

            # MSE
            value_loss = F.mse_loss(values, mb_returns)

            # total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()
            
    return float(policy_loss.item()), float(value_loss.item()), float(entropy.item())