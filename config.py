ENV = "ALE/Tetris-v5"
SEED = 42

GAMMA = 0.99
LAMBDA = 0.95

ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

CLIP_EPSILON = 0.1     # eps for clipping in PPO
PPO_EPOCHS = 4         # how many epochs per update
BATCH_SIZE = 64        # mini-batch size

VALUE_COEF = 0.5       # weight for value loss
ENTROPY_COEF = 0.01    # weight for entropy loss

ROLLOUT_STEPS = 1024   # how many env steps per PPO update
NUM_UPDATES = 100      # total number of PPO updates to perform