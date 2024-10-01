import gym
import numpy as np
from gym import spaces
import torch
import torchvision
from alpha_zero.core.simple_judge import get_judge, ConvNet
from enum import Enum
import random

class Player(Enum):
    HONEST = 0
    LIAR = 1

class PixelDebateEnv(gym.Env):
    def __init__(self, max_turns=6, precommit=True, judge: ConvNet = None):
        super(PixelDebateEnv, self).__init__()
        
        # hacky
        self.has_pass_move = False
        self.action_dim = 28 * 28
        self.pass_move = None
        
        self.legal_actions = np.ones(self.action_dim, dtype=np.int8)
        
        self.max_turns = max_turns
        self.precommit = precommit
        
        self.honest_player = Player.HONEST
        self.liar_player = Player.LIAR
        self.current_player = None # will be set in reset()
        self.last_player = None # will be set in reset()
        
        # Action space: 784 possible pixels to reveal
        self.action_space = spaces.Discrete(28 * 28)
        
        # Observation space: 28x28 image, 28x28 mask, current player
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, 28, 28), dtype=np.uint8)
        
        # Load MNIST dataset
        self.mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        
        # Load pre-trained judge model
        self.judge = judge
        
        self.reset()
        
    def is_game_over(self):
        return self.steps >= self.max_turns
    

    
    def reset(self):
        # Select random MNIST image
        self.image_idx = np.random.randint(len(self.mnist))
        self.image, self.true_label = self.mnist[self.image_idx]
        self.image = np.array(self.image)
        
        # Initialize mask
        self.mask = np.zeros((28, 28), dtype=np.uint8)
        
        # Assign labels to players
        self.honest_label = self.true_label
        self.liar_label = np.random.choice([i for i in range(10) if i != self.true_label])
        
        # Randomly assign first player
        self.last_player = None
        self.current_player = random.choice([Player.HONEST, Player.LIAR])
        
        self.steps = 0
        
        return self.observation()
    
    @property
    def to_play(self):
        return self.current_player
    
    @property 
    def opponent_player(self):
        return self.liar_player if self.current_player == self.honest_player else self.honest_player
        
    def step(self, action):
        # Reveal pixel
        row, col = divmod(action, 28)
        self.mask[row, col] = 1
        
        self.steps += 1
        self.last_player = self.current_player
        self.current_player = Player.LIAR if self.current_player == Player.HONEST else Player.HONEST
        
        done = self.steps >= self.max_turns
        reward = (0, 0)  # Default reward during the game
        
        if done:
            winner = self._determine_winner()
            # Reward tuple: (honest_player_reward, liar_reward)
            reward = (1, -1) if winner == 0 else (-1, 1)
        
        return self.observation(), reward, done, {}
    
    def render(self, mode='human'):
        # Implement visualization here
        pass
    
    def observation(self) -> np.ndarray:
        obs = np.zeros((2, 28, 28), dtype=np.uint8)
        obs[0] = self.mask
        obs[1] = self.image
        return obs
    
    def _determine_winner(self):
        # Prepare input for judge
        judge_input = torch.from_numpy(self.observation()).unsqueeze(0)
        
        # Get judge's prediction
        with torch.no_grad():
            logits = self.judge(judge_input.float())
        
        # Compare logits for honest and liar labels
        honest_logit = logits[0, self.honest_label]
        liar_logit = logits[0, self.liar_label]
        
        return self.honest_player if honest_logit > liar_logit else self.liar_player