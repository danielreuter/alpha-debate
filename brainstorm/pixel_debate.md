# MNIST Debate Environment Specification

## Overview

This document outlines the implementation of an OpenAI Gym environment for the MNIST debate game described in the paper. The game involves two agents debating over the classification of an MNIST digit by revealing pixels to a pre-trained judge.

## Environment Structure

### Observation Space

The observation space consists of three main components:

1. **MNIST Image**: A 28x28 grayscale image represented as a numpy array.
2. **Revealed Pixels Mask**: A 28x28 binary mask indicating which pixels have been revealed thus far by agents.
3. **Current Player**: A binary indicator (0 or 1) of which player's turn it is.

### Action Space

The action space is discrete, representing the index of the pixel to reveal. It ranges from 0 to 783 (28*28 - 1).

### State

The environment maintains the following state information:

- Full MNIST image
- Revealed pixels mask
- Current player
- Precommitted labels for both players (a digit class that they are assigned to argue for)
- Number of turns taken
- Maximum number of turns allowed

## Environment Dynamics

### Initialization

1. Randomly select an MNIST image from the test set.
2. Initialize the revealed pixels mask with all zeros.
3. Create two players: an honest one and a liar
4. Assign the honest one the true label
5. Assign the liar a randomly selected false label
6. Randomly assign a player to go first
7. Set the turn counter to 0.

### Step Function

The step function should:

1. Take an action (pixel index to reveal) from the current player.
2. Update the revealed pixels mask.
3. Increment the turn counter.
4. Switch the current player.
5. Check for game termination.
6. Return the new observation, reward, done flag, and info dictionary.

### Termination Conditions

The game terminates when the maximum number of turns is reached (6 for the 6-pixel game, 4 for the 4-pixel game).

### Reward Structure

Rewards are sparse and only given at the end of the game:

- +1 for the winning player
- -1 for the losing player

The winner is determined by passing the final revealed pixels to the pre-trained judge and comparing the logits for the two precommitted labels.

## Judge Integration

The pre-trained judge should be integrated as follows:

1. Load the judge model at environment initialization.
2. At the end of the game, prepare the input for the judge:
   - Create a 2-channel 28x28 input:
     - Channel 1: The revealed pixels mask (binary)
     - Channel 2: The original image pixel values, scaled to [0, 1]
   - Note: Unrevealed pixels in channel 2 should retain their original (scaled) values, not set to 0
3. Pass this input through the judge model.
4. Compare the logits for the two precommitted labels to determine the winner.

### Data Normalization

The judge expects input data to be normalized as follows:

- The input is a 2-channel 28x28 tensor.
- The first channel is a binary mask (0 or 1) indicating which pixels are revealed.
- The second channel contains the actual pixel values from the original image, scaled to the range [0, 1].
- Importantly, unrevealed pixels in the second channel retain their original (scaled) values.
- It will apply the mask itself (i.e. it will only know about the revealed pixels at inference-time)
- No additional normalization is applied to the pixel values.

## Additional Considerations

1. **Precommitment**: Implement both precommit and non-precommit versions of the game as separate environment variants.
2. **Rendering**: Implement a render function to visualize the game state, showing the partially revealed image and precommitted labels.

## Implementation Steps

1. Define the `PixelDebateEnv` class, inheriting from `gym.Env`.
2. Implement the `__init__`, `reset`, `step`, and `render` methods.
3. Define the observation and action spaces using `gym.spaces`.
4. Load and integrate the pre-trained judge model (ConvNet).
5. Implement utility functions for pixel revealing and winner determination.
6. Add options for different game variants (4-pixel vs 6-pixel, precommit vs non-precommit).

## Judge Model Details

The judge is a ConvNet with the following architecture:

- Input: 2 channels, 28x28 images
- Convolutional layer: 32 output channels, 3x3 kernel, padding=1
- ReLU activation
- Fully connected layer: 28 * 28 * 32 -> 128
- ReLU activation
- Output layer: 128 -> 10 (number of classes)

The judge was trained using the following hyperparameters:

- Batch size: 128
- Learning rate: 1e-4
- Number of batches: 30,000 for 6-pixel model, 50,000 for 4-pixel model
- Optimizer: Adam

When implementing the environment, ensure that the input to the judge model matches the format and normalization used during training.

This environment structure will allow for easy integration with existing RL algorithms and provide a flexible framework for experimenting with different debate strategies.