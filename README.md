# Reinforcement-Learning-in-Grid-World-Environment
A custom grid world environment with obstacles and implementations of Q-learning, SARSA, and Expected SARSA reinforcement learning algorithms. Agents learn to navigate the grid, avoiding obstacles and reaching the goal state, with performance visualized through reward plots.

## Introduction
This repository contains a custom gridworld environment implemented in Python using the OpenAI Gym framework. Additionally, it includes implementations of three reinforcement learning algorithms: Q-learning, SARSA, and Expected SARSA. These algorithms are used to train an agent to navigate the gridworld environment to reach a goal while avoiding obstacles.

### Gridworld Environment
The gridworld environment consists of a grid of cells, where each cell can be either empty, contain an obstacle, or represent the goal. The agent starts at a predefined starting position and must navigate through the grid to reach the goal while avoiding obstacles. The environment supports four discrete actions: up, down, left, and right.

The code implements three classic reinforcement learning algorithms:
1. Q-learning
Q-learning is an off-policy TD control algorithm that learns the optimal action-value function without requiring a model of the environment's dynamics.

2. SARSA (State-Action-Reward-State-Action)
SARSA is an on-policy TD control algorithm that learns the optimal action-value function by updating Q-values based on the agent's experience.

3. Expected SARSA
Expected SARSA is an extension of SARSA that computes the expected value of the next state-action pair instead of selecting the greedy action.

### Usage
Install the required dependencies (gym, numpy, matplotlib).
Import the necessary modules and classes from the provided code.
Create an instance of the CustomGridworldEnv class with desired parameters (grid size and number of obstacles).
Train the agent using one of the implemented algorithms by calling the respective function (q_learning, sarsa, expected_sarsa), providing the environment instance and other hyperparameters.
Test the trained agent using the corresponding test function (test_agent_q_learning, test_agent_sarsa, test_agent_expected_sarsa), passing the environment instance and the learned Q-table.
