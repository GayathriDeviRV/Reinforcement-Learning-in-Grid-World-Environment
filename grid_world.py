import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML


class CustomGridworldEnv(gym.Env):
    def __init__(self, grid_size=5, num_obstacles=5):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.obstacles = self._generate_obstacles()
        self.agent_pos = self.start
        # 4 discrete actions: up, down, left, right
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(grid_size),
            gym.spaces.Discrete(grid_size)
        ))

    def _generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < self.num_obstacles:
            obstacle = (np.random.randint(0, self.grid_size),
                        np.random.randint(0, self.grid_size))
            if obstacle != self.start and obstacle != self.goal:
                obstacles.add(obstacle)
        return obstacles

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size - 1, y + 1)

        if (x, y) in self.obstacles:
            reward = -1
        elif (x, y) == self.goal:
            reward = 10
        else:
            reward = -0.1

        self.agent_pos = (x, y)
        done = (x, y) == self.goal
        return self.agent_pos, reward, done, {}

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for obstacle in self.obstacles:
            x, y = obstacle
            grid[x, y] = -1
        x, y = self.agent_pos
        grid[x, y] = 1
        plt.imshow(grid, cmap='viridis', origin='lower',
                   extent=[-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5])
        plt.xticks(np.arange(-0.5, self.grid_size, 1))
        plt.yticks(np.arange(-0.5, self.grid_size, 1))
        plt.grid(color='w', linewidth=2)
        plt.draw()
        plt.pause(0.5)


# Q-learning algorithm
def q_learning(env, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state[0], state[1]])

            next_state, reward, done, _ = env.step(action)

            # Q-learning update rule
            best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
            q_table[state[0], state[1], action] += learning_rate * \
                (reward + discount_factor * q_table[next_state[0], next_state[1],
                                                    best_next_action] - q_table[state[0], state[1], action])

            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(
                "Q-learning Episode {}: Total Reward = {}".format(episode + 1, total_reward))

    # Plot rewards per episode
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-learning: Rewards per Episode')
    plt.show()

    return q_table


# SARSA algorithm
def sarsa(env, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        action = None
        done = False
        while not done:
            if action is not None:
                next_state, reward, done, _ = env.step(action)
                next_action = None
                if np.random.rand() < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(
                        q_table[next_state[0], next_state[1]])
                q_table[state[0], state[1], action] += learning_rate * \
                    (reward + discount_factor * q_table[next_state[0], next_state[1], next_action] -
                     q_table[state[0], state[1], action])
                total_reward += reward
                state = next_state
                action = next_action
            else:
                action = np.random.choice(env.action_space.n)
        rewards_per_episode.append(total_reward)
        if (episode + 1) % 100 == 0:
            print("SARSA Episode {}: Total Reward = {}".format(
                episode + 1, total_reward))

    # Plot rewards per episode
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SARSA: Rewards per Episode')
    plt.show()

    return q_table


# Expected SARSA algorithm
def expected_sarsa(env, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = None
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state[0], state[1]])

            next_state, reward, done, _ = env.step(action)

            # Calculate expected value of next state-action pair
            next_action_values = q_table[next_state[0], next_state[1]]
            next_action_probabilities = np.ones(
                env.action_space.n) * epsilon / env.action_space.n
            best_next_action = np.argmax(next_action_values)
            next_action_probabilities[best_next_action] += 1.0 - epsilon
            expected_next_value = np.sum(
                next_action_probabilities * next_action_values)

            # Expected SARSA update rule
            q_table[state[0], state[1], action] += learning_rate * \
                (reward + discount_factor * expected_next_value -
                 q_table[state[0], state[1], action])

            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)
        if (episode + 1) % 100 == 0:
            print("Expected SARSA Episode {}: Total Reward = {}".format(
                episode + 1, total_reward))

    # Plot rewards per episode
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Expected SARSA: Rewards per Episode')
    plt.show()

    return q_table


# Function to test the trained agent(SARSA)
def test_agent_sarsa(env, q_table):
    state = env.reset()
    env.render()
    done = False
    action = None
    while not done:
        if action is not None:
            next_state, _, done, _ = env.step(action)
            action = np.argmax(q_table[next_state[0], next_state[1]])
            state = next_state
            env.render()
        else:
            action = np.argmax(q_table[state[0], state[1]])
    env.render()

# Function to test the trained agent(Q-learning)
def test_agent_q_learning(env, q_table):
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = np.argmax(q_table[state[0], state[1]])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()

# Function to test the trained agent(Expected SARSA)
def test_agent_expected_sarsa(env, q_table):
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = np.argmax(q_table[state[0], state[1]])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()


# Create custom gridworld environment
env = CustomGridworldEnv(grid_size=5, num_obstacles=5)

# Train the agent using Q-learning
q_table_q_learning = q_learning(
    env, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

# Train the agent using SARSA
q_table_sarsa = sarsa(env, num_episodes=1000,
                      learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

# Train the agent using Expected SARSA
q_table_expected_sarsa = expected_sarsa(
    env, num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

# Test the trained agent using Q-learning
test_agent_q_learning(env, q_table_q_learning)

# Test the trained agent using SARSA
test_agent_sarsa(env, q_table_sarsa)

# Test the trained agent using Expected SARSA
test_agent_expected_sarsa(env, q_table_expected_sarsa)
