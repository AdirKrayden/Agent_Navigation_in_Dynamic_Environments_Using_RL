# Created by Adir Krayden on feb'24

## -------------------------------- Imports -------------------------------- ##
import gymnasium as gym
from gym import spaces
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from collections import deque
from time import sleep

from stable_baselines3.common.env_checker import check_env  # for debug - check if the environment works with baseline3

## -------------------------------- Environment class -------------------------------- ##
class Environment(gym.Env):
    """
    <fill later>
    """

    # Inheriting the methods and properties from gem class Env
    def __init__(self, matrix_shape=(10, 10), obs_interval=2, partial_obs=False, partial_act=False, num_agents=1,
                 num_prizes=1, reward_setup='reg', orientation=0):
        # first check that matrix_shape input
        self.matrix_shape = matrix_shape
        self.num_agents = num_agents
        self.obs_interval = obs_interval
        self.matrix = np.zeros(self.matrix_shape, dtype=np.int8)
        self.agent_pos = None
        self.num_prizes = num_prizes
        self.prev_distance = None
        while not self.is_connected():  # check if there is a path from each prize to at least one agent
            print("Trying to get a valid environment, please wait")
            self.matrix = np.zeros(self.matrix_shape, dtype=np.int8)
            self.initialize_environment()
        print("Environment is valid!, continuing...")
        self.orientation = orientation
        self.partial_act = partial_act
        if self.partial_act:  # move forward / change position
            self.action_space = Discrete(2)
        else:  # agent may move to either direction
            self.action_space = Discrete(4)  # up/down/left/right
        self.max_steps = 100
        self.current_step = 0
        self.partial_obs = partial_obs
        self.reward_setup = reward_setup
        if not self.partial_obs:
            self.observation_space = Dict({
                "agent_direction": Box(low=0, high=1, shape=(4,), dtype=np.int8),  # One-hot encoded agent direction vec
                "local_environment": Box(low=0, high=3,
                                         shape=((2 * self.obs_interval + 1) * (2 * self.obs_interval + 1),),
                                         dtype=np.int8)
            })
        else:  # less data (observations) - give only local environment
            self.observation_space = Dict({
                "local_environment": Box(low=0, high=3,
                                         shape=((2 * self.obs_interval + 1) * (2 * self.obs_interval + 1),),
                                         dtype=np.int8)
            })

    def initialize_environment(self):
        """
        Initialize the environment>
        0 indicates a free space.
        1 represents walls or boundaries.
        2 denotes obstacles.
        3 signifies the location of a prize or treasure.
        4 indicates the current position of the agent.

        :return:
        """
        # Set walls (boundaries) to 1
        self.matrix[0, :] = 1
        self.matrix[-1, :] = 1
        self.matrix[:, 0] = 1
        self.matrix[:, -1] = 1

        # Set walls (boundaries) to 1
        self.matrix[:2, :] = 1
        self.matrix[-2:, :] = 1
        self.matrix[:, :2] = 1
        self.matrix[:, -2:] = 1

        if self.obs_interval == 3:  # More walls (for complete FOV)
            # Set walls (boundaries) to 1
            self.matrix[:3, :] = 1
            self.matrix[-3:, :] = 1
            self.matrix[:, :3] = 1
            self.matrix[:, -3:] = 1

        percentage_obs = 0.12
        # Set the obstacles randomly
        num_obs_tot = ceil(percentage_obs * np.prod(self.matrix.shape))
        remaining_obs_num = num_obs_tot - (self.matrix.shape[0] - 2)
        if remaining_obs_num > 0:
            other_obstacles_pos = self.get_random_positions(remaining_obs_num)
            self.matrix[other_obstacles_pos[:, 0], other_obstacles_pos[:, 1]] = 2

        # Set treasure/reward position (cell number three)
        treasure_pos = self.get_random_positions(self.num_prizes)
        self.matrix[treasure_pos[:, 0], treasure_pos[:, 1]] = 3

        # Set agent position (cell number four)
        agent_pos = self.get_random_positions(self.num_agents)
        self.matrix[agent_pos[:, 0], agent_pos[:, 1]] = 4
        self.agent_pos = agent_pos[:, 0], agent_pos[:, 1]

        prize_pos = np.argwhere(self.matrix == 3)  # might be more than one prize
        self.prev_distance = abs(prize_pos[0][0] - self.agent_pos[0]) + abs(
            prize_pos[0][1] - self.agent_pos[1])

    def get_random_positions(self, num_positions):
        """
        A helper function to get random positions that are not already occupied
        :param num_positions:
        :return: a 2D ndarray with shape: <num_positions,2>  of randomly positions in the env's matrix
        """
        available_positions = np.argwhere(self.matrix == 0)  # returns an array with the free indices such as [0,2]
        selected_indices = np.random.choice(available_positions.shape[0], size=num_positions, replace=False)
        return available_positions[selected_indices]

    def get_obs(self):
        """
        Observe the current state of the environment (for the agent).
        """
        local_surroundings = self._get_local_surroundings()
        if self.partial_obs:
            return {"local_environment": local_surroundings}
        else:  # more data to observation space
            agent_direction_vec = self._get_agent_direction()
            return {
                "agent_direction": agent_direction_vec,
                "local_environment": local_surroundings
            }

    def _get_agent_direction(self):
        agent_pos = self.agent_pos
        prize_pos = np.array(np.where(self.matrix == 3))
        # Calculate directional information relative to the agent
        relative_direction = np.sign(prize_pos - agent_pos)
        # Encode the directional information into a one-hot vector
        directional_vector = np.zeros(4, dtype=np.int8)  # Four possible directions: up, down, left, right
        if relative_direction[0] == -1:
            directional_vector[0] = 1  # Up
        if relative_direction[0] == 1:
            directional_vector[1] = 1  # Down
        if relative_direction[1] == -1:
            directional_vector[2] = 1  # Left
        if relative_direction[1] == 1:
            directional_vector[3] = 1  # Right
        return directional_vector

    def _get_local_surroundings(self):
        """
        For debugging / visualization: remove flatten() + call self.render(local_surroundings)
        :return:
        """
        x, y = self.agent_pos
        x = x[0]
        y = y[0]
        x_min = max(0, x - self.obs_interval)
        x_max = min(self.matrix_shape[0] - 1, x + self.obs_interval)
        y_min = max(0, y - self.obs_interval)
        y_max = min(self.matrix_shape[1] - 1, y + self.obs_interval)
        # Extract the local surroundings from the matrix and encode obstacles and prize
        local_surroundings = self.matrix[x_min:x_max + 1, y_min:y_max + 1].copy()
        local_surroundings[(local_surroundings == 1) | (local_surroundings == 2)] = 1  # Encode walls and obstacles as 1
        local_surroundings[local_surroundings == 3] = 2  # Encode prize as 2
        local_surroundings[local_surroundings == 4] = 3  # Encode agent as 3
        # self.render(local_surroundings) # for DEBUG
        local_surroundings = local_surroundings.flatten()
        return local_surroundings

    def _get_info(self):
        prize_pos = np.where(self.matrix == 3)  # Assuming only one prize position
        current_distance = abs(prize_pos[0] - self.agent_pos[0]) + abs(prize_pos[1] - self.agent_pos[1])
        return {"distance": current_distance}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the environment to initial, random state
        self.matrix = np.zeros_like(self.matrix, dtype=np.int8)
        self.agent_pos = None
        self.prev_distance = None
        self.initialize_environment()
        self.max_steps = 100
        self.current_step = 0
        info = self._get_info()
        return self.get_obs(), info

    @staticmethod
    def _check_if_valid_action(content):
        if content == 1 or content == 2:  # wall / obstacle
            return False
        else:
            return True  # action is valid

    def step(self, action):
        """
        Take an action and return the next state and reward
        """
        terminated = False  # default
        truncated = False  # default
        if self.reward_setup == 'reg':
            reward_penalty = 0  # not closer/further
            reward_closer = +1
            reward_further = -1
            reward_obstacle = -3  # penalty for trying moving over an obs (cell's type - 1/2)
            reward_prize = +5  # for agent success - gets prize and the episode is done

        elif self.reward_setup == 'sparse':
            reward_penalty = 0  # not closer/further
            reward_closer = 0
            reward_further = 0
            reward_obstacle = -1  # penalty for trying moving over an obs (cell's type - 1/2)
            reward_prize = +10  # for agent success -  gets prize and the episode is done

        else:
            raise ValueError("Error: Invalid prize setup - currently supports only 'reg' and 'spare'")

        # Apply action
        new_position_x, new_position_y = self._get_new_position(action)
        content = self.matrix[new_position_x, new_position_y]  # not self.agent_pos because it might be invalid
        # Check if action is valid, if it is: update matrix (environment) + agent_pos (self)
        if self._check_if_valid_action(content):
            self._update_matrix(new_position_x,
                                new_position_y)  # first update matrix - so we have both old (self.agent_pos) and new
            self.agent_pos = new_position_x, new_position_y
            # Calculate Manhattan distance between agent and prize
            prize_pos = np.where(self.matrix == 3)  # Assuming only one prize position
            current_distance = abs(prize_pos[0] - self.agent_pos[0]) + abs(prize_pos[1] - self.agent_pos[1])

        # calculate reward
        if content == 0:  # an empty cell
            if current_distance < self.prev_distance:
                reward = reward_closer
                self.prev_distance = current_distance
            elif current_distance > self.prev_distance:
                reward = reward_further
                self.prev_distance = current_distance
            else:
                reward = reward_penalty
                self.prev_distance = current_distance
        elif content == 1 or content == 2:
            reward = reward_obstacle
        elif content == 3:
            reward = reward_prize
            terminated = True
        elif content == 4 and self.partial_act:
            reward = reward_penalty
        else:
            raise ValueError("Invalid content at agent's position within the matrix")

        # Increment step counter
        self.current_step += 1

        # Check if maximum steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        info = self._get_info()
        obs = self.get_obs()
        reward = int(reward)
        return obs, reward, terminated, truncated, info

    def _update_matrix(self, new_x_pos, new_y_pos):
        curr_x_pos, curr_y_pos = self.agent_pos
        self.matrix[curr_x_pos, curr_y_pos] = 0  # empty cell, agent left
        self.matrix[new_x_pos, new_y_pos] = 4  # set new agent position

    def _get_new_position(self, action):
        x, y = self.agent_pos
        if not self.partial_act:
            if action == 0:  # up
                return x - 1, y
            elif action == 1:  # down
                return x + 1, y
            elif action == 2:  # left
                return x, y - 1
            elif action == 3:  # right
                return x, y + 1
            else:
                raise ValueError("Invalid action's value")
        else:  # needs to check orientation
            if action == 0:  # Move forward
                if self.orientation == 0:  # If facing up
                    return x - 1, y
                elif self.orientation == 1:  # If facing right
                    return x, y + 1
                elif self.orientation == 2:  # If facing down
                    return x + 1, y
                elif self.orientation == 3:  # If facing left
                    return x, y - 1
            elif action == 1:  # Turn right (change orientation)
                self.orientation = (self.orientation + 1) % 4  # Update orientation
                return x, y  # Agent stays in the same position
            else:
                raise ValueError("Invalid action's value")

    def render(self, matrix=None, agent_pos=None, agent_orientation=None):
        # Visualization
        if matrix is None:
            matrix = self.matrix
        num_levels = len(np.unique(matrix))  # Calculate the number of unique levels
        levels = np.linspace(0, num_levels - 1, num_levels)  # Generate discrete levels
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(boundaries=levels)  # Specify discrete levels for the color bar

        if self.partial_act:
            if self.agent_pos is not None and self.orientation is not None:
                x, y = self.agent_pos

                dx, dy = 0, 0  # Initialize arrow direction
                if self.orientation == 0:  # Up
                    dx = -0.5
                elif self.orientation == 1:  # Right
                    dy = 0.5
                elif self.orientation == 2:  # Down
                    dx = 0.5
                elif self.orientation == 3:  # Left
                    dy = -0.5

                x = int(x.item())
                y = int(y.item())
                plt.arrow(y, x, dy, dx, color='red', head_width=0.2)
                plt.scatter(y, x, color='red', s=100, marker='o')


        plt.show()

    def is_connected(self):
        rows, cols = self.matrix_shape
        visited = np.zeros_like(self.matrix, dtype=bool)

        def bfs(start_row, start_col):
            queue = deque([(start_row, start_col)])
            while queue:
                row, col = queue.popleft()
                if self.matrix[row, col] == 4:
                    return True
                visited[row, col] = True
                # Check neighboring cells
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row, new_col] and self.matrix[
                        new_row, new_col] != 1 and self.matrix[new_row, new_col] != 2:
                        queue.append((new_row, new_col))
            return False

        # Find all cells with value 3
        cells_with_3 = np.argwhere(self.matrix == 3)
        for row, col in cells_with_3:
            if bfs(row, col):
                return True
        return False


## -------------------------------- Debug -------------------------------- ##

if __name__ == '__main__':
    env = Environment(num_agents=1, num_prizes=1, partial_act=True, matrix_shape=(15, 15))
    env.render()
    sleep(1)
    obs, reward, terminated, truncated, info = env.step(0)
    env.render()
    sleep(1)
    obs, reward, terminated, truncated, info = env.step(0)
    env.render()
    sleep(1)
    obs, reward, terminated, truncated, info = env.step(1)
    env.render()
    sleep(1)
    obs, reward, terminated, truncated, info = env.step(0)
    env.render()
    sleep(1)
    # env._get_local_surroundings()

