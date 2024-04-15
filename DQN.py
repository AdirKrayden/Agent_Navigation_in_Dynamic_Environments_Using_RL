# Created by Adir Krayden on 02-03-24
# -------------------------------- Imports -------------------------------- #

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import random
from typing import Dict


# -------------------------------- DQN class -------------------------------- #

class DQN:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 128,
            lr: float = 0.001,
            gamma: float = 0.95,
            epsilon_start: float = 0.9996,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.9999,
            target_update_freq: int = 250,
            memory_capacity: int = 60000,
            batch_size: int = 64,
            save_model_freq=10000
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, dropout_prob=0.2).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim, dropout_prob=0.2).to(self.device)
        self.target_network.load_state_dict(
            self.q_network.state_dict())  # Initializes the target network with the same parameters as the Q-network
        self.target_network.eval()  # Sets the target network to evaluation mode. This is important because the target network is used for inference during training and should not be updated or trained itself.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_capacity)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq  # Sets the frequency at which the target network is updated with the parameters of the Q-network.
        self.batch_size = batch_size
        self.writer = SummaryWriter()
        self.save_model_freq = save_model_freq

    def _select_action(self, state: Dict[str, np.ndarray]) -> int:
        if random.random() < self.epsilon:  # Implements an epsilon-greedy exploration strategy.
            return random.randint(0, self.action_dim - 1)

        # Extract components from the state dictionary
        local_environment = state["local_environment"]
        agent_direction = state["agent_direction"]
        state_array = np.concatenate([local_environment, agent_direction], axis=0)
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(
                state_tensor)  # Passes the preprocessed state tensor through the Q-network to obtain the Q-values for all possible actions.
        return q_values.argmax().item()  # Selects the action with the highest Q-value

    def train(self, env, num_episodes: int, max_steps: int) -> None:
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self._select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.replay_buffer.push((state, action, reward, next_state, done))
                episode_reward += reward
                state = next_state
                if len(self.replay_buffer) > self.batch_size:
                    self.update_network()
                if done:
                    break
            self.writer.add_scalar("Reward", episode_reward, episode)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if episode % self.target_update_freq == 0:
                self.update_target_network()

            if episode % self.save_model_freq == 0:  # Check if it's time to save the model
                self.save_model("model_episode_{}.pt".format(episode))
                print("Saving model: " + str(episode))

    def update_network(self) -> None:
        states_tuple, actions, rewards, next_states_tuple, dones = self.replay_buffer.sample(
            self.batch_size)  # Randomly sample a batch of transitions
        # Extract components from the observation dictionaries
        local_environment = [s["local_environment"] for s in states_tuple]
        agent_direction = [s["agent_direction"] for s in states_tuple]
        next_local_environment = [s["local_environment"] for s in next_states_tuple]
        next_agent_direction = [s["agent_direction"] for s in next_states_tuple]

        # Convert the components into tensors
        local_environment_tensor = torch.FloatTensor(local_environment).to(self.device)
        agent_direction_tensor = torch.FloatTensor(agent_direction).to(self.device)
        next_local_environment_tensor = torch.FloatTensor(next_local_environment).to(self.device)
        next_agent_direction_tensor = torch.FloatTensor(next_agent_direction).to(self.device)

        states_tensor = torch.cat((local_environment_tensor, agent_direction_tensor),
                                  dim=1)  # Concatenate the components to form the complete states
        next_states_tensor = torch.cat((next_local_environment_tensor, next_agent_direction_tensor), dim=1)

        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states_tensor).gather(1,
                                                        actions_tensor)  # Compute Q-values for the given states and actions using the Q-network
        next_q_values = self.target_network(next_states_tensor).max(dim=1, keepdim=True)[
            0].detach()  # Compute the maximum Q-value for each next state using the target network
        target_q_values = rewards_tensor + self.gamma * next_q_values * (~dones_tensor.byte())
        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath: str):
        torch.save(self.q_network.state_dict(), filepath)
