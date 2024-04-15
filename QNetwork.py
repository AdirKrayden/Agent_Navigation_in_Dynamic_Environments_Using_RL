# Created by Adir Krayden on 02-03-24
# -------------------------------- Imports -------------------------------- #
import torch
import torch.nn as nn


# -------------------------------- QNetwork class -------------------------------- #

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, dropout_prob: float) -> None:
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)
