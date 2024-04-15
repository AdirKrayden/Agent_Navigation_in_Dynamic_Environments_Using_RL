# Created by Adir Krayden on 02-03-24
# -------------------------------- Imports -------------------------------- #

import numpy as np
from collections import deque
from typing import List, Tuple


# -------------------------------- ReplayBuffer class -------------------------------- #

class ReplayBuffer:
    """
    store and sample (past) experiences: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Tuple[np.ndarray, int, int, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[List[np.ndarray], List[int], List[int], List[np.ndarray], List[bool]]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
