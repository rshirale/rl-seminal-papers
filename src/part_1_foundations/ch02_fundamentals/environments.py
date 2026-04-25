import numpy as np

class GridWorld:
    """
    Standard 4x3 Grid World MDP used for TD(0) and value estimation.
    Matches Listing 2.1 in "RL: The Seminal Papers".
    """
    def __init__(self, width=4, height=3):
        self.width = width
        self.height = height
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_map = {
            'Up': (0, 1), 'Down': (0, -1),
            'Right': (1, 0), 'Left': (-1, 0)
        }
        # Terminals: (3,2)=Goal (+10), (1,1) & (2,1)=Hazards (-10)
        self.terminals = {(3, 2): 10, (1, 1): -10, (2, 1): -10}
        self.walls = []

    def transition(self, state, action):
        if state in self.terminals:
            return state, 0, True

        dx, dy = self.action_map[action]
        next_state = (state[0] + dx, state[1] + dy)

        # Boundary and wall logic
        if (next_state[0] < 0 or next_state[0] >= self.width or
            next_state[1] < 0 or next_state[1] >= self.height or
            next_state in self.walls):
            next_state = state

        # Reward Logic: -1 step penalty to encourage speed
        reward = self.terminals.get(next_state, -1)
        done = next_state in self.terminals
        return next_state, reward, done

class CliffWalking:
    """
    12x4 Cliff Walking environment used to compare Q-Learning and SARSA.
    Introduced by Sutton & Barto (1998).
    """
    def __init__(self, width=12, height=4):
        self.width = width
        self.height = height
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_map = {
            'Up': (0, 1), 'Down': (0, -1),
            'Right': (1, 0), 'Left': (-1, 0)
        }
        self.start = (0, 0)
        self.goal = (11, 0)
        # Cliff is the bottom row excluding start and goal
        self.cliff = [(x, 0) for x in range(1, 11)]

    def transition(self, state, action):
        if state == self.goal:
            return state, 0, True
            
        dx, dy = self.action_map[action]
        next_state = (state[0] + dx, state[1] + dy)
        
        # Boundary logic
        if (next_state[0] < 0 or next_state[0] >= self.width or
            next_state[1] < 0 or next_state[1] >= self.height):
            next_state = state

        # Cliff logic: reset to start with -100 penalty
        if next_state in self.cliff:
            return self.start, -100, False
            
        # Goal logic: terminate with -1 penalty (standard)
        if next_state == self.goal:
            return next_state, -1, True

        return next_state, -1, False
