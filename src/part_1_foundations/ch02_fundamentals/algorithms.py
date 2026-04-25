import numpy as np

def td0(env, episodes=500, alpha=0.1, gamma=0.99):
    """
    Temporal Difference (TD(0)) value estimation.
    Matches Listing 2.2 in "RL: The Seminal Papers".
    """
    V = {s: 0.0 for x in range(env.width) for y in range(env.height) for s in [(x, y)]}

    for _ in range(episodes):
        state = (0, 0)
        done = False
        while not done:
            action = np.random.choice(env.actions)
            next_state, reward, done = env.transition(state, action)
            
            # Bellman expectation update
            td_target = reward + gamma * V[next_state]
            V[state] += alpha * (td_target - V[state])
            
            state = next_state
    return V

def run_agent(env, mode='qlearning', episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    generic runner for Q-Learning and SARSA agents.
    Matches Listing 2.3 in "RL: The Seminal Papers".
    """
    # Initialize Q-values to zero
    Q = {}
    for x in range(env.width):
        for y in range(env.height):
            for a in env.actions:
                Q[((x, y), a)] = 0.0

    def choose_action(state):
        if np.random.random() < epsilon:
            return np.random.choice(env.actions)
        # Greedy choice: handle ties randomly
        q_vals = [Q[(state, a)] for a in env.actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(env.actions, q_vals) if q == max_q]
        return np.random.choice(best_actions)

    cliff_falls = 0
    rewards_history = []

    for _ in range(episodes):
        # Env-specific start logic
        state = getattr(env, 'start', (0, 0))
        action = choose_action(state)
        total_reward = 0
        done = False
        
        while not done:
            # Step transition (standardized 3-tuple return)
            next_state, reward, done = env.transition(state, action)

            total_reward += reward
            if reward == -100: cliff_falls += 1
            
            next_action = choose_action(next_state)
            
            # TD Updates
            if mode == 'qlearning':
                best_next_q = max(Q[(next_state, a)] for a in env.actions)
                target = reward + gamma * best_next_q * (not done)
            else:
                target = reward + gamma * Q[(next_state, next_action)] * (not done)
                
            Q[(state, action)] += alpha * (target - Q[(state, action)])
            
            state, action = next_state, next_action
            
        rewards_history.append(total_reward)
        
    return Q, rewards_history, cliff_falls
