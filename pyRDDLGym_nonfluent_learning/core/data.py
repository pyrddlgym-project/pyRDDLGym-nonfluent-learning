import numpy as np
import gymnasium as gym
from typing import Callable, Dict, Generator, Tuple

State = Dict[str, np.ndarray]
Action = Dict[str, np.ndarray]
Transition = Tuple[State, Action, State]


def generate_rollouts(env: gym.Env, policy: Callable, episodes: int, max_steps: int
                      ) -> Transition:
    '''Generates rollouts from the given environment and policy.
    
    :param env: the environment to generate rollouts from
    :param policy: a function that takes in an observation and returns an action
    :param episodes: how many episodes to generate
    :param max_steps: the maximum number of steps per episode
    '''
    model = env.model
    states = {k: [] for k in model.state_fluents}
    actions = {k: [] for k in model.action_fluents}
    next_states = {k: [] for k in model.state_fluents}

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = policy(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            for key in states:
                states[key].append(obs[key])
                next_states[key].append(next_obs[key])
            for key in actions:
                actions[key].append(action[key])
            obs = next_obs
            done = term or trunc
            steps += 1
    
    # reshape the data into arrays of shape (num_transitions, fluent_size)
    state_shapes = {k: (-1, *np.shape(v)) for k, v in model.state_fluents.items()}
    action_shapes = {k: (-1, *np.shape(v)) for k, v in model.action_fluents.items()}
    states = {k: np.reshape(v, state_shapes[k]) for k, v in states.items()}
    actions = {k: np.reshape(v, action_shapes[k]) for k, v in actions.items()}
    next_states = {k: np.reshape(v, state_shapes[k]) for k, v in next_states.items()}
    return states, actions, next_states


def batch_sampler(states: State, actions: Action, next_states: State, batch_size: int=32,
                  max_iters: int=99999999) -> Generator[Transition, None, None]:
    '''Yields batches of transitions from the given states, actions, and next_states.
    
    :param states: a dictionary mapping state fluent names to numpy arrays
    :param actions: a dictionary mapping action fluent names to numpy arrays
    :param next_states: a dictionary mapping state fluent names to numpy arrays
    :param batch_size: how many transitions to include in each batch
    :param max_iters: the maximum number of batches to yield
    '''
    if not (len(states) == len(actions) == len(next_states)):
        raise ValueError('States, actions, and next_states must have the same keys')
    num_transitions = len(next(iter(states.values())))
    if num_transitions < batch_size:
        raise ValueError('Batch size must be less than the number of transitions')
    
    indices = np.arange(num_transitions)
    for _ in range(max_iters):
        np.random.shuffle(indices)
        for start in range(0, num_transitions - batch_size, batch_size):
            end = start + batch_size
            batch_id = indices[start:end]
            batch_states = {k: v[batch_id] for k, v in states.items()}
            batch_actions = {k: v[batch_id] for k, v in actions.items()}
            batch_next = {k: v[batch_id] for k, v in next_states.items()}
            yield batch_states, batch_actions, batch_next
