import numpy as np  
from time import sleep

def estimate(OldEstimate, StepSize, Target):
    NewEstimate = OldEstimate + StepSize * (Target - OldEstimate)
    return NewEstimate

def epsilon_greedy(value, e, seed = None):
    '''
    Implement Epsilon-Greedy policy
    
    Inputs:
    value: numpy ndarray
            A vector of values of actions to choose from
    e: float
            Epsilon
    seed: None or int
            Assign an integer value to remove the randomness
    
    Outputs:
    action: int
            Index of the chosen action
    '''
    
    assert len(value.shape) == 1
    assert 0 <= e <= 1
    
    if seed != None:
        np.random.seed(seed)
    
    ############################
    # YOUR CODE STARTS HERE
    probs = (e/len(value))*np.ones(len(value))
    max_idx = np.argmax(value)
    probs[max_idx] += (1-e)
    action_val = np.random.choice(value,size=1,p=probs)
    action = np.where(value == action_val)[0]
    if isinstance(action,np.ndarray):
        action = np.random.choice(action)
    if isinstance(action,np.ndarray):
        print('action is np.ndarray; action =',action)
    # YOUR CODE ENDS HERE
    ############################
    return action

def action_evaluation(env, gamma, v):
    '''
    
    Inputs:
    env: OpenAI Gym environment
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount value
    v: numpy ndarray
            Values of states
    Outputs:
    q: numpy ndarray
            Q values of all state-action pairs
    '''
    
    nS = env.nS
    nA = env.nA
    q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            ############################
            # YOUR CODE STARTS HERE
           	for vals in env.P[s][a]:
           	    prob = vals[0]
           	    next_state =  vals[1]
           	    reward = vals[2]
           	    q[s,a] += prob*(reward+v[next_state])
            # YOUR CODE ENDS HERE
            ############################
    return q

def action_selection(q):
    '''
    Select action from the Q value
    
    Inputs:
    q: numpy ndarray
    
    Outputs:
    actions: int
            The chosen action of each state
    '''
    
    actions = np.argmax(q, axis = 1)    
    return actions 

def render(env, policy):
    '''
    Play games with the given policy
    
    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    policy: numpy ndarray
            Maps state to action
    '''
    
    state = env.reset()
    terminal = False
    
    while not terminal:
        action = policy[state]
        state, reward, terminal, prob = env.step(action)
        env.render()
        sleep(1)
    
    print('Episode ends. Reward =', reward)
    
def human_play(env):
    '''
    Play games
    
    Inputs:
    env: OpenAI Gym environment
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    '''
    
    print('Action indices: LEFT=0, DOWN = 1, RIGHT = 2, UP = 3')
    state = env.reset()
    env.render()
    terminal = False
    
    while not terminal:
        action = int(input('Give the environment your action index:'))
        state, reward, terminal, prob = env.step(action)
        env.render()