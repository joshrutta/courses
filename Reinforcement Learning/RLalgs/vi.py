import numpy as np
from RLalgs.utils import action_evaluation

def value_iteration(env, gamma, max_iteration, theta):
    """
    Implement value iteration algorithm. 

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
            Number of iterations
    """
    nS = env.nS
    V = np.zeros(env.nS)
    numIterations = 0

    #Implement the loop part here
    ############################
    # YOUR CODE STARTS HERE
    while True:
        v = list(V)
        delta = 0
        for s in range(nS):
            for a in range(env.nA):
                vs = 0
                for vals in env.P[s][a]:
                    prob = vals[0]
                    next_state = vals[1]
                    reward = vals[2]
                    vs += prob*(reward+gamma*v[next_state])
                if vs > V[s]:
                    V[s] = vs
            delta = max(delta,abs(V[s]-v[s]))
        if delta <= theta:
            break
    # YOUR CODE ENDS HERE
    ############################
    
    #Extract the "optimal" policy from the value function
    policy = extract_policy(env, V, gamma)
    
    return V, policy, numIterations

def extract_policy(env, v, gamma):

    """ 
    Extract the optimal policy given the optimal value-function

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    v: numpy.ndarray
        value function
    gamma: float
        Discount factor. Number in range [0, 1)
    
    Outputs:
    policy: numpy.ndarray
    """

    policy = np.zeros(env.nS, dtype = np.int32)
    ############################
    # YOUR CODE STARTS HERE
    for s in range(env.nS):
        maxa = -1
        vmax = -1
        for a in range(env.nA):
            vs = 0
            for vals in env.P[s][a]:
                prob = vals[0]
                next_state = vals[1]
                reward = vals[2]
                vs += prob*(reward + gamma*v[next_state])
                if vs > vmax:
                    # print('vs =',vs,'vmax =',vmax)
                    maxa = a
                    vmax = vs
        # print('maxa = ',maxa,'; vmax =',vmax)
        policy[s] = maxa

    # YOUR CODE ENDS HERE
    ############################

    return policy