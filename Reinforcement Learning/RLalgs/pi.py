import numpy as np
from RLalgs.utils import action_evaluation

def policy_iteration(env, gamma, max_iteration, theta):
    """
    Implement Policy iteration algorithm.

    Inputs:
    env: OpenAI Gym environment.
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
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
            
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
    """

    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype = np.int32)
    policy_stable = False
    numIterations = 0
    old_policy = policy
    while not policy_stable and numIterations < max_iteration:
        #Implement it with function policy_evaluation and policy_improvement
        ############################
        # YOUR CODE STARTS HERE
        V = policy_evaluation(env,policy,gamma,theta)
        policy,policy_stable = policy_improvement(env,V,policy,gamma)
        # YOUR CODE ENDS HERE
        ############################
        numIterations += 1
        
    return V, policy, numIterations


def policy_evaluation(env, policy, gamma, theta):
    """
    Evaluate the value function from a given policy.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    gamma: float
            Discount factor.
    policy: numpy.ndarray
            The policy to evaluate. Maps states to actions.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
            The value function from the given policy.
    """
    ############################
    # YOUR CODE STARTS HERE
    nS = env.nS
    nA = env.nA
    V = np.zeros(nS)
    count = 0
    while True:
        count +=1
        delta = 0
        v = list(V)
        V = np.zeros(nS)
        for s in range(nS):
            for vals in env.P[s][policy[s]]:
                prob = vals[0]
                next_state =  vals[1]
                #only reward is if you're in terminal state
                reward = vals[2]
                V[s] += prob*(reward+gamma*v[next_state])
                # print("V[s] = ",V[s])
                # print("v[s] = ",)
            delta = max(delta,abs(v[s]-V[s]))
        if delta<=theta:
            break
        # print("v =\n",v[:4],"\n",v[4:8],"\n",v[8:12],"\n",v[12:])
        # print("V =\n",V[:4],"\n",V[4:8],"\n",V[8:12],"\n",V[12:])
        # print('count = ',count,"delta = ",delta)
    # # YOUR CODE ENDS HERE
    # ############################
    return V


def policy_improvement(env, value_from_policy, policy, gamma):
    """
    Given the value function from policy, improve the policy.

    Inputs:
    env: OpenAI Gym environment
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

    value_from_policy: numpy.ndarray
            The value calculated from the policy
    policy: numpy.ndarray
            The previous policy.
    gamma: float
            Discount factor.

    Outputs:
    new policy: numpy.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    policy_stable: boolean
            True if the "optimal" policy is found, otherwise false
    """
    ############################
    # YOUR CODE STARTS HERE
    old_policy = list(policy)
    for s in range(env.nS):
        maxv = 0
        for a in range(env.nA):
            vs = 0
            for vals in env.P[s][a]:
                prob = vals[0]
                next_state = vals[1]
                reward = vals[2]
                vs += prob*(reward+gamma*value_from_policy[next_state])
            if vs > maxv:
                maxv = vs
                policy[s] = a
            # print(policy[s])
    if (old_policy!= policy).any():
        policy_stable = False
    else:
        policy_stable = True
    # YOUR CODE ENDS HERE
    ############################

    return policy, policy_stable