import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration. Update Q at the end of every episode.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
    """

    Q = np.zeros((env.nS, env.nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    env.isd = (1/env.nS)*env.nS
    for ep in range(num_episodes):
        s = env.reset()
        # print('s =',s,'; a =',a)
        terminal = False
        while not terminal:
            # choosing a from S with eps-greedy policy
            a = epsilon_greedy(Q[s],e = e)
            nextstate,reward,terminal,info = env.step(a)
            # choosing next_a from S' with greedy policy
            next_a = epsilon_greedy(Q[nextstate],e = 0)
            Q[s,a] += lr*(reward + gamma*(Q[nextstate,next_a])-Q[s,a])
            s = nextstate
        # print("ep =",ep)

    # YOUR CODE ENDS HERE
    ############################

    return Q