### Frances Ding, Lily Zhang, Kimberley Yu

# to run:
# python game.py --QLearnerPlus --epsilon 0.1 --gamma 0.9 --alpha 0.05 --numtrain 1000
# all arguments are optional
# order of arguments doesn't matter
# use --QLearnerPlus if you want to use our version with the two additional features.
# use --epsilon, --gamma, --alpha, and/or --numtrain if you want to adjust those parameters. The first 3 must be floats and the last must be an int.

import gym
import qlearningagent
import numpy as np
import matplotlib.pyplot as plt
import argparse

TOP_PADDLE_ROW = 189
SCREEN_L = 8
SCREEN_R = 152
PADDLE_LEN = 16

def ballFell(state):
    """ determine from the image if the ball fell off the screen"""
    for i in range(SCREEN_L, SCREEN_R):
        if state[TOP_PADDLE_ROW][i][0] == 0 and state[TOP_PADDLE_ROW][i+1][0] != 0 and state[TOP_PADDLE_ROW][i+3][0] == 0:
            return True
    return False

## input command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', dest='epsilon', default=0.05, required=False, type=float, help='exploration rate for epsilon-greedy')
parser.add_argument('--gamma', default=0.99, required=False, type=float, help='discount factor')
parser.add_argument('--alpha', default=0.02, required=False, type=float, help='learning rate')
parser.add_argument('--numtrain', default=10000, required=False, type=int, help='number of training episodes')

parser.add_argument('--QLearnerPlus', dest='qplus', action='store_true', help="Add this argument if you want to run QLearnerPlus.")
parser.set_defaults(feature=False).

args = parser.parse_args()

epsilon = args.epsilon
gamma = args.gamma
alpha = args.alpha
numTraining = args.numtrain
qplus = args.qplus

# create the environment for Breakout
env = gym.make('Breakout-v0')

# for monitoring how we are doing
# env.monitor.start('breakout-experiment-3')

# get the legal actions as a list
legalActions = range(env.action_space.n)
# qplus is our agent which the additional features of where the ball will land relative to the paddle
# qplus takes in the current and previous state to determine the best action
if qplus:
    agent = qlearningagent.QLearnerPlus(legalActions, epsilon=epsilon,gamma=gamma,alpha=alpha, numTraining=numTraining)
# else, we use only use our base features
else:
    agent = qlearningagent.QLearner(legalActions, epsilon=epsilon,gamma=gamma,alpha=alpha, numTraining=numTraining)
# lists to record the lengths of the episodes, total rewards, and weights for graphing
lengths = []
rewards = []
weights_over_time = []
# extra precaution in place to prevent crashes during instances where weights blow up
end = False

# play the game 200 times
for i_episode in range(200):
    # keep track of the total reward for a game
    r = 0
    # reset the environment for the beginninng of each game
    state = env.reset()
    if qplus:
        prev_state = state
    # iterate through each timestep t in a game
    for t in range(10000):
        # render the environment at that timestep
        env.render()
        # get an action
        if qplus:
            action = agent.getAction(state, prev_state)
        else:
            action = agent.getAction(state)
        # take the action and get the reward and nextState from the environment
        nextState, reward, done, info = env.step(action)
        # update the total reward count
        r +=reward
        # include a negative reward if the ball falls
        if ballFell(nextState):
            reward = -1
        # update the weights
        if qplus:
            weights = agent.update(state, prev_state, action, nextState, reward)
        else:
            weights = agent.update(state, action, nextState, reward)
        # keep track of the weights over time
        weights_over_time.append([weights['paddlex'],weights['ballx'], weights['bally']])
        
        if abs(weights["ballx"]) > 10**305:
            print("Episode finished after {} timesteps with {} reward".format(t+1, r))
            lengths.append(t+1)
            rewards.append(r)
            end = True
            break
        if qplus:
            nextState = state
            prev_state = state
            state = nextState
        # end if done
        if done:
            print("Episode finished after {} timesteps with {} reward".format(t+1, r))
            lengths.append(t+1)
            rewards.append(r)
            break
    if end:
        break
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lengths)
fig.savefig('lengthslonglanding.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(rewards)
fig.savefig('rewardslonglanding.png')

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(weights_over_time)
fig.savefig('weightslonglanding.png')
print "Lengths: mean", np.mean(lengths), "std", np.std(lengths)
print "Rewards: mean", np.mean(rewards), "std", np.std(rewards)
# env.monitor.close()
