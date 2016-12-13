### Frances Ding, Lily Zhang, Kimberley Yu

# to run:
# ex. python game.py --version 3 --epsilon 0.1 --gamma 0.9 --alpha 0.05 --numtrain 1000
# all arguments are optional
# order of arguments doesn't matter
# use --version to choose between version 1, 2, and 3.
# For all versions: use --epsilon, --gamma, --alpha, and/or --numtrain if you want to adjust those parameters.
    #The first 3 must be floats and the last must be an int.
# For version 3: --lambd adjusts the lambda decay rate for the eligibility traces.

import gym
import qlearningagent
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

# constants for screen pixel configurations
BOTTOM_BLOCK_ROW = 93
TOP_PADDLE_ROW = 189
SCREEN_L = 8
SCREEN_R = 152
MIDDLE_X = (SCREEN_L+SCREEN_R)/2
MIDDLE_Y = (BOTTOM_BLOCK_ROW+TOP_PADDLE_ROW)/2

PADDLE_LEN = 16

def ballFell(state):
    """ determine from the image if the ball fell off the screen"""
    for i in range(SCREEN_L, SCREEN_R):
        if state[TOP_PADDLE_ROW][i][0] == 0 and state[TOP_PADDLE_ROW][i+1][0] != 0 and state[TOP_PADDLE_ROW][i+3][0] == 0:
            return True
    return False

def ballHit(state):
    """ determine from the image if the paddle hit the ball (x positions match)"""
    paddle_xpos = state[TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0]
    paddlex = next((i for i, x in enumerate(paddle_xpos) if x), MIDDLE_X)

    ball_xpos = np.sum(state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=0)
    ballx = next((i for i, x in enumerate(ball_xpos) if x != 0), MIDDLE_X)

    ball_ypos = np.sum(state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=1)
    # unlike featureExtractor, add BOTTOM_BLOCK_ROW because bally otherwise offset relative to TOP_PADDLE_ROW
    # (don't need to do this for ballx and paddlex because they're in the same frame of reference (only black space considered))
    bally = next((i for i, x in enumerate(ball_ypos) if x != 0), MIDDLE_Y) + BOTTOM_BLOCK_ROW

    ##reward if exact hit: ballx matches paddlex and bally matches paddle y
    if ballx - paddlex < PADDLE_LEN and ballx - paddlex > 0 and abs(bally - TOP_PADDLE_ROW) < 10 :
    ## alternatively, reward if ball really close to hitting paddle
    #if abs(ballx - paddlex) < 20 and abs(bally - TOP_PADDLE_ROW) < 15 :
        return True
    return False

## input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', dest='epsilon', default=0.05, required=False, type=float, help='exploration rate for epsilon-greedy')
parser.add_argument('--gamma', default=0.99, required=False, type=float, help='discount factor')
parser.add_argument('--alpha', default=0.02, required=False, type=float, help='learning rate')
parser.add_argument('--numtrain', default=10000, required=False, type=int, help='number of training episodes')

parser.add_argument('--lambd', default=0.9, required=False, type=float, help='decay rate of eligibility traces')
parser.add_argument('--version', default=1, required=False, type=int, help='1 for QLearner; 2 for QLearnerPlus; 3 for QLearnerPlusLambda')

args = parser.parse_args()

epsilon = args.epsilon
gamma = args.gamma
alpha = args.alpha
numTraining = args.numtrain
lambd = args.lambd
version = args.version

# create the environment for Breakout
env = gym.make('Breakout-v0')

# testing only: to monitor how we are doing
# env.monitor.start('breakout-experiment-3')

# get the legal actions as a list
legalActions = range(env.action_space.n)

# initiatlize corresponding agents for each version
if version==1:
    agent = qlearningagent.QLearner(legalActions, epsilon=epsilon,gamma=gamma,alpha=alpha, numTraining=numTraining)
elif version==2:
    agent = qlearningagent.QLearnerPlus(legalActions, epsilon=epsilon,gamma=gamma,alpha=alpha, numTraining=numTraining)
elif version==3:
    agent = qlearningagent.QLearnerPlusLambda(legalActions, epsilon=epsilon,gamma=gamma,alpha=alpha, numTraining=numTraining, lambd=lambd)
else:
    print "NOT A VALID VERSION. PLEASE CHOOSE 1, 2, OR 3. Defaulting to version 1."
    version = 1
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
    # reset the environment for the beginning of each game
    state = env.reset()

    # extra initializations for more complext versions
    if version>1:
        prev_state = state
    if version==3:
        action = 0 # initialize a first action

    # iterate through each timestep t in a game
    for t in range(10000):
        # render the environment at that timestep
        env.render()
        # get an action
        if version==1:
            action = agent.getAction(state)
        elif version==2:
            action = agent.getAction(state, prev_state)
        # take the action and get the reward and nextState from the environment
        nextState, reward, done, info = env.step(action)

        # update the total reward count
        r +=reward
        # include a negative reward if the ball falls
        if ballFell(nextState):
            reward = -1
        # include a positive reward if ball hit by paddle
        if ballHit(nextState):
            reward += 1

        # update the weights
        if version==1:
            weights = agent.update(state, action, nextState, reward)
        elif version==2:
            weights = agent.update(state, prev_state, action, nextState, reward)
        elif version==3:
            nextAction = agent.getAction(nextState, state)
            weights = agent.update(state, prev_state, nextState, action, nextAction, reward)

        # keep track of the weights over time
        weights_over_time.append([weights['paddlex'],weights['ballx'], weights['bally'], weights['directionDown'], weights['landing']])

        # a check to prevent an error in the case of the weights blowing up
        # instead, we stop the game
        if abs(weights["ballx"]) > 10**305:
            print("Episode finished after {} timesteps with {} reward".format(t+1, r))
            lengths.append(t+1)
            rewards.append(r)
            end = True
            break

        # if version 2 or 3, update the prev_state; if version 3, also update the action
        if version > 1:
            prev_state = state
            if version==3:
                action = nextAction
        # update state to next state
        state = nextState

        # end and print timesteps and reward if done
        if done:
            print("Episode finished after {} timesteps with {} reward".format(t+1, r))
            lengths.append(t+1)
            rewards.append(r)
            break

        # to slow down the game for testing purposes
        #time.sleep(0.1)

    # if we need to preemptively end the game because of the weights blowing up
    if end:
        break

# plot the lengths of the episodes, the total rewards, and the weights over time
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lengths)
fig.savefig('lengths-v%s-e%s-g%s-a%s-n%s-l%s.png' % (version, epsilon, gamma, alpha, numTraining, lambd))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(rewards)
fig.savefig('rewards-v%s-e%s-g%s-a%s-n%s-l%s.png' % (version, epsilon, gamma, alpha, numTraining, lambd))

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(weights_over_time)
fig.savefig('weights-v%s-e%s-g%s-a%s-n%s-l%s.png' % (version, epsilon, gamma, alpha, numTraining, lambd))

# in case I need all lengths for significance test / t-test purposes
print "All lengths recorded: ", lengths
print "All rewards recorded: ", rewards

print "Lengths: mean", np.mean(lengths), "std", np.std(lengths)
print "Rewards: mean", np.mean(rewards), "std", np.std(rewards)

# env.monitor.close()
