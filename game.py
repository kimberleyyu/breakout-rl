import gym
import qlearningagent
import numpy as np

TOP_PADDLE_ROW = 189
SCREEN_L = 8
SCREEN_R = 152
PADDLE_LEN = 16

def ballFell(state):
    # check black red black
    # red_pixels = 0
    # for i in range(SCREEN_L, SCREEN_R):
    #     if state[TOP_PADDLE_ROW][i][0] != 0:
    #         red_pixels += 1
    # print red_pixels
    # if red_pixels > PADDLE_LEN:
    #     return True
    # return False
    for i in range(SCREEN_L, SCREEN_R):
        if state[TOP_PADDLE_ROW][i][0] == 0 and state[TOP_PADDLE_ROW][i+1][0] != 0 and state[TOP_PADDLE_ROW][i+3][0] == 0:
            return True
    return False

env = gym.make('Breakout-v0')
# for monitoring how we are doing
# env.monitor.start('breakout-experiment-2')
legalActions = range(env.action_space.n)
agent = qlearningagent.QLearner(legalActions, featureVersion = 2, epsilon=0.05,gamma=0.8,alpha=0.02, numTraining=10000)
# play the game 5 times
for i_episode in range(1000):
    state = env.reset()  
    for t in range(10000):
        env.render()
        action = agent.getAction(state)
        nextState, reward, done, info = env.step(action)
        if ballFell(nextState):
            reward = -1
            print reward
        agent.update(state, action, nextState, reward)
        nextState = state
        # end if done
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
# env.monitor.close()


