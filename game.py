import gym
import qlearningagent
import numpy as np
import matplotlib.pyplot as plt

TOP_PADDLE_ROW = 189
SCREEN_L = 8
SCREEN_R = 152
PADDLE_LEN = 16

def ballFell(state):
    for i in range(SCREEN_L, SCREEN_R):
        if state[TOP_PADDLE_ROW][i][0] == 0 and state[TOP_PADDLE_ROW][i+1][0] != 0 and state[TOP_PADDLE_ROW][i+3][0] == 0:
            return True
    return False


env = gym.make('Breakout-v0')
# for monitoring how we are doing
# env.monitor.start('breakout-experiment-3')
legalActions = range(env.action_space.n)
agent = qlearningagent.QLearner(legalActions, featureVersion = 2, epsilon=0.05,gamma=0.8,alpha=0.02, numTraining=10000)
lengths = []
rewards = []

# play the game 5 times
for i_episode in range(200):
    r = 0
    state = env.reset()  
    for t in range(10000):
        env.render()
        action = agent.getAction(state)
        nextState, reward, done, info = env.step(action)
        r +=reward
        if ballFell(nextState):
            reward = -1
        agent.update(state, action, nextState, reward)
        nextState = state
        # end if done
        if done:
            print("Episode finished after {} timesteps with {} reward".format(t+1, r))
            lengths.append(t+1)
            rewards.append(r)
            break
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lengths)
fig.savefig('lengths.png')

fig = plt.figure()
ax = fig.add_subplot(112)
ax.plot(rewards)
fig.savefig('rewards.png')
print "Lengths: mean", np.mean(lengths), "std", np.std(lengths)
print "Rewards: mean", np.mean(rewards), "std", np.std(rewards)
# env.monitor.close()


