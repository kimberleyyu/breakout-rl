import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Breakout-v0')
# observation = env.reset()
# print np.unique(observation)
# all the colors
# colors = set([])
# for i in range(210):
# 	for j in range(160):
# 		colors.add(tuple(observation[i][j]))
# print colors

# discretizing parts of the screen
# for i in range(210):
# 	if observation[i][80][0] == 0:
# 		print i

# 0 - 16
# 32 - 209

# paddle size 
# paddle = []
# for i in range(8, 152+1):
# 	if observation[189][i][0] != 0:
# 		paddle.append(i)
# print paddle
# print len(paddle)

# # screen L and screen R
# for i in range(160):
# 	if observation[188][i][0] != 0:
# 		paddle.append(i)
# print paddle
# print len(paddle)

# print np.unique(observation[0][0][0])

## top of paddle 
# for i in range(8, 152):

#> Discrete(2)
# print "obs space", env.observation_space
# print "obs space high", env.observation_space.high
#> array([ 2.4       ,         inf,  0.20943951,         inf])
# print "obs space high", env.observation_space.low
# for i in range(20):
#     env.render()
#     # action = env.action_space.sample()
#     # print action
#     # env.step(env.action_space.sample()) # take a random action
#     env.step(0)
#     print 0
lengths = []
rewards = []


for i_episode in range(200):
    observation = env.reset()
    r = 0
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        r +=reward
        if done:
            print("Episode finished after {} timesteps with {} reward".format(t+1, r))
            lengths.append(t+1)
            rewards.append(r)
            break

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lengths)
fig.savefig('lengthsran.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(rewards)
fig.savefig('rewardsran.png')
print "Lengths: mean", np.mean(lengths), "std", np.std(lengths)
print "Rewards: mean", np.mean(rewards), "std", np.std(rewards)
# for i in range(1000):
# 	env.render()
# 	env.step(1)
# for i in range(100):
# 	env.render()
# 	env.step(2)
# 	print 2
# for i in range(10):
# 	env.render()
# 	env.step(3)
# 	print 3
# for i in range(10):
# 	env.render()
# 	env.step(4)
# 	print 4
# for i in range(10):
# 	env.render()
# 	env.step(5)
# 	print 5