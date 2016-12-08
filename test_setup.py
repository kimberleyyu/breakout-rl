import gym
import numpy as np

env = gym.make('Breakout-v0')
observation = env.reset()
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
paddle = []
# for i in range(8, 152+1):
# 	if observation[189][i][0] != 0:
# 		paddle.append(i)
# print paddle
# print len(paddle)

# screen L and screen R
for i in range(160):
	if observation[188][i][0] != 0:
		paddle.append(i)
print paddle
print len(paddle)

print np.unique(observation[0][0][0])
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

for i in range(1000):
	env.render()
	env.step(1)
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