import gym

env = gym.make('Breakout-v0')
observation = env.reset()
print "action space", env.action_space
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

# for i in range(10):
# 	env.render()
# 	env.step(1)
# 	print 1
for i in range(100):
	env.render()
	env.step(2)
	print 2
for i in range(10):
	env.render()
	env.step(3)
	print 3
for i in range(10):
	env.render()
	env.step(4)
	print 4
for i in range(10):
	env.render()
	env.step(5)
	print 5