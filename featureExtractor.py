# feature extractor script
# CS182 Project
# Kimberley Yu
# 12-04-2016

# take openAI gym breakout environment and generate features for approximate Q-learning
import numpy as np
# np.set_printoptions(threshold=np.nan) # so that I can see full array
# import gym
# env = gym.make('Breakout-v0')
#
# # possible action space
# print(env.action_space) # Discrete(6)
# # possible state space (they call their states "observations")
# print(env.observation_space) # Box(210, 160, 3) # each state is 210x160x3 grid
#
# for i_episode in range(20): # number of "rounds"
#     state = env.reset()
#     for _ in range(1000): # number of time steps
#         env.render()
#
#         # get and print features
#         featExtractor = FeatureExtractor() ## how do i call the class in the file? or maybe it has to be in other file?
#         features = featExtractor.getFeatures(env, state)
#         print features
#
#         env.step(env.action_space.sample()) # take a random action


# returns dict with keys [paddle, ballx, bally]
def getFeatures(self, state):

    ## paddle position (get from pixel image by color and location)
    self.features = dict()

    paddle_possible_xpos = state[189, 8:152, :] # only 189 necessary because all heights are same
    paddle_allx = [i for i, RGB_num in enumerate(paddle_possible_xpos) if list(RGB_num) == [200, 72, 72]]
    paddlex_leftx = paddle_allx[0]

    self.features["paddle"] = paddle_leftx

    ## ball x and y positions
    ball_possible_pos = state[93:193, 8:152, :] # blocks stop at 31; we're only considering if ball is below blocks

    allx = list()
    ally = list()
    for i, col in enumerate(ball_possible_pos):
        for j, RGB_num in enumerate(col):
            if list(RGB_num) == [200, 72, 72]:
                allx.append(i)
                ally.append(j)
    print allx, ally

    ball_xpos = np.median(allx) # if we do center of ball
    ball_ypos = np.median(ally)

    self.features["ballx"] = ball_xpos # ball x position
    self.features["bally"] = ball_ypos # ball y position

    return self.features
