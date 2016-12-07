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
# def getFeatures(state, action): ## didn't use the action at all here

#     ## paddle position (get from pixel image by color and location)
#     features = dict()

#     paddle_possible_xpos = state[189, 8:152, :] # only 189 necessary because all heights are same
#     paddle_allx = [i for i, RGB_num in enumerate(paddle_possible_xpos) if list(RGB_num) == [200, 72, 72]]
#     paddle_leftx = paddle_allx[0]

#     features["paddle"] = paddle_leftx

#     ## ball x and y positions
#     ball_possible_pos = state[93:193, 8:152, :] # blocks stop at 31; we're only considering if ball is below blocks

#     allx = list()
#     ally = list()
#     for i, col in enumerate(ball_possible_pos):
#         for j, RGB_num in enumerate(col):
#             if list(RGB_num) == [200, 72, 72]:
#                 allx.append(i)
#                 ally.append(j)
    
#     ball_xpos = np.median(allx) # if we do center of ball
#     ball_ypos = np.median(ally)

#     features["ballx"] = ball_xpos # ball x position
#     features["bally"] = ball_ypos # ball y position

#     return features
BOTTOM_BLOCK_ROW = 93
TOP_PADDLE_ROW = 189
SCREEN_L = 8
SCREEN_R = 152
MIDDLE_X = (SCREEN_L+SCREEN_R)/2
MIDDLE_Y = (BOTTOM_BLOCK_ROW+TOP_PADDLE_ROW)/2

def getFeatures(version, state, action): ## didn't use the action at all here
    features = dict()
    if version >= 0:
        ## get possible paddle positions (get from pixel image by color and location)
        paddle_xpos = state[TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0] 
        # find the first non-zero value in the list (i.e. the first non-black pixel in that row) 
        features["paddlex"] = next((i for i, x in enumerate(paddle_xpos) if x), MIDDLE_X)
        ## get possible ball x positions between the bottom block row and top paddle row
        ball_xpos = np.sum(state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=0)
        # find the first non-zero value in the list (i.e. the first non-black pixel in that row) 
        features["ballx"] = next((i for i, x in enumerate(ball_xpos) if x), MIDDLE_X)
    
    if version >= 1:
        ball_ypos = state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, features["ballx"], 0]
        features["bally"] = next((i for i, x in enumerate(ball_ypos) if x), MIDDLE_Y)
    
    if version >= 2:
        features["paddlex"] = features["paddlex"]/32
        features["ballx"] = features["ballx"]/32
        features["bally"] = features["bally"]/50
        print "features", features
    # else:

    return features