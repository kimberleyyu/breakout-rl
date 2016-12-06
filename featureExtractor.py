# feature extractor script
# CS182 Project
# Kimberley Yu
# 12-04-2016

# take openAI gym breakout environment and generate features for approximate Q-learning
import numpy as np
import gym
env = gym.make('Breakout-v0')

# possible action space
print(env.action_space) # Discrete(6)
# possible state space (they call their states "observations")
print(env.observation_space) # Box(210, 160, 3) # each state is 210x160x3 grid

for i_episode in range(20): # number of "rounds"
    state = env.reset()
    for _ in range(1000): # number of time steps
        env.render()
        print type(state) # already of type ndarray
        print np.ndarray.nonzero(state) # gets indices of non-zero states
        # (array([  5,   5,   5, ..., 195, 195, 195]), array([36, 36, 36, ...,  7,  7,  7]), array([0, 1, 2, ..., 0, 1, 2]))
        # i think gray goes 0:5 and then 
        # get and print features
        #features = getFeatures(env, prev_state, cur_state)
        #print features

        #env.step(env.action_space.sample()) # take a random action

# returns feature vector of size _ x 1
# paddle: paddle position
# where position is of top left corner
# prev_ball: previous ball position
# where position is of center of ball
# cur_ball: current ball position
# etc: nothing yet, but we might implement more things

## colors: (BGR, not RGB)
#142, 142, 142 = gray (there's a gray border around the top, left, and right sides)
#0, 0, 0 = black
#2, 5, 5 = white
# 255, 0, 0 = blue
# 0, 255, 0 = green
# 0, 0, 255 = red
def getFeatures(self, prev_state, cur_state):

    ## paddle position (get from pixel image by color and location)
    self.features = dict()

    self.features["paddle"] = paddle_pos

    ## previous ball position


    self.features["prev_ball"] = prev_ball_pos

    ## current ball position
    self.features["cur_ball"] = cur_ball_pos

    ## not doing this anymore:
    ## next ball position (after taking action)
    # 6 possible actions: 0,1 = don't move; 3,5 = L; 2,4 = R

    return self.features


## how to define bricks vs. ball vs. paddle??? like how do I create that feature space if idk where the ball is?



## explanations of how breakout works
#https://www.nervanasys.com/openai/
## pacman version:
# http://ai.berkeley.edu/reinforcement.html
