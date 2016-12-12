### Frances Ding, Lily Zhang, Kimberley Yu

import numpy as np
np.set_printoptions(threshold=np.nan)

BOTTOM_BLOCK_ROW = 93
TOP_PADDLE_ROW = 189
SCREEN_L = 8
SCREEN_R = 152
MIDDLE_X = (SCREEN_L+SCREEN_R)/2
MIDDLE_Y = (BOTTOM_BLOCK_ROW+TOP_PADDLE_ROW)/2

def getFeatures(state, action): ## didn't use the action at all here
    features = dict()
    ## get possible paddle positions (get from pixel image by color and location)
    paddle_xpos = state[TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0]
    # find the first non-zero value in the list (i.e. the first non-black pixel in that row)
    # that is the leftmost position of the paddle
    # else, give the paddle the position of the middle of the screen
    features["paddlex"] = next((i for i, x in enumerate(paddle_xpos) if x != 0), MIDDLE_X)
    
    ## get possible ball x positions between the bottom block row and top paddle row
    ball_xpos = np.sum(state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=0)
    # find the first non-zero value in the list (i.e. the first non-black pixel in that row)
    # that is the leftmost position of the ball
    # else, give the ball the position of the middle of the screen in the x-direction
    features["ballx"] = next((i for i, x in enumerate(ball_xpos) if x != 0), MIDDLE_X)

    ## get the possible y positions of the ball given where we know the x position to be
    ball_ypos = np.sum(state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=1)
    # find the first non-zero value in the list (i.e. the first non-black pixel in that row)
    # that is the topmost position of the ball
    # else, give the ball the position of the middle of the screen in the y-direction
    features["bally"] = next((i for i, x in enumerate(ball_ypos) if x != 0), MIDDLE_Y)
    
    # discretize the feature space
    # tested for various discretizations
    features["paddlex"] = features["paddlex"]/32
    features["ballx"] = features["ballx"]/32
    features["bally"] = features["bally"]/50

    return features

def getFeaturesPlus(state, prev_state, action):
    features = getFeatures(state, action)
    # add exactly where the ball will land
    ball_prev_xpos = np.sum(state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=0)
    feature_ballx_prev = next((i for i, x in enumerate(ball_prev_xpos) if x), MIDDLE_X)
    ball_prev_ypos = state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, feature_ballx_prev, 0]
    feature_bally_prev = next((i for i, x in enumerate(ball_prev_ypos) if x), MIDDLE_Y)
    # if ball is falling downward:
    if feature_bally_prev < features['bally']:
        ## feature for the y-direction of the ball 
        features['directionDown'] = 1

        ## get the landing position of the ball
        y_to_descend = TOP_PADDLE_ROW - features['bally']
        steps_left = y_to_descend/(features['bally']- feature_bally_prev)
        # get the landing position of the ball
        land_x = features['ballx'] + (features['ballx'] - feature_ballx_prev)*steps_left
        # if ball will land to the left of the current paddle position
        if features['paddlex'] > land_x:
            features['landing'] = 1
        # if ball will land to the right of the current paddle position
        elif features['paddlex'] < land_x:
            features['landing'] = -1
        # if ball will land right on top of the current paddle position
        else:
            features['landing'] = 0
    # else if the ball is bouncing upward
    else:
        features['directionDown'] = 0
        
        features['landing'] = 0

    return features
