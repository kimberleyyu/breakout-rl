import numpy as np

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
        # features["paddlex"] = features["paddlex"]/8
        # features["ballx"] = features["ballx"]/8
        # features["bally"] = features["bally"]/10
    # if version >= 3:
        

    return features