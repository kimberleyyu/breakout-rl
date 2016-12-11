import numpy as np

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
    features["paddlex"] = next((i for i, x in enumerate(paddle_xpos) if x), MIDDLE_X)
    ## get possible ball x positions between the bottom block row and top paddle row
    ball_xpos = np.sum(state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, SCREEN_L:SCREEN_R, 0], axis=0)
    # find the first non-zero value in the list (i.e. the first non-black pixel in that row)
    features["ballx"] = next((i for i, x in enumerate(ball_xpos) if x), MIDDLE_X)

    ball_ypos = state[BOTTOM_BLOCK_ROW:TOP_PADDLE_ROW, features["ballx"], 0]
    features["bally"] = next((i for i, x in enumerate(ball_ypos) if x), MIDDLE_Y)

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
    # if ball on downward slope:
    if feature_bally_prev < features['bally']:
        features['directionDown'] = 1
        y_to_descend = TOP_PADDLE_ROW - features['bally']
        steps_left = y_to_descend/(features['bally']- feature_bally_prev)
        # features['landing'] = features['ballx'] + (features['ballx'] - feature_ballx_prev)*steps_left
        land_x = features['ballx'] + (features['ballx'] - feature_ballx_prev)*steps_left
        if features['paddlex'] > land_x:
            features['landing'] = 1
        elif features['paddlex'] < land_x:
            features['landing'] = -1
        else:
            features['landing'] = 0
    else:
        features['directionDown'] = 0
        # features['landing'] = features['ballx']
        features['landing'] = 0

    return features
