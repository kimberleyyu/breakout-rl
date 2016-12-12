### Frances Ding, Lily Zhang, Kimberley Yu

# q learning with eligibility traces
# http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node78.html
# http://stackoverflow.com/questions/40862578/how-to-understand-watkinss-q%CE%BB-learning-algorithm-in-suttonbartos-rl-book/40899753#40899753

import featureExtractor, util
import random
import itertools


# Basic Q learner with simple features:
# x and y position of ball + x position of paddle
class QLearner:
    def __init__(self, legalActions, epsilon=0.05,gamma=0.99,alpha=0.2, numTraining=1000):
        """
            Initializes the QLearner.
        """
        self.legalActions = legalActions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.numTraining = numTraining
        self.weights = util.Counter()

    def getQValue(self, state, action):
        """
            Calculates and returns the Q-value for the (state, action) pair
            using the feature extractor. (Approximate Q-Learning).
        """
        features = featureExtractor.getFeatures(state, action) # THIS DEPENDS ON FEATURE EXTRACTOR
        feature_keys = features.keys()
        feature_keys.sort()
        q_sum = 0
        for feature in feature_keys:
            q_sum = q_sum + self.weights[feature]*features[feature] #increment q value sum
        return q_sum

    def computeActionFromQValues(self, state):
        """
          Computes the best action to take in a state.  If no legal actions,
          eg. at the terminal state, returns None.
        """
        actions = self.legalActions
        vals = [self.getQValue(state, a) for a in actions]
        maxVal = max(vals)
        bestActions = [a for a in actions if self.getQValue(state, a) == maxVal]
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Computes the action to take in the current state.  With
          probability self.epsilon, it takes a random action and
          take the best policy action otherwise.  If there are
          no legal actions, eg. at the terminal state, returns None.
        """
        # Pick Action
        actions = self.legalActions
        action = None
        if util.flipCoin(self.epsilon):
            return random.choice(actions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
            Performs the Q-learning update function and returns the new weights
            for all features.
        """
        # extract features
        features = featureExtractor.getFeatures(state, action) #THIS DEPENDS ON FEATURE EXTRACTOR INTERFACE
        feature_keys = features.keys()
        feature_keys.sort()
        # first we find the max Q-value over possible actions
        actions = self.legalActions
        if actions:
            max_value = -float("inf")
            for action2 in actions:
                if self.getQValue(nextState,action2) > max_value:
                    max_value = self.getQValue(nextState,action2)
        # if there are no legal actions:
        else:
            max_value = 0.0
        # calculate difference between predicted and observed value
        difference = (reward + self.gamma*max_value) - self.getQValue(state,action)
        # loop over features to update their weights
        for feature in feature_keys:
            self.weights[feature] = self.weights[feature] + self.alpha*difference*features[feature]
        self.numTraining -= 1
        if self.numTraining < 0:
            self.epsilon = 0
        return self.weights

# Inherits from QLearner class.
# Added features:
# Predicted landing x position of ball + relative position of ball to paddle
class QLearnerPlus(QLearner):

    def getQValue(self, state, prev_state, action):
        """
            Calculates and returns the Q-value for the (state, action) pair
            using the feature extractor. (Approximate Q-Learning).
        """
        features = featureExtractor.getFeaturesPlus(state, prev_state, action) # THIS DEPENDS ON FEATURE EXTRACTOR
        feature_keys = features.keys()
        feature_keys.sort()
        q_sum = 0
        for feature in feature_keys:
            q_sum = q_sum + self.weights[feature]*features[feature] #increment q value sum
        return q_sum

    def computeActionFromQValues(self, state, prev_state):
        """
          Computes the best action to take in a state.  If no legal actions,
          eg. at the terminal state, returns None.
        """
        actions = self.legalActions
        vals = [self.getQValue(state, prev_state, a) for a in actions]
        maxVal = max(vals)
        bestActions = [a for a in actions if self.getQValue(state, prev_state, a) == maxVal]
        return random.choice(bestActions)

    def getAction(self, state, prev_state):
        """
          Computes the action to take in the current state.  With
          probability self.epsilon, it takes a random action and
          take the best policy action otherwise.  If there are
          no legal actions, eg. at the terminal state, returns None.
        """
        # Pick Action
        actions = self.legalActions
        action = None
        if util.flipCoin(self.epsilon):
            return random.choice(actions)
        else:
            action = self.computeActionFromQValues(state, prev_state)
        return action

    def update(self, state, prev_state, action, nextState, reward):
        """
            Performs the Q-learning update function and returns the new weights
            for all features.
        """
        # extract features
        features = featureExtractor.getFeaturesPlus(state, prev_state, action) #THIS DEPENDS ON FEATURE EXTRACTOR INTERFACE
        feature_keys = features.keys()
        feature_keys.sort()
        # first we find the max Q-value over possible actions
        actions = self.legalActions
        if actions:
            max_value = -float("inf")
            for action2 in actions:
                if self.getQValue(nextState,state, action2) > max_value:
                    max_value = self.getQValue(nextState,state, action2)
        # if there are no legal actions:
        else:
            max_value = 0.0
        # calculate difference between predicted and observed value
        difference = (reward + self.gamma*max_value) - self.getQValue(state,prev_state, action)
        # loop over features to update their weights
        for feature in feature_keys:
            self.weights[feature] = self.weights[feature] + self.alpha*difference*features[feature]
        self.numTraining -= 1
        if self.numTraining < 0:
            self.epsilon = 0
        return self.weights

# Inherits from QLearnerPlus class.
# Includes eligibility traces
# lambda parameter controls decay of eligibility trace
class QLearnerPlusLambda(QLearnerPlus):
    # def __init__(self, lambd=0.9):
    #     """
    #         Initializes the QLearner with additional lambda parameter.
    #     """
    #     super(QLearnerPlusLambda, self).__init__()
    #     self.lambd = lambd
    #     self.eligibility = util.Counter()

    def __init__(self, legalActions, epsilon=0.05,gamma=0.99,alpha=0.2, numTraining=1000, lambd=0.9):
        """
            Initializes the QLearner.
        """
        self.legalActions = legalActions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.numTraining = numTraining
        self.weights = util.Counter()

        self.lambd = lambd
        self.eligibility = util.Counter()

    def getQValue(self, state, prev_state, action):
        """
            Calculates and returns the Q-value for the (state, action) pair
            using the feature extractor. (Approximate Q-Learning).
        """
        features = featureExtractor.getFeaturesPlus(state, prev_state, action) # THIS DEPENDS ON FEATURE EXTRACTOR
        feature_keys = features.keys()
        feature_keys.sort()
        q_sum = 0
        for feature in feature_keys:
            q_sum = q_sum + self.weights[feature]*features[feature] #increment q value sum
        return q_sum

    def computeActionFromQValues(self, state, prev_state):
        """
          Computes the best action to take in a state.  If no legal actions,
          eg. at the terminal state, returns None.
        """
        actions = self.legalActions
        vals = [self.getQValue(state, prev_state, a) for a in actions]
        maxVal = max(vals)
        bestActions = [a for a in actions if self.getQValue(state, prev_state, a) == maxVal]
        return random.choice(bestActions)

    def getAction(self, state, prev_state):
        """
          Computes the action to take in the current state.  With
          probability self.epsilon, it takes a random action and
          take the best policy action otherwise.  If there are
          no legal actions, eg. at the terminal state, returns None.
        """
        # Pick Action
        actions = self.legalActions
        action = None
        if util.flipCoin(self.epsilon):
            return random.choice(actions)
        else:
            action = self.computeActionFromQValues(state, prev_state)
        return action

    def update(self, state, prev_state, nextState, action, nextAction, reward):
        """
            Performs the Q-learning update function and returns the new weights
            for all features.
        """

        astar = self.computeActionFromQValues(nextState, state)
        delta = reward + self.gamma*self.getQValue(nextState, state, astar) - self.getQValue(state, prev_state, action)

        # extract features
        features = featureExtractor.getFeaturesPlus(state, prev_state, action) #THIS DEPENDS ON FEATURE EXTRACTOR INTERFACE
        feature_keys = features.keys()
        feature_keys.sort()
        for feature in feature_keys:
            # have an eligibility trace for each feature?
            self.eligibility[feature] = self.eligibility[feature] + 1

        # for all features
        ## wait lols don't think this is necessary
        # features_space = itertools.product(range(8, 152)/32, range(93,189)/50, range(8, 152)/32, range(8, 152)/32)
        # for feat_values in features_space:
        #     feat_dict = dict(zip(feature_keys, feat_values))

        for feature in feature_keys:
            self.weights[feature] = self.weights[feature] + self.alpha * delta * self.eligibility[feature]
            if nextAction == astar:
                self.eligibility[feature] = self.gamma * self.lambd * self.eligibility[feature]
            else:
                self.eligibility[feature] = 0

        self.numTraining -= 1
        if self.numTraining < 0:
            self.epsilon = 0

        return self.weights
