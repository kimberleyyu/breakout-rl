### Frances Ding, Lily Zhang, Kimberley Yu

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
        features = featureExtractor.getFeatures(state, action)
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
        # with probability self.epsilon, explore a random action
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
        features = featureExtractor.getFeatures(state, action)
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
        # update numTraining and stop exploring once you have trained the specified number of times
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
        features = featureExtractor.getFeaturesPlus(state, prev_state, action)
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
        # with probability self.epsilon, explore a random action
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
        features = featureExtractor.getFeaturesPlus(state, prev_state, action)
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
# Includes eligibility traces as a means to propagate rewards through timesteps.
# lambda parameter controls decay of eligibility trace.
class QLearnerPlusLambda(QLearnerPlus):
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
        features = featureExtractor.getFeaturesPlus(state, prev_state, action)
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
        # with probability self.epsilon, explore a random action
        if util.flipCoin(self.epsilon):
            return random.choice(actions)
        else:
            action = self.computeActionFromQValues(state, prev_state)
        return action

    def update(self, state, prev_state, nextState, action, nextAction, reward):
        """
            Performs the Q-learning update function and returns the new weights
            for all features. Uses eligibility traces to do so.
        """
        # Q(lambda) update function, as according to the Watkins algorithm

        # store the best action from state to nextState
        astar = self.computeActionFromQValues(nextState, state)
        # calculate the q-update difference
        delta = reward + self.gamma*self.getQValue(nextState, state, astar) - self.getQValue(state, prev_state, action)

        # extract features
        features = featureExtractor.getFeaturesPlus(state, prev_state, action)
        feature_keys = features.keys()
        feature_keys.sort()
        for feature in feature_keys:
            # update eligibility trace by 1 for current feature...
            self.eligibility[feature] = self.eligibility[feature] + 1
        for feature in feature_keys:
            # update weights based on difference (delta), previous weight, learning rate, and eligibility trace
            self.weights[feature] = self.weights[feature] + self.alpha * delta * self.eligibility[feature]
            # if the nextAction is the best action (not chosen randomly),
            # decay the eligibility trace accordingly for the next lookahead step
            if nextAction == astar:
                self.eligibility[feature] = self.gamma * self.lambd * self.eligibility[feature]
            # else if random exploration action was chosen, restart lookahead
            else:
                self.eligibility[feature] = 0

        # update numTraining
        self.numTraining -= 1
        if self.numTraining < 0:
            self.epsilon = 0

        return self.weights
