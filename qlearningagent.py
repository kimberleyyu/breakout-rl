# the Q learning code
import featureExtractor, util

class QLearner:
    def __init__(self, legalActions, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.numTraining = numTraining
        self.weights = util.Counter()
        self.legalActions = legalActions

    # I'm not sure about this function in the context of the Open AI gyme

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        features = featureExtractor.getFeatures(state, action) # THIS DEPENDS ON FEATURE EXTRACTOR
        feature_keys = features.keys()
        feature_keys.sort()
        q_sum = 0
        for feature in feature_keys:
            q_sum = q_sum + self.weights[feature]*features[feature] #increment q value sum
        return q_sum

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action) where the max is over
          legal actions.  If no legal actions, returns a value of 0.0.
        """
        actions = self.legalActions
        vals = [self.getQValue(state, a) for a in actions]
        return max(vals)

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

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def update(self, state, action, nextState, reward):
        # extract features
        features = featureExtractor.getFeatures(state,action) #THIS DEPENDS ON FEATURE EXTRACTOR INTERFACE
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
        difference = (reward + self.discount*max_value) - self.getQValue(state,action)
        # loop over features to update their weights
        for feature in feature_keys:
            self.weights[feature] = self.weights[feature] + self.alpha*difference*features[feature]
