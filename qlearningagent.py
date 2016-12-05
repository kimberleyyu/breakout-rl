# the Q learning code
import featureExtractor, util

class QLearner:
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.numTraining = numTraining
        self.weights = util.Counter()
#from pacman, need to change
class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state,action) # get feature dict
        feature_keys = features.sortedKeys()
        q_sum = 0
        for feature in feature_keys:
            q_sum = q_sum + self.weights[feature]*features[feature] #increment q value sum
        return q_sum


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # extract features
        features = self.featExtractor.getFeatures(state,action)
        feature_keys = features.sortedKeys()
        # first we find the max Q-value over possible actions
        legalActions = self.getLegalActions(nextState)
        if legalActions:
            max_value = -float("inf")
            for action2 in legalActions:
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
