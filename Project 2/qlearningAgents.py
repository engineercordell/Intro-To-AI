# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # the Q-table that will hold all Q(s,a) pairs and their respective values
        self.dicStates = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # all this function does is get a particular Q(s,a) from the Q-table and return its inherent goodness
        
        # isNotVisited = False
        # for pair in self.dicStates:
        #     if self.dicStates[pair] == 0.0:

        if self.dicStates.get(state, action) == 0.0:
            return 0.0
        else:
            return self.dicStates[(state, action)]
        
        # could probably just return self.dic[(state, action)] itself without using a conditional bc
        # the dictionary that stores all Q(s,a) pairs (self.dicStates) value initializes everything to 0 which means
        # if the agent never visits a state, it's guaranteed to be 0 anyways..


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # this is essentially the max_a(Q(s', a')) of the Q-learning update equation
 
        # get all possible actions..
        legalActions = self.getLegalActions(state)
        if not legalActions: # not if legalActions.isEmpty(), keep thinking this is java..
            return 0.0
        
        # make a list to hold all these Q-values for the next state
        Qvalues = []

        # get all the Q-values of the Q(s',a') pairs
        for a in legalActions:
            Qvalues.append(self.getQValue(state, a))
        
        # if not Qvalues.isEmpty():
        #   highestQValue = max(Qvalues)
        #   return highestQValue
        # else:
        #   return 0
        
        # get the best value
        highestQValue = max(Qvalues)

        # return estimate of optimal future value
        return highestQValue


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        
        # IDEA: There can be multiple actions with similar Qvalues. The selected one is a random choice to the agent

        # to find the best action we must initialize 2 things:
        # 1. variable to hold highest Q-value
        highestQvalue = float('-inf')
        # 2. list to hold multiple highest Q-values
        actions = []
        
        for a in legalActions: # for every single action the agent could perform
            Qvalue = self.getQValue(state, a) # compute a particular QValue for that state

            if highestQvalue < Qvalue:
                highestQvalue = Qvalue # set the new highest value
                # actions = []
                actions = [a] # set actions equal to a new list of just the action, since this new Qvalue is greater than every single one we already have stored in actions
            elif highestQvalue == Qvalue:
                actions.append(a) #...otherwise we add the action to one of the possible ones the agent could perform, since they are all equal

        actionToTake = random.choice(actions) # pick whatever the agent wants to do
        return actionToTake # return that action

        # if some Qvalue > highest Q value, we must do 2 things:
        # 1. clear the list
        # 2. update the highest Q value



    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # this function is this step: Choose Action - Agent selects action a in current state s using an exploration strategy (ϵ-greedy)
        # the agent has to select the action for the CURRENT state, not the next one like max_a(Q(s',a')) would suggest
        # all of these functions are building the Q-update equation piece by piece..

        # get all possible actions
        legalActions = self.getLegalActions(state)
        "*** YOUR CODE HERE ***"
        if not legalActions: # if not possible, return from function and continue Q-learning iteration
            return None
        
        # agent either chooses exploration or exploitation
        if util.flipCoin(self.epsilon): # Exploration: Agent chooses random action with probability ϵ 
            return random.choice(legalActions) # picks any legal action without using Q-value insight
        else: # Exploitation: Agent chooses action with highest Q-value with probability 1−ϵ
            return self.computeActionFromQValues(state) # makes a more informed selection 
              

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # throw everything together from the Q-Learning update equation..

        firstPart = (1 - self.alpha) * self.getQValue(state, action)
        secondPart = self.alpha * reward
        thirdPart = self.alpha * self.discount * self.computeValueFromQValues(nextState)

        self.dicStates[(state, action)] = firstPart + secondPart + thirdPart

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


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
        features = self.featExtractor.getFeatures(state, action)
        "*** YOUR CODE HERE ***"
        return self.weights * features

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        "*** YOUR CODE HERE ***"

        y = reward + self.discount * self.computeValueFromQValues(nextState)
        delta = y - self.getQValue(state, action)

        alphaAndDelta = self.alpha * delta

        scaledFeatures = features

        for pair in features:
            scaledFeatures[pair] = features[pair] * alphaAndDelta

        # self.weights is a counter
        # scaledFeatures is also a counter

        # self.weight K: (s,a) V: value 
        
        self.weights = self.weights + scaledFeatures

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
