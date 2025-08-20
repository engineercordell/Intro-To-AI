# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # error = 0 not needed here bc running for fixed num iterations and then stopping, no matter the condition
        # self.values - value functions for each state
        
        # first, initialize all state values to 0...
        listStates = self.mdp.getStates()
        for eachState in listStates:
            self.values[eachState] = 0
        
        # # how do I initialize the terminal state values??? does it matter?
        for i in range(self.iterations): # iterate i times
            
        #     for eachState in listStates: # on each iteration, each state has to be updated
        #         if self.mdp.isTerminal(eachState):
        #             continue
                
        #         highestValue = float('-inf') # always assume highest value is -infinity bc R+K book 
        #         # shows that state values can become negative on successive iterations before stabilization

        #         currValues = util.Counter() # store the current iteration values for each action
        #         for a in self.mdp.getPossibleActions(eachState):
        #             value = self.computeQValueFromValues(eachState, a)
        #             currValues[eachState] = value
                    
        #         stateWithHighestValue = max(currValues, key=currValues.get)
        #         highestValue = currValues[stateWithHighestValue]

        #         if highestValue != float('-inf'):
        #             currValues[eachState] = highestValue

        #         self.values[eachState] = highestValue

            currValues = util.Counter()

            for eachState in listStates:
                #if self.mdp.isTerminal(eachState): not actually needed because it doesn't contain inherent value within the code itself, so it's irrelevant. there's only 1 terminal state
                #    currValues[eachState] = self.values[eachState]
                #    continue

                highestValue = float('-inf')

                for a in self.mdp.getPossibleActions(eachState):
                    value = self.computeQValueFromValues(eachState, a)
                    if value > highestValue:
                        highestValue = value

                if highestValue != float('-inf'):
                    currValues[eachState] = highestValue

            self.values = currValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        newQValue = 0
        nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
        # If getTSAP((1,1), right), then it should return something that looks like [((2,1), 0.8), ((1,1), 0.1), ((1,2), 0.1)]

        # we need a for loop for each P(s'|s,a)*Vk(s') to account for action noise

        for newState, probability in nextStates:
            rewardTransition = self.mdp.getReward(state, action, newState)
            newQValue += probability * (rewardTransition + self.discount * self.values[newState])

        return newQValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # gotta remember that the goal of this function is to build a policy
        # so we need to start somewhere (state, the passed in param)
        # see what actions CAN be taken
        # determine which action is the best to take based on state Q-values

        possibleActions = self.mdp.getPossibleActions(state) # get all possible actions in this state
        if not possibleActions: # go ahead and check if we are in the terminal state
            return None
        
        # get all surrounding QValues for each next state and pair it with its action
        act_val = util.Counter()
        for a in possibleActions:
            act_val[a] = self.computeQValueFromValues(state, a)

        # we need to select for the action that gets us to the highest utility state
        maxAction = max(act_val, key=act_val.get)

        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
