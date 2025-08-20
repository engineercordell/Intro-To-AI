import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined
import util

class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """


        if self.total() == 0.0:
            return

        tot = self.total()
        for key in self.keys():
            self[key] = float(self[key])/tot




    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """


        s_seq = []
        s_weights = []

        for item in self.items():
            s_seq.append(item[0])
            s_weights.append(float(item[1])/float(self.total()))

        x = random.random()

        for i, val in enumerate(s_seq):
            if x<=s_weights[i]:
                return val
            x-=s_weights[i]



class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """

        "*** YOUR CODE HERE ***"

        if noisyDistance == None and jailPosition == ghostPosition:
            return 1
        elif noisyDistance == None and jailPosition != ghostPosition:
            return 0
        elif noisyDistance != None and jailPosition ==ghostPosition:
            return 0


        obs = busters.getObservationProbability(noisyDistance, manhattanDistance(pacmanPosition,ghostPosition))
        return obs

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.
        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.
        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.
        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        # 1. pacman receives reading
        # 2. this function gets called
        # 3. belief is updated at that particular position

        # we collect evidence for each position and update our belief, so this method shouldn't return anything
        # self.beliefs is a DiscreteDistribution with keys for each legal position and associated weights
        # pacman's current position can change, but this position is given
        # self.allPositions: POSSIBLE ghost positions + jail position
        # before ANY READINGS, pacman believes chosts could be anywhere with P(pos) = 1/num positions
        # pacman then receives a reading, a sensor value (like "ok so what's around me?")
        # pacman must then update its self.beliefs = DiscreteDistribution() ("ok, I see the distances to each ghost, now I think each pos has each ghost with X probability")

        # each position has two states: contains ghost, doesn't contain ghost

        # iterate through every single position and update its distribution based on evidence after every reading
        # readings are abstracted by other methods though. this method will simply recompute the distribution once more evidence is collected
        # the goal of this method is to use 'observations' from the pacman, from the reading, to then iterate through every state, and update...
        # ...the likelihood of a ghost being on that square P(ghost = true | mahattan, pacman position)? WRONG!
        # we have a general idea of where the ghosts are via manhattan distance, but we don't know exactly where.
        

        # self.getObservationProb(noisyDistance, pacmanPosition, ghostPosition, jailPosition) returns P(noisyDistance | pacmanPosition, ghostPosition)
        # What is the probability of noisyDistance being within some sort of acceptable range given pacmanPosition and ghostPosition?

        # There's an inference module for each ghost, as stated in the pdf. 
            # self.index is the index of the ghost a particular inference module is tracking, unique for each ghost.
            # initialized with ghost agent
            # distances stores noisy distances to all ghosts on screen
            # self.index - 1 accesses a particular ghost

        observedDistance = observation
        pacmanPos = gameState.getPacmanPosition() # we get pacman's position

        jailPos = self.getJailPosition() # we get jail position

        for position in self.allPositions: # for every single position on this board bc self.allPositions is legalPos + jailPos
            probability = self.getObservationProb(observedDistance, pacmanPos, position, jailPos) # prediction step as said in the book: compute the probability of P(noisyDistance | pacmanPosition, ghostPosition)
            self.beliefs[position] *= probability # self.beliefs[position] = self.beliefs[position] * P(position) (update * prediction)

        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.
        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.

        Your agent has access to the action distribution for the ghost through self.getPositionDistribution.
        In order to obtain the distribution over new positions for the ghost, given its previous position,
        use this line of code:

        newPosDist = self.getPositionDistribution(gameState, oldPos)

        Where oldPos refers to the previous ghost position.
        newPosDist is a DiscreteDistribution object, where for each position p in self.allPositions, newPosDist[p] is the probability
        that the ghost is at position p at time t + 1, given that the ghost is at position oldPos at time t

        """
        "*** YOUR CODE HERE ***"
        # pacman isn't observing the ghost here at all.
        # instead, pacman's beliefs are updated purely over time based, instead, on gameState variables (how much time has passed + valid ghost movements)

        # kind of a copy and paste of observeUpdate? nope

        # observedDistance = observation
        # pacmanPos = gameState.getPacmanPosition() # we get pacman's position

        # # jailPos = self.getJailPosition() # we get jail position
        # newPosDist = self.getPositionDistribution(gameState, pacmanPos)

        # for position in self.allPositions: # for every single position on this board bc self.allPositions is legalPos + jailPos
        #     probability = newPosDist[position]
        #     self.beliefs[position] *= probability # self.beliefs[position] = self.beliefs[position] * P(position) Bayes' rule

        # self.beliefs.normalize()

        # after a time step (from t to t + 1) has passed, what must predict where a ghost could be on the board.
        # we are quite literally using the ghost's transition models P(s'|s,a) ...
        # ...to feed pacman more information over time more information about where each ghost is

        # do this for now...
        updatedBeliefs = DiscreteDistribution() # actually we have to use a copy bc we want the calculations to be done only on the intial state of self.beliefs
        # if we don't do this, the calculations will kinda compound over time, leading to large errors

        for oldPos in self.allPositions: # for every single position on the board...
            ghostDist = self.getPositionDistribution(gameState, oldPos) # we obtain for this particular old position the new states possible the ghost
            # could go to in addition to their associated probabilities

            for newPos, prob in ghostDist.items(): # each new position the ghost could go has a probability associated with it
                updatedBeliefs[newPos] += self.beliefs[oldPos] * prob # for all available actions in that state for the ghost, we compute a sum that pretty
                # much computes the expectation of the ghost being in a given new position.

        updatedBeliefs.normalize() # normalize each position
        self.beliefs = updatedBeliefs


    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = [] # not a DiscreteDistribution()
        # getBeliefDistribution() takes the list of particles and converts it into a DiscreteDistribution obj not implemented
        "*** YOUR CODE HERE ***"
        # self.beliefs = DiscreteDistribution()
        # particles correspond to positions
        i = 0
        # gameState wasn't used for ExactInference, so we shouldn't need to use it here (pacman.py)
        positions = len(self.legalPositions)

        while i < self.numParticles: # we want to add x num particles given y board positions, but what if x > y?
            # if there are more particles x than board positions y, then we must cycle back through the board and 
            # continue to uniformly add these particles
            # simple iteration should work? there should only ever be a difference of 1 b/w each particle for every board position p(x_i) - 1 = p(x_i+1)
            # on any jth iteration through the entire board...
            # ...which is as uniform as it's going to get
            # so essentially we have to use mod operator to implement this behavior?
            self.particles.append(self.legalPositions[i % positions])
            i += 1

    def observeUpdate(self, observation, gameState):
        """
        Resample particles based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.

        This method constructs a weight distribution over self.particles where the weight of a
        particle is the probability of the observation given Pacman’s position and that particle location.
        Then, we resample from this weighted distribution to construct our new list of particles.

        You should again use the function self.getObservationProb to find the probability of an observation
        given Pacman’s position, a potential ghost position, and the jail position.
        The sample method of the DiscreteDistribution class will also be useful.
        As a reminder, you can obtain Pacman’s position using gameState.getPacmanPosition(), and the jail position using self.getJailPosition()
        

        """
        "*** YOUR CODE HERE ***"
        # belief[particle] = weight of particle = P(noise | pacman pos, particle's location)
        # 

        observedDistance = observation
        pacmanPos = gameState.getPacmanPosition() # we get pacman's position
        jailPos = self.getJailPosition() # we get jail position

        tmp = DiscreteDistribution() # distribution for particles and their respective weights
        
        for particle in self.particles: # each particle...
            probability = self.getObservationProb(observedDistance, pacmanPos, particle, jailPos) # compute weight: probability of P(noisyDistance | pacmanPosition, ghostPosition)
            tmp[particle] += probability # sum weight for each particle
        
        # do all particles have 0 weight?
        if tmp.total() == 0: # if the discretedistribtion values sum to 0 then we must reinitialize
            self.initializeUniformly(gameState)
        else: # if not
            # do resampling
            newParticles = [] # reinitialize the list of particles
            i = 0
            while i < self.numParticles: # for each particle in our curr list
                newParticles.append(tmp.sample()) # get a random particle (location), add it to new list
                i += 1

            tmp.normalize()
            self.particles = newParticles # set particle list equal to this list

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.

        As in the elapseTime method of the ExactInference class, you should use:

        newPosDist = self.getPositionDistribution(gameState, oldPos)

        This line of code obtains the distribution over new positions for the ghost, given its previous position (oldPos).
        The sample method of the DiscreteDistribution class will also be useful.


        """
        "*** YOUR CODE HERE ***"
        # prediction step, no different from what's mentioned in the book
        # particles move based on ghost transition model
        newParticles = [] # new list to arrange particles
        for part in self.particles: # we have to move each particle based on ghost distribution model
            ghostDist = self.getPositionDistribution(gameState, part) # this will return another DiscreteDistribution that maps K: new ghost position --> 
            # V: probability
            newParticles.append(ghostDist.sample())

        self.particles = newParticles

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        beliefs = DiscreteDistribution()
        for particle in self.particles: # for every particle
            if particle not in beliefs: # if it's not already a key in the DiscreteDistribution
                beliefs[particle] = 1 # add it with a value of 1
            else:
                beliefs[particle] += 1 # increment the number of particles for this position
        
        beliefs.normalize()
        return beliefs

class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"

        raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.

        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Resample particles based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.
        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.

        To loop over all the ghosts, use:
            for i in range(self.numGhosts):

        You can still obtain Pacman’s position using gameState.getPacmanPosition(), but to get the jail
        position for a ghost, use self.getJailPosition(i), since now there are multiple ghosts each with their own jail positions.

        As in the update method for the ParticleFilter class, you should again use the function self.getObservationProb
        to find the probability of an observation given Pacman’s position, a potential ghost position, and the jail position.
        The sample method of the DiscreteDistribution class will also be useful.

        """
        "*** YOUR CODE HERE ***"





        raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.

        As in the last question, you can loop over the ghosts using:
            for i in range(self.numGhosts):

        Then, assuming that i refers to the index of the ghost, to obtain the distributions over new positions
        for that single ghost, given the list (prevGhostPositions) of previous positions of all of the ghosts, use:

        newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])

        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            raiseNotDefined()





            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist