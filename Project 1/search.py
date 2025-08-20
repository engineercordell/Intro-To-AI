# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    
    frontier = util.Stack()
    frontier.push([problem.getStartState(), []]) # frontier <-- stack with problem.initial as elem
    # print("This is the stack: ", frontier)
    result = [] # result <-- actions to execute for the agent
    visited = set() # visited <-- explored nodes

    while not frontier.isEmpty(): # loop until state space is exhausted, essentially..
        node = frontier.pop() # node <-- highest priority elem from frontier (node holds a state essentially with (pos, action, cost) params)
        if problem.isGoalState(node[0]): # goal test: if the goal is reached...
            result = node[1] # result <-- long list of actions...
            break
        if node[0] not in visited:
            visited.add(node[0]) # visited <-- last element on the stack (frontier)
            # result.append(problem.getSuccessors(node))
            for w in problem.getSuccessors(node[0]):
                if w[0] not in visited:
                    frontier.push([w[0], node[1] + [w[1]]]) 

    return result # the only way this could return is if a solution isn't found, therefore empty list? (nah)

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    # literally just copy and paste dfs but change frontier data structure lol
    frontier = util.Queue()
    frontier.push((problem.getStartState(), [])) # frontier <-- queue with problem.initial as elem
    # print("This is the stack: ", frontier)
    result = [] # result <-- actions to execute for the agent
    visited = [] # visited <-- explored nodes (wanted this to be a set, but this was causing me headaches w/ q5 bc the state was a list of (state, []) and not
    # just (state). So when the algorithm checked to see if it was visited, python would complain that 'state' was unhashable..

    while not frontier.isEmpty(): # loop until state space is exhausted, essentially..
        state, actions = frontier.pop() # node <-- highest priority elem from frontier (node holds a state essentially with (pos, action, cost) params)
        if problem.isGoalState(state): # goal test: if the goal is reached...
            result = actions # result <-- long list of actions...
            break
        if state not in visited:
            visited.append(state) # visited <-- first elem added (frontier)
            # result.append(problem.getSuccessors(node))
            for new_state, action, cost in problem.getSuccessors(state):
                if new_state not in visited:
                    frontier.push([new_state, actions + [action]]) 

    return result # the only way this could return is if a solution isn't found, therefore empty list? (nah)


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"

    """ COUPLE NOTES FOR MY LEARNING: 
    1. We need to explicitly push 3 params (state, actionm cost) onto the PQ instead of 2 (state, action) on each iteration bc otherwise
    there wouldn't be a way to access cost associated for each node on the PQ (i.e. when you do frontier.push([state, action]), total_cost), 
    how would you get access to total_cost for each node? You can't.
    2. state, actions, total_cost = node --> Python syntax that allows you to get extract elems from a list into separate variables in their respective order:
    elem1, elem2, ... , elemN = set
    3. Point 2, but you can do the same thing in for loops, pretty cool
    4. Tried to copy and paste my DFS + BFS code here, but doing that doesn't work bc for UCS, you must keep track of the total path cost, not just the cost
    from the last state to the next. Interestingly, test case ucs0 seems to check for this, but all the other test cases were working just fine. I guess it
    makes sense, bc the pacman search is much more local and you would never really backtrack.  
    """

    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), [], 0), 0) # frontier <-- PQ with problem.initial as elem
    # print("This is the stack: ", frontier)
    result = [] # result <-- actions to execute for the agent
    visited = [] # visited <-- explored nodes

    while not frontier.isEmpty(): # loop until state space is exhausted, essentially..
        node = frontier.pop() # node <-- highest priority elem from frontier (node holds a state essentially with (pos, action, cost) params)
        state, actions, total_cost = node

        if problem.isGoalState(state): # goal test: if the goal is reached...
            result = actions # result <-- long list of actions...
            break
        if state not in visited:
            visited.append(state) # visited <-- highest prior elem (frontier)
            # result.append(problem.getSuccessors(node))
            for new_node, action, cost in problem.getSuccessors(state):
                if new_node not in visited:
                    frontier.push([new_node, actions + [action], total_cost + cost], total_cost + cost)
                    # print("Cost:", w[2]) 

    return result # the only way this could return is if a solution isn't found, therefore empty list? (nah)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    """frontier = util.PriorityQueue()
    frontier.push([problem.getStartState(), []], 0) # frontier <-- PQ with problem.initial as elem
    # print("This is the stack: ", frontier)
    result = [] # result <-- actions to execute for the agent
    visited = set() # visited <-- explored nodes

    while not frontier.isEmpty(): # loop until state space is exhausted, essentially..
        node = frontier.pop() # node <-- highest priority elem from frontier (node holds a state essentially with (pos, action, cost) params)
        
        if problem.isGoalState(node[0]): # goal test: if the goal is reached...
            result = node[1] # result <-- long list of actions...
            break
        if node[0] not in visited:
            visited.add(node[0]) # visited <-- highest prior elem (frontier)
            # result.append(problem.getSuccessors(node))
            for w in problem.getSuccessors(node[0]):
                if w[0] not in visited:
                    frontier.push([w[0], node[1] + [w[1]]], w[2] + heuristic(w[0], problem))
                    # print("Cost:", w[2]) 

    return result # the only way this could return is if a solution isn't found, therefore empty list? (nah)
    """

    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), [], 0), 0) # frontier <-- PQ with problem.initial as elem
    # print("This is the stack: ", frontier)
    result = [] # result <-- actions to execute for the agent
    visited = [] # visited <-- explored nodes

    while not frontier.isEmpty(): # loop until state space is exhausted, essentially..
        node = frontier.pop() # node <-- highest priority elem from frontier (node holds a state essentially with (pos, action, cost) params)
        state, actions, total_cost = node

        if problem.isGoalState(state): # goal test: if the goal is reached...
            result = actions # result <-- long list of actions...
            break
        if state not in visited:
            visited.append(state) # visited <-- highest prior elem (frontier)
            # result.append(problem.getSuccessors(node))
            for new_node, action, cost in problem.getSuccessors(state):
                if new_node not in visited:
                    frontier.push([new_node, actions + [action], total_cost + cost], cost + total_cost + heuristic(new_node, problem))
                    # print("Cost:", w[2]) 

    return result # the only way this could return is if a solution isn't found, therefore empty list? (nah)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
