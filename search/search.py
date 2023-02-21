# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    global action
    "*** YOUR CODE HERE ***"
    f = util.Stack()
    choices = []

    base = problem.getStartState()
    start = (base, [])

    f.push(start)

    while f.isEmpty() == False:
        state, action = f.pop()

        if state not in choices:
            choices.append(state)

            if problem.isGoalState(state):
                return action
            print(state)
            winrace = problem.getSuccessors(state)
            for state_i, move, _ in winrace:
                tmp = [move]
                newNode = (state_i, action + tmp)
                f.push(newNode)

    return action


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    global action
    "*** YOUR CODE HERE ***"
    f = util.Queue()
    choices = []

    base = problem.getStartState()
    start = (base, [], 0)

    f.push(start)

    while not f.isEmpty():
        # print(1)
        state, action, cost = f.pop()
        # print('state', state)
        if state not in choices:
            choices.append(state)

            if problem.isGoalState(state):
                return action

            winrace = problem.getSuccessors(state)
            for state_i, move, cost_i in winrace:
                tmp = action + [move]
                newcost = cost + cost_i
                newNode = (state_i, tmp, newcost)
                f.push(newNode)

    return action


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """Search the shallowest nodes in the search tree first."""
    global action
    "*** YOUR CODE HERE ***"
    f = util.PriorityQueue()
    choices = {}

    base = problem.getStartState()
    start = (base, [], 0)

    f.push(start, 0)

    while not f.isEmpty():
        # print(1)
        state, action, cost = f.pop()
        # print('state', state)
        if (state not in choices) or (cost < choices[state]):
            choices[state] = cost
            if problem.isGoalState(state):
                return action

            winrace = problem.getSuccessors(state)
            for state_i, move, cost_i in winrace:
                tmp = action + [move]
                newcost = cost + cost_i
                newNode = (state_i, tmp, newcost)
                f.update(newNode, newcost)
    return action


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    global action
    f = util.PriorityQueue()
    choices = []

    base = problem.getStartState()
    start = (base, [], 0)

    f.push(start, 0)

    while not f.isEmpty():
        state, action, cost = f.pop()

        choices.append((state, cost))

        if problem.isGoalState(state):
            return action

        winrace = problem.getSuccessors(state)
        for state_i, move, cost_i in winrace:
            tmp = action + [move]
            newcost = cost + cost_i
            newNode = (state_i, tmp, newcost)

            moved_bool = False
            for choice in choices:
                # examine each explored node tuple
                state_j, cost_j = choice

                if (state_i == state_j) and (newcost >= cost_j):
                    moved_bool = True

            # if this successor not explored, put on frontier and explored list
            if not moved_bool:
                f.push(newNode, newcost + heuristic(state_i, problem))
                choices.append((state_i, newcost))

    return action


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
