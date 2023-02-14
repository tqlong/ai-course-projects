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


def extractTrace(pathTrace, goalState):
    _goalState = goalState
    path = []

    while _goalState is not None:
        path.insert(0, _goalState)

        _goalState = pathTrace[_goalState] if _goalState in pathTrace else None

    return path

def transformPath(path, start):
    from game import Directions

    location = start

    for i in range(0, len(path)):
        nextLocation = path[i]

        if nextLocation[0] > location[0]:
            path[i] = Directions.EAST
        elif nextLocation[0] < location[0]:
            path[i] = Directions.WEST
        elif nextLocation[1] > location[1]:
            path[i] = Directions.NORTH
        elif nextLocation[1] < location[1]:
            path[i] = Directions.SOUTH

        location = nextLocation

    path.remove(start)

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

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
    "*** YOUR CODE HERE ***"

    frontier = util.Stack()
    frontier.push(problem.getStartState())
    pathTrace = {}
    visited = {}
    solutionFound = False
    goalState = None

    while frontier.isEmpty() is False:
        testState = frontier.pop()

        if testState in visited:
            continue;

        visited[testState] = True

        if problem.isGoalState(testState):
            goalState = testState
            solutionFound = True
            break

        successors = problem.getSuccessors(testState)

        for successor in successors:
            nextState = successor[0]

            if nextState not in visited:
                frontier.push(nextState)
                pathTrace[nextState] = testState


    if solutionFound is False:
        return []

    path = extractTrace(pathTrace, goalState)

    transformPath(path, problem.getStartState())

    return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    frontier = util.Queue()
    frontier.push(problem.getStartState())
    pathTrace = {}
    visited = {}
    solutionFound = False
    goalState = None

    while frontier.isEmpty() is False:
        testState = frontier.pop()

        if testState in visited:
            continue;

        visited[testState] = True

        if problem.isGoalState(testState):
            goalState = testState
            solutionFound = True
            break

        successors = problem.getSuccessors(testState)

        for successor in successors:
            nextState = successor[0]

            if nextState not in visited:
                frontier.push(nextState)
                pathTrace[nextState] = testState

    if solutionFound is False:
        return []

    path = extractTrace(pathTrace, goalState)

    transformPath(path, problem.getStartState())

    return path


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), 0), 0)
    pathTrace = {}
    visited = {}
    solutionFound = False
    goalState = None

    while frontier.isEmpty() is False:
        (testState, testStateCost) = frontier.pop()

        if testState in visited:
            continue;

        visited[testState] = True

        if problem.isGoalState(testState):
            goalState = testState
            solutionFound = True
            break

        successors = problem.getSuccessors(testState)

        for successor in successors:
            nextState = successor[0]
            nextStateCost = successor[2] + testStateCost

            if nextState not in visited:
                frontier.push((nextState, nextStateCost), -nextStateCost)
                pathTrace[nextState] = testState

    if solutionFound is False:
        return []

    path = extractTrace(pathTrace, goalState)

    transformPath(path, problem.getStartState())

    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    startState = problem.getStartState();
    openList = set([startState])
    closedList = set([])
    pathTrace = {}
    g = {}
    solutionFound = False
    goalState = None

    g[startState] = 0

    while len(openList) != 0:
        testState = None

        for state in openList:
            if testState == None or g[state] + heuristic(state, problem) < g[testState] + heuristic(testState, problem):
                testState = state

        if testState == None:
            break

        if problem.isGoalState(testState):
            solutionFound = True
            goalState = testState
            break

        successors = problem.getSuccessors(testState)

        for successor in successors:
            (nextState, _, actionCost) = successor
            
            if nextState not in openList and nextState not in closedList:
                openList.add(nextState)
                pathTrace[nextState] = testState
                g[nextState] = g[testState] + actionCost
            else:
                if g[nextState] > g[testState] + actionCost:
                    g[nextState] = g[testState] + actionCost
                    pathTrace[nextState] = testState

                    if nextState in closedList:
                        closedList.remove(nextState)
                        openList.add(nextState)

        openList.remove(testState)
        closedList.add(testState)

    if solutionFound is False:
        return []

    path = extractTrace(pathTrace, goalState)

    transformPath(path, problem.getStartState())

    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
