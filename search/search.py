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


def generateMoves(node, parent):
    action = node[1]
    if action is None:
        return []
    return generateMoves(parent[node], parent) + [action]


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
    """
    "*** YOUR CODE HERE ***"

    parent = util.Counter()
    existed = set()
    queue = util.Stack()

    init_state = problem.getStartState()
    start_action = None
    start_cost = 0

    start_node = (init_state, start_action, start_cost)

    queue.push(start_node)

    while not queue.isEmpty():
        current_node = queue.pop()

        if (problem.isGoalState(current_node[0])):
            return generateMoves(current_node, parent)

        next_moves = problem.getSuccessors(current_node[0])

        if current_node[0] not in existed:
            existed.add(current_node[0])

            for move in next_moves:
                if move[0] not in existed:
                    queue.push(move)
                    parent[move] = current_node

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    parent = util.Counter()
    existed = set()
    queue = util.Queue()

    init_state = problem.getStartState()
    start_action = None
    start_cost = 0

    start_node = (init_state, start_action, start_cost)

    queue.push(start_node)

    while not queue.isEmpty():
        current_node = queue.pop()
        if (problem.isGoalState(current_node[0])):
            return generateMoves(current_node, parent)

        if current_node[0] not in existed:
            existed.add(current_node[0])
            for move in problem.getSuccessors(current_node[0]):
                if move[0] not in existed:
                    queue.push(move)
                    parent[move] = current_node

    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    queue = util.PriorityQueue()
    cost = util.Counter()
    existed = set()
    parent = util.Counter()

    init_state = problem.getStartState()
    start_action = None
    start_cost = 0
    cost[init_state] = start_cost

    queue.push((init_state, start_action), start_cost)

    while not queue.isEmpty():
        current_node = queue.pop()
        current_state = current_node[0]

        if (problem.isGoalState(current_state)):
            return generateMoves(current_node, parent)

        if (current_state not in existed):
            existed.add(current_state)
            for move in problem.getSuccessors(current_state):
                next_move_state = move[0]
                next_move_cost = move[2]
                if (next_move_state not in existed):
                    g = cost[current_state] + next_move_cost
                    next_node = (next_move_state, move[1])
                    queue.update(next_node, g)
                    cost[next_move_state] = g
                    parent[next_node] = current_node
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()
    cost = util.Counter()
    existed = set()
    parent = util.Counter()

    init_state = problem.getStartState()
    start_action = None
    start_cost = 0
    cost[init_state] = start_cost

    queue.push((init_state, start_action), start_cost)
    while not queue.isEmpty():
        current_node = queue.pop()
        current_state = current_node[0]

        if (problem.isGoalState(current_state)):
            return generateMoves(current_node, parent)

        if (current_state not in existed):
            existed.add(current_state)
            for move in problem.getSuccessors(current_state):
                next_move_state = move[0]
                next_move_cost = move[2]
                if (next_move_state not in existed):
                    g = cost[current_state] + next_move_cost
                    next_node = (next_move_state, move[1])
                    queue.update(next_node, g +
                                 heuristic(next_move_state, problem))
                    cost[next_move_state] = g
                    parent[next_node] = current_node
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
