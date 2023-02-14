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
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    

    fringe = util.Stack()
    visited_nodes = []
    fringe.push((problem.getStartState(), [], 1))

    while not fringe.isEmpty():
        node = fringe.pop()
        # print('node', node[0])
        state = node[0]
        direction = node[1]

        if problem.isGoalState(state): 
            # print('y', direction)
            return direction

        if state not in visited_nodes:
            visited_nodes.append(state)
            # tham cac dinh con
            children = problem.getSuccessors(state)
            # print('children', children)
            # print('parent action', direction)
            # duyet qua cac node con
            for child in children:
                # neu child chua duoc tham
                if child[0] not in visited_nodes:
                    # print('child1', child[1])
                    # print('child action', direction + [child[1]])
                    newDirect = direction + [child[1]]
                    fringe.push((child[0], newDirect, 1))

                
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    # tao mang chua cac nodes
    fringe = util.Queue()
    visited_nodes = []
    fringe.push((problem.getStartState(), [], 1))

    # lap khi mang khong rong
    while not fringe.isEmpty():
        # lay ra phan tu dau stack
        node = fringe.pop()
        # print('node', node[0])
        state = node[0]
        direction = node[1]

        # kiem tra xem da dung dich chua
        if problem.isGoalState(state): 
            # print('y', direction)
            return direction

        if state not in visited_nodes:
            visited_nodes.append(state)
            # tham cac dinh con
            children = problem.getSuccessors(state)
            # print('children', children)
            # print('parent action', direction)
            # duyet qua cac node con
            for child in children:
                # neu child chua duoc tham
                if child[0] not in visited_nodes:
                    # print('child1', child[1])
                    # print('child action', direction + [child[1]])
                    newDirect = direction + [child[1]]
                    fringe.push((child[0], newDirect, 1))
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    fringe = util.PriorityQueue()
    visited_nodes_cost = {}
    fringe.push((problem.getStartState(), [], 0), 0)

    while not fringe.isEmpty():
        node = fringe.pop()
        # print('node', node)
        state = node[0]
        direction = node[1]
        cost = node[2]
        # print('state', state, direction, cost)

        if (state not in visited_nodes_cost) or (cost < visited_nodes_cost[state]):
            visited_nodes_cost[state] = cost

            if problem.isGoalState(state): 
                # print('y', direction)
                return direction
            
            else:
                children = problem.getSuccessors(state)
                for child in children:
                    newDirection = direction + [child[1]]
                    newCost = cost + child[2]
                    fringe.push((child[0], newDirection, newCost), newCost)

    return direction
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    visited_nodes = []
    nulHeu = heuristic(problem.getStartState(), problem)
    fringe.push((problem.getStartState(), [], 0), nulHeu)

    while not fringe.isEmpty():
        node = fringe.pop()
        # print('node', node)
        state = node[0]
        direction = node[1]
        cost = node[2]
        # print('state', state, direction, cost)

        if problem.isGoalState(state): 
            # print('y', direction)
            return direction

        if (state not in visited_nodes):
            visited_nodes.append(state)
            children = problem.getSuccessors(state)
            for child in children:
                if (child[0] not in visited_nodes):
                    child_action = direction + [child[1]]
                    child_cost = problem.getCostOfActions(child_action) + heuristic(child[0], problem)
                    fringe.push((child[0], child_action, 0), child_cost)

    return direction
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
