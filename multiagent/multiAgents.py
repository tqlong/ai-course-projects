# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_distances = []
        for food in newFood.asList():
            distance_to_food = manhattanDistance(newPos, food)
            food_distances.append(distance_to_food)

        if food_distances:
            min_distance_food = min(food_distances)
        else:
            min_distance_food = 0

        scared_ghost_distances = []
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                distance_to_ghost = manhattanDistance(newPos, ghost.getPosition())
                scared_ghost_distances.append(distance_to_ghost)

        if scared_ghost_distances:
            min_distance_ghost = min(scared_ghost_distances)
        else:
            min_distance_ghost = -1

        score = successorGameState.getScore()
        score -= min_distance_food / 2
        if min_distance_ghost != -1:
            score -= 6 / (min_distance_ghost + 2)

        return score

    
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        ghosts = gameState.getNumAgents() - 1
        ghostIdx = list(range(1, ghosts + 1))
        def is_terminal(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def min_value(state, depth, ghost):
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            value = float('inf')
            for action in state.getLegalActions(ghost):
                if ghost == ghostIdx[-1]:
                    value = min(value, max_value(state.generateSuccessor(ghost, action), depth + 1))
                else:
                    value = min(value, min_value(state.generateSuccessor(ghost, action), depth, ghost + 1))

            return value

        def max_value(state, depth):
            if is_terminal(state, depth):
                return self.evaluationFunction(state)

            value = float('-inf')
            for action in state.getLegalActions(0):
                value = max(value, min_value(state.generateSuccessor(0, action), depth, ghostIdx[0]))

            return value

        res = [(action, min_value(gameState.generateSuccessor(0, action), 0, ghostIdx[0])) for action in gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return max_value(gameState, depth, agentIndex, alpha, beta)
            else:
                return min_value(gameState, depth, agentIndex, alpha, beta)

        def max_value(gameState, depth, agentIndex, alpha, beta):
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, minimax(successor, depth, agentIndex+1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(gameState, depth, agentIndex, alpha, beta):
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents()-1:
                    v = min(v, minimax(successor, depth-1, 0, alpha, beta))
                else:
                    v = min(v, minimax(successor, depth, agentIndex+1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        v = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            temp = minimax(successor, self.depth, 1, alpha, beta)
            if temp > v:
                v = temp
                bestAction = action
            alpha = max(alpha, v)
        return bestAction


        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max_value(gameState, depth, agentIndex)
            else:
                return exp_value(gameState, depth, agentIndex)

        def max_value(gameState, depth, agentIndex):
            v = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                v = max(v, expectimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
            return v


        def exp_value(gameState, depth, agentIndex):
            v = 0
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v += expectimax(gameState.generateSuccessor(agentIndex, action), depth - 1, 0)
                else:
                    v += expectimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            return v / len(gameState.getLegalActions(agentIndex))

        v = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            temp = expectimax(gameState.generateSuccessor(0, action), self.depth, 1)
            if temp > v:
                v = temp
                bestAction = action
        return bestAction


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    food_value = 10
    ghost_value = -10
    scared_ghost_value = 100
    score = currentGameState.getScore()

    # Compute the minimum distance to food
    if food_list:
        min_dist_food = min(manhattanDistance(pacman_pos, food) for food in food_list)
        score += food_value / min_dist_food
    else:
        score += food_value

    # Compute the score contribution of each ghost
    for ghost in ghost_states:
        dist_ghost = manhattanDistance(pacman_pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            score += scared_ghost_value / dist_ghost
        elif dist_ghost > 0:
            score += ghost_value / dist_ghost
        else:
            score += ghost_value

    return score


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
