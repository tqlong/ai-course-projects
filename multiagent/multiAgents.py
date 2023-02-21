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


from util import Queue, manhattanDistance
from game import Directions
import random
import util

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        val = 500

        if (successorGameState.isWin()):
            return 99999
        if (successorGameState.isLose()):
            return -99999

        # Seeking for food, i want it to be BFS
        currentFoodPos = currentGameState.getFood().asList()
        nearestFood = min([manhattanDistance(newPos, food)
                          for food in currentFoodPos])

        val -= nearestFood * 10

        # Run from ghost if distance is relatively small
        currentGhostStates = currentGameState.getGhostStates()
        nearestGhostLater = min([manhattanDistance(
            newPos, ghostState.getPosition()) for ghostState in newGhostStates])
        nearestGhostCurrent = min([manhattanDistance(
            newPos, ghostState.getPosition()) for ghostState in currentGhostStates])

        if (nearestGhostLater < 2):
            val -= 50

        if (nearestGhostLater < nearestGhostCurrent):
            val += 100
        else:
            val -= 50

        # Which causes me LOSE most of the time
        if (action == Directions.STOP):
            val -= 10

        return val


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        MAX_AGENT = 0
        PACMAN = 0
        FIRST_GHOST = 1
        LAST_GHOST = gameState.getNumAgents() - 1

        def value(gameState, depth, agentIndex):
            # NOTE:  score the leaves with evaluationFunction
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if (agentIndex == MAX_AGENT):
                return max_value(gameState, depth)
            else:
                return min_value(gameState, depth, agentIndex)

        def max_value(gameState, depth):
            val = -99999
            for action in gameState.getLegalActions(0):
                successorGameState = gameState.generateSuccessor(0, action)
                val = max(val, value(successorGameState, depth, FIRST_GHOST))
            return val

        def min_value(gameState, depth, agentIndex):
            val = 99999
            # FIXME: Forgot to make all ghosts moving, and call successor with only 1 ghost
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                # All ghosts have moved
                if (agentIndex == LAST_GHOST):
                    val = min(val, value(successor, depth + 1, MAX_AGENT))
                else:
                    # In case 1 ghost move, the game reach leaf -> have to take min value
                    val = min(val, value(successor, depth, agentIndex + 1))

            return val

        finalAction = Directions.STOP
        finalValue = -99999

        # forgot to add 0
        for action in gameState.getLegalActions(PACMAN):
            nextState = gameState.generateSuccessor(PACMAN, action)
            val = value(nextState, 0, FIRST_GHOST)
            finalValue, finalAction = max(
                (finalValue, finalAction),
                (val, action))

        return finalAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        MAX_AGENT = 0
        PACMAN = 0
        FIRST_GHOST = 1
        LAST_GHOST = gameState.getNumAgents() - 1

        def value(gameState, depth, agentIndex, alpha, beta):
            # NOTE:  score the leaves with evaluationFunction
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if (agentIndex == MAX_AGENT):
                return max_value(gameState, depth, alpha, beta)
            else:
                return min_value(gameState, depth, agentIndex, alpha, beta)

        def max_value(gameState, depth, alpha, beta):
            val = -99999
            for action in gameState.getLegalActions(0):
                successorGameState = gameState.generateSuccessor(0, action)
                val = max(val, value(successorGameState,
                                     depth, FIRST_GHOST, alpha, beta))
                if (val > beta):
                    return val
                alpha = max(alpha, val)
            return val

        def min_value(gameState, depth, agentIndex, alpha, beta):
            val = 99999
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                # All ghosts have moved
                if (agentIndex == LAST_GHOST):
                    val = min(val, value(successor, depth + 1,
                                         MAX_AGENT, alpha, beta))
                else:
                    # In case 1 ghost move, the game reach leaf
                    val = min((val), value(successor, depth,
                                           agentIndex + 1, alpha, beta))

                if (val < alpha):
                    return val

                beta = min(beta, val)

            return val

        finalAction = Directions.STOP
        finalValue = -99999

        alpha = -99999
        beta = 99999

        # NOTE: Wrong pseudo-code: it's either greater or less, never stop when equal
        for action in gameState.getLegalActions(PACMAN):
            nextState = gameState.generateSuccessor(PACMAN, action)
            val = value(nextState, 0, FIRST_GHOST, alpha, beta)
            finalValue, finalAction = max(
                (finalValue, finalAction),
                (val, action))
            if (finalValue > beta):
                break
            alpha = max(alpha, finalValue)

        return finalAction


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
        MAX_AGENT = 0
        PACMAN = 0
        FIRST_GHOST = 1
        LAST_GHOST = gameState.getNumAgents() - 1

        def value(gameState, depth, agentIndex):
            # NOTE:  score the leaves with evaluationFunction
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if (agentIndex == MAX_AGENT):
                return max_value(gameState, depth)
            else:
                return min_value(gameState, depth, agentIndex)

        def max_value(gameState, depth):
            val = -99999
            for action in gameState.getLegalActions(0):
                successorGameState = gameState.generateSuccessor(0, action)
                val = max(val, value(successorGameState,
                                     depth, FIRST_GHOST))
            return val

        def min_value(gameState, depth, agentIndex):
            val = 99999
            numVal = 0.0
            totalVal = 0.0
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                # All ghosts have moved
                if (agentIndex == LAST_GHOST):
                    val = min(val, value(successor, depth + 1,
                                         MAX_AGENT))
                    totalVal += value(successor, depth + 1, MAX_AGENT)
                    numVal += 1
                else:
                    # In case 1 ghost move, the game reach leaf
                    val = min(val, value(successor, depth, agentIndex + 1))

            # NOTE: Expected Value
            if numVal != 0:
                val = totalVal / numVal
            return val

        finalAction = Directions.STOP
        finalValue = -99999

        # NOTE: Wrong pseudo-code: it's either greater or less, never stop when equal
        for action in gameState.getLegalActions(PACMAN):
            nextState = gameState.generateSuccessor(PACMAN, action)
            val = value(nextState, 0, FIRST_GHOST)
            finalValue, finalAction = max(
                (finalValue, finalAction),
                (val, action))

        return finalAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    position = currentGameState.getPacmanPosition()
    foodPosition = currentGameState.getFood().asList()
    numFoodRemain = len(foodPosition)
    ghostStates = currentGameState.getGhostStates()
    remainingCapsules = len(currentGameState.getCapsules())

    val = 0

    foodDis = [manhattanDistance(position, food) for food in foodPosition]
    if (numFoodRemain != 0):
        val -= min(foodDis)

    val -= 20 * remainingCapsules
    val -= 4 * numFoodRemain

    val += currentGameState.getScore()

    nearestGhostLater = min([manhattanDistance(
        position, ghostState.getPosition()) for ghostState in ghostStates])
    nearestGhostCurrent = min([manhattanDistance(
        position, ghostState.getPosition()) for ghostState in ghostStates])

    if (nearestGhostLater < 2):
        val -= 50

    if (nearestGhostLater < nearestGhostCurrent):
        val += 100
    else:
        val -= 50
    return val


# Abbreviation
better = betterEvaluationFunction
