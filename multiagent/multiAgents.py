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
        foodList = newFood.asList()
        if len(foodList) == 0:
            return successorGameState.getScore()
        elif len(foodList) == 1:
            nearestFoodDistance = util.manhattanDistance(newPos, foodList[0])
        else:
            foodDistance = [util.manhattanDistance(newPos, food) for food in foodList]
            nearestFoodDistance = min(foodDistance)
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        nearestGhostDistance = min([util.manhattanDistance(newPos, ghost) for ghost in ghostPositions])
        capsules = currentGameState.getCapsules()
        capsuleDistance = [util.manhattanDistance(newPos, capsule) for capsule in capsules]
        if len(capsuleDistance) == 0:
            capsuleDistance = [float('inf')]
        nearestCapsuleDistance = min(capsuleDistance)
        score = successorGameState.getScore()
        if nearestGhostDistance <= 1:
            score -= 1000
        elif nearestCapsuleDistance < nearestGhostDistance:
            score += 100
        else:
            score += 10 / nearestFoodDistance
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
        def maxValue(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minValue(successor, depth, agentIndex + 1))
            return v

        def minValue(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, maxValue(successor, depth - 1, 0))
                else:
                    v = min(v, minValue(successor, depth, agentIndex + 1))
            return v

        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        bestValue = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minValue(successor, self.depth, 1)
            if value > bestValue:
                bestAction = action
                bestValue = value
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                v = max(v, minValue(successor, depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, maxValue(successor, depth + 1, alpha, beta))
                else:
                    v = min(v, minValue(successor, depth, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        bestScore = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minValue(successor, 0, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, agentIndex, currentDepth):
        """
        Returns the expectimax value of a given gameState.
        """
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            maxEval = float("-inf")
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                eval = self.expectimax(successor, agentIndex + 1, currentDepth)
                maxEval = max(maxEval, eval)
            return maxEval
        else:
            expectedEval = 0
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    expectedEval += self.expectimax(successor, 0, currentDepth + 1)
                else:
                    expectedEval += self.expectimax(successor, agentIndex + 1, currentDepth)
            return expectedEval / len(legalActions)
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        bestEval = float("-inf")
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            eval = self.expectimax(successor, 1, 0)
            if eval > bestEval:
                bestEval = eval
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    numFood = len(foodList)
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    if numFood > 0:
        closestFood = min(foodList, key=lambda x: manhattanDistance(pacmanPosition, x))
        distanceToFood = manhattanDistance(pacmanPosition, closestFood)
        score += 1.0 / distanceToFood
    score += numFood
    closestGhostDistance = min([manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates])
    if closestGhostDistance > 0:
        score += 10.0 / closestGhostDistance

    return score

# Abbreviation
better = betterEvaluationFunction
