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


import sys
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

        "* YOUR CODE HERE *"
        return successorGameState.getScore()

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

    You do not need to make any changes here, but you can if you want to
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
    def maxValue(self, gameState, depth):
        pacmanActions = gameState.getLegalActions(0)

        if depth > self.depth or gameState.isWin() or not pacmanActions:
            return self.evaluationFunction(gameState)

        actionCosts = []
        for action in pacmanActions:
            successor = gameState.generateSuccessor(0, action)
            actionCosts.append((self.minValue(successor, 1, depth), action))
            
        return max(actionCosts)

    def minValue(self, gameState, agentIndex, depth):
        ghostActions = gameState.getLegalActions(agentIndex)

        if not ghostActions or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == gameState.getNumAgents() - 1:
            return min([self.maxValue(successor, depth + 1) for successor in [gameState.generateSuccessor(agentIndex, action) for action in ghostActions]])
        else:
            return min([self.minValue(successor, agentIndex + 1, depth) for successor in [gameState.generateSuccessor(agentIndex, action) for action in ghostActions]])


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
        "* YOUR CODE HERE *"
        return self.maxValue(gameState, 1)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self, gameState, depth, alpha, beta):  
        pacmanActions = gameState.getLegalActions(0)

        if depth > self.depth or gameState.isWin() or not pacmanActions:
            return self.evaluationFunction(gameState), Directions.STOP

        value = -sys.maxint
        bestAction = Directions.STOP

        for action in pacmanActions:
            successor = gameState.generateSuccessor(0, action)
            cost = self.minValue(successor, 1, depth, alpha, beta)[0]
            if cost > value:
                value = cost
                bestAction = action
            if value > beta:
                return value, bestAction
            alpha = max(alpha, value)

        return value, bestAction

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        ghostActions = gameState.getLegalActions(agentIndex)

        if not ghostActions or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        value = sys.maxint
        bestAction = Directions.STOP

        for action in ghostActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                cost = self.maxValue(successor, depth + 1, alpha, beta)[0]
            else:
                cost = self.minValue(successor, agentIndex + 1, depth, alpha, beta)[0]

            if value > cost:
                value = cost
                bestAction = action
            if alpha > value:
                return value, bestAction
            beta = min(beta, value)

        return value, bestAction

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "* YOUR CODE HERE *"
        return self.maxValue(gameState, 1, -sys.maxint, sys.maxint)[1]

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
        "* YOUR CODE HERE *"
    def expectiMax(totalAgents, indexAgent, currentGameState, currentDepth):
        if totalAgents == indexAgent:
            indexAgent = 0
            currentDepth += 1
        if currentDepth == self.depth:
            return self.evaluationFunction(currentGameState)

        possibleActions = currentGameState.getLegalActions(indexAgent)
        if not possibleActions:
            return self.evaluationFunction(currentGameState)

        if indexAgent == 0:
            maxPossibilites = []
            for possibleAction in possibleActions:
                successorGameState = currentGameState.generateSuccessor(indexAgent, possibleAction)
                valueMini = expectiMax(totalAgents, indexAgent + 1, successorGameState, currentDepth)
                maxPossibilites.append([valueMini, possibleAction])
            if currentDepth == 0:
                return max(maxPossibilites)[1]
            else:
                return max(maxPossibilites)[0]

        elif indexAgent > 0:
            valueRandom = 0
            equalProbability = 1.0 / len(possibleActions)

            for possibleAction in possibleActions:
                successorGameState = currentGameState.generateSuccessor(indexAgent, possibleAction)
                valueExpectiMaxNext = expectiMax(totalAgents, indexAgent + 1, successorGameState, currentDepth)
                valueRandom += equalProbability * valueExpectiMaxNext
            return valueRandom
        
        totalAgents = gameState.getNumAgents()
        valueReturn = expectiMax(totalAgents, 0, gameState, 0)
        return valueReturn

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "* YOUR CODE HERE *"
    gameScore = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    maxFoodDistanceScore = max(map(lambda x: 1.0 / manhattanDistance(x, pacmanPos), currentGameState.getFood().asList()) + [0])

    return currentGameState.getScore() + maxFoodDistanceScore

# Abbreviation
better = betterEvaluationFunction