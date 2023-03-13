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
        newFoodList = newFood.asList()
        nearestFoodDistance = -1
        for food in newFoodList:
            foodDistance = util.manhattanDistance(newPos, food)
            if foodDistance < nearestFoodDistance or nearestFoodDistance == -1:
                nearestFoodDiatance = foodDistance

        totalGhostDistance = 1
        proximityGhosts = 0
        for ghostState in successorGameState.getGhostPositions():
            ghostDistance = util.manhattanDistance(newPos, ghostState)
            totalGhostDistance += ghostDistance
            if ghostDistance <= 1:
                proximityGhosts += 1
        return successorGameState.getScore() + (1/float(nearestFoodDistance)) - (1/float(totalGhostDistance)) - proximityGhosts

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
        def miniMax(totalAgents, indexAgent, currentGameState, currentDepth):
            if totalAgents == indexAgent:
                indexAgent = 0  
                currentDepth += 1
            if currentDepth == self.depth:
                return self.evaluationFunction(currentGameState)

            possibleActions = currentGameState.getLegalActions(indexAgent)
            if not possibleActions:
                return self.evaluationFunction(currentGameState)

            if indexAgent == 0:
                maxPossibilities = []
                for possibleAction in possibleActions:
                    successorGameState = currentGameState.generateSuccessor(indexAgent, possibleAction)
                    valueMini = miniMax(totalAgents, indexAgent + 1, successorGameState, currentDepth)
                    maxPossibilities.append([valueMini, possibleAction])

                if currentDepth == 0:
                    return max(maxPossibilities)[1]
                else:
                    return max(maxPossibilities)[0]

            elif indexAgent > 0:
                minPossibilities = []
                for possibleAction in possibleActions:
                    successorGameState = currentGameState.generateSuccessor(indexAgent, possibleAction)
                    nextValueMini = miniMax(totalAgents, indexAgent + 1, successorGameState, currentDepth)
                    minPossibilities.append(nextValueMini)
                return min(minPossibilities)

        totalAgents = gameState.getNumAgents()
        valueReturn = miniMax(totalAgents, 0, gameState, 0)
        return valueReturn

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "* YOUR CODE HERE *"
        def alphaBetaMiniMax(totalAgents, indexAgent, currentGameState, currentDepth, alpha, beta):
            if totalAgents == indexAgent:
                indexAgent = 0
                currentDepth += 1

            if currentDepth == self.depth:
                return self.evaluationFunction(currentGameState)

            possibleActions = currentGameState.getLegalActions(indexAgent)
            if not possibleActions:
                return self.evaluationFunction(currentGameState)

            # Pacman
            if indexAgent == 0:
                maxPossibilities = []
                bestValue = float("-inf")
                for possibleAction in possibleActions:
                    successorGameState = currentGameState.generateSuccessor(indexAgent, possibleAction)
                    valueMini = alphaBetaMiniMax(totalAgents, indexAgent + 1, successorGameState, currentDepth, alpha, beta)
                    maxPossibilities.append([valueMini, possibleAction])
                    bestValue = max(bestValue, valueMini)
                    if bestValue > beta:
                        return bestValue
                    alpha = max(alpha, bestValue)

                if currentDepth == 0:
                    return max(maxPossibilities)[1]
                else:
                    return max(maxPossibilities)[0]

            # Ghost
            elif indexAgent > 0:
                minPossibilities = []
                bestValue = float("inf")
                for possibleAction in possibleActions:
                    successorGameState = currentGameState.generateSuccessor(indexAgent, possibleAction)
                    valueMiniNext = alphaBetaMiniMax(totalAgents, indexAgent + 1, successorGameState, currentDepth, alpha, beta)
                    minPossibilities.append(valueMiniNext)
                    bestValue = min(bestValue, valueMiniNext)

                    if bestValue < alpha:
                        return bestValue

                    beta = min(beta,bestValue)
                return min(minPossibilities)
        totalAgents = gameState.getNumAgents()
        valueReturn = alphaBetaMiniMax(totalAgents, 0, gameState, 0, float("-inf"), float("inf"))
        return valueReturn

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

            # Pacman
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

            #Ghost
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
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    nearestFoodDistance = -1

    for food in newFoodList:
        foodDistance = util.manhattanDistance(newPos, food)
        if foodDistance < nearestFoodDistance or nearestFoodDistance == -1:
            nearestFoodDistance = foodDistance

    totalGhostDistance = 1
    proximityGhosts = 0
    for ghostState in currentGameState.getGhostPositions():
        ghostDistance = util.manhattanDistance(newPos, ghostState)
        totalGhostDistance += ghostDistance
        if ghostDistance <= 1:
            proximityGhosts += 1

    currentCapsules = currentGameState.getCapsules()
    numberCurrentCapsules = len(currentCapsules)
    return currentGameState.getScore() + (1/float(nearestFoodDistance)) - (1/float(totalGhostDistance)) - proximityGhosts - numberCurrentCapsules

# Abbreviation
better = betterEvaluationFunction