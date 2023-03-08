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
        nearestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if nearestGhost:
            disGhost = -10/nearestGhost
        else:
            disGhost = -1000
        foodList = newFood.asList()
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
        return (-2*nearestGhost) + disGhost - (100*len(foodList))

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
        def minimax(state, agentIndex, depth):
            if agentIndex == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return minimax(state, 0, depth+1)
            else:
                action = state.getLegalActions(agentIndex)
                if len(action) == 0:
                    return self.evaluationFunction(state)
                value = (minimax(state.generateSuccessor(agentIndex, newAction), agentIndex+1, depth) for newAction in action)
                if agentIndex == 0:
                    return max(value)
                else:
                    return min(value)
        return max(gameState.getLegalActions(0),key=lambda x: minimax(gameState.generateSuccessor(0, x), 1, 1))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(agentIndex, depth, state, a, b):
            v = float("-inf")
            for newState in state.getLegalActions(agentIndex):
                v = max(v, alphabeta(1, depth, state.generateSuccessor(agentIndex, newState), a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def minValue(agentIndex, depth, state, a, b):
            v = float("inf")
            nextAgent = agentIndex + 1 
            if state.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            for newState in state.getLegalActions(agentIndex):
                v = min(v, alphabeta(nextAgent, depth, state.generateSuccessor(agentIndex, newState), a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v
    
        def alphabeta(agentIndex, depth, state, a, b):
            if state.isLose() or state.isWin() or depth == self.depth: 
                return self.evaluationFunction(state)
            if agentIndex == 0: 
                return maxValue(agentIndex, depth, state, a, b)
            else: 
                return minValue(agentIndex, depth, state, a, b)
        
        bestValue = float("-inf")
        action = None
        a = float("-inf")
        b = float("inf")
        for state in gameState.getLegalActions(0):
            value = alphabeta(1, 0, gameState.generateSuccessor(0, state), a, b)
            if value > bestValue:
                bestValue = value
                action = state
            if bestValue > b:
                return bestValue
            a = max(a, bestValue)

        return action

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
        def maxValue(gameState, depth):
            if len(gameState.getLegalActions(0))==0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            bestValue = float("-inf")
            action = None
            for newState in gameState.getLegalActions(0):
                value = expectimax(gameState.generateSuccessor(0, newState), 1, depth)[0]
                if (bestValue < value): bestValue, action = value, newState
            return (bestValue, action)

        def expectimax(gameState, agentIndex, depth):
            if len(gameState.getLegalActions(agentIndex)) == 0:
              return (self.evaluationFunction(gameState), None)
            bestValue = 0
            action = None
            for newState in gameState.getLegalActions(agentIndex):
                if (agentIndex == gameState.getNumAgents()-1):
                    value = maxValue(gameState.generateSuccessor(agentIndex, newState), depth+1)[0]
                else:
                    value = expectimax(gameState.generateSuccessor(agentIndex, newState), agentIndex+1, depth)[0]

                bestValue += value/len(gameState.getLegalActions(agentIndex))
            return (bestValue, action)

        return maxValue(gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFoodGrid = currentGameState.getFood()
    newFoodList = newFoodGrid.asList()
    ghostStates = currentGameState.getGhostStates()
    value = currentGameState.getScore()
    if len(newFoodList) > 0:
        nearestFood = min(newFoodList, key=lambda x: manhattanDistance(newPos, x))
        disFood = manhattanDistance(newPos, nearestFood)
        value += 1.0/disFood
    value += len(newFoodList)
    disNearestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in ghostStates])
    if disNearestGhost > 0:
        value += 10.0/disNearestGhost
    return value
# Abbreviation
better = betterEvaluationFunction
