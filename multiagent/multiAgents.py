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

        import math;

        minDisToFood = math.inf
        minDisToGhost = math.inf
        score = 0
        penalty = 0

        for food in newFood.asList():
            minDisToFood = min(minDisToFood, manhattanDistance(newPos, food))
        for ghost in newGhostStates:
            minDisToGhost = min(minDisToGhost, manhattanDistance(newPos, ghost.getPosition()))

        if action == 'Stop':
            penalty -= 50

        score = minDisToGhost / minDisToFood

        return successorGameState.getScore() + penalty + score
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
        PACMAN = 0

        def maxAgent(state, depth):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            bestScore = float("-inf")
            score = bestScore
            bestAction = Directions.STOP
            for action in actions:
                score = minAgent(state.generateSuccessor(PACMAN, action), depth, 1)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            if depth == 0:
                return bestAction
            else:
                return bestScore

        def minAgent(state, depth, agent):
            if state.isLose() or state.isWin():
                return state.getScore()
            nextAgent = agent + 1
            if agent == state.getNumAgents() - 1:
                nextAgent = PACMAN
            actions = state.getLegalActions(agent)
            bestScore = float("inf")
            score = bestScore
            for action in actions:
                if nextAgent == PACMAN:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(agent, action))
                    else:
                        score = maxAgent(state.generateSuccessor(agent, action), depth + 1)
                else:
                    score = minAgent(state.generateSuccessor(agent, action), depth, nextAgent)
                if score < bestScore:
                    bestScore = score
            return bestScore

        return maxAgent(gameState, 0)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        PACMAN = 0

        def maxAgent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            bestScore = float("-inf")
            score = bestScore
            bestAction = Directions.STOP
            for action in actions:
                score = minAgent(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                alpha = max(alpha, bestScore)
                if bestScore > beta:
                    return bestScore
            if depth == 0:
                return bestAction
            else:
                return bestScore

        def minAgent(state, depth, agent, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            nextAgent = agent + 1
            if agent == state.getNumAgents() - 1:
                nextAgent = PACMAN
            actions = state.getLegalActions(agent)
            bestScore = float("inf")
            score = bestScore
            for action in actions:
                if nextAgent == PACMAN:  # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(agent, action))
                    else:
                        score = maxAgent(state.generateSuccessor(agent, action), depth + 1, alpha, beta)
                else:
                    score = minAgent(state.generateSuccessor(agent, action), depth, nextAgent, alpha, beta)
                if score < bestScore:
                    bestScore = score
                beta = min(beta, bestScore)
                if bestScore < alpha:
                    return bestScore
            return bestScore

        return maxAgent(gameState, 0, float("-inf"), float("inf"))

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

        def maxValue(gameState, depth, agent):
            value = float('-inf')
            for action in gameState.getLegalActions(agent):
                value = max(value, minimax(gameState.generateSuccessor(agent, action), depth, agent + 1))
            return value

        def minValue(gameState, depth, agent):
            value = 0
            for action in gameState.getLegalActions(agent):
                value += minimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
            return value / len(gameState.getLegalActions(agent))

        def minimax(gameState, depth, agent):
            if agent == gameState.getNumAgents():
                agent = 0
                depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth or len(gameState.getLegalActions(agent)) == 0:
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maxValue(gameState, depth, agent)
            else:
                return minValue(gameState, depth, agent)

        bestScore = float('-inf')
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            score = minimax(gameState.generateSuccessor(0, action), 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    score = currentGameState.getScore()
    foodList = currentFood.asList()
    if len(foodList) > 0:
        minDisToFood = min([manhattanDistance(currentPos, food) for food in foodList])
        score += 1 / minDisToFood

    ghostPos = currentGameState.getGhostPositions()
    if len(ghostPos) > 0:
        minDisToGhost = min([manhattanDistance(currentPos, ghost) for ghost in ghostPos])
        maxScaredTime = max(currentScaredTimes)
        if minDisToGhost <= 1:
            score -= 150
        else:
            score += maxScaredTime / minDisToGhost

    return score

# Abbreviation
better = betterEvaluationFunction
