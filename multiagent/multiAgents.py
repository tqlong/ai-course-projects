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
        # focusing on eating food.When ghost near don't go,
        newFood = successorGameState.getFood().asList()
        minFoodist = float("inf")
        for food in newFood:
            minFoodist = min(minFoodist, manhattanDistance(newPos, food))

        # avoid ghost if too close
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
        # reciprocal
        return successorGameState.getScore() + 1.0 /minFoodist

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
        return self.maxAgent(gameState, 1,0)[1]

    def maxAgent(self, gameState, depth, agentIndex =0):
        value = -float('inf')
        
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
    
        legalMoves = gameState.getLegalActions()

        players = gameState.getNumAgents()
        V = []
        for i in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, i)
            V.append(self.minAgent(successor, depth, 1, players)[0])
            value = max(value, self.minAgent(successor, depth, 1, players)[0])
        bestScore = value
        bestIndices = [index for index in range(len(V)) if V[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return bestScore, legalMoves[chosenIndex]

    def minAgent(self, gameState, depth, agentIndex, players):
        value = +float('inf')
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        V = []
        p = players-1
        if agentIndex < p:
            legalMoves = gameState.getLegalActions(agentIndex)
            for i in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, i)
                V.append(self.minAgent(successor, depth, agentIndex+1, players-1)[0])
                value = min(value, self.minAgent(successor, depth, agentIndex+1, players-1)[0])
            bestScore = value
            bestIndices = [index for index in range(len(V)) if V[index] == bestScore]
            chosenIndex = random.choice(bestIndices)
            return bestScore, legalMoves[chosenIndex]
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            for i in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, i)
                V.append(self.maxAgent(successor,depth+1,  0)[0])
                value = min(value, self.maxAgent(successor, depth+1, 0)[0])
            bestScore = value
            bestIndices = [index for index in range(len(V)) if V[index] == bestScore]
            chosenIndex = random.choice(bestIndices)
            return bestScore, legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.maxval(gameState, agentIndex, depth, alpha, beta)[1]
        else:
            return self.minval(gameState, agentIndex, depth, alpha, beta)[1]

    def maxval(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("max",-float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.alphabeta(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
            bestAction = max(bestAction,succAction,key=lambda x:x[1])

            # Prunning
            if bestAction[1] > beta: return bestAction
            else: alpha = max(alpha,bestAction[1])

        return bestAction

    def minval(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.alphabeta(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
            bestAction = min(bestAction,succAction,key=lambda x:x[1])

            # Prunning
            if bestAction[1] < alpha: return bestAction
            else: beta = min(beta, bestAction[1])

        return bestAction

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
        def expectimax(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            # If it's the player's turn, return the maximum value and corresponding action
            if agentIndex == 0:
                return max_value(state, agentIndex, depth)
            # If it's a ghost's turn, return the expected value
            else:
                return expect_value(state, agentIndex, depth)

        def max_value(state, agentIndex, depth):
            v = float('-inf')
            bestAction = None

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                successor_value, _ = expectimax(successor, 1, depth)
                if successor_value > v:
                    v = successor_value
                    bestAction = action

            # Return the maximum value and corresponding action
            return v, bestAction

        def expect_value(state, agentIndex, depth):
            v = 0
            numActions = len(state.getLegalActions(agentIndex))

            # If the ghost has no legal actions, return the score of the current state and no action
            if numActions == 0:
                return self.evaluationFunction(state), None

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    successor_value, _ = expectimax(successor, 0, depth + 1)
                else:
                    successor_value, _ = expectimax(successor, agentIndex + 1, depth)
                v += successor_value

            return v / numActions, None

        # Call the expectimax function with the initial game state, the player agent (agentIndex = 0), and a depth of 0
        # Return the corresponding action
        _, bestAction = expectimax(gameState, 0, 0)
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentScore = currentGameState.getScore()
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()

    # Calculate the distance to the nearest food
    if currentFood:
        nearestFood = min(currentFood, key=lambda food: util.manhattanDistance(currentPos, food))
        foodDistance = util.manhattanDistance(currentPos, nearestFood)
    else:
        foodDistance = 0

    # Get the number of remaining food pellets
    remainingFood = len(currentFood)

    # Calculate the evaluation score as a weighted sum of the current score, the distance to the nearest food, and the number of remaining food pellets
    evaluationScore = currentScore - foodDistance - 10 * remainingFood

    return evaluationScore

# Abbreviation
better = betterEvaluationFunction
