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
        minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList], default=float('inf'))
        return successorGameState.getScore() + 1.0 / minFoodDistance

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
        legalMoves = gameState.getLegalActions()
        successors = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.minimax(successor, 1, self.depth) for successor in successors]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]
    
    def minimax(self, state, agentIndex, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        legalMoves = state.getLegalActions(agentIndex)
        successors = [state.generateSuccessor(agentIndex, action) 
                    for action in legalMoves]
        scores = [self.minimax(successor, nextAgent, nextDepth)
                  for successor in successors]

        if agentIndex == 0:  # pacman turn
            selectScore = max
        else:  # ghost turn
            selectScore = min

        return selectScore(scores)

# class AlphaBetaAgent(MultiAgentSearchAgent):
#     """
#       Your minimax agent with alpha-beta pruning (question 3)
#     """

#     def maxValue(self, gameState, agent, depth, alpha, beta):
#         bestValue = float("-inf")
#         for action in gameState.getLegalActions(agent):
#             successor = gameState.generateSuccessor(agent, action)
#             v = self.minimax(successor, agent + 1, depth, alpha, beta)
#             bestValue = max(bestValue, v)
#             if depth == 1 and bestValue == v: self.action = action
#             if bestValue > beta: return bestValue
#             alpha = max(alpha, bestValue)
#         return bestValue

#     def minValue(self, gameState, agent, depth, alpha, beta):
#         bestValue = float("inf")
#         for action in gameState.getLegalActions(agent):
#             successor = gameState.generateSuccessor(agent, action)
#             v = self.minimax(successor, agent + 1, depth, alpha, beta)
#             bestValue = min(bestValue, v)
#             if bestValue < alpha: return bestValue
#             beta = min(beta, bestValue)
#         return bestValue

#     def minimax(self, gameState, agent=0, depth=0,
#                 alpha=float("-inf"), beta=float("inf")):

#         agent = agent % gameState.getNumAgents()

#         if gameState.isWin() or gameState.isLose():
#             return self.evaluationFunction(gameState)

#         if agent == 0:
#             if depth < self.depth:
#                 return self.maxValue(gameState, agent, depth+1, alpha, beta)
#             else:
#                 return self.evaluationFunction(gameState)
#         else:
#             return self.minValue(gameState, agent, depth, alpha, beta)

#     def getAction(self, gameState):
#         """
#           Returns the minimax action using self.depth and self.evaluationFunction
#         """
#         "*** YOUR CODE HERE ***"
#         self.minimax(gameState)
#         return self.action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # legalMoves = gameState.getLegalActions()
        # successors = (gameState.generateSuccessor(0, action) for action in legalMoves)
        # scores = [self.minimax(successor, 1, self.depth, alpha=float("-inf"), beta=float("inf")) for successor in successors]
        # bestScore = max(scores)
        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices) # Pick randomly among the best)
        # return legalMoves[chosenIndex]
        self.minimax(gameState, 0,  self.depth, alpha=float("-inf"), beta=float("inf"))
        return self.action

    def maxValue(self, state, agentIndex, depth, alpha, beta):
        maxV = float("-inf")
        legalMoves = state.getLegalActions(agentIndex)
        # successors = [state.generateSuccessor(agentIndex, action) 
        #             for action in legalMoves]
        for action in legalMoves:
            successor = state.generateSuccessor(agentIndex, action)
            v = self.minimax(successor, agentIndex + 1, depth, alpha, beta)
            maxV = max(maxV, v)
            if depth == self.depth and maxV == v: self.action = action
            if maxV > beta: return maxV
            alpha = max(alpha, maxV)
        return maxV

    def minValue(self, state, agentIndex, depth, alpha, beta):
        minV = float("inf")
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = (depth - 1) if nextAgent == 0 else depth

        legalMoves = state.getLegalActions(agentIndex)
        # successors = [state.generateSuccessor(agentIndex, action) 
        #             for action in legalMoves]
        for action in legalMoves:
            successor = state.generateSuccessor(agentIndex, action)
            v = self.minimax(successor, nextAgent, nextDepth, alpha, beta)
            minV = min(minV, v)
            if minV < alpha: return minV
            beta = min(beta, minV)
        return minV

    def minimax(self, state, agentIndex, depth, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agentIndex == 0:
            return self.maxValue(state, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(state, agentIndex, depth, alpha, beta)

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
        legalMoves = gameState.getLegalActions()
        successors = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.expectimax(successor, 1, self.depth) for successor in successors]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def expectimax(self, state, agentIndex, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth - 1 if nextAgent == 0 else depth

        legalMoves = state.getLegalActions(agentIndex)
        successors = [state.generateSuccessor(agentIndex, action) for action in legalMoves]
        scores = [self.expectimax(successor, nextAgent, nextDepth)
                  for successor in successors]
        if agentIndex == 0: 
            return max(scores)
        else:  
            return sum(scores) / len(legalMoves)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    infinity = float('inf')
    position = currentGameState.getPacmanPosition()
    currentScore = currentGameState.getScore()
    # ghostStates = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()

    # if currentGameState.isWin() or currentGameState.isLose(): return currentScore

    if currentGameState.isWin(): return infinity
    if currentGameState.isLose(): return -infinity

    minFood = min([manhattanDistance(position, food) for food in foodList], 
                default=infinity)

    minCapsule = min([manhattanDistance(position, capsule) for capsule in capsuleList], 
                    default=infinity)

    return 10.0/minFood + 1.0/minCapsule + currentScore

# Abbreviation
better = betterEvaluationFunction
