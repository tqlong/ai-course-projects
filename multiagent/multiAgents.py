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
        return self.maxval(gameState, 0, 0)[0]

    def minimax(self, gameState, agentIndex, depth):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.maxval(gameState, agentIndex, depth)[1]
        else:
            return self.minval(gameState, agentIndex, depth)[1]

    def maxval(self, gameState, agentIndex, depth):
        bestAction = ("max",-float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = max(bestAction,succAction,key=lambda x:x[1])
        return bestAction

    def minval(self, gameState, agentIndex, depth):
        bestAction = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = min(bestAction,succAction,key=lambda x:x[1])
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
        return self.alphabetaSearch(gameState, 0, self.depth, -1e9, 1e9)[1]

    def alphabetaSearch(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            ret = self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            ret = self.alphasearch(gameState, agentIndex, depth, alpha, beta)
        else:
            ret = self.betasearch(gameState, agentIndex, depth, alpha, beta)
        return ret

    def alphasearch(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        max_score, max_action = -1e9, Directions.STOP
        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.alphabetaSearch(successor_game_state, next_agent, next_depth,\
                                             alpha, beta)[0]
            if new_score > max_score:
                max_score, max_action = new_score, action
            if new_score > beta:
                return new_score, action
            alpha = max(alpha, max_score)
        return max_score, max_action

    def betasearch(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        min_score, min_action = 1e9, Directions.STOP
        for action in actions:
            successor_game_state = gameState.generateSuccessor(agentIndex, action)
            new_score = self.alphabetaSearch(successor_game_state, next_agent, next_depth,\
                                             alpha, beta)[0]
            if new_score < min_score:
                min_score, min_action = new_score, action
            if new_score < alpha:
                return new_score, action
            beta = min(beta, min_score)
        return min_score, min_action

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
        def expectimax(gameState, agentIndex, depth):
            if agentIndex == gameState.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(gameState)
                else: 
                    return expectimax(gameState, 0, depth + 1)
            else:
                moves = gameState.getLegalActions(agentIndex)
                if len(moves) == 0:
                    return self.evaluationFunction(gameState)
                next = [expectimax(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth)
                        for move in moves]
                if agentIndex == 0:
                    return max(next)
                else:
                    return sum(next)/len(next)
        return max(gameState.getLegalActions(0), key=lambda x: expectimax(gameState.generateSuccessor(0, x), 1, 1))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodList = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestFood = min([manhattanDistance(newPos, food) for food in newFoodList]) if len(newFoodList) > 0 else 0
    distancesToGhost = [manhattanDistance(ghost.getPosition(), newPos) if ghost.scaredTimer == 0 else -10 for ghost in newGhostStates]
    closestGhost = min(distancesToGhost)

    result = currentGameState.getScore() - 6/(closestGhost + 1) - (closestFood/3)
    return result

# Abbreviation
better = betterEvaluationFunction
