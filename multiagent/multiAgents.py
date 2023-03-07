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
        from util import manhattanDistance

        score = successorGameState.getScore()
        foodList = newFood.asList()
        if len(foodList) > 0:
            minDistance = min([manhattanDistance(newPos, food) for food in foodList])
            score += 1 / minDistance
        
        ghostPos = successorGameState.getGhostPositions()
        if len(ghostPos) > 0:
            minDistance = min([manhattanDistance(newPos, ghost) for ghost in ghostPos])
            maxScaredTime = max(newScaredTimes)
            if minDistance <= 1:
                score -= 100
            else:
                if maxScaredTime > 0:
                    score += 1 / minDistance
                else:
                    score -= 1 / minDistance

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

        def max_value(gameState, depth, agentIndex):
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                v = max(v, minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
            return v
        
        def min_value(gameState, depth, agentIndex):
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
            return v

        def minimax(gameState, depth, agentIndex):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth or len(gameState.getLegalActions(agentIndex)) == 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max_value(gameState, depth, agentIndex)
            else:
                return min_value(gameState, depth, agentIndex)
            
        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            score = minimax(gameState.generateSuccessor(0, action), 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
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

        def max_value(gameState, depth, agentIndex, alpha, beta):
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                v = max(v, minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(gameState, depth, agentIndex, alpha, beta):
            v = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                v = min(v, minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
        def minimax(gameState, depth, agentIndex, alpha, beta):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth or len(gameState.getLegalActions(agentIndex)) == 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max_value(gameState, depth, agentIndex, alpha, beta)
            else:
                return min_value(gameState, depth, agentIndex, alpha, beta)
            
        bestScore = float('-inf')
        bestAction = Directions.STOP
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            score = minimax(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
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

        def max_value(gameState, depth, agentIndex):
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                v = max(v, minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
            return v
        
        def min_value(gameState, depth, agentIndex):
            v = 0
            for action in gameState.getLegalActions(agentIndex):
                v += minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            return v / len(gameState.getLegalActions(agentIndex))

        def minimax(gameState, depth, agentIndex):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if gameState.isWin() or gameState.isLose() or depth == self.depth or len(gameState.getLegalActions(agentIndex)) == 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return max_value(gameState, depth, agentIndex)
            else:
                return min_value(gameState, depth, agentIndex)
            
        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            score = minimax(gameState.generateSuccessor(0, action), 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    "*** YOUR CODE HERE ***"
    from util import manhattanDistance

    score = currentGameState.getScore()
    foodList = currentFood.asList()
    if len(foodList) > 0:
        minDistance = min([manhattanDistance(currentPos, food) for food in foodList])
        score += 1 / minDistance
    
    ghostPos = currentGameState.getGhostPositions()
    if len(ghostPos) > 0:
        minDistance = min([manhattanDistance(currentPos, ghost) for ghost in ghostPos])
        maxScaredTime = max(currentScaredTimes)
        if minDistance <= 1:
            score -= 100
        else:
            if maxScaredTime > 0:
                score += 1 / minDistance
            else:
                score -= 1 / minDistance

    return score

# Abbreviation
better = betterEvaluationFunction
