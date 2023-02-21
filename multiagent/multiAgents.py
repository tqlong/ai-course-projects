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

MIN_VALUE = float("-inf")
MAX_VALUE = float("inf")

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
        curPos = currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newGhostDistances = list(map(lambda state: manhattanDistance(state.getPosition(), newPos), newGhostStates))
        newFoodDistances = list(map(lambda food: manhattanDistance(food, newPos), newFood.asList()))

        newFoodDistances.append(float("inf"))

        closestFood = min(newFoodDistances, default=1e-6)
        closestGhost = min(newGhostDistances)

        score = successorGameState.getScore()

        capsuleList = successorGameState.getCapsules()
        # curCap = min(map(lambda capsule: manhattanDistance(capsule, curPos)), default=0)
        posCap = min(map(lambda capsule: manhattanDistance(capsule, newPos), capsuleList), default=0)

        if currentGameState.hasFood(newPos[0], newPos[1]):
            score += 100

        if closestGhost <= 2:
            score -= closestGhost * 20

        if action == Directions.STOP:
            score -= 50

        return score + 1.0 / (closestFood + posCap * 1.75)

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

        def _value(agentIndex, gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == gameState.getNumAgents():
                depth += 1
                if depth >= self.depth:
                    return self.evaluationFunction(gameState)
                return _max(0, gameState, depth)
            elif agentIndex == self.index:
                return _max(agentIndex, gameState, depth)
            else:
                return _min(agentIndex, gameState, depth)
        
        def _max(agentIndex, gameState, depth):
            # if depth >= self.depth:
            #     return self.evaluationFunction(gameState)
            val = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                val = max(val, _value(agentIndex + 1, successor, depth))
            
            return val
    
        def _min(agentIndex, gameState, depth):
            # if depth >= self.depth:
            #     return self.evaluationFunction(gameState)
            val = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                val = min(val, _value(agentIndex + 1, successor, depth))
            
            return val

        best_val = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, action)
            val = _value(self.index + 1, successor, 0)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def _value(agentIndex, gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == gameState.getNumAgents():
                depth += 1
                if depth >= self.depth:
                    return self.evaluationFunction(gameState)
                return _max(0, gameState, depth, alpha, beta)
            elif agentIndex == self.index:
                return _max(agentIndex, gameState, depth, alpha, beta)
            else:
                return _min(agentIndex, gameState, depth, alpha, beta)
        
        def _max(agentIndex, gameState, depth, alpha, beta):
            val = MIN_VALUE
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                val = max(val, _value(agentIndex + 1, successor, depth, alpha, beta))
                if val > beta:
                    return val

                alpha = max(alpha, val)
            return val
    
        def _min(agentIndex, gameState, depth, alpha, beta):
            val = MAX_VALUE
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                val = min(val, _value(agentIndex + 1, successor, depth, alpha, beta))
                if val < alpha:
                    return val

                beta = min(beta, val)
            return val

        best_val = MIN_VALUE
        best_action = None
        alpha, beta = MIN_VALUE, MAX_VALUE
        # a re-write of _max
        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, action)
            val = _value(self.index + 1, successor, 0, alpha, beta)
            if val > best_val:
                best_val = val
                best_action = action

            if best_val > beta:
                return best_action

            alpha = max(alpha, best_val)

        return best_action

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
        def _value(agentIndex, gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == gameState.getNumAgents():
                depth += 1
                if depth >= self.depth:
                    return self.evaluationFunction(gameState)
                return _max(0, gameState, depth)
            elif agentIndex == self.index:
                return _max(agentIndex, gameState, depth)
            else:
                return _expect(agentIndex, gameState, depth)
        
        def _max(agentIndex, gameState, depth):
            # if depth >= self.depth:
            #     return self.evaluationFunction(gameState)
            val = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                val = max(val, _value(agentIndex + 1, successor, depth))
            
            return val
    
        def _expect(agentIndex, gameState, depth):
            # if depth >= self.depth:
            #     return self.evaluationFunction(gameState)
            val = 0
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                val += _value(agentIndex + 1, successor, depth)
            
            return val / len(gameState.getLegalActions(agentIndex))

        best_val = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, action)
            val = _value(self.index + 1, successor, 0)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    closestFood = min(map(lambda food: manhattanDistance(food, curPos), foodList), default=1e-6)

    score = 0
    score -= closestFood

    # if ghost is close, very bad situation
    # the more ghost scared, the better 
    for ghost in newGhostStates:
        if ghost.scaredTimer > 0:
            score += 25
        else:
            score -= 100 ** (1.6 - manhattanDistance(ghost.getPosition(), curPos))
    
    _pow = 1.2 ** 10
    for i in range(currentGameState.getNumFood()):
        score -= _pow * 10
        _pow /= 1.2
    return  score + currentGameState.getScore() * 1.2

# Abbreviation
better = betterEvaluationFunction
