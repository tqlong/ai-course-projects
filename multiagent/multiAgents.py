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
        REMAIN_FOOD_COUNT_WEIGHT = 1000
        FOOD_DISTANCE_WEIGHT = 10
        DANGER_DISTANCE = 2
        GHOST_DANGER_WEIGHT = 5000
        TARGET_FOOD_NUM = 4
        WIN_SCORE = float('inf')

        score = successorGameState.getScore()
        foods = newFood.asList()
        numFood = len(foods)
   
        if numFood == 0:
            return WIN_SCORE
        score -= numFood * REMAIN_FOOD_COUNT_WEIGHT
        
        foodDistance = [manhattanDistance(newPos, food) for food in foods]
        foodDistance.sort()
        score -= min(foodDistance[:min(len(foodDistance), TARGET_FOOD_NUM)]) * FOOD_DISTANCE_WEIGHT
    
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        ghostDistances = [manhattanDistance(newPos, ghost) for ghost in ghostPositions]
        for i in range(len(ghostDistances)):
            if ghostDistances[i] < DANGER_DISTANCE and newScaredTimes[i] == 0:
                score -= GHOST_DANGER_WEIGHT * (DANGER_DISTANCE - ghostDistances[i])

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
        pacmanActions = gameState.getLegalActions(0)
 
        values = []
        for action in pacmanActions:
            v = self.value(gameState.generateSuccessor(0, action), self.depth, 1)
            values.append(v)
        
        return pacmanActions[values.index(max(values))]
    
    def value(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.minValue(gameState, depth, agentIndex)
        
    def maxValue(self, gameState, depth, agentIndex):
        v = -float('inf')
        for action in gameState.getLegalActions(agentIndex):
            v = max(v, self.value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
        return v
    
    def minValue(self, gameState, depth, agentIndex):
        v = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = min(v, self.value(gameState.generateSuccessor(agentIndex, action), depth - 1, 0))
            else:
                v = min(v, self.value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
        return v
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacmanActions = gameState.getLegalActions(0)
     
        beta = float('inf')
        alpha = -float('inf')
        values = []
        for action in pacmanActions:
            v = self.value(gameState.generateSuccessor(0, action), self.depth, alpha, beta, 1)
            if v > beta:
                return action
            alpha = max(alpha, v)
            values.append(v)
        
        return pacmanActions[values.index(max(values))]
    
    def value(self, gameState, depth, alpha, beta, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState, depth, alpha, beta, agentIndex)
        else:
            return self.minValue(gameState, depth, alpha, beta, agentIndex)
        
    def maxValue(self, gameState, depth, alpha, beta, agentIndex):
        v = -float('inf')
        for action in gameState.getLegalActions(agentIndex):
            v = max(v, self.value(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta, agentIndex + 1))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def minValue(self, gameState, depth, alpha, beta, agentIndex):
        v = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = min(v, self.value(gameState.generateSuccessor(agentIndex, action), depth - 1, alpha, beta, 0))
            else:
                v = min(v, self.value(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta, agentIndex + 1))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

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
        pacmanActions = gameState.getLegalActions(0)
         
        values = []
        for action in pacmanActions:
            v = self.value(gameState.generateSuccessor(0, action), self.depth, 1)
            values.append(v)
        
        return pacmanActions[values.index(max(values))]
        # util.raiseNotDefined()

    def value(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.expValue(gameState, depth, agentIndex)
        
    def maxValue(self, gameState, depth, agentIndex):
        v = -float('inf')
        for action in gameState.getLegalActions(agentIndex):
            v = max(v, self.value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
        return v
    

    def expValue(self, gameState, depth, agentIndex):
        v = 0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v += self.value(gameState.generateSuccessor(agentIndex, action), depth - 1, 0)
            else:
                v += self.value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
        return v / len(gameState.getLegalActions(agentIndex))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    food = currentGameState.getFood()
    foodList = food.asList()
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    NO_MORE_CAPSULES_SCORE = 1000
    FOOD_SCORE = 100
    CAPSULE_SCORE = 200
    GHOST_SCARED_SCORE = 10
    NEAR_GHOST_PENALTY = 1000
    DANGER_DISTANCE = 2

    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')
    
    if len(capsules) == 0:
        score += NO_MORE_CAPSULES_SCORE

    if pacmanPosition in foodList:
        score += FOOD_SCORE
    
    if pacmanPosition in capsules:
        score += CAPSULE_SCORE

    for food in foodList:
        score += 1.0 / manhattanDistance(pacmanPosition, food)
    
    for capsule in capsules:
        score += 1.0 / manhattanDistance(pacmanPosition, capsule)
    
    for ghostState in ghostStates:
        if ghostState.scaredTimer > 0:
            score += GHOST_SCARED_SCORE
        else:
            if manhattanDistance(pacmanPosition, ghostState.getPosition()) <= DANGER_DISTANCE:
                score -= NEAR_GHOST_PENALTY

    return score

# Abbreviation
better = betterEvaluationFunction
