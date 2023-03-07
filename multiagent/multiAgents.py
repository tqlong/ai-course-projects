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
        ghostScore = float('inf');
        distToClostFood = float('inf');
        def distToPacman(obj):
            return manhattanDistance(obj, newPos)
        def solveGhost(ghost):
            dist = distToPacman(ghost.getPosition())
            if ghost.scaredTimer > dist:
                return float('inf')
            if dist <= 1:
                return float('-inf')
            return 0
        for ghost in newGhostStates:
            ghostScore = min(solveGhost(ghost), ghostScore)
        for food in newFood.asList():
            distToClostFood =  min(distToPacman(food), distToClostFood)
        foodScore = 1.0 / (1.0 + distToClostFood)
        return successorGameState.getScore() + ghostScore + foodScore

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
        maxVal = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            val = self.getMin(gameState.generateSuccessor(0, action))
            if val is not None and val > maxVal:
                maxVal = val
                best_action = action
        return best_action
    
    def getMax(self, gameState, depth = 0, agentIndex = 0):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)
        maxVal = float('-inf')
        for action in actions:
            value = self.getMin(gameState.generateSuccessor(agentIndex, action),depth, 1)
            if value is not None and value > maxVal:
                maxVal = value
        return maxVal

    def getMin(self, gameState, depth = 0, agentIndex = 1):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)
        minVal = float('inf')
        for action in actions:
            if agentIndex == gameState.getNumAgents() - 1:
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
            else:
                value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if value is not None and value < minVal:
                minVal = value
        return minVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxVal, bestAction = self.getMax(gameState)
        return bestAction
        util.raiseNotDefined()
    def getMax(self, gameState, depth = 0, agentIndex = 0, alpha = float('-inf'), beta = float('inf')):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions)==0:
            return self.evaluationFunction(gameState), None
        maxVal = None
        bestAction = None
        for action in actions:
            value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, 1, alpha, beta)[0]
            if value is not None and (maxVal == None or value > maxVal):
                maxVal = value
                bestAction = action
            if value is not None and value > beta:
                return value, action
            if value is not None and value > alpha:
                alpha = value
        return maxVal, bestAction
    
    def getMin(self, gameState, depth = 0, agentIndex = 0, alpha = -float('inf'), beta = float('inf')):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions)==0:
            return self.evaluationFunction(gameState), None
        minVal = None
        bestAction = None
        for action in actions:
            if agentIndex >= gameState.getNumAgents() - 1:
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth+1, 0, alpha, beta)[0]
            else:
                value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1, alpha, beta)[0]
            if value is not None and (minVal == None or value < minVal):
                minVal = value
                bestAction = action
            if value is not None and value < alpha:
                return value, action
            if value is not None and value < beta:
                beta = value
        return minVal, bestAction
    
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
        return self.getMax(gameState)
        util.raiseNotDefined()
    
    def getMax(self, gameState, depth = 0, agentIndex = 0):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions)==0:
            return self.evaluationFunction(gameState)
        maxVal = None
        bestAction = None
        for action in actions:
            value = self.getExp(gameState.generateSuccessor(agentIndex, action), depth, 1)
            if value is not None and (maxVal == None or value > maxVal):
                maxVal = value
                bestAction = action 
        if depth is 0 and agentIndex is 0:
            return bestAction
        else:
            return maxVal
        
    def getExp(self, gameState, depth = 0, agentIndex = 0):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions)==0:
            return self.evaluationFunction(gameState) 
        totalUtil = 0
        numActions = len(actions)
        for action in actions:
            if agentIndex >= gameState.getNumAgents() - 1:
                totalUtil += self.getMax(gameState.generateSuccessor(agentIndex, action), depth+1, 0)
            else:
                totalUtil += self.getExp(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1)
        return totalUtil / float(numActions)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    INF = float('inf')  # Infinite value
    WEIGHT_FOOD = 10.0  # Food base value
    WEIGHT_GHOST = -10.0  # Ghost base value
    WEIGHT_SCARED_GHOST = 100.0  # Scared ghost base value

    score = currentGameState.getScore()

    # Evaluate closest food
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(distancesToFoodList) > 0:
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        score += WEIGHT_FOOD

    # Evaluate distance to ghosts
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:  # If scared, add points
                score += WEIGHT_SCARED_GHOST / distance
            else:  # If not, decrease points
                score += WEIGHT_GHOST / distance
        else:
            return -INF  # Pacman is dead at this point

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
