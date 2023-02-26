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

        ghost = 1e9;
        for state in newGhostStates:
            ghost = min(ghost, manhattanDistance(state.getPosition(), newPos))

        if(ghost <= 1): return -1e9;

        numFood = successorGameState.getNumFood();
        prevNumFood = currentGameState.getNumFood();

        distanceFood = 1e9;
        for food in newFood.asList(): 
            distanceFood = min(distanceFood, manhattanDistance(newPos, food))

        if len(newFood.asList()) == 0: distanceFood = 0
        
        return -distanceFood + successorGameState.getScore() + 100*(prevNumFood - numFood)

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
        return value(gameState, 0, self.depth, self.evaluationFunction)[1]
        
def value(gameState, agentIndex, depth, function):
    if depth == 0 or gameState.isLose() or gameState.isWin():
        return (function(gameState),Directions.STOP)
    
    if agentIndex == 0: 
        compare = MAX
        res = -1e9,None
    else: 
        compare = MIN
        res = 1e9, None

    return sol(gameState, agentIndex, depth, function, value, compare, res)
    
def sol(gameState, agentIndex, depth, function, func, compare, res):
    actions = gameState.getLegalActions(agentIndex)
    agent, next_depth = agentIndex + 1, depth

    if agentIndex + 1 == gameState.getNumAgents():
        agent, next_depth = 0, depth - 1

    for action in actions:
        state = gameState.generateSuccessor(agentIndex, action)
        score = func(state, agent, next_depth, function)[0], action
        if compare(score, res):
            res = score
    
    return res

def MAX(x, y):
    return x[0] > y[0]

def MIN(x, y):
    return x[0] < y[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return alphaBetaValue(gameState, 0, self.depth, self.evaluationFunction, -1e9, 1e9)[1]
        
def alphaBetaValue(gameState, agentIndex, depth, function, alpha, beta):
    if depth == 0 or gameState.isLose() or gameState.isWin():
        return (function(gameState),Directions.STOP)
    
    if agentIndex == 0: alphaBeta = alphaValue
    else: alphaBeta = betaValue

    return alphaBeta(gameState, agentIndex, depth, function, alpha, beta)
    
def alphaValue(gameState, agentIndex, depth, function, alpha, beta):
    actions = gameState.getLegalActions(agentIndex)
    res = -1e9, None
    agent, next_depth = agentIndex + 1, depth

    if agentIndex + 1 == gameState.getNumAgents():
        agent, next_depth = 0, depth - 1

    for action in actions:
        state = gameState.generateSuccessor(agentIndex, action)
        score = alphaBetaValue(state, agent, next_depth, function, alpha, beta)[0], action
        if beta < score[0]: return score
        if res[0] < score[0]:
            res = score
            alpha = max(alpha, res[0])
    
    return res

def betaValue(gameState, agentIndex, depth, function, alpha, beta):
    actions = gameState.getLegalActions(agentIndex)
    res = 1e9, None
    agent, next_depth = agentIndex + 1, depth

    if agentIndex + 1 == gameState.getNumAgents():
        agent, next_depth = 0, depth - 1

    for action in actions:
        state = gameState.generateSuccessor(agentIndex, action)
        score = alphaBetaValue(state, agent, next_depth, function, alpha, beta)[0], action
        if score[0] < alpha: return score
        if res[0] > score[0]:
            res = score
            beta = min(beta, res[0])
    
    return res

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
        return expectimax(gameState, 0, self.depth, self.evaluationFunction)[1]

def expectimax(gameState, agentIndex, depth, function):
    if depth == 0 or gameState.isLose() or gameState.isWin():
        return (function(gameState),Directions.STOP)
    
    if agentIndex == 0:
        return sol(gameState, agentIndex, depth, function, expectimax, MAX, (-1e9, None))
    else:
        return exp(gameState, agentIndex, depth, function)
    
def exp(gameState, agentIndex, depth, function):
    actions = gameState.getLegalActions(agentIndex)
    agent, next_depth = agentIndex + 1, depth

    if agentIndex + 1 == gameState.getNumAgents():
        agent, next_depth = 0, depth - 1

    res = 0
    for action in actions:
        state = gameState.generateSuccessor(agentIndex, action)
        res += expectimax(state, agent, next_depth, function)[0]
    
    res = res * 1.0 / len(actions)
    
    return res, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    "*** YOUR CODE HERE ***"

    ghost = 1e9;
    for state in ghostStates:
        ghost = min(ghost, manhattanDistance(state.getPosition(), pos))

    if ghost <= 1 and len(scaredTimes) == 0: return -1e9;

    distanceFood = 1e9
    for food in foods.asList(): 
        distanceFood = min(distanceFood, manhattanDistance(pos, food))

    if len(foods.asList()) == 0: distanceFood = 0
    
    return -distanceFood + currentGameState.getScore() + 100*(ghost < 1 and len(scaredTimes) > 0)

# Abbreviation
better = betterEvaluationFunction
