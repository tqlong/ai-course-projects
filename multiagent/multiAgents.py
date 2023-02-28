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
        if action == 'STOP':
            return -999999999
        
        currentPosition = list(newPos)

        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0 and ghostState.getPosition() == tuple(currentPosition): # CurrentPo has a ghost
                return -999999999
            
        min = 999999999
        v = 0
        currentFood = currentGameState.getFood()
        foodList = currentFood.asList()
        for i in range(len(foodList)):
            v = (manhattanDistance(foodList[i], currentPosition))
            if v < min:
                min = v
        min = -min
        return min

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
        action, score = self.minimax(0, 0, gameState) 
        return action

    def minimax(self, currentDepth, agentIndex, gameState):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            currentDepth += 1
            
        if currentDepth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        bestScore, chosenAction = None, None
        if agentIndex == 0:  # pacman
            for action in gameState.getLegalActions(agentIndex): 
                newState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.minimax(currentDepth, agentIndex + 1, newState)
                if bestScore is None or score > bestScore:
                    bestScore, chosenAction = score, action
        else:  # ghost
            for action in gameState.getLegalActions(agentIndex):  
                newState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.minimax(currentDepth, agentIndex + 1, newState)
                if bestScore is None or score < bestScore:
                    bestScore, chosenAction = score, action
       
        if bestScore is None:
            return None, self.evaluationFunction(gameState)
        
        return chosenAction, bestScore  

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, score = self.alpha_beta(0, 0, gameState, -999999999, 999999999)
        return action

    def alpha_beta(self, currentDepth, agentIndex, gameState, alpha, beta):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            currentDepth += 1
        
        if currentDepth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        bestScore, chosenAction = None, None
        if agentIndex == 0:  # pacman
            for action in gameState.getLegalActions(agentIndex):
                newState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.alpha_beta(currentDepth, agentIndex + 1, newState, alpha, beta)
                if bestScore is None or score > bestScore:
                    bestScore, chosenAction = score, action
                alpha = max(alpha, score)
                if alpha > beta:
                    break
        else:  # ghost
            for action in gameState.getLegalActions(agentIndex):
                newState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.alpha_beta(currentDepth, agentIndex + 1, newState, alpha, beta)
                if bestScore is None or score < bestScore:
                    bestScore, chosenAction = score, action
                beta = min(beta, score)
                if beta < alpha:
                    break
        
        if bestScore is None:
            return None, self.evaluationFunction(gameState)
        
        return chosenAction, bestScore  

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
        action, score = self.expectimax(0, 0, gameState)
        return action

    def expectimax(self, currentDepth, agentIndex, gameState):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            currentDepth += 1
        
        if currentDepth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        bestScore, chosenAction = None, None
        if agentIndex == 0:  # pacman
            for action in gameState.getLegalActions(agentIndex):
                newState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.expectimax(currentDepth, agentIndex + 1, newState)
                if bestScore is None or score > bestScore:
                    bestScore, chosenAction = score, action
        else: # exp
            ghostActions = gameState.getLegalActions(agentIndex)
            if len(ghostActions) is not 0:
                probability = 1.0 / len(ghostActions)
            for action in gameState.getLegalActions(agentIndex):
                newState = gameState.generateSuccessor(agentIndex, action)
                _, score = self.expectimax(currentDepth, agentIndex + 1, newState)
                if bestScore is None:
                    bestScore = 0.0
                bestScore += probability * score
                chosenAction = action
        
        if bestScore is None:
            return None, self.evaluationFunction(gameState)
        
        return chosenAction, bestScore

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Adding distance between capsules and pacman, excluding distance between ghost and pacman from current score. 
    """
    "*** YOUR CODE HERE ***"
    capsules = currentGameState.getCapsules()   
    ghostStates = currentGameState.getGhostStates()
    currentPos = currentGameState.getPacmanPosition()
    currentScore = currentGameState.getScore()
    capScore = 0

    for capsule in capsules:
        capDistance = min([manhattanDistance(currentPos, capsule)])
        if capDistance == 0:
            capScore = 1.0/capDistance
        else:
            capScore = -100
    
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDistance = manhattanDistance(currentPos, ghostPos)

    return currentScore - ghostDistance + capScore


# Abbreviation
better = betterEvaluationFunction
