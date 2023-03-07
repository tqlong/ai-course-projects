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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        minFoodist = float("inf")
        for food in newFood:
            minFoodist = min(minFoodist, manhattanDistance(newPos, food))

        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
        return successorGameState.getScore() + 1.0/minFoodist


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
        def maximizer(state, depth):            
            if depth==self.depth or state.isWin() or state.isLose():                
                return self.evaluationFunction(state)            
            value = float("-inf")            
            legalMoves = state.getLegalActions()            
            for action in legalMoves:                
                value = max(value, minimizer(state.generateSuccessor(0, action), depth, 1))            
            return value        
    
        def minimizer(state, depth, agentIndex):            
            if depth==self.depth or state.isWin() or state.isLose():                
                return self.evaluationFunction(state)            
            value = float("inf")            
            legalMoves = state.getLegalActions(agentIndex)            
            if agentIndex==state.getNumAgents()-1:                
                for action in legalMoves:                    
                    value = min(value, maximizer(state.generateSuccessor(agentIndex, action), depth+1))
            else:                
                for action in legalMoves:                    
                    value = min(value, minimizer(state.generateSuccessor(agentIndex, action), depth, agentIndex+1))            
            return value        

        legalMoves = gameState.getLegalActions()        
        move = Directions.STOP        
        value = float("-inf")        
        for action in legalMoves:            
            temp = minimizer(gameState.generateSuccessor(0, action), 0, 1)            
            if temp > value:                
                value = temp                
                move = action        
        return move
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
        def maximizer(state, depth):
            if depth==self.depth or state.isWin() or state.isLose():                
                return self.evaluationFunction(state)            
            value = float("-inf")            
            legalMoves = state.getLegalActions()            
            for action in legalMoves:                
                value = max(value, expecter(state.generateSuccessor(0, action), depth, 1))
            return value        
     
        def expecter(state, depth, agentIndex):
            if depth==self.depth or state.isWin() or state.isLose():                
                return self.evaluationFunction(state)            
            value = 0            
            legalMoves = state.getLegalActions(agentIndex)            
            if agentIndex==state.getNumAgents()-1:                
                for action in legalMoves:                    
                    value +=  maximizer(state.generateSuccessor(agentIndex, action), depth+1)
            else:                
                for action in legalMoves:                    
                    value += expecter(state.generateSuccessor(agentIndex, action), depth, agentIndex+1)
            return value/len(legalMoves)        

        legalMoves = gameState.getLegalActions()        
        move = Directions.STOP        
        value = float("-inf")        
        for action in legalMoves:            
            temp = expecter(gameState.generateSuccessor(0, action), 0, 1)
            if temp > value:                
                value = temp                
                move = action        
        return move

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
    weightFood = 10.0
    weightGhost = -10.0
    weightScaredGhost = 100.0
    INF = 100000000.0

    score = currentGameState.getScore()

    for foodPos in newFood:
        disToFood = [manhattanDistance(newPos, foodPos)]
        if len(disToFood) > 0:
            score += weightFood / min(disToFood)
        else:
            score += weightFood
    for ghost in newGhostStates:
        dis = manhattanDistance(newPos, ghost.getPosition())
        if dis > 0:
            if ghost.scaredTimer > 0:
                score += weightScaredGhost / dis
            else:
                score += weightGhost / dis
        else:
            return -INF 
    return score
    

# Abbreviation
better = betterEvaluationFunction
