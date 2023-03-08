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

        #Calculate distance min(pos, Food)
        dClosetFood = min(manhattanDistance(newPos,food) for food in newFood)
        dClosetGhost = min(manhattanDistance(newFood,ghost) for ghost in newGhostStates)
        foodRemain = len(newFood)
        timeScare = min(newScaredTimes)

        # Good to close food
        food = 1/(dClosetFood+0.5)
        ghost = -2/(dClosetGhost + 0.5) if timeScare == 0 else 1/(dClosetGhost+1)


        "*** YOUR CODE HERE ***"
        return successorGameState.getScore() +food + ghost + timeScare*0.5

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
        return self.minimaxSearch(gameState, agentId= 0, depth=self.depth)[1]
        util.raiseNotDefined()
    def minimaxSearch(self, gameState, agentId, depth): #return score, act
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        
        if agentId == 0:
            return self.findingMax(gameState,agentId, depth)
        
        return self.findingMin(gameState, agentId, depth)
            
    def findingMin(self, gameState , agentId : int, depth):
        actions = gameState.getLegalActions(agentId)
        if agentId == gameState.getNumAgents() - 1: #final agent
            next_agent, next_depth = 0, depth - 1
        else :
            next_agent,next_depth = agentId + 1, depth
        
        min_score = 1e9
        min_act = actions[0]
        for action in actions :
            successor_gamestate = gameState.generateSuccessor(agentId,action)
            score = self.minimaxSearch(successor_gamestate,next_agent,next_depth)[0] #return score
            if score < min_score:
                min_score, min_act = score, action
        
        return min_score, min_act
    
    def findingMax(self, gameState, agentId, depth):
        actions = gameState.getLegalActions(agentId)
        if agentId == gameState.getNumAgents()-1:
            next_agent, next_depth = 0, depth -1
        else :
            next_agent, next_depth = agentId+1, depth
        
        max_score = -1e9
        max_act = actions[0]
        for action in actions :
            successor_gamestate = gameState.generateSuccessor(agentId,action)
            score = self.minimaxSearch(successor_gamestate,next_agent,next_depth)[0]
            if score > max_score:
                max_score, max_act = score, action
        
        return max_score, max_act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabetaSearch(gameState, 0 , self.depth, -1e9,1e9)[1]
        util.raiseNotDefined()
    
    def alphabetaSearch(self, gameState, agentId, depth, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        
        if agentId == 0:
            return self.findingMax(gameState,agentId, depth, alpha, beta) #beta cat nhanh 
        
        return self.findingMin(gameState, agentId, depth, alpha, beta) #alpha cat nhanh 

    def findingMax(self, gameState, agentId, depth, alpha, beta):
        actions = gameState.getLegalActions(agentId)
        if agentId == gameState.getNumAgents()-1:
            next_agent, next_depth = 0, depth -1
        else :
            next_agent, next_depth = agentId+1, depth
        
        max_score = -1e9
        max_act = actions[0]
        for action in actions:
            successor_gameState = gameState.generateSuccessor(agentId, action)
            score = self.alphabetaSearch(successor_gameState,next_agent,next_depth,alpha,beta)[0]

            if score > max_score:
                max_score, max_act = score, action
            if score > beta:
                return score, action
            alpha = max(alpha, max_score)
        return max_score, max_act
    
    def findingMin(self, gameState, agentId, depth, alpha, beta):
        actions = gameState.getLegalActions(agentId)
        if agentId == gameState.getNumAgents()-1:
            next_agent, next_depth = 0, depth -1
        else :
            next_agent, next_depth = agentId+1, depth
        
        min_score = 1e9
        min_act = actions[0]
        for action in actions:
            successor_gameState = gameState.generateSuccessor(agentId, action)
            score = self.alphabetaSearch(successor_gameState,next_agent,next_depth,alpha,beta)[0]

            if score  < min_score:
                min_score, min_act = score, action
            if score < alpha:
                return score, action
            beta = min(beta, min_score)
        return min_score, min_act

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
        return self.expectimaxSearch(gameState, 0, self.depth)[1]
        util.raiseNotDefined()
    def expectimaxSearch(self, gameState, agentId, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), Directions.STOP
        
        if agentId == 0:
            return self.findingMax(gameState,agentId, depth) #beta cat nhanh 
        
        return self.expectation(gameState, agentId, depth) #alpha cat nhanh
    
    def findingMax(self, gameState, agentId, depth):
        actions = gameState.getLegalActions(agentId)
        if agentId == gameState.getNumAgents()-1:
            next_agent, next_depth = 0, depth -1
        else :
            next_agent, next_depth = agentId+1, depth
        
        max_score = -1e9
        max_act = actions[0]
        for action in actions :
            successor_gamestate = gameState.generateSuccessor(agentId,action)
            score = self.expectimaxSearch(successor_gamestate,next_agent,next_depth)[0]
            if score > max_score:
                max_score, max_act = score, action
        
        return max_score, max_act
    
    def expectation(self, gameState, agentId, depth):
        actions = gameState.getLegalActions(agentId)
        if agentId == gameState.getNumAgents()-1:
            next_agent, next_depth = 0, depth -1
        else :
            next_agent, next_depth = agentId+1, depth
        
        exp_score, exp_act = 0, Directions.STOP
        for action in actions:
            successor_gamestate = gameState.generateSuccessor(agentId,action)
            exp_score += self.expectimaxSearch(successor_gamestate,next_agent,next_depth)[0]

        exp_score /= len(actions)
        return exp_score, exp_act
    

        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pac_position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    scare_times = [ghost.scaredTimer for ghost in ghosts]

    pac_ghost_distances = [manhattanDistance(ghost.getPosition(),pac_position) if ghost.scaredTimer == 0 else -10 for ghost in ghosts ]
    nearest_ghost = min(pac_ghost_distances)
    if foods: 
        pac_food_distances = [manhattanDistance(food,pac_position)  for food in foods]
        nearest_food = min(pac_food_distances)
    else:
        nearest_food = 0
    
    return currentGameState.getScore() - 5/ (nearest_ghost+1) - nearest_food/3

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
