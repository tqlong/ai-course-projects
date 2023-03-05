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
        # give score to step that have min_food_dist small
        min_food_dist = float("inf")
        for food in newFood.asList():
            min_food_dist = min(min_food_dist, (manhattanDistance(newPos, food)))
        food_score = 10 / min_food_dist 

        # give score to step that have min_ghost_dist big (devided by 10 cuz giving this feature a small weight)
        ghost_score = 0
        min_ghost_dist = float("inf")
        for new_ghost_pos in successorGameState.getGhostPositions():
            min_ghost_dist = min(min_ghost_dist, manhattanDistance(newPos, new_ghost_pos))
        ghost_score += min_ghost_dist / 10

        # force pacman not to get too close to the ghost
        if min_ghost_dist < 2:
            ghost_score -= 99999
        
        # force the pacman to minimize the remaining food
        remaining_food_score = len(successorGameState.getFood().asList())
        
        return successorGameState.getScore() + food_score + ghost_score - remaining_food_score

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
        return self.value(gameState, 0, 0)[1]
        util.raiseNotDefined()
    
    def value(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        
        # pacman's turn 
        if index == 0:
            return self.max_value(gameState, index, depth)
        # ghost's turn
        else:
            return self.min_value(gameState, index, depth)
    
    def max_value(self, gameState, index, depth):
        legalActions = gameState.getLegalActions(index)
        max_value = float("-Inf")
        max_action = legalActions[0]

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update index and depth
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_value = self.value(successor, successor_index, successor_depth)[0]
            
            if(curr_value > max_value):
                max_value = curr_value
                max_action = action
            
        return max_value, max_action

    def min_value(self, gameState, index, depth):
        legalActions = gameState.getLegalActions(index)
        min_value = float("Inf")
        min_action = legalActions[0]
        
        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update index and depth
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            
            curr_value = self.value(successor, successor_index, successor_depth)[0]

            if(curr_value < min_value):
                min_value = curr_value
                min_action = action
        
        return min_value, min_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0, float("-Inf"), float("Inf"))[1]

    def value(self, gameState, index, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        
        # pacman's turn 
        if index == 0:
            return self.max_value(gameState, index, depth, alpha, beta)
        # ghost's turn
        else:
            return self.min_value(gameState, index, depth, alpha, beta)
    
    def max_value(self, gameState, index, depth, alpha, beta):
        legalActions = gameState.getLegalActions(index)
        max_value = float("-Inf")
        max_action = legalActions[0]

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update index and depth
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_value = self.value(successor, successor_index, successor_depth, alpha, beta)[0]
            
            if(curr_value > max_value):
                max_value = curr_value
                max_action = action
            
            if(curr_value > beta):
                return curr_value, action, alpha, beta
            
            alpha = max(alpha, curr_value)
        return max_value, max_action, alpha, beta

    def min_value(self, gameState, index, depth, alpha, beta):
        legalActions = gameState.getLegalActions(index)
        min_value = float("Inf")
        min_action = legalActions[0]
        
        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update index and depth
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            
            curr_value = self.value(successor, successor_index, successor_depth, alpha, beta)[0]

            if(curr_value < min_value):
                min_value = curr_value
                min_action = action

            if(curr_value < alpha):
                return curr_value, action, alpha, beta
            
            beta = min(beta, curr_value)
        return min_value, min_action, alpha, beta

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
        return self.value(gameState, 0, 0)[1]
        util.raiseNotDefined()
    
    def value(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP
        
        # pacman's turn 
        if index == 0:
            return self.max_value(gameState, index, depth)
        # ghost's turn
        else:
            return self.min_value(gameState, index, depth)
    
    def max_value(self, gameState, index, depth):
        legalActions = gameState.getLegalActions(index)
        max_value = float("-Inf")
        max_action = legalActions[0]

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update index and depth
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            curr_value = self.value(successor, successor_index, successor_depth)[0]
            
            if(curr_value > max_value):
                max_value = curr_value
                max_action = action
            
        return max_value, max_action

    def min_value(self, gameState, index, depth):
        legalActions = gameState.getLegalActions(index)
        expected_value = 0
        expected_action = ""
        
        succesor_prob = 1.0 / len(legalActions)

        for action in legalActions:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update index and depth
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1
            
            curr_value, curr_action = self.value(successor, successor_index, successor_depth)

            expected_value += succesor_prob * curr_value
        
        return expected_value, expected_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # How to choose weights?
    pacman_pos = currentGameState.getPacmanPosition()
    ghost_pos = currentGameState.getGhostPositions()

    food_list = currentGameState.getFood().asList()
    capsule_list = currentGameState.getCapsules()
    # give score to step that have min_food_dist small
    min_food_dist = float("inf")
    for food in food_list:
        min_food_dist = min(min_food_dist, (manhattanDistance(pacman_pos, food)))
    
    food_score = 20 / min_food_dist 

    # give score to step that have min_ghost_dist big (devided by 10 cuz giving this feature a small weight)
    ghost_score = 0
    min_ghost_dist = float("inf")
    for ghost in ghost_pos:
        min_ghost_dist = min(min_ghost_dist, manhattanDistance(pacman_pos, ghost))
    ghost_score += min_ghost_dist / 10

    # force pacman not to get too close to the ghost
    if min_ghost_dist < 2:
        ghost_score -= 99999
    
    # force the pacman to minimize the remaining food
    remaining_food_score = len(food_list)
    remaining_capsule_score = len(capsule_list)
    return 100*currentGameState.getScore() + food_score + ghost_score - 10 * remaining_food_score - remaining_capsule_score
    # return currentGameState.getScore() + food_score + ghost_score - remaining_food_score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
