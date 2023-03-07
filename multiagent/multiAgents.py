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
import random
import util

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        currentFoods = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score_final = successorGameState.getScore()
        new_ghosts_positions = successorGameState.getGhostPositions()
        current_food_list = currentFoods.asList()
        new_food_list = newFood.asList()
        closest_food = float('+Inf')
        closest_ghost = float('+Inf')
        add_score_01 = 0
        if newPos in current_food_list:
            add_score_01 += 10.0

        distance_from_food = [manhattanDistance(
            newPos, food_position) for food_position in new_food_list]
        total_available_food = len(new_food_list)
        if len(distance_from_food):
            closest_food = min(distance_from_food)

        score_final += 10.0 / closest_food - 4.0 * total_available_food + add_score_01

        for ghost_position in new_ghosts_positions:
            distance_from_ghost = manhattanDistance(newPos, ghost_position)
            closest_ghost = min([closest_ghost, distance_from_ghost])

        if closest_ghost < 2:
            score_final -= 50.0

        return score_final
# return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        action = self.max_value(gameState=gameState, depth=0, agent_idx=0)[1]
        return action

        # util.raiseNotDefined()
    def is_terminal_state(self, gameState, depth, agent_idx):
        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agent_idx) is 0:
            return gameState.getLegalActions(agent_idx)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth
    
    def max_value(self, gameState, depth, agent_idx):
        value = (float('-Inf'), None)
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = (depth + 1) % number_of_agents
            value = max([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player), action)], key=lambda idx: idx[0])
        return value
    
    def min_value(self, gameState, depth, agent_idx):
        value = (float('+Inf'), None)
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = (depth + 1) % number_of_agents
            value = min([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player), action)], key=lambda idx: idx[0])
        
        return value
    
    def value(self, gameState, depth, agent_idx):
        if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
            return self.evaluationFunction(gameState)
        elif agent_idx is 0:
            return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx)[0]
        else:
            return self.min_value(gameState=gameState, depth=depth, agent_idx=agent_idx)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-Inf')
        beta = float('+Inf')
        depth = 0
        action = self.max_value(gameState=gameState, depth=depth, agent_idx=0, alpha=alpha, beta=beta)
        return action[1]
    
    def is_terminal_state(self, gameState, depth, agent_idx):
        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agent_idx) is 0:
            return gameState.getLegalActions(agent_idx)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth
    
    def max_value(self, gameState, depth, agent_idx,alpha,beta):
        value = (float('-Inf'), None)
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = expand % number_of_agents
            value = max([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player, alpha=alpha, beta=beta), action)], key=lambda idx: idx[0])
            if value[0] > beta:
                return value
            alpha = max(alpha, value[0])

        return value
    
    def min_value(self, gameState, depth, agent_idx,alpha,beta):
        value = (float('+Inf'), None)
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = expand % number_of_agents
            value = min([value, (self.value(gameState=successor_state, depth=expand, agent_idx=current_player, alpha=alpha, beta=beta), action)], key=lambda idx: idx[0])
            if value[0] < alpha:
                return value
            beta = min(beta, value[0])

        
        return value
    
    def value(self, gameState, depth, agent_idx,alpha,beta):
        if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
            return self.evaluationFunction(gameState)
        elif agent_idx is 0:
            return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx,alpha=alpha, beta=beta)[0]
        else:
            return self.min_value(gameState=gameState, depth=depth, agent_idx=agent_idx, alpha=alpha, beta=beta)[0]





        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
        # calling expectimax with the depth we are going to investigate
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, "expect", maxDepth, 0)[0]

    def expectimax(self, gameState, action, depth, agentIndex):

        if depth is 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))

        # if pacman (max agent) - return max successor value
        if agentIndex is 0:
            return self.maxvalue(gameState,action,depth,agentIndex)
        # if ghost (EXP agent) - return probability value
        else:
            return self.expvalue(gameState,action,depth,agentIndex)

    def maxvalue(self,gameState,action,depth,agentIndex):
        bestAction = ("max", -(float('inf')))
        for legalAction in gameState.getLegalActions(agentIndex):
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            succAction = None
            if depth != self.depth * gameState.getNumAgents():
                succAction = action
            else:
                succAction = legalAction
            succValue = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                        succAction,depth - 1,nextAgent)
            bestAction = max(bestAction,succValue,key = lambda x:x[1])
        return bestAction

    def expvalue(self,gameState,action,depth,agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        averageScore = 0
        propability = 1.0/len(legalActions)
        for legalAction in legalActions:
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            bestAction = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                         action, depth - 1, nextAgent)
            averageScore += bestAction[1] * propability
        return (action, averageScore)

import math

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
      Evaluate state by  :
            * closest food
            * food left
            * capsules left
            * distance to ghost
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()

    minFoodist = float('inf')
    for food in newFood:
        minFoodist = min(minFoodist, manhattanDistance(newPos, food))

    ghostDist = 0
    for ghost in currentGameState.getGhostPositions():
        ghostDist = manhattanDistance(newPos, ghost)
        if (ghostDist < 2):
            return -float('inf')

    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())

    foodLeftMultiplier = 950050
    capsLeftMultiplier = 10000
    foodDistMultiplier = 950

    additionalFactors = 0
    if currentGameState.isLose():
        additionalFactors -= 50000
    elif currentGameState.isWin():
        additionalFactors += 50000

    return 1.0/(foodLeft + 1) * foodLeftMultiplier + ghostDist + \
           1.0/(minFoodist + 1) * foodDistMultiplier + \
           1.0/(capsLeft + 1) * capsLeftMultiplier + additionalFactors

# Abbreviation
better = betterEvaluationFunction