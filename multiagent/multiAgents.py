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


from functools import reduce
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
        currentFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        # custom code
        # print(newFood)
        # print('-----------------------------------')
        

        "*** YOUR CODE HERE ***"
        
        # mdistances = []
        # mdscore = []
        # mdglist = [manhattanDistance(newPos, i.getPosition()) for i in newGhostStates]
        # mdgav = reduce(lambda x, y: x+y, mdglist)/len(mdglist)
        
        # if not newFood.asList():
        #     mdscore = 0
        # else:
        #     for x_2 in newFood.asList():
        #         mdistances.append(manhattanDistance(newPos, x_2))
        #         mdscore = min(mdistances)
        
        # return successorGameState.getScore() + min(newScaredTimes) + 1/(mdscore+0.1) - (1/(mdgav +0.1))
        
        
        
        #================================
        
        score = successorGameState.getScore()
        new_ghost_positions = successorGameState.getGhostPositions()
        current_food_list = currentFood.asList()
        new_food_list = newFood.asList()
        closest_food = float('+Inf')
        closest_ghost = float('+Inf')
        add_score = 0
        
        
        
        if newPos in current_food_list:
            add_score += 10.0

        distance_from_food = [manhattanDistance(newPos, food_position) for food_position in new_food_list]
        total_available_food = len(new_food_list)
        if len(distance_from_food):
            closest_food = min(distance_from_food)

        score += 10.0 / closest_food  - 4.0 * total_available_food + add_score

        # ? TODO: Write some comments about the implementation.

        for ghost_position in new_ghost_positions:
            distance_from_ghost = manhattanDistance(newPos, ghost_position)
            closest_ghost = min([closest_ghost, distance_from_ghost])

        if closest_ghost < 2:
            score -= 50.0

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
        
        best_action = self.max_value(gameState = gameState, depth =0, agent_idx = 0)[1]
        
        return best_action
        
        util.raiseNotDefined()
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
        best_action = self.max_value(gameState = gameState, depth = depth, agent_idx = 0, alpha = alpha, beta = beta )
        return  best_action[1]
        util.raiseNotDefined()
    
    def is_terminal_state(self, gameState, depth, agent_idx):
        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agent_idx) is 0:
            return gameState.getLegalActions(agent_idx)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth 
        
    
    def value(self, gameState, depth, agent_idx, alpha, beta):
        if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
            return self.evaluationFunction(gameState)
        elif agent_idx is 0:
            return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx,alpha=alpha, beta=beta)[0]
        else:
            return self.min_value(gameState=gameState, depth=depth, agent_idx=agent_idx, alpha=alpha, beta=beta)[0]
    
    def max_value(self, gameState, depth, agent_idx, alpha, beta):
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
    def min_value(self, gameState, depth, agent_idx, alpha, beta):
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
        best_action = self.max_value(gameState=gameState, depth=0, agent_idx=0)[1]
        return best_action

        util.raiseNotDefined()
        
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

    def expected_value(self, gameState, depth, agent_idx):
        value = list()
        legal_actions = gameState.getLegalActions(agent_idx)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_idx, action)
            number_of_agents = gameState.getNumAgents()
            expand = depth + 1
            current_player = (depth + 1) % number_of_agents
            value.append(self.value(gameState=successor_state, depth=expand, agent_idx=current_player))
        expected_value = sum(value) / len(value)
        return expected_value

    def value(self, gameState, depth, agent_idx):
        if self.is_terminal_state(gameState=gameState, depth=depth, agent_idx=agent_idx):
            return self.evaluationFunction(gameState)
        elif agent_idx is 0:
            return self.max_value(gameState=gameState, depth=depth, agent_idx=agent_idx)[0]
        else:
            return self.expected_value(gameState=gameState, depth=depth, agent_idx=agent_idx)




def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_position = currentGameState.getPacmanPosition()
    food_positions = currentGameState.getFood().asList()
    capsules_positions = currentGameState.getCapsules()
    ghost_positions = currentGameState.getGhostPositions()
    ghost_states = currentGameState.getGhostStates()
    scared_ghosts_timer = [ghost_state.scaredTimer for ghost_state in ghost_states]
    remaining_food = len(food_positions)
    remaining_capsules = len(capsules_positions)
    scared_ghosts = list()
    enemy_ghosts = list()
    enemy_ghost_positions = list()
    scared_ghosts_positions = list()
    score = currentGameState.getScore()

    closest_food = float('+Inf')
    closest_enemy_ghost = float('+Inf')
    closest_scared_ghost = float('+Inf')

    distance_from_food = [manhattanDistance(pacman_position, food_position) for food_position in food_positions]
    if len(distance_from_food) is not 0:
        closest_food = min(distance_from_food)
        score -= 1.0 * closest_food

    for ghost in ghost_states:
        if ghost.scaredTimer is not 0:
            enemy_ghosts.append(ghost)
        else:
            scared_ghosts.append(ghost)

    for enemy_ghost in enemy_ghosts:
        enemy_ghost_positions.append(enemy_ghost.getPosition())

    if len(enemy_ghost_positions) is not 0:
        distance_from_enemy_ghost = [manhattanDistance(pacman_position, enemy_ghost_position) for enemy_ghost_position in enemy_ghost_positions]
        closest_enemy_ghost = min(distance_from_enemy_ghost)
        score -= 2.0 * (1 / closest_enemy_ghost)

    for scared_ghost in scared_ghosts:
        scared_ghosts_positions.append(scared_ghost.getPosition())

    if len(scared_ghosts_positions) is not 0:
        distance_from_scared_ghost = [manhattanDistance(pacman_position, scared_ghost_position) for scared_ghost_position in scared_ghosts_positions]
        closest_scared_ghost = min(distance_from_scared_ghost)
        score -= 3.0 * closest_scared_ghost

    score -= 20.0 * remaining_capsules
    score -= 4.0 * remaining_food
    return score

    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
