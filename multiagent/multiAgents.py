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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        if successorGameState.isWin():
            return 1000000
        closetFood = min([manhattanDistance(newPos, food)
                          for food in newFood.asList()])
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0 and manhattanDistance(ghost.getPosition(), newPos) < 2:
                return -1000000
        return successorGameState.getScore() + 1.0 / closetFood


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

    def minimax(self, state, depth, agent=0, maxing=True):

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions(agent)

        if maxing == True:
            scores = [self.minimax(state.generateSuccessor(agent, action), depth - 1, 1, False)[0] for action in
                      actions]
            bestScore = max(scores)
            bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
            return bestScore, actions[bestIndices[0]]

        elif maxing == False:
            scores = []

            if agent == state.getNumAgents() - 1:
                scores = [self.minimax(state.generateSuccessor(agent, action), depth - 1, 0, True)[0] for action in
                          actions]

            else:
                scores = [self.minimax(state.generateSuccessor(agent, action), depth, agent + 1, False)[0] for action in
                          actions]

            bestScore = min(scores)
            bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
            return bestScore, actions[bestIndices[0]]

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
        return self.minimax(gameState, self.depth * 2, 0, True)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        val = -10000.0
        alpha = -10000.0
        beta = 10000.0
        actionSeq = []
        moves = gameState.getLegalActions(0)
        for move in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, move)
            t = self.minimaxPrune(1, range(gameState.getNumAgents()), state, self.depth, self.evaluationFunction,
                                  alpha, beta)
            if t > val:
                val = t
                actionSeq = move
            if val > beta:
                return actionSeq
            alpha = max(alpha, val)
        return actionSeq

    def minimaxPrune(self, agent, agents, state, depth, eval_function, alpha, beta):
        if depth <= 0 or state.isWin() or state.isLose():
            return eval_function(state)

        val = -9999999.0 if agent == 0 else 9999999.0

        for move in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, move)
            if agent == agents[-1]:
                val = min(val, self.minimaxPrune(agents[0], agents, successor, depth - 1, eval_function, alpha, beta))
                beta = min(beta, val)
                if val < alpha:
                    return val
            elif agent == 0:
                val = max(val,
                          self.minimaxPrune(agents[agent + 1], agents, successor, depth, eval_function, alpha, beta))
                alpha = max(alpha, val)
                if val > beta:
                    return val
            else:
                val = min(val,
                          self.minimaxPrune(agents[agent + 1], agents, successor, depth, eval_function, alpha, beta))
                beta = min(beta, val)
                if val < alpha:
                    return val
        return val


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxLevel(gameState, depth):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions_to_check = gameState.getLegalActions(0)
            for action_to_check in actions_to_check:
                successor = gameState.generateSuccessor(0, action_to_check)
                maxvalue = max(maxvalue, expectLevel(successor, currDepth, 1))
            return maxvalue

        def expectLevel(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            actions_in_method = gameState.getLegalActions(agentIndex)
            totalexpectedvalue = 0
            numberofactions = len(actions_in_method)
            for action_to_check in actions_in_method:
                successor = gameState.generateSuccessor(agentIndex, action_to_check)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maxLevel(successor, depth)
                else:
                    expectedvalue = expectLevel(successor, depth, agentIndex + 1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue
            if numberofactions == 0:
                return 0
            return float(totalexpectedvalue) / float(numberofactions)

        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            score = expectLevel(nextState, 0, 1)
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    new_scared_times = [ghostState.scaredTimer for ghostState in newGhostStates]

    food_distance = [0]
    for pos in newFood.asList():
        food_distance.append(manhattanDistance(newPos, pos))

    ghost_distance = [0]
    for pos in [ghost.getPosition() for ghost in newGhostStates]:
        ghost_distance.append(manhattanDistance(newPos, pos))

    number_of_power_pellets = len(currentGameState.getCapsules())

    score = 0
    distance = 1.0 / sum(food_distance) if sum(food_distance) > 0 else 0
    sumScaredTimes = sum(new_scared_times)
    sumGhostDistance = sum(ghost_distance)
    numberOfNoFoods = len(newFood.asList(False))

    score += currentGameState.getScore() + distance + numberOfNoFoods

    if sumScaredTimes > 0:
        score += sumScaredTimes - number_of_power_pellets - sumGhostDistance
    else:
        score += sumGhostDistance + number_of_power_pellets
    return score


# Abbreviation
better = betterEvaluationFunction
