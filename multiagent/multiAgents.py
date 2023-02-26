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


from typing import List
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
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
        food_distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        min_distance_food = 0
        if len(food_distances) > 0:
            min_distance_food = min(food_distances)

        scaredGhosts = [ghost for ghost in newGhostStates if ghost.scaredTimer == 0]
        if len(scaredGhosts) > 0:
            min_distance_ghost = min(
                [
                    manhattanDistance(newPos, ghost.getPosition())
                    for ghost in scaredGhosts
                ]
            )
        else:
            min_distance_ghost = -1

        return (
            successorGameState.getScore()
            - min_distance_food / 2
            - 6 / (min_distance_ghost + 2)
        )


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
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

        def maxValue(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            return max(
                [
                    minValue(
                        state.generateSuccessor(agentIndex, action),
                        depth,
                        agentIndex + 1,
                    )
                    for action in state.getLegalActions(agentIndex)
                ]
            )

        def minValue(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            return min(
                [
                    minValue(
                        state.generateSuccessor(agentIndex, action),
                        depth,
                        agentIndex + 1,
                    )
                    if agentIndex < state.getNumAgents() - 1
                    else maxValue(
                        state.generateSuccessor(agentIndex, action), depth - 1, 0
                    )
                    for action in state.getLegalActions(agentIndex)
                ]
            )

        return max(
            gameState.getLegalActions(0),
            key=lambda x: minValue(gameState.generateSuccessor(0, x), self.depth, 1),
        )


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(state, alpha, beta, depth, number_min_layer):
            if state.isWin() or state.isLose() or depth >= self.depth:
                return self.evaluationFunction(state)
            return (
                max_value(state, alpha, beta, depth, 0)
                if number_min_layer >= state.getNumAgents() - 1
                else min_value(
                    state,
                    alpha,
                    beta,
                    depth + 1
                    if number_min_layer == state.getNumAgents() - 2
                    else depth,
                    number_min_layer + 1,
                )
            )

        def max_value(state, alpha, beta, depth, number_min_layer):
            # rewrite this in one return line
            max_value = float("-inf")
            for action in state.getLegalActions(0):
                max_value = max(
                    max_value,
                    value(
                        state.generateSuccessor(0, action),
                        alpha,
                        beta,
                        depth,
                        number_min_layer,
                    ),
                )
                if max_value > beta:
                    return max_value
                alpha = max(alpha, max_value)
            return max_value

        def min_value(state, alpha, beta, depth, number_min_layer):
            min_value = float("inf")
            for action in state.getLegalActions(number_min_layer):
                min_value = min(
                    min_value,
                    value(
                        state.generateSuccessor(number_min_layer, action),
                        alpha,
                        beta,
                        depth,
                        number_min_layer,
                    ),
                )
                if min_value < alpha:
                    return min_value
                beta = min(beta, min_value)
            return min_value

        max_v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        minimax_action = None
        for action in gameState.getLegalActions(0):
            minimax_value = value(
                gameState.generateSuccessor(0, action), alpha, beta, 0, 0
            )
            if max_v < minimax_value:
                max_v = minimax_value
                minimax_action = action
            if minimax_value > beta:
                return minimax_action
            alpha = max(alpha, minimax_value)

        return minimax_action


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

    def value(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        return (
            self.maxValue(gameState, depth, agentIndex)
            if agentIndex == 0
            else self.expectedValue(gameState, depth, agentIndex)
        )

    def maxValue(self, gameState, depth, agentIndex):
        return max(
            [
                self.value(
                    gameState.generateSuccessor(agentIndex, action),
                    depth,
                    agentIndex + 1,
                )
                for action in gameState.getLegalActions(agentIndex)
            ]
        )

    def expectedValue(self, gameState, depth, agentIndex):
        value = 0
        value = sum(
            [
                self.value(
                    gameState.generateSuccessor(agentIndex, action), depth - 1, 0
                )
                if agentIndex == gameState.getNumAgents() - 1
                else self.value(
                    gameState.generateSuccessor(agentIndex, action),
                    depth,
                    agentIndex + 1,
                )
                for action in gameState.getLegalActions(agentIndex)
            ]
        )
        return value / len(gameState.getLegalActions(agentIndex))


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghost_state.scaredTimer for ghost_state in ghost_states]

    min_distance_food = (
        min(manhattanDistance(position, food) for food in foods)
        if len(foods) > 0
        else 0
    )
    min_distance_ghost = (
        min(
            manhattanDistance(position, ghost.getPosition())
            if ghost.scaredTimer == 0
            else float("inf")
            for ghost in ghost_states
        )
        if (len(ghost_states) > 0) and (0 in scared_times)
        else -2
    )

    return (
        currentGameState.getScore()
        - min_distance_food / 5
        - 9 / (min_distance_ghost + 1)
    )


# Abbreviation
better = betterEvaluationFunction
