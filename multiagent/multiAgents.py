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

        minimumFoodist = min([manhattanDistance(newPos, food) for food in newFood],
                         default=float('inf'))

        # reciprocal
        return successorGameState.getScore() + 1.0 / (minimumFoodist + 1)

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
        def miniMax(state, depth, agent):
            next_depth = depth-(agent == 0)

            if next_depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            bestaction = None
            next_agent = (agent + 1) % state.getNumAgents()
            bestval = -float('inf') if agent == 0 else float('inf')
           

            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                valOfAction, _ = miniMax(successorState, next_depth, next_agent)
                if (agent == 0 and bestval < valOfAction) or (
                        agent != 0 and bestval > valOfAction):
                    bestval = valOfAction
                    bestaction = action
            return bestval, bestaction

        _, action = miniMax(gameState, self.depth + 1, self.index)

        return action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(state,
                      depth,
                      agent,
                      alpha=-float('inf'),
                      beta=float('inf')):
            next_depth = depth - (agent == 0)
            if next_depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            next_agent = (agent + 1) % state.getNumAgents()
            bestval = -float('inf') if (agent == 0) else float('inf')
            bestaction = None
        #    check

            for action in state.getLegalActions(agent):
                successorstate = state.generateSuccessor(agent, action)
                valofaction, _ = alphaBeta(successorstate, next_depth,
                                           next_agent, alpha, beta)
                if agent == 0:
                    if bestval < valofaction:
                        bestaction = action
                        bestval = valofaction
                      
                    if bestval > beta:
                        return bestval, bestaction
                    alpha = max(alpha, bestval)
                else:
                    if bestval > valofaction:
                        bestaction = action
                        bestval = valofaction
                    if bestval < alpha:
                        return bestval, bestaction
                    beta = min(beta, bestval)

            return bestval, bestaction

        _, action = alphaBeta(gameState, self.depth + 1, self.index)
        return action
        util.raiseNotDefined()

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
        selfagent = self.index
        if selfagent != 0:
            return random.choice(gameState.getLegalActions(selfagent))

        def expectimax(state, depth, selfagent):
            '''
                Returns the best value-action pair for the agent
            '''
            next_depth = depth - (selfagent == 0)
            if next_depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            bestval = -float('inf')
            bestaction = None
            next_agent = (selfagent + 1) % state.getNumAgents()

            if selfagent != 0:
                value = 0.0
                for action in state.getLegalActions(selfagent):
                    successorState = state.generateSuccessor(selfagent, action)
                    expVal, _ = expectimax(successorState, next_depth,
                                           next_agent)
                    value += expVal
                return value / float(len(state.getLegalActions(selfagent))), None
            else:
                for action in state.getLegalActions(selfagent):
                    successorState = state.generateSuccessor(selfagent, action)
                    expVal, _ = expectimax(successorState, next_depth,
                                           next_agent)
                    if max(bestval, expVal) == expVal:
                        bestval, bestaction = expVal, action
                return bestval, bestaction

        _, action = expectimax(gameState, self.depth + 1, self.index)
        return action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newpos = currentGameState.getPacmanPosition()
    newfood = currentGameState.getFood().asList()

    min_foodist = min([manhattanDistance(newpos, food) for food in newfood],default=float('inf'))

    foodleft = len(newfood)
    capsleft = len(currentGameState.getCapsules())
    foot_dis_multiplier = 1000
    caps_left_multiplier = 10000
    foot_left_multiplier = 1000000
    endgamescore = 0
    if currentGameState.isLose():
        endgamescore -= 500
    elif currentGameState.isWin():
        endgamescore += 500
    return 1.0/(min_foodist + 1) * foot_dis_multiplier + \
           1.0/(foodleft + 1) * foot_left_multiplier + \
           1.0/(capsleft + 1) * caps_left_multiplier + endgamescore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
