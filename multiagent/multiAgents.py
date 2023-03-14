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
        Food = newFood.asList()
        gPos = successorGameState.getGhostPositions()
        FoodDist = []
        GhostDist = []

        for food in Food:
            FoodDist.append(manhattanDistance(food, newPos))
        for ghost in gPos:
            GhostDist.append(manhattanDistance(ghost, newPos))

        if (currentGameState.getPacmanPosition() == newPos):
            return (-(float("inf")))
        
        for dist in GhostDist:
            if (dist < 2):
                return (-(float("inf")))
        if len(FoodDist) == 0:
            return float("inf")
        
        return 1000/sum(FoodDist) + 10000/len(FoodDist)

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
        def minimax(agentIndex, depth, gameState):
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                return self.evaluationFunction(gameState)
            if (agentIndex == 0 ):
                nextAgent = agentIndex + 1
                actions = gameState.getLegalActions(agentIndex)
                maxValue = max(minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action)) for action in actions)
                return maxValue
            else:
                nextAgent = agentIndex + 1
                if (gameState.getNumAgents() == nextAgent):
                    nextAgent = 0
                if (nextAgent == 0):
                    depth += 1
                actions = gameState.getLegalActions(agentIndex)
                minValue = min(minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action)) for action in actions)
                return minValue
        maxScore = float("-inf")
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            score = minimax(1, 0, gameState.generateSuccessor(0, action))
            if score > maxScore:
                maxScore = score
                bestAction = action
        return bestAction
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, agentIndex, a, b):
            maximum = ["", -float("inf")]
            actions = gameState.getLegalActions(agentIndex)

            if not actions:
                return self.evaluationFunction(gameState)
            
            for action in actions:
                curState = gameState.generateSuccessor(agentIndex, action)
                current = value(curState, depth, agentIndex + 1, a, b)
                newVal = current[1]
                if newVal > maximum[1]:
                    maximum = [action, newVal]
                if newVal > b:
                    return [action, newVal]
                a = max(a, newVal)
            return maximum
        
        def minValue(gameState, depth, agentIndex, a, b):
            minimum = ["", float("inf")]
            actions = gameState.getLegalActions(agentIndex)

            if not actions:
                return self.evaluationFunction(gameState)
            
            for action in actions:
                curState = gameState.generateSuccessor(agentIndex, action)
                current = value(curState, depth, agentIndex + 1, a, b)
                newVal = current[1]
                if newVal < minimum[1]:
                    minimum = [action, newVal]
                if newVal < a:
                    return [action, newVal]
                b = min(b, newVal)
            return minimum

        def value(gameState, depth, agentIndex, a, b):
            if agentIndex >= gameState.getNumAgents():
                depth += 1
                agentIndex = 0
            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return ["", self.evaluationFunction(gameState)]
            elif (agentIndex == 0):
                return maxValue(gameState, depth, agentIndex, a, b)
            else:
                return minValue(gameState, depth, agentIndex, a, b)
        action = value(gameState, 0, 0, float("-inf"), float("inf"))[0]
        return action
    
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
        def expectimax(gameState, agentIndex, depth):
            if agentIndex == gameState.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(gameState)
                else: 
                    return expectimax(gameState, 0, depth + 1)
            else:
                moves = gameState.getLegalActions(agentIndex)
                if len(moves) == 0:
                    return self.evaluationFunction(gameState)
                next = [expectimax(gameState.generateSuccessor(agentIndex, move), agentIndex + 1, depth)
                        for move in moves]
                if agentIndex == 0:
                    return max(next)
                else:
                    return sum(next)/len(next)
        return max(gameState.getLegalActions(0), key=lambda x: expectimax(gameState.generateSuccessor(0, x), 1, 1))

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
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    capsuleScore = 0
    ghostDist = 0
    if len(currentGameState.getCapsules()) != 0:
        closestCapsule = min([manhattanDistance(capsule, newPos) for capsule in currentGameState.getCapsules()])
        capsuleScore += 200 / closestCapsule

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    foodList = newFood.asList()
    if (len(foodList) != 0):
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0

    if (newGhostStates[0].scaredTimer != 0):
        capsuleScore = -300
        if closestGhost <= 1:
            ghostDist += 350
        else:
            ghostDist = 250 / closestGhost
        score = (-1.3 *  closestFood) + ghostDist - (3 * len(foodList)) + capsuleScore
    else:
        if closestGhost <= 1:
            ghostDist -= 1000
        else:
            ghostDist = -13/closestGhost
        score = (-1.3 * closestFood) + ghostDist - (95 * len(foodList)) + capsuleScore
    return score
# Abbreviation
better = betterEvaluationFunction
