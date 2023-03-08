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

        score = 0

        # tinh khoang cach gan nhat tu pacman den ghost
        closestGhost = newGhostStates[0].configuration.pos
        minDistanceToGhost = manhattanDistance(newPos, closestGhost)

        minDistanceToFood = float('inf')
        for item in newFood.asList():
            # tinh khoang cach tu pacman den thuc an va cap nhat khoang cach gan nhat
            distance = manhattanDistance(newPos, item)
            minDistanceToFood = min(minDistanceToFood, distance)

        if action == 'Stop':
            score -= float('inf')
        return successorGameState.getScore() + minDistanceToGhost / (minDistanceToFood * 10) + score

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

    # minimax la gia tri tot nhat dat duoc khi doi thu choi hop ly
    def minimax(self, gameState, depth, agent): 
        # reset gia tri cua agent va tang do sau depth khi cac agent da choi xong luot trong 1 nuoc di
        if agent >= gameState.getNumAgents():
            agent = 0
            depth += 1

        # khi da dat do sau toi da (duyet het cay trang thai) thi tra ve gia tri ham danh gia
        if depth == self.depth:
            return None, self.evaluationFunction(gameState)
 
        score = None
        action = None
        if agent == 0:  # neu la luot choi toi da cua pacman
            pacmanActions = gameState.getLegalActions(agent)
            for item in pacmanActions:  # duyet cac hanh dong cua pacman
                # lay diem minimax cua luot tiep theo
                nextGameState = gameState.generateSuccessor(agent, item)

                # trang thai va diem cua pacman sau khi thuc hien hanh dong
                tempAction, tempScore = self.minimax(nextGameState, depth, agent + 1)

                # cap nhat lai score va action tot nhat
                if score is None or tempScore > score:
                    score = tempScore
                    action = item

        # neu do la luot choi nho nhat cua ghost
        else: 
            actionList = gameState.getLegalActions(agent)
            for item in actionList: 
                 # lay diem minimax cua successor
                nextGameState = gameState.generateSuccessor(agent, item)

                # tang gia tri agent cho luot choi tiep theo
                tempAction, tempScore = self.minimax(nextGameState, depth, agent + 1)

                # cap nhat lai score va action tot nhat
                if score is None or tempScore < score:
                    score = tempScore
                    action = item
        
        # neu do la trang thai trong hang cuoi cung, khong co trang thai ke tiep thi tra ve gia tri ham danh gia
        if score is None:
            return None, self.evaluationFunction(gameState)
        return action, score  

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        (neu agentIndex=0 thi tra ve hanh dong cua pacman, >=1 thi tra ve hanh dong cua ghost)

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

        # lay ra gia tri diem va hanh dong cho pacman
        action, score = self.minimax(gameState, 0, 0) 
        return action      
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, agent, depth, a, b):  # maximizer function
            # khoi tao gia tri v
            v = float("-inf")

            # lay ra danh sach hanh dong cua tac tu
            actionList = gameState.getLegalActions(agent)

            # duyet danh sach hanh dong
            for action in actionList:
                successor = gameState.generateSuccessor(agent, action)

                # cap nhat lai gia tri v
                v = max(v, self.alphaBetaPrune(successor, 1, depth, a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v
    
    def minValue(self, gameState, agent, depth, a, b):  
            # khoi tao gia tri v
            v = float("inf")

            nextAgent = agent + 1  

            # tinh agent tiep theo va depth tuong ung
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1

            actionList = gameState.getLegalActions(agent)

            # duyet cac hanh dong cua tac tu
            for action in actionList:
                successor = gameState.generateSuccessor(agent, action)

                # cap nhat lai gia tri min
                v = min(v, self.alphaBetaPrune(successor, nextAgent, depth, a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v
    
    def alphaBetaPrune(self, gameState, agent, depth, a, b):
            
            # tra ve ham danh gia khi chua dat den do sau xac dinh hoac khi game thang/ thua
            if depth == self.depth or gameState.isWin() or gameState.isLose(): 
                return self.evaluationFunction(gameState)

            if agent == 0: # neu la pacman
                return self.maxValue(gameState, agent, depth, a, b)
            else:  
                return self.minValue(gameState, agent, depth, a, b)
            
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """Performing maximizer function to the root node i.e. pacman using alpha-beta pruning."""

        # khoi tao cac gia tri
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        action = Directions.WEST

        for state in gameState.getLegalActions(0):
            nextGameState = gameState.generateSuccessor(0, state)
            ghostValue = self.alphaBetaPrune(nextGameState, 1, 0, alpha, beta)
            if ghostValue > v:
                v = ghostValue
                action = state
            if v > beta:
                return v
            alpha = max(alpha, v)

        return action

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
        """
        "*** YOUR CODE HERE ***"
        action, score = self.expectimax(gameState, 0, 0)
        return action
        # util.raiseNotDefined()

    def expectimax(self, gameState, depth, agent):

        # tang chi so cua tac tu va do sau khi cac agent da choi xong luot cua ho
        if agent >= gameState.getNumAgents():
            agent = 0
            depth += 1

        if depth == self.depth: # khi dat do sau toi da thi tra ve ham danh gia
            return None, self.evaluationFunction(gameState) 
        
        # khoi tao bien
        score = None
        action = None

        if agent == 0: # neu la luot choi toi da cua pacman
            actionList = gameState.getLegalActions(agent)
            for item in actionList: # duyet cac hanh dong cua pacman
                # doi voi moi hanh dong, lay ra so diem cua successor va chuyen luot choi sang ghost
                nextGameState = gameState.generateSuccessor(agent, item)
                tempAction, tempScore = self.expectimax(nextGameState, depth, agent + 1)

                # cap nhat diem va hanh dong tot nhat
                if score is None or score < tempScore:
                    score = tempScore
                    action = item

        else: # neu la luot choi min cua ghost
            ghostAction = gameState.getLegalActions(agent)
            n = len(ghostAction)
            if n is not 0:
                probability = 1.0 / n
            
            # duyet cac hanh dong cua ghost
            for item in ghostAction:
                nextGameState = gameState.generateSuccessor(agent, item)
                # lay so diem ki vong cua successor
                tempAction, tempScore = self.expectimax(nextGameState, depth, agent + 1)

                if score is None:
                    score = 0.0

                # cap nhat gia tri diem
                score += probability * tempScore
                action = item
        
        # neu do la trang thai trong dong cuoi cung, khong co cac trang thai con
        if score is None:
            return None, self.evaluationFunction(gameState)
        
        return action, score

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # lay ra danh sach thuc an 
    foodList = currentGameState.getFood().asList()

    # lay ra vi tri pacman
    pacmanPos = currentGameState.getPacmanPosition()

    # khoi tao bien minDistanceToFood la khoang cach gan nhat tu pacman den thuc an
    minDistanceToFood = -1

    # tinh khoang cach tu pacman den thuc an gan nhat
    for food in foodList:
        distance = util.manhattanDistance(pacmanPos, food)
        if minDistanceToFood == -1 or distance < minDistanceToFood:
            minDistanceToFood = distance

    # tinh khoang cach tu pacman den ghost 
    ghostList = currentGameState.getGhostPositions()
    distanceToGhost = 1
    isCollision = 0

    # duyet danh sach cac ghost
    for ghost in ghostList:
        # tinh khoang cach tu pacman den ghost
        distance = util.manhattanDistance(pacmanPos, ghost)
        distanceToGhost += distance

        # neu khoang cach tu pacman den ghost < 1 thi coi nhu va cham, tang so luong va cham len
        if distance <= 1:
            isCollision += 1

    # tinh so luong vien thuoc
    numberOfDrug = len(currentGameState.getCapsules())
    return currentGameState.getScore() + 1/float(minDistanceToFood) - 1/float(distanceToGhost) - isCollision - numberOfDrug

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
