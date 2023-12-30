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

        # Initialize evalution function
        evaluation = 0

        # Evaluate closest food
        closestFoodDistance = float('inf')
        for food in newFood.asList():
            currFoodDist = util.manhattanDistance(newPos, food)
            closestFoodDistance = min(closestFoodDistance, currFoodDist)
        evaluation += 1 / closestFoodDistance

        # Evaluate distance to ghost
        currDistToGhost = float('inf')
        for ghost in newGhostStates:
            currDistToGhost = util.manhattanDistance(newPos, ghost.getPosition())
            minDistanceFromPacman = 3
            if currDistToGhost < minDistanceFromPacman: # is Pacman within 3 grid blocks from current ghost?
                evaluation -= float('inf') # Then back away by reducing adversarial search
            elif newScaredTimes[0] > 0:
                evaluation += currDistToGhost + minDistanceFromPacman

        return successorGameState.getScore() + evaluation

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
        value = float("-inf")
        minimaxAction = None
        for action in gameState.getLegalActions(0): # Search Pacman's Actions
            successor = gameState.generateSuccessor(0, action)
            minVal = self.minValue(successor, self.depth, 1)
            if value < minVal:
                minimaxAction, value = action, minVal

        return minimaxAction

    def maxValue(self, currState, currDepth, agentIndex):
        v = float("-inf")
        if currDepth == 0 or currState.isWin() or currState.isLose(): # Base Case: Give the evaluation function at win or lose state
            return self.evaluationFunction(currState)
        
        for action in currState.getLegalActions(agentIndex):
            successor = currState.generateSuccessor(agentIndex, action)
            v = max(v, self.minValue(successor, currDepth, 1))

        return v
    
    def minValue(self, currState, currDepth, agentIndex):
        v = float("inf")
        if currState.isWin() or currState.isLose(): # Base Case: Give the evaluation function at win or lose state
            return self.evaluationFunction(currState)
        
        for action in currState.getLegalActions(agentIndex):
            successor = currState.generateSuccessor(agentIndex, action)
            lastGhostIndex = currState.getNumAgents() - 1
            if agentIndex == lastGhostIndex:
                v = min(v, self.maxValue(successor, currDepth - 1, 0)) # Minimize based on Pacman's chosen Action
            else:
                v = min(v, self.minValue(successor, currDepth, agentIndex + 1)) # Minimize based on Succeeding Ghost Action

        return v
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Alpha and Beta
        alpha = float("-inf")
        beta = float("inf")
        minimaxAction = None
        
        for action in gameState.getLegalActions(0): # Search Pacman's Actions
            successor = gameState.generateSuccessor(0, action)
            minVal = self.minValue(successor, self.depth, 1, alpha, beta)
            if minVal > alpha:
                alpha, minimaxAction = minVal, action

        return minimaxAction
    
    def maxValue(self, currState, currDepth, agentIndex, alpha, beta):
        v = float("-inf")
        if currDepth == 0 or currState.isWin() or currState.isLose(): # Base Case: Give the evaluation function at state
            return self.evaluationFunction(currState)
        
        for action in currState.getLegalActions(agentIndex):
            successor = currState.generateSuccessor(agentIndex, action)
            v = max(v, self.minValue(successor, currDepth, 1, alpha, beta))
            if v > beta: 
                return v
            alpha = max(alpha, v)

        return v
    
    def minValue(self, currState, currDepth, agentIndex, alpha, beta):
        v = float("inf")
        if currState.isWin() or currState.isLose(): # Base Case: Give the evaluation function at state
            return self.evaluationFunction(currState)
        
        for action in currState.getLegalActions(agentIndex):
            successor = currState.generateSuccessor(agentIndex, action)
            lastGhostIndex =  currState.getNumAgents() - 1
            if agentIndex == lastGhostIndex:
                v = min(v, self.maxValue(successor, currDepth - 1, 0, alpha, beta)) # Minimize based on Pacman's chosen Action
                if v < alpha: 
                    return v
                beta = min(beta, v)
            else:
                v = min(v, self.minValue(successor, currDepth, agentIndex + 1, alpha, beta)) # Minimize based on Succeeding Ghost Action
                if v < alpha: 
                    return v
                beta = min(beta, v)

        return v

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
        value = float("-inf")
        expectedAction = None
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            expVal = self.expValue(successor, self.depth, 1)
            if value < expVal:
                value, expectedAction = expVal, action

        return expectedAction
    
    def maxValue(self, currState, currDepth, agentIndex):
        v = float("-inf")
        if currDepth == 0 or currState.isWin() or currState.isLose(): # Base Case: Give the evaluation function at state
            return self.evaluationFunction(currState)
        
        for action in currState.getLegalActions(agentIndex):
            successor = currState.generateSuccessor(agentIndex, action)
            v = max(v, self.expValue(successor, currDepth, 1))

        return v
    
    def expValue(self, currState, currDepth, agentIndex):
        v = 0
        if currState.isWin() or currState.isLose(): # Base Case: Give the evaluation function at state
            return self.evaluationFunction(currState)
        
        ghostActions = currState.getLegalActions(agentIndex)
        for action in ghostActions:
            p = 1 / len(ghostActions) # Equal probability across all ghost actions
            successor = currState.generateSuccessor(agentIndex, action)
            lastGhostIndex =  currState.getNumAgents() - 1
            if agentIndex == lastGhostIndex:
                v += p * self.maxValue(successor, currDepth - 1, 0)
            else:
                v += p * self.expValue(successor, currDepth, agentIndex + 1)

        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
        This evaluation function is virtually the same as the 
        evaluation function in the 'Reflex Agent' class.
        However, I made some slight changes to variable values 
        and some extra conditions to ensure Pacman is maximizing
        his score output while keeping his distance from the ghosts
        at his most minimal.

        For example, I changed the minimum distance Pacman must be from
        the ghost from 3 to 1. This way, he'll still be able to avoid ghosts
        accordingly by subtracting from his evaluation function. Since a distance
        of 1 is less common than 3, it will incentivize greater evaluation outcomes.

        Another thing I changed was always having Pacman accumulate his score if the
        scared timers aren't active and if the ghosts aren't within minimum distance range.
        This way, Pacman can average a score past 1000. Without that condition, he
        would still be clearing the mazes properly, but he won't reach his maximum
        evaluated values.
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    evaluation = 0

    # Evaluate closest food
    closestFoodDistance = float('inf')
    for food in newFood.asList():
        currFoodDist = util.manhattanDistance(newPos, food)
        closestFoodDistance = min(closestFoodDistance, currFoodDist)
    evaluation += 1 / closestFoodDistance

    # Evaluate distance to ghost
    currDistToGhost = float('inf')
    for ghost in newGhostStates:
        currDistToGhost = util.manhattanDistance(newPos, ghost.getPosition())
        minDistanceFromPacman = 1 # Changed from 3 -> 1 to maximize his evaluation while still maintaining distance from ghost
        if currDistToGhost < minDistanceFromPacman: # is Pacman within 3 grid blocks from current ghost?
            evaluation -= float('inf') # Then back away by reducing adversarial search
        elif newScaredTimes[0] > 0:
            evaluation += currDistToGhost + minDistanceFromPacman
        else:
            evaluation += currentGameState.getScore() # Now simply have Pacman always accumulate his score when not accounting for Ghosts or Timers

    return currentGameState.getScore() + evaluation
    

# Abbreviation
better = betterEvaluationFunction
