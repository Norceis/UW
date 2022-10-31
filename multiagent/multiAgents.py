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
import numpy as np
from multiagentTestClasses import MultiagentTreeState
from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        if currentGameState.isLose():
            return -float("inf")
        elif currentGameState.isWin():
            return float("inf")

        no_action = 0
        danger = 0
        has_food = 0
        point_score = 0

        if action == 'Stop':
            no_action -= 20

        for ghost in newGhostStates:
            if manhattan_distance(ghost.getPosition(), newPos) < 3:
                danger = -40
                break

        if currentGameState.hasFood(newPos[0], newPos[1]):
            has_food = 20

        for food_x in range(newFood.width):
            for food_y in range(newFood.height):
                if successorGameState.hasFood(food_x, food_y):
                    point_score += np.exp(-manhattan_distance((food_x, food_y), newPos) + 1)

        score = 1 * danger + 1 * has_food + 2 * point_score + no_action
        return score


def manhattan_distance(point_1: tuple, point_2: tuple):
    return (abs(point_1[0] - point_2[0]) + abs(point_1[1] - point_2[1]))


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        legal_moves = gameState.getLegalActions()
        results = []

        for move in legal_moves:
            state = gameState.generateSuccessor(self.index, move)
            results.append(self.step_down(state=state,
                                          depth=0,
                                          agent_index=self.index + 1, ))

        return legal_moves[results.index(max(results))]

    def step_down(self, state, depth, agent_index):
        if state.isWin() or state.isLose() or self.depth * 2 - 1 == depth:
            return self.evaluationFunction(state)

        legal_moves = state.getLegalActions(agent_index)
        results = []
        if not depth % 2:

            ghost_number = state.getNumAgents() - 1
            for move in legal_moves:
                new_state = state.generateSuccessor(agent_index, move)
                if agent_index == ghost_number:
                    results.append(self.step_down(state=new_state,
                                                  depth=depth + 1,
                                                  agent_index=0))
                else:
                    results.append(self.step_down(state=new_state,
                                                  depth=depth,
                                                  agent_index=agent_index + 1))

            return min(results)

        for move in legal_moves:
            new_state = state.generateSuccessor(0, move)
            results.append(self.step_down(state=new_state,
                                          depth=depth + 1,
                                          agent_index=self.index + 1))

        return max(results)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legal_moves = gameState.getLegalActions()
        results = []
        v, alpha, beta = float('-inf'), float('-inf'), float('inf')

        for move in legal_moves:
            state = gameState.generateSuccessor(self.index, move)
            new_state = self.step_down(state=state,
                                       depth=0,
                                       agent_index=self.index + 1,
                                       alpha=alpha,
                                       beta=beta)
            v = max(v, new_state)
            results.append(new_state)
            if v > beta:
                return state
            alpha = max(alpha, v)

        return legal_moves[results.index(max(results))]

    def step_down(self, state, depth, agent_index, alpha, beta):
        if state.isWin() or state.isLose() or self.depth * 2 - 1 == depth:
            return self.evaluationFunction(state)

        legal_moves = state.getLegalActions(agent_index)

        if not depth % 2:
            v = float('inf')
            ghost_number = state.getNumAgents() - 1
            for move in legal_moves:
                new_state = state.generateSuccessor(agent_index, move)
                if agent_index == ghost_number:
                    v = min(v, self.step_down(state=new_state,
                                              depth=depth + 1,
                                              agent_index=0,
                                              alpha=alpha,
                                              beta=beta))
                else:
                    v = min(v, self.step_down(state=new_state,
                                              depth=depth,
                                              agent_index=agent_index + 1,
                                              alpha=alpha,
                                              beta=beta))
                if v < alpha:
                    return v

                beta = min(beta, v)

            return v

        v = float('-inf')
        for move in legal_moves:
            new_state = state.generateSuccessor(self.index, move)
            v = max(v, self.step_down(state=new_state,
                                       depth=depth + 1,
                                       agent_index=self.index + 1,
                                       alpha=alpha,
                                       beta=beta))

            if v > beta:
                return v

            alpha = max(alpha, v)

        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legal_moves = gameState.getLegalActions()
        results = []

        for move in legal_moves:
            state = gameState.generateSuccessor(self.index, move)
            results.append(self.step_down(state=state,
                                          depth=0,
                                          agent_index=self.index + 1, ))

        return legal_moves[results.index(max(results))]

    def step_down(self, state, depth, agent_index):
        if state.isWin() or state.isLose() or self.depth * 2 - 1 == depth:
            return self.evaluationFunction(state)

        legal_moves = state.getLegalActions(agent_index)
        results = []
        if not depth % 2:

            ghost_number = state.getNumAgents() - 1
            for move in legal_moves:
                new_state = state.generateSuccessor(agent_index, move)
                if agent_index == ghost_number:
                    results.append(self.step_down(state=new_state,
                                                  depth=depth + 1,
                                                  agent_index=0))
                else:
                    results.append(self.step_down(state=new_state,
                                                  depth=depth,
                                                  agent_index=agent_index + 1))

            return sum(results) / len(results)

        for move in legal_moves:
            new_state = state.generateSuccessor(0, move)
            results.append(self.step_down(state=new_state,
                                          depth=depth + 1,
                                          agent_index=self.index + 1))

        return max(results)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    current_score = currentGameState.getScore()

    if currentGameState.isLose():
        return -float("inf")
    elif currentGameState.isWin():
        return float("inf")

    danger = 0
    point_score = 0

    for ghost in newGhostStates:
        if manhattan_distance(ghost.getPosition(), newPos) < 3:
            danger = -40
            break


    for food_x in range(newFood.width):
        for food_y in range(newFood.height):
            if currentGameState.hasFood(food_x, food_y):
                point_score += np.exp(-manhattan_distance((food_x, food_y), newPos) + 1)

    score = 1 * danger + 1 * point_score + 10 * current_score
    return score


# Abbreviation
better = betterEvaluationFunction
