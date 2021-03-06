import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        oldCapsules = currentGameState.getCapsules()

        score = 0

        # Pacman does not like ghost unless if he ate a capsule!
        for ghostState in successorGameState.getGhostStates():
            dist = distance.manhattan(newPosition, ghostState.getPosition())
            if dist <= 2:
                score += - 1000 if ghostState.getScaredTimer() == 0 else 500
            else:
                score += - 5 / dist if ghostState.getScaredTimer() == 0 else 10 / dist

        # Pacman likes food
        if newPosition in oldFood:
            score += 100
        else:
            for food in oldFood:
                score += 5 / distance.manhattan(food, newPosition)

        # Pacman likes capsules :)
        if newPosition in oldCapsules:
            score += 200
        else:
            for capsule in oldCapsules:
                score += 20 / distance.manhattan(capsule, newPosition)

        # And pacman does not like to stop
        if action == Directions.STOP:
            score -= 3

        # If only pacman needs to know his way
        # print(str(action) + " " + str(score))

        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """ returns the minimax action from the current gameState using getTreeDepth
        """
        return self.__minimax__(state, self.getTreeDepth())

    def getEvaluationFunction(self):
        return super().getEvaluationFunction()

    def __minimax__(self, state, depth):
        bestAction = Directions.STOP
        bestScore = - float("inf")

        for action in self.__getLegalActions__(state, 0):
            if action == Directions.STOP:
                continue
            successor = state.generateSuccessor(0, action)
            score = self.__maxValue__(successor, depth)
            if score > bestScore:
                bestAction = action
                bestScore = score

        return bestAction

    def __maxValue__(self, state, depth):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)

        v = - float("inf")
        for action in self.__getLegalActions__(state, 0):
            successor = state.generateSuccessor(0, action)
            v = max(v, self.__minValue__(successor, depth, 1))
        return v

    def __minValue__(self, state, depth, ghostId):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)

        v = float("inf")
        for action in self.__getLegalActions__(state, ghostId):
            successor = state.generateSuccessor(ghostId, action)
            if ghostId == state.getNumAgents() - 1:
                v = min(v, self.__maxValue__(successor, depth - 1))
            else:
                v = min(v, self.__minValue__(successor, depth, ghostId + 1))
        return v

    def __getLegalActions__(self, state, agentId):
        actions = state.getLegalActions(agentId)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        return actions

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        return self.__minimax__(state, self.getTreeDepth())

    def __minimax__(self, state, depth):
        bestAction = Directions.STOP
        bestScore = - float("inf")

        for action in self.__getLegalActions__(state, 0):
            successor = state.generateSuccessor(0, action)
            score = self.__maxValue__(successor, depth, - float("inf"), float("inf"))
            if score > bestScore:
                bestAction = action
                bestScore = score

        return bestAction

    def __maxValue__(self, state, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)

        maxValue = - float("inf")
        for action in self.__getLegalActions__(state, 0):
            successor = state.generateSuccessor(0, action)
            currentValue = self.__minValue__(successor, depth, alpha, beta, 1)
            maxValue = max(maxValue, currentValue)

            # alpha beta pruning
            alpha = max(alpha, currentValue)
            if beta <= alpha:
                break

        return maxValue

    def __minValue__(self, state, depth, alpha, beta, ghostId):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)

        minValue = float("inf")
        for action in self.__getLegalActions__(state, ghostId):
            successor = state.generateSuccessor(ghostId, action)
            if ghostId == state.getNumAgents() - 1:
                currentValue = self.__maxValue__(successor, depth - 1, alpha, beta)
            else:
                currentValue = self.__minValue__(successor, depth, alpha, beta, ghostId + 1)
            minValue = min(minValue, currentValue)

            # alpha beta pruning
            beta = min(beta, currentValue)
            if beta <= alpha:
                break

        return minValue

    def __getLegalActions__(self, state, agentId):
        actions = state.getLegalActions(agentId)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        return actions

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        bestAction = Directions.STOP
        bestScore = - float("inf")
        actions = self.__getLegalActions__(state, 0)

        for action in actions:
            successor = state.generateSuccessor(0, action)
            score = self.__expectiminimax__(successor, self.getTreeDepth(), 1)
            if score > bestScore:
                bestAction = action
                bestScore = score

        return bestAction

    def __expectiminimax__(self, state, depth, agentId):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)

        actions = self.__getLegalActions__(state, agentId)

        if agentId == 0:
            value = - float("inf")
            for action in actions:
                successor = state.generateSuccessor(agentId, action)
                value = max(value, self.__expectiminimax__(successor, depth - 1, agentId + 1))
            return value

        else:
            value = 0
            for action in actions:
                successor = state.generateSuccessor(agentId, action)
                nextAgent = (agentId + 1) % state.getNumAgents()
                value += self.__expectiminimax__(successor, depth, nextAgent)
            # v1 * 1/p + v2 * 1/p + ... + vk * 1/p = (v1 + v2 + ... + vk) / p
            return value / len(actions)

    def __getLegalActions__(self, state, agentId):
        actions = state.getLegalActions(agentId)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        return actions

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION:

    I took the work that I did in my reflex agent, and I improved it:

    1) I need to take into account the terminal states (won or lost)

    2) It is not really relevant to sum up the distance with all the remaining foods. I just need
    to tell to pacman where is the closest one.

    3) It was a good idea to tell to pacman to eat the ghosts if he could, or to run away fast from
    them if it was not the case. The calculation works the same way as for the food, but we change
    the operator (+ or -) according to the state of the closest ghost (scared or not).

    4) I had to clearly show the coefficients so that I can update it easily if needed. That's why
    the score is computed in detail at the end. It is ordered in ascending priority.

    5) The capsules are not needed to win the game, but I need to tell to pacman to eat them if he
    is close enough. That's why I don't compute the closest capsule distance. I prefer that pacman
    focuses on eating food and on dodging the ghosts than trying to get the capsules. So counting
    the remaining capsules is a really good solution, because if pacman sees a capsule next to him,
    he will jump on it.
    """

    pacman = currentGameState.getPacmanPosition()

    # We need to specify the terminal states
    if currentGameState.isWin():
        return + float("inf")
    elif currentGameState.isLose():
        return - float("inf")

    # We want to know where is the closest food
    foods = currentGameState.getFood().asList()
    closestFoodDistance = distance.manhattan(pacman, foods[0])
    for food in foods:
        dist = distance.manhattan(pacman, food)
        if dist < closestFoodDistance:
            closestFoodDistance = dist

    # We want the closest ghost distance. The distance will be negative
    # if the ghost is not scared, positive otherwise
    ghosts = currentGameState.getGhostStates()
    closestGhostDistance = distance.manhattan(pacman, ghosts[0].getPosition())
    for ghost in ghosts:
        dist = distance.manhattan(pacman, ghost.getPosition())
        if dist < abs(closestGhostDistance):
            closestGhostDistance = - dist if ghost.getScaredTimer() == 0 else dist

    # coefficients * parameters ordered in ascending priority
    score = 0
    score += + 1.0 * currentGameState.getScore()
    score += - 2.0 * closestFoodDistance
    score += + 3.0 * closestGhostDistance
    score += - 4.0 * len(currentGameState.getCapsules())

    return score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        if kwargs:
            print("[ContestAgent/WARNING]: You used parameters, but we will ignore depth and "
                + "eval function.")

        self._treeDepth = 3
        self._evaluationFunction = self.__eval__

    def getAction(self, state):
        bestAction = Directions.STOP
        bestScore = - float("inf")
        actions = self.__getLegalActions__(state, 0)

        for action in actions:
            successor = state.generateSuccessor(0, action)
            score = self.__expectiminimax__(successor, self.getTreeDepth(), 1)
            if score > bestScore:
                bestAction = action
                bestScore = score

        return bestAction

    def __expectiminimax__(self, state, depth, agentId):
        if state.isWin() or state.isLose() or depth == 0:
            return self.getEvaluationFunction()(state)

        actions = self.__getLegalActions__(state, agentId)

        if agentId == 0:
            value = - float("inf")
            for action in actions:
                successor = state.generateSuccessor(agentId, action)
                value = max(value, self.__expectiminimax__(successor, depth - 1, agentId + 1))
            return value

        else:
            # Here comes the change. The ghosts do not take random actions now. They want to
            # eat our friend pacman. We need to save him by telling him that the ghosts will
            # prefer taking actions that reduce the distance with him. Yes, we hacked into the
            # ghosts' code!
            pacman = state.getPacmanPosition()
            oldGhostDistance = distance.manhattan(pacman, state.getGhostPosition(agentId))
            coefficient = 10  # How much will take seriously the optimal ghost action
            n = 0  # The number to devide the total (to get the state average)

            value = 0
            for action in actions:
                successor = state.generateSuccessor(agentId, action)
                nextAgent = (agentId + 1) % state.getNumAgents()
                newGhostDistance = distance.manhattan(pacman, successor.getGhostPosition(agentId))

                # We will increase the weight of a state that the ghost prefers
                isScared = state.getGhostState(agentId).getScaredTimer() == 0
                if ((newGhostDistance < oldGhostDistance and isScared)
                        or (newGhostDistance > oldGhostDistance and not isScared)):
                    # He is not afraid and wants to go close to pacman OR
                    # He is afraid and wants to go far from pacman
                    value += self.__expectiminimax__(successor, depth, nextAgent) * coefficient
                    n += coefficient
                else:
                    value += self.__expectiminimax__(successor, depth, nextAgent) * 1
                    n += 1

            return value / n

    def __getLegalActions__(self, state, agentId):
        actions = state.getLegalActions(agentId)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        return actions

    def __eval__(self, currentGameState):
        """
        It is the same evaluation function as for question 4, but with coefficient
        changes.
        """

        pacman = currentGameState.getPacmanPosition()

        # We need to specify the terminal states
        if currentGameState.isWin():
            return + float("inf")
        elif currentGameState.isLose():
            return - float("inf")

        # We want to know where is the closest food
        foods = currentGameState.getFood().asList()
        closestFoodDistance = distance.manhattan(pacman, foods[0])
        for food in foods:
            dist = distance.manhattan(pacman, food)
            if dist < closestFoodDistance:
                closestFoodDistance = dist

        # We want the closest ghost distance. The distance will be negative
        # if the ghost is not scared, positive otherwise
        ghosts = currentGameState.getGhostStates()
        closestGhostDistance = distance.manhattan(pacman, ghosts[0].getPosition())
        for ghost in ghosts:
            dist = distance.manhattan(pacman, ghost.getPosition())
            if dist < abs(closestGhostDistance):
                closestGhostDistance = - dist if ghost.getScaredTimer() == 0 else dist

        # coefficients * parameters ordered in ascending priority
        score = 0
        score += + 1.0 * currentGameState.getScore()
        score += - 2.0 * closestFoodDistance
        score += + 2.0 * closestGhostDistance
        score += - 4.0 * len(currentGameState.getCapsules())

        return score
