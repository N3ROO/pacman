"""
This file contains incomplete versions of some agents that can be selected to
control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.search import heuristic
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions
from pacai.core import distance


class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

    def startingState(self):
        """
        Answers the question:
        Where should the search start?

        note: Returns the start state (in your search space,
            NOT a `pacai.core.gamestate.AbstractGameState`)

        Returns the starting state for the search problem.
        """

        return (self.startingPosition,) + (False, False, False, False)

    def isGoal(self, state):
        """
        Answers the question:
        Is this state a goal?

        note: Returns whether this search state is a goal state of the problem

        Returns True if and only if the state is a valid goal state.
        """

        goal = state[1:5] == (True, True, True, True)

        if goal:
            # Register the locations we have visited.
            # This allows the GUI to highlight them.
            self._visitedLocations.add(state[0])
            self._visitHistory.append(state[0])

        return goal

    def successorStates(self, state):
        """
        Answers the question:
        What moves are possible from this state?

        note: Returns successor states, the actions they require, and a cost
        of 1. The following code snippet may prove useful

        Returns a list of tuples with three values:
        (successor state, action, cost of taking the action).
        """

        successors = []

        for action in Directions.CARDINAL:
            # bl = bottom left corner
            # tl = top left corner
            # br = bottom right corner
            # tr = top right corner
            pos, bl, tl, br, tr = state
            x, y = pos
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                newState = (
                        (nextx, nexty),
                        (nextx, nexty) == self.corners[0] or state[1],  # bl
                        (nextx, nexty) == self.corners[1] or state[2],  # tl
                        (nextx, nexty) == self.corners[2] or state[3],  # br
                        (nextx, nexty) == self.corners[3] or state[4]   # tr
                    )

                successors.append(
                    (
                        newState,
                        action,
                        1
                    )
                )

        # Bookkeeping for display purposes (the highlight in the GUI).
        self._numExpanded += 1
        if (state not in self._visitedLocations):
            self._visitedLocations.add(state[0])
            self._visitHistory.append(state[0])

        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full
    credit.)
    """

    # ---------
    # First try: following the walls (not exactly that, but it was the idea)
    # Result: 1897
    # ----------

    # position = state[0]
    # neighboors = []

    # for i in range(-1, 2):
    #     for j in range(-1, 2):
    #         if i == j == 0:
    #             continue
    #         neighboors.append(problem.walls[i + position[0]][j + position[1]])

    # return neighboors.count(False)

    # ---------
    # Second try: heuristic = distance to closest corner
    # Result: 2838 nodes...
    # ----------

    # We will get a tuple telling us which corner has already been visited: (T/F, T/F, T/F, T/F)
    # With: (bottom left, top left, bottom right, top right)
    # cornersStatus = state[1:5]
    # minDistance = 9999999999999999999
    # for index, cornerStatus in enumerate(cornersStatus):
    #     if not cornerStatus:  # if not visited yet
    #         dist = distance.manhattan(state[0], problem.corners[index])
    #         if dist < minDistance:
    #             minDistance = dist

    # return minDistance

    # ---------
    # Third try: heuristic = the shortest path between corners and pacman
    # Result: 427 nodes!
    # Note: it can be optimized by changing the state design
    # ----------

    heuristic = 0

    # First, we find all the corners that we still need to visit
    cornersToVisit = []
    for index, cornerStatus in enumerate(state[1:5]):
        if not cornerStatus:
            cornersToVisit.append(problem.corners[index])

    # This is our reference position, it will be updated with the closest
    # corner of the remaining corners
    comparaisonPos = state[0]

    # Now we will find the closest corner, and then the closest one to the
    # closest one until we checked all the corners that we need to visit
    while len(cornersToVisit) != 0:
        # Find closest corner
        closestCorner = cornersToVisit[0]
        minDistance = distance.manhattan(comparaisonPos, closestCorner)
        for cornerToVisit in cornersToVisit[1:]:
            # [1:] -> we don't want the first one because we assume that it is
            # the closest one
            if distance.manhattan(comparaisonPos, cornerToVisit) < minDistance:
                closestCorner = cornerToVisit
        # We found the closest corner, so we add its distance to the heurisitc
        heuristic += minDistance
        # Now we will loop once again with the last corner as the ref position
        comparaisonPos = closestCorner
        # And we don't forget to remove this corner from the corners to visit
        cornersToVisit.remove(closestCorner)

    return heuristic


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state

    # *** Your Code Here ***
    return heuristic.null(state, problem)  # Default to the null heuristic.


class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        raise NotImplementedError()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start=None):
        super().__init__(gameState, goal=None, start=start)

        # Store the food for later reference.
        self.food = gameState.getFood()


class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
