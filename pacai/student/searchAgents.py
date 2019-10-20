"""
This file contains incomplete versions of some agents that can be selected to
control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions
from pacai.core import distance
from pacai.student.search import breadthFirstSearch

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
            pos, blVisited, tlVisited, brVisited, trVisited = state
            x, y = pos
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                newState = ((nextx, nexty),
                            (nextx, nexty) == self.corners[0] or blVisited,
                            (nextx, nexty) == self.corners[1] or tlVisited,
                            (nextx, nexty) == self.corners[2] or brVisited,
                            (nextx, nexty) == self.corners[3] or trVisited)

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
    # Result: 692 nodes!
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
            dist = distance.manhattan(comparaisonPos, cornerToVisit)
            if dist < minDistance:
                closestCorner = cornerToVisit
                minDistance = dist
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

    # ---------
    # First try: we find the two furthest fruits, because we will need to get them anyway.
    # Then, we need to get one of those two. We will obviously take the closest one. Then
    # It should be working very well because we will take the food in order. However, it
    # may not be optimal when we will get to eat the first food.
    # Result: 7.8 seconds 7537 nodes -> 3/4, admissible and consistent!
    # Note: Could do better by improving the path from start to first further food?
    # Requierement: from pacai.core.distanceCalculator import Distancer
    # ---------

    # position, foodGrid = state

    # # Used to get the real distance (not manhattan or euclidian)
    # distancer = Distancer(problem)
    # foodList = foodGrid.asList()

    # # In case if there is only one food remaining, we don't need to find the furthers ones
    # if len(foodList) == 1:
    #     return distancer.getDistance(position, foodList[0])
    # # If there are no more foods, we can't calculate the heuristic
    # elif len(foodList) == 0:
    #     return 0

    # # We compare every food to every other food to retrieve the ones that are the most spaced
    # furtherFood1, furtherFood2 = foodList[0], foodList[1]
    # distBetweenFurtherFoods = distancer.getDistance(furtherFood1, furtherFood2)
    # i, j = 0, 0
    # while i < len(foodList):
    #     j = i + 1
    #     while j < len(foodList):
    #         dist = distancer.getDistance(foodList[i], foodList[j])
    #         if dist > distBetweenFurtherFoods:
    #             distBetweenFurtherFoods = dist
    #             furtherFood1, furtherFood2 = foodList[i], foodList[j]
    #         j += 1
    #     i += 1

    # # Now we find the distance from current Pacman position to the closer of previous two fruits
    # dist1 = distancer.getDistance(position, furtherFood1)
    # dist2 = distancer.getDistance(position, furtherFood2)

    # return distBetweenFurtherFoods + min(dist1, dist2)

    # ---------
    # Second try: the same thing, but with a better distance calculation, because I did not see
    # that "maze" function in the first try. I should not have used getDistance from Distancer
    # because it does not give the *real* maze distance. It works so well now!!!!
    # Result: 12.6 seconds 376 nodes!!!!!!!!!! admissible and consistent!
    #
    # Optimization idea:
    # It may be possible to optimize it by pre-computing the distances into the dict given. The
    # difficulty is to update the computed values if a food was eaten. We have to register the
    # original foodList, and then at each iteration, we have to make a diff, and we need to
    # update all the distances that were computed thanks to these eaten foods.
    # We can't directly update the distances computed because the dict is static, and will be the
    # same for each call to the heuristic. So we may call the heuristic with one food eaten, and
    # the in next call, the food can be back on the grid.
    # 1) If dict key does not exist: compute all the distances and register foodList in dict
    # 2) Make diff between registeredFoodList and foodList (we can do it with sets in python)
    # 3) LOCALLY update the computed distances by taking in account the eaten foods
    # 4) We can use the distances to calculate the heuristic: we want the biggest distance, so
    #      we may use a sort of priority list where the first one is the one with the bigger dist
    # Ideas:
    # computedDistances = list of tuples (distance, foodA, foodB)
    # With foodB being the FURTHER point from foodA, and distance, the distance between those two.
    # Remark: if the further food from foodA is foodB, it does not mean that the further food from
    # foodB is foodA.
    # ---------

    # We will need to calculate the distances using distance.maze. But it requires a SearchProblem
    # that has "getWalls()" method. Sadly, FoodSearchProblem does not have "getWalls" method, so we
    # will add dynamically this method to the problem instance it is a sort of hack, but the results
    # are fantastic

    # we convert the walls given by problem.walls into the good format
    wallList = [[problem.walls[y][x] for x in range(0, problem.walls.getHeight())]
                for y in range(0, problem.walls.getWidth())]

    # problem.walls is not callable, so we need to create a function that returns the list
    def getWalls():
        # I love python
        return wallList

    # We can now dynamically add a new method to the problem instance
    problem.getWalls = getWalls

    # Then, the algorithm is the same, but we use distance.maze!
    position, foodGrid = state

    # Used to get the real distance (not manhattan or euclidian)
    # distancer = Distancer(problem)
    foodList = foodGrid.asList()

    # In case if there is only one food remaining, we don't need to find the furthers ones
    if len(foodList) == 1:
        return distance.maze(position, foodList[0], problem)
    # If there are no more foods, we can't calculate the heuristic
    elif len(foodList) == 0:
        return 0

    # We compare every food to every other food to retrieve the ones that are the most spaced
    furtherFood1, furtherFood2 = foodList[0], foodList[1]
    distBetweenFurtherFoods = distance.maze(furtherFood1, furtherFood2, problem)
    i, j = 0, 0
    while i < len(foodList):
        j = i + 1
        while j < len(foodList):
            dist = distance.maze(foodList[i], foodList[j], problem)
            if dist > distBetweenFurtherFoods:
                distBetweenFurtherFoods = dist
                furtherFood1, furtherFood2 = foodList[i], foodList[j]
            j += 1
        i += 1

    # Now we find the distance from current Pacman position to the closer of previous two fruits
    dist1 = distance.maze(position, furtherFood1, problem)
    dist2 = distance.maze(position, furtherFood2, problem)

    return distBetweenFurtherFoods + min(dist1, dist2)

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

        return breadthFirstSearch(AnyFoodSearchProblem(gameState))


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

    def isGoal(self, state):
        if (state not in self.food.asList()):
            return False

        # Register the locations we have visited.
        # This allows the GUI to highlight them.
        self._visitedLocations.add(state)
        self._visitHistory.append(state)

        return True

class ApproximateSearchAgent(BaseAgent):
    """
    We will eat the food one by one. At each iteration, we will
    find which food is *really* the closest one by using BFS. And
    then, we will eat it!
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
        self._actionIndex = 0
        self._actions = []

    def getAction(self, state):
        """
        No modifications here.
        """

        if (self._actionIndex >= (len(self._actions))):
            return Directions.STOP

        action = self._actions[self._actionIndex]
        self._actionIndex += 1

        return action

    def registerInitialState(self, state):
        """
        At each iteration, we will find and eat the closest food.
        """

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestFood(currentState)
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestFood returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def optimizedMazeDistance(self, state, pos1, pos2, limit=-1):
        """
        It computes the REAL distance between the pacman and a position in the maze by
        using distance.mazeDistance().
        The limit parameter is used as an optimization. If the manhattan distance is
        bigger than the limit, we are sure that the real distance won't be better.
        We know that manhattan distance >= real distance by logic

        Without limit optimization: ~41 seconds
        With limit optimization: ~1.5 seconds
        """
        if limit != -1:
            if distance.manhattan(pos1, pos2) > limit:
                return 9999999
        return len(breadthFirstSearch(PositionSearchProblem(state, start=pos1, goal=pos2)))

    def findPathToClosestFood(self, state):
        """
        It finds the path to the closest food by taking in account the maze layout. It
        does not use euclidian distances. It is also optimized (see realDistance(...)).
        """
        foods = state.getFood().asList()

        pacman = state.getPacmanPosition()
        closestFood = foods[0]
        minDist = self.optimizedMazeDistance(state, pacman, closestFood)
        for food in foods:
            dist = self.optimizedMazeDistance(state, pacman, food, limit=minDist)
            if dist < minDist:
                minDist = dist
                closestFood = food

        return breadthFirstSearch(PositionSearchProblem(state, goal=closestFood))
