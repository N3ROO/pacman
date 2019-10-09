"""
In this file, you will implement generic search algorithms which are called by
Pacman agents.
"""

from pacai.util.stack import Stack


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the
    goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].
    """

    # We will push nodes in this stack
    stack = Stack()

    # We need to keep track of visited node so we don't run into an infinite
    # loop
    visited = []

    # The first state does not have any direction or cost, so we need to create
    # them
    stack.push((problem.startingState(), [], 0))

    while not stack.isEmpty():
        position, path, cost = stack.pop()

        if problem.isGoal(position):
            return path

        if position in visited:
            continue

        visited.append(position)

        successors = problem.successorStates(position)
        for successor in successors:
            # We keep track of the path for each node here by adding their
            # direction to the current node's path. A successor is a tuple
            # (position, direction, cost)
            stack.push(
                (
                    successor[0],
                    path + [successor[1]],
                    successor[2]
                )
            )

    # No path found!
    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    raise NotImplementedError()


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()
