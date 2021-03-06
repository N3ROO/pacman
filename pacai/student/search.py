"""
In this file, you will implement generic search algorithms which are called by
Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


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
        state, path, cost = stack.pop()

        if problem.isGoal(state):
            return path

        if state in visited:
            continue

        visited.append(state)

        successors = problem.successorStates(state)
        for successor in successors:
            # We keep track of the path for each node here by adding their
            # direction to the current node's path. A successor is a tuple
            # (state, direction, cost)
            stack.push(
                (
                    successor[0],
                    path + [successor[1]],
                    successor[2]
                )
            )

    # No path found!
    return None


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # For explainations, look at DFS code, it is the same algorithm but we
    # don't pop nodes in the same order
    queue = Queue()
    visited = []

    queue.push((problem.startingState(), [], 0))

    while not queue.isEmpty():
        state, path, cost = queue.pop()

        if problem.isGoal(state):
            return path

        if state in visited:
            continue

        visited.append(state)

        successors = problem.successorStates(state)
        for successor in successors:
            queue.push(
                (
                    successor[0],
                    path + [successor[1]],
                    successor[2]
                )
            )

    return None


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # The algorithm looks like DFS and BFS, the major difference is that we pop
    # nodes according to their cost. We will take the cheaper at each iteration

    pqueue = PriorityQueue()
    visited = []

    pqueue.push(
        (
            problem.startingState(),
            [],
            0
        ),
        0
    )

    while not pqueue.isEmpty():
        state, path, cost = pqueue.pop()

        if problem.isGoal(state):
            return path

        if state in visited:
            continue

        visited.append(state)

        successors = problem.successorStates(state)
        for successor in successors:
            pqueue.push(
                (
                    successor[0],
                    path + [successor[1]],
                    cost + successor[2]
                ),
                cost
            )

    return None


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # It is an improvement of UCS by taking in account heuristics.
    # good heuristic = good results!

    pqueue = PriorityQueue()
    visited = []

    pqueue.push(
        (
            problem.startingState(),
            [],
            0
        ),
        heuristic(problem.startingState(), problem)
    )

    while not pqueue.isEmpty():
        state, path, cost = pqueue.pop()

        if problem.isGoal(state):
            return path

        if state in visited:
            continue

        visited.append(state)

        successors = problem.successorStates(state)
        for successor in successors:
            pqueue.push(
                (
                    successor[0],
                    path + [successor[1]],
                    cost + successor[2]
                ),
                successor[2] + cost + heuristic(successor[0], problem)
            )

    return None
