from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0

        # The states won't change, so we can get them before
        states = self.mdp.getStates()

        # First, we need to make sure that we make enough iterations
        for i in range(self.iters):
            # We need a tmp values dict because we wil use self.values in our computations
            tmp = counter.Counter()

            states = self.mdp.getStates()
            for state in states:
                value = self.__getBestActionAndValue(state)[1]
                tmp[state] = 0 if value == - float("inf") else value

            # Now that we computed everything, we can update the actual values
            for state in states:
                self.values[state] = tmp[state]

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    def getPolicy(self, state):
        """
        returns the best action according to computed values.
        """

        return self.__getBestActionAndValue(state)[0]

    def getQValue(self, currentState, action):
        """
        Returns the q-value of the (state, action) pair.
        """

        # We need to compute that: SUM(p(s', a) * (R(s, a, s') + lambda * V(s'))
        # With:
        # - s being the current state,
        # - s' the next one,
        # - p(s', a) the probability to go to s' by taking the action a
        # - R(s, a, s') the reward to go to s' by taking the action a from s
        # - lambda the discount rate
        # - V(s') the value of the state s'

        # list of (nextState, prob) that we have by taking <action> from <currentState>
        transitions = self.mdp.getTransitionStatesAndProbs(currentState, action)
        qValue = 0

        for transition in transitions:
            nextState, probability = transition[0], transition[1]

            # We need to get R(s, a, s'):
            reward = self.mdp.getReward(currentState, action, nextState)
            # We already have lamda which is the discountRate
            drate = self.discountRate
            # And we already have V(s') by calling self.values[nextState]
            v = self.values[nextState]

            # Finally, we can compute the sum
            qValue += probability * (reward + drate * v)

        return qValue

    def __getBestActionAndValue(self, state):
        """
        We will use this code multiple time, so we will use only one function to do so.
        returns a pair of (best action, best value) from the state
        It returns (None, - float("inf")) if we are in a terminal state
        """

        # We need to compute V(s) = (a)max(Q(s, a))
        # In brief, we want the qValue for the best action

        actions = self.mdp.getPossibleActions(state)
        bestAction = None  # it will happen if we are in the terminal state
        bestQValue = - float("inf")

        for action in actions:
            qValue = self.getQValue(state, action)

            if qValue > bestQValue:
                bestQValue = qValue
                bestAction = action

        return bestAction, bestQValue
