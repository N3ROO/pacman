import random

from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import counter
from pacai.util import probability

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        self.qValues = counter.Counter()

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        # If the (state, action) pair has never been seen then it is 0 by default (see init()).
        return self.qValues[(state, action)]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        value = self.__getBestActionAndValue(state)[1]
        return value if value != - float("inf") else 0

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """

        actions = self.__getBestActionAndValue(state)[0]
        return random.choice(actions) if actions is not None else None

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
        we should take a random action and take the best policy action otherwise.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should choose None as the action.
        """

        action = None
        legalActions = self.getLegalActions(state)

        # We have to make sure that there are some legal actions
        if legalActions:
            if probability.flipCoin(self.getEpsilon()):
                # In this case we take the random action
                action = random.choice(legalActions)
            else:
                # Otherwise we take the best one
                action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a state transition and reward.
        You should do your Q-Value update here.
        Note that you should never call this function, it will be called on your behalf.
        """

        # Okay, so we don't want to return anything
        # 1) We just need to update the qValues
        # 2) Now we have to know HOW to update these values
        # 3) After taking a look at the wonderful lecture material, we know the formula:
        # V(s) = (1 - alpha) * V(s) + alpha * [R(s, policy(s), s') + lamda * V(s')]
        # 4) In the commens we can see that the function will be called on our behalf, so now we
        # have everything to code

        a = self.getAlpha()
        vs = self.getQValue(state, action)  # vs stands for V(s)
        vss = self.getValue(nextState)  # vss stands for V(s')
        drate = self.getDiscountRate()

        self.qValues[(state, action)] = (1 - a) * vs + a * (reward + drate * vss)

    def __getBestActionAndValue(self, state):
        """
        We will use this code multiple time, so we will use only one function to do so.
        returns a pair of ([list of best action], best value) from the state
        It returns (None, - float("inf")) if we are in a terminal state

        PS: it is the same function as the one in valueIterationAgent BUT it returns an array of
        possible actions.
        """

        # We need to compute V(s) = (a)max(Q(s, a))
        # In brief, we want the qValue for the best action

        actions = self.getLegalActions(state)
        bestActions = []  # it will happen if we are in the terminal state
        bestQValue = - float("inf")

        for action in actions:
            qValue = self.getQValue(state, action)

            if qValue > bestQValue:
                bestQValue = qValue
                bestActions = [action]
            elif qValue == bestQValue:  # I spent more time on that than I would admit
                bestActions.append(action)

        return bestActions if bestActions else None, bestQValue

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            raise NotImplementedError()
