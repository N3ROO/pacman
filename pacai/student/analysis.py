"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    If we want the agent to cross the bridge, we just have to remove the noise, so that the agent
    never falls.
    """

    answerDiscount = 0.9
    answerNoise = 0  # updated

    return answerDiscount, answerNoise

def question3a():
    """
    We just need to change the living reward to a negative one to force the agent to finish fast.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -3  # updated

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    First, we need to change the living reward to a negative one so that the agent wants to finish
    fast. But we don't want him to fall. So we need to update the noise to tell him that it is not
    a good idea to go on the bridge. Then we need to change the discount rate. I have to admit that
    it was a bit random for this one. But it works with 0.5!
    """

    answerDiscount = 0.5  # updated
    answerNoise = 0.1  # updated
    answerLivingReward = -1  # updated

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    We just need a negative living reward (not too big so that it do won't go in +1)
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -1  # updated

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    We just need a negative living reward, but just a bit! So that it takes the longest path
    possible.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -0.1  # updated

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    We just have to tell that it is better for him to not end the game. So we put a big living
    reward.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 2

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    [Enter a description of what you did here.]
    """

    answerEpsilon = 0.3
    answerLearningRate = 0.5

    return answerEpsilon, answerLearningRate

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
