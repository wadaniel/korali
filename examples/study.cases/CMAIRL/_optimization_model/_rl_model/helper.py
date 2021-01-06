def linear_reward(param, feature):
    nParam = len(param)
    nFeature = len(feature)

    nParamRequired = nFeature + 1
    assert nParam == nParamRequired, "Wrong amount of param provided."

    piterator = iter(param)
    reward = 0.

    # linear terms
    for i in range(nFeature):
        reward += next(piterator)*feature[i]

    # bias term
    reward += next(piterator)

    return reward


def quadratic_reward(param, feature):
    nParam = len(param)
    nFeature = len(feature)

    nParamRequired = 2*nFeature + 1 
    assert nParam == nParamRequired, "Wrong amount of param provided."

    piterator = iter(param)
    reward = 0.

    # quadratic terms
    for i in range(nFeature):
            reward += next(piterator)*feature[i]*feature[i]

    # linear terms
    for i in range(nFeature):
        reward += next(piterator)*feature[i]

    # bias term
    reward += next(piterator)

    return reward

def cross_quadratic_reward(param, feature):
    nParam = len(param)
    nFeature = len(feature)

    # 2 features : 6 param
    # 3 features : 10 param
    # 4 features : 15 param 
    nParamRequired = (nFeature+1)*nFeature/2 + nFeature + 1 
    assert nParam == nParamRequired, "Wrong amount of param provided."

    piterator = iter(param)
    reward = 0.

    # quadratic and cross terms
    for i in range(nFeature):
        for j in range(i, nFeature):
            reward += next(piterator)*feature[i]*feature[i]

    # linear terms
    for i in range(nFeature):
        reward += next(piterator)*feature[i]

    # bias term
    reward += next(piterator)

    return reward

