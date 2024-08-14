import numpy as np

# zero inflated negative binomial distribution, pi is probability for 0 draw
def zinb(size=(1000,100)):
    pi = 0.25
    p = np.random.uniform(0.1,0.9, size=size[1])
    p = np.repeat(np.expand_dims(p,axis=0),size[0],axis=0)
    X = np.random.negative_binomial(10,p)
    X = np.random.binomial(1,1-pi,size)*X
    return X

# zinb but makes data compositional
def zinb_comp(size=(1000,100)):
    X =  zinb(size)
    return np.expand_dims(1/np.sum(X,axis=1),axis=1)*X