import numpy as np


class IInspectior(object):
    def error(i): pass
    def discrepancy(i, j): pass
    def variance(i): pass    
    def weight(i): pass
    def functional(): pass
    def check(): pass
    def which_is_dominated_clf(cclf1, cclf2): pass
    def which_is_dominated_feature(feature1, feature2): pass
    
    def __init__(weights, clf):
        self.weights = weights
        self.clf = clf
        

class MaxCorrelationInspector(object):
    _epsilon = 1e-3
    single_functional_description = 'pearson'
    complex_functional_description = 'pearson'
    
    def __init__(self, sample, feature_subset):
        self.alpha, self.beta, self.gamma = [np.double() for x in xrange(3)]
        self.DE, self.DI = [], []
        self.cs = np.double()
        self.B0, self.B1, self.B2 = [np.double() for x in xrange(3)]        
        self.sample, self.subset = sample, feature_subset
        
    def subset_weights(reversed_):
        subsize = len(self.feature_subset)
        weights = [np.double() for x in xrange(subsize)]
        functional = None
        varC = self.varC #  TODO: figure out what is varC
        discrepancies = self.discrepancies #  TODO: figure out what is discrepancies
        variances = self.variances #  TODO: figure out what is variances
        
        if subsize == 2:            
            v1, v2 = variances
            rho = discrepancies[0, 1]            
            if (np.square(v1-v2) - rho * (v1 + v2)): return None # TODO ???
            c1 = (v2*v2 - v1*v2 + v2*rho) /\
                ((v1-v2)*(v1-v2) - rho*(v1+v2))
            if not 0 <= c1 <= 1: return None            
            weights = [c1, 1-c1]
            functional = (c1 * (v1-v2) + v2) /\
                np.sqrt((c1 * (v1-v2) + v2 - c1 * (1-c1) * rho) * varC)
        else:
            #phi = lambda idx: beta * DI[i] - gamma * DE[i]
            #psi = lambda idx: beta * DE[i] - alpha * DI[i]
            
            DI = np.sum(reversed_, axis = 1)
            DE = reversed_.dot(variances)
            cs = beta*beta - alpha*gamma
            
            alpha = np.inner(variance, DE)
            beta = np.sum(DE)
            gamma = np.sum(DI)
                        
            # bounds            
            if cs == 0: return None
            phi_ = beta * DI - gamma * DE
            psi_ = beta * DE - alpha * DI
            val = -psi_ / phi_
            if cs > 0:
                left  = np.hstack(val[div > 0], -np.inf).max()
                right = np.hstack(val[div < 0], +np.inf).min()
            else
                left  = np.hstack(val[div < 0], -np.inf).max()
                right = np.hstack(val[div > 0], +np.inf).min()
            if left > right : return None
            
            B0, B1, B2 = 0, 0, 0
            B0 = psi_.dot(discrepancies).dot(psi_)
            B2 = phi_.dot(discrepancies).dot(phi_)
            B1 = psi_.dot(discrepancies).dot(phi_) +\
                phi_.dot(discrepancies).dot(psi_)
            K = lambda theta: theta / np.sqrt(varC *\
                (theta - B0 - B1*theta - B2*theta*theta))
            
            theta = (2 * B0) / (1 - B1)
            
            functional = 0
            best_theta = None
            
            # check borders
            for testK, val in ((K(val), val) for val in (left, right)):
                if testK > functional: functional, best_theta = testK, val
            
            # check range
            if left < theta < right:
                testK = K(theta)
                if testK > functional: functional, best_theta = testK, theta
            
            if best_theta is None: return None
            
            weights = best_theta * phi_ / cs + psi_ / cs
            if (weights <= self._epsilon).any(): return 0
            
        return functional, weights
    
    
    