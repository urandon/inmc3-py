import numpy as np
import functools

class Sample(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.size, self.n_features = X.shape


class Classifier(object):
    
    _epsilon = np.double(0.01)
    
    def __init__(self, sample, feature_subset, sample_subset):
        self.X = sample.X
        self.y = sample.y
        self.feature_subset = feature_subset
        self.sample_subset = sample_subset
        
        self.n_samples = len(self.sample_subset)
        self.n_features = len(self.feature_subset)
        
        self.alpha = np.zeros((self.n_features))
        self.beta = np.zeros((self.n_features))
        
        if self.n_samples != 0:
            self._find_linear_coefficients()        
    
    def _find_linear_coefficients(self):
        y_ = np.nan_to_num(self.y[self.sample_subset])
        X_ = np.nan_to_num(self.X[self.sample_subset,:][:,self.feature_subset])
        
        x = np.sum(X_, axis=0)
        xy = np.dot(y_, X_)
        x2 = np.sum(np.square(X_), axis=0)
        y = np.sum(y_)
        
        m = np.abs(x2 - x*x / self.n_samples) > self._epsilon
        self.alpha[m] = (xy[m] - x[m]*y/self.n_samples) /\
            (x2[m] - np.square(x[m])/self.n_samples)
        self.beta = (y - self.alpha * x) / self.n_samples

    def X_sub(self):
        return self.X[self.sample_subset,:][:,self.feature_subset]
    
    def y_sub(self):
        return self.y[self.sample_subset,:][:,self.feature_subset]
                    
    # precedent is a np.ndarray of size (samples,features) or (features,)
    # classify using one feature per a time
    # precedents bellow should be passed with tripped features by self.feature_subset
    def classify_one(self, feature_idx, precedent):
        if len(precedent.shape) == 1:
            return self.alpha[feature_idx] * precedent[feature_idx] +\
            self.beta[feature_idx]
        else:
            return self.alpha[feature_idx] * precedent[:,feature_idx] +\
            self.beta[feature_idx]
    
    def classify(self, weights, precedent):
        return np.inner(weights, self.alpha * precedent + self.beta)
    
    # object_idx is an int or list
    def _X_(self, object_idxs): # get a subsubsample
        return self.X[[self.sample_subset[x] for x in object_idxs],:]
        
    def classify_training_one(self, feature_idx, object_idxs):
        return self.alpha[feature_idx] * self._X_(object_idx)[:,feature_idxs] +\
        self.beta[feature_idx]
    
    def classify_training(self, weights, object_idxs):
        return np.inner(weights, self.alpha *\
                        self._X_(object_idx)[:,self.feature_subset] +\
                        self.beta)

    def classify_training_all(self, weights):
        return np.inner(weights, self.alpha *\
                        self.X[self.sample_subset,:][:,self.feature_subset] + self.beta)
        

class ComplexClassifier(object):
    
    def __init__(self, weights, multiplier = 1, classifier = None):
        self.weights = weights
        self.multiplier = multiplier
        self.clf = classifier
        self.n_samles, self.n_features = 0, 0
        self.alpha, self.beta, self.variance, self.error =\
            (np.double(0) for x in xrange(4))
        
        if classifier is not None:
            self._set_classifier(classifier)        

    def set_classifier(self, cl):
        self.n_samples, self.n_features = cl.n_samples, cl.n_features
        self.alpha, self.beta = self._find_alpha_beta()
        
        X_ = self.classify_training_all()
        y_ = self.clf.y_sub()
        predicted = self.alpha * X_ + self.beta
        
        self.error = np.mean(np.square(predicted-y_))
        self.variance = np.var(predicted)
        
        
    def  _find_alpha_beta(self):
        X_ = np.nan_to_num(self.clf.classify_training_all(self.weights))
        y_ = self.clf.y_sub()
        nonnan_cnt = y.size - np.count_nonzero(np.isnan(y_))
        y_ = np.nan_to_num(y)

        x = np.sum(X_, axis=0)
        xy = np.dot(y_, X_)
        x2 = np.sum(np.square(X_), axis=0)
        y = np.sum(y_)
        
        alpha = (xy - x*y / nonnan_cnt) / (x2 - x*x / nonnan_cnt)
        beta = (y - alpha * x) / nonnan_cnt
        return alpha, beta                                
   
    def _raw_classify_training(self, object_idx):
        return self.clf.classify_training(self.weights, object_idx)
    
    def _raw_classify_training_all(self):
        return self.clf.classify_training_all(self.weights)
    
    def classify_training(self, object_idx):
        return self.alpha * _raw_classify_training(object_idx) + self.beta

    def classify_training_all(self):
        return self.alpha * _raw_classify_training_all() + self.beta
        
    def classify(self, precedent):
        return self.alpha * self.clf.classify(precedent) + self.beta
    
