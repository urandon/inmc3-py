import numpy as np
from enum import Enum


class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

class ITrainer(object):    
    def train(sample, voting_quality_threshold, comparision_threshold,\
              filtering_type, combining_type, skip_selection,\
              logger = None):
        pass
    
    def forecast(train_sample, test_sample, logger = None):
        pass
    
    def get_gescription(voting_quality_threshold):
        pass
    
class NullLogger(object):
    def push(string):
        pass

    def flush():
        pass
    
    
class PrintLogger(object):
    def push(string):
        print string
    
    def flush():
        pass

        
class FileLogger(object):
    def __init__(self, filename):
        self.fo = open(filename, 'w')
    
    def push(string):
        fo.write(string)
        fo.write('\n')
    
    def flush():
        fo.flush()
            
    def __del__(self):
        self.fo.close()
    
    
class MaxCorrelationTrainer(object):    
    class FilteringType(Enum):
        Normalization = 0
        Domination = 1
    
    class CombiningType(Enum):
        Weighing = 0
        MNK = 1
    
    initial_single_functional = -np.inf
    best_functional_msg_template = 'Best {} ({}): {: .3f}'
    epsilon = 1e-4
    
    def __init__(self, voting_quality_threshold = 1e-3,\
                comparision_threshold = 1e-3,\
                filtering_type = FilteringType.Normalization,\
                combining_type = CombiningType.Weighing,\
                skip_selection = False, logger = PrintLogger()):
        self.voting_quality_threshold = voting_quality_threshold
        self.comparision_threshold = comparision_threshold
        self.filtering_type = filtering_type
        self.combining_type = combining_type
        self.skip_selection = skip_selection
        self.logger = logger
        
        self.n_features = None
        self.history = []
        self.dominating_combinations = None
        self.classifiers = []
        
        self.best_functional = None
        self.initial_single_functional = None
        
    def get_inspector(self, sample, subset):
        return MaxCorrelationInspector(sample, subset)
        #pass
    
    def initial_combinations_functional(self, best_single):
        return best_single
        #pass
    
    def classifier_multiplier(self, functional):
        return 1 / (1 - np.square(functional))
        #pass
    
    def is_functional_better(self, old_functional, new_functional):
        return new_functional > old_functional
        #pass
    
    def is_functional_not_worse(self, old_functional, new_functional, threshold):
        return is_functional_better(old_functional * (1 - threshold), new_functional)
        #pass
    
    def get_resulting_weights(self):
        if self.n_features == None: return []
        res_weights = np.zeros(self.n_features)
        for clf in self.classifiers:
            res_weights += clf.weights * clf.multiplier
        return res_weights
    
    def __str__(self):
        return '; '.join(map(lambda v: '{: .3f}'.format(v),\
                             self.get_resulting_weights()))
    
    def log_func(self, idx, functional, single=False):
        if single:
            logger.push(best_functional_msg_template.\
            format(MaxCorrelationInspector.single_functional_description,\
            idx, best_functional))
        else 
            logger.push(best_functional_msg_template.\
            format(MaxCorrelationInspector.complex_functional_description,\
            idx, best_functional))
    
    
    def train(self, sample, logger = self.logger): # sample is X, y tuple
        n_objects, n_features = sample.X.shape
        self.n_features = n_features
        pairs = [[] for x in xrange(n_features)]
        
        best_combination = None
        best_weights = None

        combinations = []
        # use all the features w/o selection
        features = range(n_features)
                
        self.best_functional = self.initial_single_functional
        
        for feature in features:
            subset = [feature]
            self.combinations.append(subset)
            tested = self.get_inspector(sample, subset)
            tested.check()
            
            functional = tested.functional
            self.history.append(tested)
            
            if self.enableSelection and functional > self.best_functional:
                self.best_functional = functional
                best_combination = subset
                best_weights = tested.weights
                    
        if self.enable_selection:
            log_func(1, self.best_functional, single=True)
            best_functional = initial_combinations_functional(self.best_functional)
            
            # create pair map
            for pair in ([x, y] for x in xrange(n_features)\
                         for y in xrange(x+1, n_features)):
                tested = self.get_inspector(sample, pair)
                if tested.check():
                    pairs[pair[0]].append(pair[1])
            
            # add list of combinations from the pair map
            for f_idx in xrange(1, n_features):
                best_prev_func = self.best_functional
                best_curr_func = self.initial_single_functional
                new_combinationations = []
                
                for combo in combinations:
                    last = combo[f_idx - 1]                    
                    if last == features[-1]:
                        continue
                    
                    for fpair in pairs[last]:
                        subset = combo + [fpair]
                        tested = self.get_inspector(sample, subset)
                        if not tested.check():
                            continue
                        functional = tested.functional
                        if not functional * self.comparision_threshold > best_prev_func:
                            continue
                        new_combinationations.append(subset)
                        history.append(tested)
                        if functional * self.comparision_threshold > best_curr_func:
                            best_curr_func = functional
                        if functional * self.comparision_threshold > self.best_functional:
                            self.best_functional = functional
                            best_combination = subset
                            best_weights = tested.weights
                if len(new_combinationations) <= 1: break
                combinationations = new_combinationations
                log_func(features[f_idx]+1, best_curr_func)
            # training results
            log_func('_', self.best_functional)
            logger.push('Best combination: ' + '; '.join(map(str, best_combination)))
            log_func('_')
            logger.push('Weights: ' + '; '.join(map(str, best_weights)))
        # show must go on
        high_resulted_combinations = [] # contains complex classifier
        log.push('All combinations: ')
        for spector in history:
            if is_functional_not_worse(best_functional, spector.functional,\
                                      self.voting_quality_threshold):
                weights_repr = ('{}({})'.format(i, w) for (i, w) in\
                                enumerate(spector.weights) if w > 0)
                log.push('{}: '.format(spector.functional()) +\
                        '; '.join(weights_repr))
                high_resulted_combinations.append(classifier.ComplexClassifier(\
                    np.maximum(spector.weights, 0)))
        log.flush()
        
        # TODO: todo is there
        exclude = np.zeros((len(high_resulted_combinations)), dtype=bool)
        self.dominating_combinations = []
        for idx, hrcombo in enumerate(high_resulted_combinations):
            if not exclude[idx]:
                self.dominating_combinations.append(hrcombo)
        
        
    def forecast(train_sample, test_sample, logger = self.logger):
        if self.dominating_combinations is None or\
        self.dominating_combinations == []:
            raise Exception # method hadn't trained jet
        res = np.zeros((test_sample.size))
        res_accepted = np.zeros((test_sample.size), dtype=bool)
        norms = np.zeros((test_sample.size))
        
        for cclf in self.dominating_combinations:
            _, feature_subset = np.nonzero(cclf.weights > 0)
            train_subset, test_subset = map(lambda sample:\
                np.nonzero(~np.isnan(sample.X[:,feature_subset].any(axis=1))),\
                (train_sample, test_sample))
                        
            weights = cclf.weights[feature_subset]
            nclf = classifier.ComplexClassifier(weights, multiplier=1)
            nclf.set_classifier(classifier.Classifier(\
                train_sample, feature_subset, train_subset))

            result = nclf.classify(test_sample[test_subset,:][:,feafeature_subset])
            result = np.nan_to_num(result)
            res[test_subset] += result * cclf.multiplier
            norms[test_subset] += cclf.multiplier
            res_accepted[test_subset] = True
        
        class_errors = np.zeros(2)
        counts = [0, 0]
        
        rejects = np.sum(~res_accepted)
        if rejects == test_sample.size:
            return None
        
        res[(norms > epsilon) & res_accepted] /= norms[(norms > epsilon) & res_accepted]
        y_test_predicted = np.double(res > 0.5)
        y_test = test_sample.y
        
        error = np.mean((y_test_predicted != y_test)[res_accepted])
        for class_ in [0, 1]:
            class_errors[class_] = np.sum(\
                ((y_test == class_) & (y_test_predicted != class_))[res_accepted])
        
        [[var_result, cov], [_, var_C]] = np.cov(\
            y_test_predicted[res_accepted], y_test[res_accepted])
        deviation = np.square((y_test-y_test_predicted)[res_accepted]).sum()
        
        stats = Struct(error=error, class_errors=class_errors, cov=cov,\
                       deviation=deviation, var_result=var_result, var_C=var_C)
        
        return res, stats
        
    
                                
        
                                
        
        
       
    
    
    