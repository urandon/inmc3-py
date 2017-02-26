import numpy as np
import itertools as it

from itertools import cycle
try:
    from itertools import izip
except ImportError: # python3 compatibility
    izip = zip

from . import classifier
from . import inspector
from . import storage
from . import utils


class MaxCorrelationTrainer(object):

    initial_single_functional = -np.inf
    best_functional_msg_template = 'Best {} ({}): {: .3f}'
    epsilon = 1e-4

    def __init__(self,
                 selection_threshold=(1 - 1e-2),
                 generation_threshold=(1 - .5 * 1e-2),
                 filtering_type='domination',
                 combining_type='mnk',
                 skip_selection=False, logger=utils.logger,
                 parallel_profile=None,
                 iterable_map=True):
        self.selection_threshold = selection_threshold
        self.generation_threshold = generation_threshold
        self.filtering_type = filtering_type
        self.combining_type = combining_type
        self.enable_selection = not skip_selection
        self.logger = logger

        self.n_features = None
        self.noncollapsed_combinations = storage.TreeStorage(data_handled=True)
        self.dominating_combinations = None
        self.classifiers = []

        self.best_functional = 0.0
        self.initial_single_functional = 0.0
        self.mapper = utils.Mapper(parallel_profile)
        self.iterable_map = iterable_map

    @staticmethod
    def get_inspector(sample, subset):
        return inspector.MaxCorrelationInspector(sample, subset)

    @staticmethod
    def initial_combinations_functional(best_single):
        return best_single

    @staticmethod
    def classifier_multiplier(functional):
        return 1 / (1 - np.square(functional))

    @staticmethod
    def is_functional_better(old_functional, new_functional):
        return new_functional > old_functional

    @staticmethod
    def is_functional_not_worse(old_functional, new_functional, threshold):
        return MaxCorrelationTrainer.is_functional_better(
            old_functional * (1 - threshold), new_functional)

    def get_resulting_weights(self):
        if self.n_features is None:
            return []
        res_weights = np.zeros(self.n_features)
        for clf in self.classifiers:
            res_weights += clf.weights * clf.multiplier
        return res_weights

    def __str__(self):
        return '; '.join(map(lambda v: '{: .3f}'.format(v),
                             self.get_resulting_weights()))

    def log_func(self, idx, functional, single=False):
        descr = inspector.MaxCorrelationInspector.\
            single_functional_description if single else\
            inspector.MaxCorrelationInspector.complex_functional_description
        self.logger.push(self.best_functional_msg_template.
                         format(descr, idx, functional))
        self.logger.flush()

    def train(self, sample, force_garbage_collector=True):
        logger, log_func = self.logger, self.log_func
        n_objects, n_features = sample.X.shape
        self.n_features = n_features
        pairs = [[] for x in range(n_features)]

        best_combination = None
        best_weights = None

        combinations = storage.TreeStorage(data_handled=False)
        # use all the features w/o selection
        features = range(n_features)

        self.best_functional = self.initial_single_functional

        def hist_push(inspctr):
            self.noncollapsed_combinations.add_node(
                inspctr.feature_subset,
                data=(inspctr.functional, inspctr.weights))

        for feature in features:
            subset = [feature]
            combinations.append(subset)
            tested = self.get_inspector(sample, subset)
            if tested.check():
                functional = tested.functional
                hist_push(tested)

            if self.enable_selection and functional > self.best_functional:
                self.best_functional = functional
                best_combination = subset
                best_weights = tested.weights

        log_func(1, self.best_functional, single=True)
        self.best_functional = self.initial_combinations_functional(
            self.best_functional)

        pmap = self.mapper.imap if self.iterable_map\
            else self.mapper.map
        self.mapper.push(sample=sample.copy())

        for first in range(n_features):

            def pair_check(pair):
                if self.get_inspector(sample, pair).check():
                    return pair[1]
                return None

            second_check = pmap(pair_check, izip(cycle([first]), range(first + 1, n_features)))
            pairs[first] = list(filter(None, second_check))
        logger.push('pairs found = {}'.format(
            sum((len(pair) for pair in pairs))))

        def combo_pair_iter(combos):
            for combo in combos:
                last = combo[-1]
                for second in pairs[last]:
                    yield combo + [second]

        def test_check(combo):
            tested = self.get_inspector(sample, combo)
            if not tested.check():
                return None
            if not tested.functional * self.generation_threshold > best_prev_func:
                return None
            return utils.Struct(feature_subset=tested.feature_subset,
                                functional=tested.functional,
                                weights=tested.weights)

        for iter_idx in range(1, n_features):
            best_prev_func = self.best_functional
            new_combinations = storage.TreeStorage(data_handled=False)
            if force_garbage_collector:
                self.mapper.gc_collect()

            testeds = pmap(test_check, combo_pair_iter(combinations))
            for (combo, tested) in izip(combo_pair_iter(combinations),
                                           testeds):
                if tested is None:
                    continue
                hist_push(tested)
                new_combinations.append(combo)
                if tested.functional > self.best_functional:
                    self.best_functional = tested.functional
                    best_combination = combo
                    best_weights = tested.weights
            if len(new_combinations) <= 1:
                break

            del combinations
            combinations = new_combinations

            log_func(iter_idx + 1, self.best_functional)
            logger.push('\tcombinations to process: {}'.
                        format(len(combinations)))

        if self.enable_selection:
            top_combinations = utils.top_combos_thresh(
                self.noncollapsed_combinations,
                self.best_functional * self.selection_threshold
            )
        else:
            top_combinations = self.noncollapsed_combinations

        # training results
        log_func('_', self.best_functional)
        logger.push(
            'Best combination: {c}\n'
            'Weights: {w}\n'
            '#Selected combinations {nsc}\n'
            '#Total combinations: {nnc}\n'.format(
                c='; '.join(map(str, best_combination)),
                w='; '.join(map(str, best_weights)),
                nsc=len(top_combinations),
                nnc=len(self.noncollapsed_combinations)
            )).flush()

        return top_combinations

    '''
    DEPRECATED
    '''
    def forecast(self, train_sample, test_sample, all_results=True):
        # logger = self.logger
        if self.dominating_combinations is None or\
                self.dominating_combinations == []:
            raise Exception  # method hadn't trained jet
        res = np.zeros((test_sample.size))
        res_accepted = np.zeros((test_sample.size), dtype=bool)
        norms = np.zeros((test_sample.size))

        dominating_results = np.zeros((
            test_sample.size, len(self.dominating_combinations)))

        for cidx, cclf in enumerate(self.dominating_combinations):
            # _, feature_subset = np.nonzero(cclf.weights > 0)
            # weights = cclf.weights[feature_subset]
            feature_subset, weights = cclf.feature_subset, cclf.weights
            train_subset, test_subset = map(
                lambda sample: np.nonzero(
                    ~np.isnan(sample.X[:, feature_subset].any(axis=1)))[0],
                (train_sample, test_sample))

            nclf = classifier.ComplexClassifier(weights, multiplier=1,
                                                feature_subset=feature_subset)
            nclf.set_classifier(classifier.Classifier(
                train_sample, feature_subset, train_subset))

            result = nclf.classify(test_sample.X[test_subset, :])
            result = np.nan_to_num(result)

            dominating_results[:, cidx] = result

            res[test_subset] += result * cclf.multiplier
            norms[test_subset] += cclf.multiplier
            res_accepted[test_subset] = True

        if all_results:  # other stats will be estimated by user
            return dominating_results

        # estimate stats
        class_errors = np.zeros(2)

        rejects = np.sum(~res_accepted)
        if rejects == test_sample.size:
            return None

        epsilon = self.epsilon
        res[(norms > epsilon) & res_accepted] /= \
            norms[(norms > epsilon) & res_accepted]
        y_test_predicted = np.double(res > 0.5)
        y_test = test_sample.y

        error = np.mean((y_test_predicted != y_test)[res_accepted])
        for class_ in [0, 1]:
            class_errors[class_] = np.sum(
                ((y_test == class_) & (y_test_predicted != class_))
                [res_accepted])

        [[var_result, cov], [_, var_C]] = \
            np.cov(y_test_predicted[res_accepted], y_test[res_accepted])
        deviation = np.square((y_test - y_test_predicted)[res_accepted]).sum()

        stats = utils.Struct(error=error, class_errors=class_errors,
                             cov=cov, deviation=deviation,
                             var_result=var_result, var_C=var_C)
        return res, stats


class FeatureGenerator(object):

    @staticmethod
    def from_combinations(train_sample, dest_sample, combinations):
        ncombos = len(combinations)
        ndest_samples = dest_sample.size
        dest_new_X = np.zeros((ndest_samples, ncombos))
        dest_new_X[:] = np.nan
        for cidx, (feature_subset, (_, weights)) in enumerate(combinations):
            train_subset = np.nonzero(~np.isnan(
                train_sample.X[:, feature_subset]).any(axis=1))[0]
            dest_subset = np.nonzero(~np.isnan(
                dest_sample.X[:, feature_subset]).any(axis=1))[0]

            cclf = classifier.ComplexClassifier(weights, multiplier=1,
                                                feature_subset=feature_subset)
            cclf.set_classifier(classifier.Classifier(
                train_sample, feature_subset, train_subset))
            dest_new_X[dest_subset, cidx] = np.nan_to_num(
                cclf.classify(dest_sample.X[dest_subset, :]))
        return dest_new_X
