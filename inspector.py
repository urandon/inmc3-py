import numpy as np
from . import classifier


class IInspectior(object):
    def error(i): pass

    def discrepancy(i, j): pass

    def variance(i): pass

    def weight(i): pass

    def functional(): pass

    def check(): pass

    def which_is_dominated_clf(cclf1, cclf2): pass

    def which_is_dominated_feature(feature1, feature2): pass

    def __init__(self, weights, clf):
        self.weights = weights
        self.clf = clf


class Inspector(object):

    DETERMINANT_EPS = 1e-5

    def __init__(self, sample, feature_subset):
        self.pearson = [None]
        self.weights = [None]
        self.functional = None

        self.sample = sample
        self.feature_subset = feature_subset
        self.n_features = len(feature_subset)
        self.feature_mapping = {sub: idx for (idx, sub)
                                in enumerate(feature_subset)}
        self.sample_subset = np.nonzero(
            ~np.isnan(self.sample.X[:, feature_subset].any(axis=1)))
        if type(self.sample_subset) is tuple:
            self.sample_subset = self.sample_subset[0]
        self.sample_mapping = {sub: idx for (idx, sub)
                               in enumerate(feature_subset)}
        self.n_samples = len(self.sample_subset)

        # train totally
        self.clf = classifier.Classifier(sample, self.feature_subset,
                                         self.sample_subset)
        subC = sample.y[self.sample_subset][:, np.newaxis]
        values = self.clf.classify_one(
            range(len(self.feature_subset)),
            sample.X[self.sample_subset, :][:, self.feature_subset])

        # get stats
        self.expecteds = values.mean(axis=0)
        self.errors = np.square(values - subC).mean(axis=0)
        self.variances = np.square(values-self.expecteds).mean(axis=0)
        self.discrepancies = np.zeros((self.n_features, self.n_features))

        tmp = np.square(values).sum(axis=0)[np.newaxis]
        self.discrepancies = tmp + tmp.T - 2 * np.dot(values.T, values)
        self.discrepancies /= (self.n_samples + 5)

        self.eC = np.nanmean(subC)
        self.varC = np.square(np.nanstd(subC)).mean()
        self.pearson = np.zeros(self.n_features)
        # pearson = [self.pearson(k, values) for in xrange(self.n_features)]
        e1, e2 = self.expecteds[np.newaxis], self.eC
        v1, v2 = self.variances, self.varC
        self.pearson = np.dot((self.sample.y-e2)[np.newaxis], values-e1) /\
                             (self.n_samples * np.sqrt(v1 * v2))

    def get_expected_val(self, values):
        return np.nanmean(values)

    def get_expected_f(self, feature):
        return self.expecteds[feature]

    def get_w_expected(self, weights):
        return self.clf.classify(self.sample.X).mean()

    def get_variance_feature(self, feature):
        return self.variances[feature]

    def get_variance_values(self, values):
        return np.nanstd(values).mean() ** 2

    def pearson(self, feature, values):
        # todo: checks
        e1, e2 = self.get_expected_f(feature), self.eC
        v1, v2 = self.get_variance_feature(feature), self.varC
        return np.inner(values[:, feature] - e1, self.sample.y - e2) /\
                       (self.n_samples * np.sqrt(v1 * v2))

    def check(self):
        if len(self.feature_subset) > 1:
            try:
                if np.abs(np.linalg.det(self.discrepancies))\
                        < self.DETERMINANT_EPS:
                    return False
                revrsd = np.linalg.inv(self.discrepancies)
            except np.linalg.LinAlgError:
                # print np.linalg.LinAlgError.__name__, ' got:',\
                #      '\nfeature_subset:', self.feature_subset,\
                #      '\ndiscrepancies:\n', self.discrepancies
                return False

            check_ = self.subset_weights(revrsd)
            if check_ is None:
                return False
            self.weights, self.functional = check_
            return True
        else:
            self.weights = np.array([1])
            self.functional = self.pearson[0][0]
            return True

    def which_is_dominated_clf(self, clf1, clf2):
        v1, v2 = clf1.variance, clf2.variance
        rho = np.square(clf1.classify_training-clf2.classify_training).mean()
        if (np.square(v1-v2) - rho * (v1+v2)) == 0:
            return None  # TODO: what to do?
        c1 = (v2*v2 - v1*v2 - v2*rho) / (np.square(v1-v2) - rho*(v1+v2))
        if c1 < 0:
            return clf1
        elif c1 > 1:
            return clf2
        return None

    def which_is_dominated_feature(self, feature1, feature2):
        if self.discrepancies[feature1][feature2] <\
                np.abs(self.errors[feature1], self.errors[feature2]):
            return feature1 if self.errors[feature1] >\
                self.errors[feature2] else feature2
        else:
            return None


class MaxCorrelationInspector(Inspector):
    _epsilon = 1e-3
    single_functional_description = 'pearson'
    complex_functional_description = 'pearson'

    def __init__(self, sample, feature_subset):
        super(MaxCorrelationInspector, self).__init__(sample, feature_subset)
        self.alpha, self.beta, self.gamma = [np.double() for x in xrange(3)]
        self.cs = np.double()
        self.B0, self.B1, self.B2 = [np.double() for x in xrange(3)]
        self.sample, self.subset = sample, feature_subset

    def subset_weights(self, reversed_):
        subsize = len(self.feature_subset)
        weights = [np.double() for x in xrange(subsize)]
        functional = None
        varC = self.varC
        discrepancies = self.discrepancies
        variances = self.variances

        if subsize == 2:
            v1, v2 = variances
            rho = discrepancies[0, 1]
            if (np.square(v1-v2) - rho * (v1 + v2) == 0):
                return None  # TODO ???
            c1 = (v2*v2 - v1*v2 - v2*rho) /\
                ((v1-v2)*(v1-v2) - rho*(v1+v2))
            if not 0 <= c1 <= 1:
                return None
            weights = [c1, 1-c1]
            functional = (c1 * (v1-v2) + v2) /\
                np.sqrt((c1 * (v1-v2) + v2 - c1 * (1-c1) * rho) * varC)
        else:
            # phi = lambda idx: beta * DI[i] - gamma * DE[i]
            # psi = lambda idx: beta * DE[i] - alpha * DI[i]
            DI = np.sum(reversed_, axis=1)
            DE = reversed_.dot(variances)

            self.alpha = alpha = np.inner(variances, DE)
            self.beta = beta = np.sum(DE)
            self.gamma = gamma = np.sum(DI)

            cs = beta*beta - alpha*gamma

            # bounds
            if cs == 0:
                return None
            phi_ = beta * DI - gamma * DE
            psi_ = beta * DE - alpha * DI
            val = -psi_ / phi_
            div = phi_
            if cs > 0:
                left = val[div > 0].max() if (div > 0).any() else -np.inf
                right = val[div < 0].min() if (div < 0).any() else +np.inf
            else:
                left = val[div < 0].max() if (div < 0).any() else -np.inf
                right = val[div > 0].min() if (div > 0).any() else +np.inf
            if left > right:
                return None

            B0, B1, B2 = 0, 0, 0
            norm_by = 2 * cs * cs
            B0 = psi_.dot(discrepancies).dot(psi_) / norm_by
            B2 = phi_.dot(discrepancies).dot(phi_) / norm_by
            B1 = (psi_.dot(discrepancies).dot(phi_) +
                  phi_.dot(discrepancies).dot(psi_)) / norm_by

            def K(theta):
                return theta / np.sqrt(
                    varC * (theta - B0 - B1*theta - B2*theta*theta))

            theta = (2 * B0) / (1 - B1)

            functional = 0
            best_theta = None

            # check borders
            for testK, val in ((K(val), val) for val in (left, right)):
                if testK > functional:
                    functional, best_theta = testK, val

            # check range
            if left < theta < right:
                testK = K(theta)
                if testK > functional:
                    functional, best_theta = testK, theta

            if best_theta is None:
                return None

            weights = best_theta * phi_ / cs + psi_ / cs
            if (weights <= self._epsilon).any():
                return None

        return weights, functional
