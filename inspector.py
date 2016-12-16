import numpy as np
from . import classifier


class DatasetInspectorStatsHolder(object):
    '''
    The class is used for caching purposes in the case
    if feature subset leads to extracting submatrix of
    discrepancies and subsetting corresponding values
    without reevaluation.
    It can be done if `sample_subset` is all the dataset,
    i.e. data without missing values
    '''
    def __init__(self, sample):
        # train totally -- have to be eliminated if possible, replased by cache
        n_features, n_samples = sample.n_features, sample.size
        feature_subset = np.arange(n_features)
        sample_subset = np.arange(n_samples)

        clf = classifier.Classifier(sample, feature_subset, sample_subset)
        subC = sample.y[:, np.newaxis]
        values = clf.classify_one(feature_subset, sample.X)

        self.expecteds = values.mean(axis=0)
        self.variances = np.square(values - self.expecteds).mean(axis=0)
        self.discrepancies = np.zeros((n_features, n_features))

        tmp = np.square(values).sum(axis=0)[np.newaxis]
        self.discrepancies = tmp + tmp.T - 2 * np.dot(values.T, values)
        self.discrepancies /= n_samples

        eC = np.nanmean(subC)
        self.varC = np.square(np.nanstd(subC)).mean()
        self.pearson = np.zeros(n_features)

        e1, e2 = self.expecteds[np.newaxis], eC
        v1, v2 = self.variances, self.varC
        self.pearson = np.dot((sample.y - e2)[np.newaxis], values - e1)\
            / (n_samples * np.sqrt(v1 * v2))


class Inspector(object):

    DETERMINANT_EPS = 1e-30

    def __init__(self, sample, feature_subset):
        if sample.cachable:
            if sample.cache is None:
                sample.cache = DatasetInspectorStatsHolder(sample)
            self.expecteds = sample.cache.expecteds[feature_subset]
            self.variances = sample.cache.variances[feature_subset]
            self.discrepancies = sample.cache.discrepancies[feature_subset, :][:, feature_subset]
            self.varC = sample.cache.varC
            self.pearson = sample.cache.pearson[:, feature_subset]

            self.n_features = len(feature_subset)
            self.feature_subset = feature_subset
        else:
            self._init_noncached(sample, feature_subset)

    def _init_noncached(self, sample, feature_subset):
        self.pearson = [None]
        self.weights = [None]
        self.functional = None

        self.feature_subset = feature_subset
        self.n_features = len(feature_subset)
        self.sample_subset = np.nonzero(
            ~np.isnan(sample.X[:, feature_subset]).any(axis=1))
        if type(self.sample_subset) is tuple:
            self.sample_subset = self.sample_subset[0]
        self.n_samples = len(self.sample_subset)

        # train totally -- have to be eliminated if possible, replased by cache
        self.clf = classifier.Classifier(sample, self.feature_subset,
                                         self.sample_subset)
        subC = sample.y[self.sample_subset][:, np.newaxis]
        values = self.clf.classify_one(
            range(len(self.feature_subset)),
            sample.X[self.sample_subset, :][:, self.feature_subset])

        # get stats -- should be cached in case of data without missing values
        self.expecteds = values.mean(axis=0)
        self.variances = np.square(values - self.expecteds).mean(axis=0)
        self.discrepancies = np.zeros((self.n_features, self.n_features))

        tmp = np.square(values).sum(axis=0)[np.newaxis]
        self.discrepancies = tmp + tmp.T - 2 * np.dot(values.T, values)
        self.discrepancies /= self.n_samples

        self.eC = np.nanmean(subC)
        self.varC = np.square(np.nanstd(subC)).mean()
        self.pearson = np.zeros(self.n_features)

        e1, e2 = self.expecteds[np.newaxis], self.eC
        v1, v2 = self.variances, self.varC
        self.pearson = np.dot(
            (sample.y[self.sample_subset] - e2)[np.newaxis], values - e1)\
            / (self.n_samples * np.sqrt(v1 * v2))

    def check(self):
        if self.n_features > 1:
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


class MaxCorrelationInspector(Inspector):
    _epsilon = 1e-3
    single_functional_description = 'pearson'
    complex_functional_description = 'pearson'

    def __init__(self, sample, feature_subset):
        super(MaxCorrelationInspector, self).__init__(sample, feature_subset)
        self.alpha, self.beta, self.gamma = [np.double() for x in xrange(3)]
        self.cs = np.double()
        self.B0, self.B1, self.B2 = [None for x in xrange(3)]
        self.subset = feature_subset

    def subset_weights(self, reversed_):
        weights = [np.double() for x in xrange(self.n_features)]
        functional = None
        varC = self.varC
        discrepancies = self.discrepancies
        variances = self.variances

        if self.n_features == 2:
            v1, v2 = variances
            rho = discrepancies[0, 1]
            if (np.square(v1 - v2) - rho * (v1 + v2) == 0):
                return None  # TODO ???
            c1 = (v2 * v2 - v1 * v2 - v2 * rho) /\
                ((v1 - v2) * (v1 - v2) - rho * (v1 + v2))
            if not 0 <= c1 <= 1:
                return None
            weights = [c1, 1 - c1]
            functional = (c1 * (v1 - v2) + v2) /\
                np.sqrt((c1 * (v1 - v2) + v2 - c1 * (1 - c1) * rho) * varC)
            # theta = -2*v1*v2*rho / ((v1-v2)**2 - rho * (v1+v2))
            # functional = (theta/sqrt(varC)) /\
            #   np.sqrt(theta + rho * (theta-v2) * (theta-v1) / (v1-v2) ** 2)
        else:
            # phi = lambda idx: beta * PSI[i] - gamma * PHI[i]
            # psi = lambda idx: beta * PHI[i] - alpha * PSI[i]
            PSI = np.sum(reversed_, axis=1)  # refers to PSI in article
            PHI = reversed_.dot(variances)   # refers to PHI in article

            self.alpha = alpha = np.inner(variances, PHI)
            self.beta = beta = np.sum(PHI)
            self.gamma = gamma = np.sum(PSI)

            cs = beta * beta - alpha * gamma

            # bounds
            if cs == 0:
                return None
            phi_ = beta * PSI - gamma * PHI     # refers to Gamma1/cs
            psi_ = beta * PHI - alpha * PSI     # refers to Gamma0/cs
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
            self.B0, self.B1, self.B2 = B0, B1, B2

            def corr(theta):
                Q_2 = B0 + B1 * theta + B2 * theta * theta
                return theta / np.sqrt(varC * (theta - Q_2))

            theta = (2 * B0) / (1 - B1)

            functional = 0
            best_theta = None

            # check borders
            for testK, val in ((corr(val), val) for val in (left, right)):
                if testK > functional:
                    functional, best_theta = testK, val

            # check range
            if left < theta < right:
                testK = corr(theta)
                if testK > functional:
                    functional, best_theta = testK, theta

            if best_theta is None:
                return None

            weights = best_theta * phi_ / cs + psi_ / cs
            if (weights <= self._epsilon).any():
                return None

        return weights, functional
