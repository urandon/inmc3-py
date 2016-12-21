'''
Different useful functions, classes, methods are there
'''

import itertools
import numpy as np


def gc_collect():
    import gc
    gc.collect()


def top_combos(combos, k=40):
    return sorted(combos, key=lambda (c, (f, w)): f)[-k:]


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Sample(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X.flags.writeable = False
        self.y.flags.writeable = False
        self.size, self.n_features = X.shape

        self.cachable = not self.has_missing_values()
        if self.cachable:
            self.cache = None

    def has_missing_values(self):
        return np.isnan(self.X).any() or np.isnan(self.y).any()

    def copy(self):
        return Sample(self.X.copy(), self.y.copy())


class DatasetFrequientProblemChecker(object):
    @classmethod
    def check_all(cls, sample):
        failed_results = {
            checker for checker in [cls.check_duplicates]
            if not checker(sample)
        }

        for failed_check in failed_results:
            print('Check {} failed'.format(failed_check.__name__))
        return failed_check

    @staticmethod
    def check_duplicates(sample):
        return True  # TODO


# Loggers

class NullLogger(object):
    def __init__(self):
        pass

    def push(self, string, flush=True):
        return self

    def flush(self):
        return self


class PrintLogger(NullLogger):
    def __init__(self):
        import sys
        self.fo = sys.stdout

    def push(self, string, flush=True):
        self.fo.write(string)
        self.fo.write('\n')
        if flush:
            self.flush()
        return self

    def flush(self):
        self.fo.flush()
        return self


class FileLogger(NullLogger):
    def __init__(self, filename):
        self.fo = open(filename, 'w')

    def push(self, string, flush=False):
        self.fo.write(string)
        self.fo.write('\n')
        return self

    def flush(self):
        self.fo.flush()
        return self

    def __del__(self):
        self.fo.close()


# Mappers

class DummyMapperImpl(object):
    def __init__(self, parallel_profile=None):
        pass

    def map(self, *args, **kwargs):
        return map(*args, **kwargs)

    def imap(self, *args, **kwargs):
        return itertools.imap(*args, **kwargs)

    def gc_collect(self):
        gc_collect()

    def push(self, **kwargs):
        pass


class PoolMapperImpl(DummyMapperImpl):
    def __init__(self, n_threads):
        from multiprocessing.pool import ThreadPool
        self.n_threads = n_threads
        self.pool = ThreadPool(processes=self.n_threads)
        self.pool._maxtasksperchild = 10**5
        logger.push('Running parallel in {} threads'.
                    format(self.n_threads))

    def imap(self, *args, **kwargs):
        return self.pool.imap(*args, **kwargs)

    def map(self, *args, **kwargs):
        return self.pool.map(*args, **kwargs)


class IPyClusterMapperImpl(DummyMapperImpl):
    def __init__(self, parallel_profile):
        from ipyparallel import Client
        self.rc = Client(profile=parallel_profile)
        self.dv = self.rc.direct_view()
        self.lbv = self.rc.load_balanced_view()
        with self.db.sync_imports():
            import sys
        self.dv['sys.path'] = sys.path
        with self.dv.sync_imports():
            from . import trainer, inspector, classifier, storage
        logger.push('Running parallel on cluster with {} nodes'.
                    format(len(self.dv)))

    def map(self, *args, **kwargs):
        return self.dv.map_sync(*args, **kwargs)

    def imap(self, *args, **kwargs):
        return self.dv.imap(*args, **kwargs)

    def gc_collect(self):
        self.dv.apply(gc_collect)

    def push(self, **kwargs):
        self.dv.push(kwargs)


class Mapper(object):
    def __init__(self, parallel_profile=None):
        self._parallel_profile = parallel_profile
        if parallel_profile is None:
            self._impl = DummyMapperImpl()
        elif parallel_profile.startswith('threads-'):
            n_threads = int(parallel_profile[len('threads-'):])
            self._impl = PoolMapperImpl(n_threads)
        else:
            self._impl = IPyClusterMapperImpl(parallel_profile)

    def map(self, *args, **kwargs):
        return self._impl.map(*args, **kwargs)

    def imap(self, *args, **kwargs):
        return self._impl.imap(*args, **kwargs)

    def gc_collect(self):
        self._impl.gc_collect()

    def push(self, *args, **kwargs):
        self._impl.push(*args, **kwargs)


logger = PrintLogger()
