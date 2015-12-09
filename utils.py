'''
Different useful functions, classes, methods are there
'''

import itertools


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
        self.size, self.n_features = X.shape


class NullLogger(object):
    def __init__(self):
        pass

    def push(self, string):
        return self

    def flush(self):
        return self


class PrintLogger(NullLogger):
    def __init__(self):
        import sys
        self.fo = sys.stdout

    def push(self, string):
        self.fo.write(string)
        self.fo.write('\n')
        return self

    def flush(self):
        self.fo.flush()
        return self


class FileLogger(NullLogger):
    def __init__(self, filename):
        self.fo = open(filename, 'w')

    def push(self, string):
        self.fo.write(string)
        self.fo.write('\n')
        return self

    def flush(self):
        self.fo.flush()
        return self

    def __del__(self):
        self.fo.close()


class Mapper(object):
    def __init__(self, parallel_profile=None):
        self._parallel_profile = parallel_profile
        if parallel_profile is None:
            pass
        elif str.startswith(parallel_profile, 'threads-'):
            from multiprocessing.pool import ThreadPool
            self.n_threads = int(parallel_profile[len('threads-'):])
            self.pool = ThreadPool(processes=self.n_threads)
            self.pool._maxtasksperchild = 10**5
            logger.push('Running parallel in {} threads'.
                        format(self.n_threads))
        else:
            from ipyparallel import Client
            self.rc = Client(profile=parallel_profile)
            self.dv = self.rc.direct_view()
            self.lbv = self.rc.load_balanced_view()
            with self.dv.sync_imports():
                import sys
            self.dv['path'] = sys.path
            with self.dv.sync_imports():
                from . import trainer, inspector, classifier, storage
            logger.push('Running parallel on cluster on {} cores'.format(
                        len(self.dv)))

    def imap(self):
        if self._parallel_profile is None:
            return itertools.map
        elif str.startswith(self._parallel_profile, 'threads-'):
            return self.pool.imap
        else:
            return self.dv.imap

    def map(self):
        if self._parallel_profile is None:
            return map
        elif str.startswith(self._parallel_profile, 'threads-'):
            return self.pool.map
        else:
            return self.dv.map_sync

    def gc_collect(self):
        if self._parallel_profile is None:
            gc_collect()
        elif str.startswith(self._parallel_profile, 'threads-'):
            gc_collect()
        else:
            self.dv.apply(gc_collect)

logger = PrintLogger()
