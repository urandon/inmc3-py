'''
Different useful functions, classes, methods are there
'''


def gc_collect():
    import gc
    gc.collect()


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


class PrintLogger(object):
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


class FileLogger(object):
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
