__author__ = 'wuyihao'

import gzip
import pickle
import time
import operator


class PersonalRank:
    def __init__(self, epsilon):
        self._rank = dict()
        self.map = dict()
        self._r = dict()
        self._epsilon = epsilon
        self.test = dict()
        self._s = None

    # TODO randomize, weight by time, How to separate two sets

    def personal_rank(self, d, s):
        self._s = s
        self._rank = dict.fromkeys(self.map.iterkeys(), 0.0)
        self._rank[s] = 1

        while True:
            # The iteration is based on the previous vector
            # We need a 2-rolling pair (tmp, rank), to avoid circular dependency in one roll 's inner loop
            if not __debug__:
                print sorted(self._rank.values(), reverse=True)[0:5]
                time.sleep(0.1)
            tmp = dict.fromkeys(self.map.iterkeys(), 0.0)
            # All the other options have the probability: d
            # The option to start again on 's' is : 1 - d
            for u, vv in self.map.iteritems():
                for v in vv:
                    tmp[v] += 1.0 * d * self._rank[u] / len(self.map[u])
            tmp[s] += 1 - d
            # Quadratic sum as the error ratio
            delta = sum(map(lambda x, y: (x-y)*(x-y), tmp.itervalues(), self._rank.itervalues()))
            self._rank = tmp
            if delta < self._epsilon:
                break

    def predict(self, k):
        # Top K ranking candidates
        return map(operator.itemgetter(0),
                   sorted(self._rank.iteritems(), key=operator.itemgetter(1), reverse=True)[0:k])

    def verify(self, predicted):
        s = 0
        c = 0
        for u in self.test.iterkeys():
            if u in predicted:
                m = 0
                # For u, Predicted & Happened ratio Predicted
                for p in predicted[u]:
                    if p in self.test[u]:
                        m += 1
                # The numerator
                s += 1.0 * m / len(predicted[u])
                # The denominator
                c += 1
        # For all, average
        return 1.0 * s / c


class Loader:
    def __init__(self, mapp, test):
        self._map = mapp
        self._test = test
        self._hash = {}
        self._re_hash = {}
        self._hash_count = 0

    def hash_put(self, u):
        if u not in self._hash:
            self._hash_count += 1
            self._hash[u] = self._hash_count
            self._re_hash[self._hash_count] = u
        return self._hash[u]

    def load_train(self, day):
        db_dir = 'experiments'
        data_file = '{}/watch_day{}.pkl.gz'.format(db_dir, day)
        events = pickle.load(gzip.open(data_file, 'r'))
        for d in events:
            u = d['actor']
            v = d['repo']

            u = self.hash_put(u)
            v = self.hash_put(v)

            if v not in self._map:
                self._map[v] = set()
            self._map[v].add(u)

            if u not in self._map:
                self._map[u] = set()
            self._map[u].add(v)

    def load_test(self, day):
        db_dir = 'experiments'
        data_file = '{}/watch_day{}.pkl.gz'.format(db_dir, day)
        events = pickle.load(gzip.open(data_file, 'r'))
        for d in events:
            u = d['actor']
            v = d['repo']

            # Using integer as dict index is faster
            u = self.hash_put(u)
            v = self.hash_put(v)

            if u not in self._test:
                self._test[u] = set()
            self._test[u].add(v)

if __name__ == '__main__':
    pr = PersonalRank(1.0)
    ld = Loader(pr.map, pr.test)
    for i in range(7):
        ld.load_train(i)
    ld.load_test(7)
    predicts = {}
    for i in list(set(pr.test.iterkeys()) & set(pr.map.iterkeys())):
        pr.personal_rank(0.2, i)
        predicts[i] = pr.predict(5)
    precise = pr.verify(predicts)
    print 'Precise %f' % precise
