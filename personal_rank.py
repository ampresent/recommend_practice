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

    # TODO randomize, weight by time

    def personal_rank(self, d, s):
        self._rank = dict.fromkeys(self.map.iterkeys(), 0.0)
        self._rank[s] = 1

        while True:
            # The iteration is based on the previous vector
            # We need a 2-rolling pair (tmp, rank), to avoid circular dependency in one roll 's inner loop
            if __debug__:
                print self._rank.values()[0:5]
                time.sleep(0.1)
            tmp = dict.fromkeys(self.map.iterkeys(), 0.0)
            for v in self.map.iterkeys():
                for u in self.map[v]:
                    tmp[v] += 1.0 * d * self._rank[u] / len(self.map[u])
                # All the other options have the probability: d
                # The option to start again on 's' is : 1 - d
                tmp[s] += 1 - d
            delta = reduce(operator.add, map(lambda x, y: (x-y)*(x-y), tmp.itervalues(), self._rank.itervalues()))
            self._rank = tmp
            if delta < self._epsilon:
                break

class Loader:
    def __init__(self, map):
        self._map = map

    def load_daily_events(self, day):
        db_dir = 'experiments'
        data_file = '{}/watch_day{}.pkl.gz'.format(db_dir, day)
        events = pickle.load(gzip.open(data_file, 'r'))
        for d in events:
            u = d['actor']
            v = d['repo']
            if v not in self._map:
                self._map[v] = set()
            self._map[v].add(u)

            if u not in self._map:
                self._map[u] = set()
            self._map[u].add(v)

if __name__ == '__main__':
    pr = PersonalRank(1e-10)
    ld = Loader(pr.map)
    for i in range(8):
        ld.load_daily_events(i)
    pr.personal_rank(0.2, 'ronhuang')
