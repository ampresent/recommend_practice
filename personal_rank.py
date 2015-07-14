__author__ = 'wuyihao'

import gzip
import pickle
import time


class PersonalRank:
    def __init__(self):
        self.rev_map_ = dict()
        self._rank = dict()
        self.map = dict()
        self._r = dict()

    # TODO randomize, weight by time

    def personal_rank(self, d, s):
        self._rank = dict.fromkeys(self.map.keys() + self.rev_map_.keys(), 0.0)
        self._rank[s] = 1

        while True:
            # The iteration is based on the previous vector
            # We need a 2-rolling pair (tmp, rank), to avoid circular dependency in one roll 's inner loop
            if __debug__:
                print sorted(self._rank.values(), reverse=True)[0:15]
                time.sleep(1)
            tmp = dict.fromkeys(self.map.keys() + self.rev_map_.keys(), 0.0)
            for v in self.rev_map_.keys():
                for u in self.rev_map_[v]:
                    tmp[v] += 1.0 * d * self._rank[u] / len(self.map[u])
                # All the other options have the probability: d
                # The option to start again on 's' is : 1 - d
                tmp[s] += (1 - d)
            self._rank = tmp

class Loader:
    def __init__(self, map, rev_map):
        self._map = map
        self._rev_map_ = rev_map

    def load_daily_events(self, day):
        db_dir = 'experiments'
        data_file = '{}/watch_day{}.pkl.gz'.format(db_dir, day)
        events = pickle.load(gzip.open(data_file, 'r'))
        for d in events:
            u = d['actor']
            v = d['repo']
            if v not in self._rev_map_:
                self._rev_map_[v] = set()
            self._rev_map_[v].add(u)

            if u not in self._map:
                self._map[u] = set()
            self._map[u].add(v)

if __name__ == '__main__':
    pr = PersonalRank()
    ld = Loader(pr.map, pr.rev_map_)
    for i in range(8):
        ld.load_daily_events(i)
    pr.personal_rank(0.8, 'ronhuang')
