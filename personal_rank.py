__author__ = 'wuyihao'

import gzip
import pickle
import time
import operator
import math


class PersonalRank:
    def __init__(self, epsilon):
        self._rank = dict()
        self.map = dict()
        self._r = dict()
        self._epsilon = epsilon
        self.test = dict()
        self._s = None

    # TODO randomize, weight by time

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
            for u, vw in self.map.iteritems():
                for v, w in vw.iteritems():
                    tmp[v] += 1.0 * d * self._rank[u] * w
            tmp[s] += 1 - d
            # Quadratic sum as the error ratio
            delta = sum(map(lambda x, y: (x-y)*(x-y), tmp.itervalues(), self._rank.itervalues()))
            self._rank = tmp
            if delta < self._epsilon:
                break

    def predict(self, k):
        # Top K ranking candidates
        result = map(operator.itemgetter(0),
                     sorted(filter(lambda x: x[0] < 0 and x[1] > 0.0, self._rank.iteritems()),
                            key=operator.itemgetter(1), reverse=True)[0:k])
        # TODO it seems the recommend results covers both user and respositories
        '''
        if self._s in result:
            result.remove(self._s)
        '''
        '''
        if len(result) > 0:
            print 'DEBUG', self._s
            print result[0:3]
        '''
        return result

    def verify(self, predicted):
        m = 0
        c1 = 0
        c2 = 0
        for u in self.test.iterkeys():
            if u in predicted:
                for p in predicted[u]:
                    if p in self.test[u]:
                        m += 1
                c1 += len(predicted[u])
                c2 += len(self.test.keys())

        accuracy = 1.0 * m / c1
        recall = 1.0 * m / c2
        return accuracy, recall


class Loader:
    ACTOR = 0
    REPO = 1

    def __init__(self, mapp, test, weighting='norm', arg=()):
        self._map = mapp
        self._test = test
        self._hash = {}
        '''
        self._re_hash = {}
        '''
        self._hash_count_ACTOR = 0
        self._hash_count_REPO = 0
        self._weighting = weighting
        self._arg_ = arg

    def hash_put(self, u, hash_type):
        if u not in self._hash:
            if hash_type == Loader.ACTOR:
                self._hash_count_ACTOR += 1
                self._hash[u] = self._hash_count_ACTOR
            elif hash_type == Loader.REPO:
                self._hash_count_REPO -= 1
                self._hash[u] = self._hash_count_REPO
            '''
            self._re_hash[self._hash_count] = u
            '''
        return self._hash[u]

    def load_train(self, day):
        db_dir = 'experiments'
        data_file = '{}/watch_day{}.pkl.gz'.format(db_dir, day)
        events = pickle.load(gzip.open(data_file, 'r'))
        coefficient = 0

        # Initialize weighting method
        if self._weighting == 'time':
            theta = self._arg_
            #mu = time.time()
            # TODO the data sets are too far from now, so it's not accurate
            mu = time.mktime(time.strptime('2012-01-08T23:59:19Z','%Y-%m-%dT%H:%M:%SZ'))
            coefficient = 1.0 / (theta * math.sqrt(math.pi * 2.0))
        for d in events:
            u = d['actor']
            v = d['repo']
            u = self.hash_put(u, Loader.ACTOR)
            v = self.hash_put(v, Loader.REPO)
            if v not in self._map:
                self._map[v] = dict()
            if u not in self._map:
                self._map[u] = dict()

            if self._weighting == 'time':
                # Happening time
                x = time.mktime(time.strptime(d['created_at'], '%Y-%m-%dT%H:%M:%SZ'))
                # Gaussian distribution
                w = coefficient * math.exp(-(x-mu)*(x-mu)/(2.0*theta*theta))
            elif self._weighting == 'norm':
                w = 1
            self._map[v][u] = w
            self._map[u][v] = w

        # It seems very slow
        for u, vw in self._map.iteritems():
            s = 1.0 * sum(self._map[u])
            for v, w in vw.items():
                self._map[u][v] /= s

    def load_test(self, day):
        db_dir = 'experiments'
        data_file = '{}/watch_day{}.pkl.gz'.format(db_dir, day)
        events = pickle.load(gzip.open(data_file, 'r'))
        for d in events:
            u = d['actor']
            v = d['repo']

            # Using integer as dict index is faster
            u = self.hash_put(u, Loader.ACTOR)
            v = self.hash_put(v, Loader.REPO)

            if u not in self._test:
                self._test[u] = set()
            self._test[u].add(v)

if __name__ == '__main__':
    pr = PersonalRank(0.1)
    ld = Loader(pr.map, pr.test, 'norm', ())
    for i in range(0, 2):
        ld.load_train(i)
    for i in range(2, 8):
        ld.load_test(i)
    predicts = {}
    for i in list(set(pr.test.iterkeys()) & set(pr.map.iterkeys())):
        pr.personal_rank(0.2, i)
        # TOO much predictions reduce the precision!
        predicts[i] = pr.predict(3)
    precise, recall = pr.verify(predicts)
    print 'Precise %f   Recall %f' % (precise, recall)
