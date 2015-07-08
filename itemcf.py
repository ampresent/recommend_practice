__author__ = 'wuyihao'
import operator

import math

class ItemCF:
    def __init__(self):
        self.user_item = {}
        self.item_user = {}
        self.item_item = {}
        self.rank = {}
        self.N = {}

    def read_from_file(self, filename):
        with open(filename) as f:
            for line in f.readlines():
                line = map(int, line.split())
                if line[1] not in self.item_user:
                    self.item_user[line[1]] = set()
                self.item_user[line[1]].add(line[0])

                if line[0] not in self.user_item:
                    self.user_item[line[0]] = set()
                self.user_item[line[0]].add(line[1])

    def train(self):
        # Rehash from origin data set
        for u in self.item_user.iterkeys():
            if u not in self.item_item:
                self.item_item[u] = dict()
            for v in self.item_user.iterkeys():
                if u != v:
                    if v not in self.item_item[u]:
                        self.item_item[u][v] = 0
                    self.item_item[u][v] += 1.0 * len(self.item_user[u] & self.item_user[v])\
                        / math.sqrt((len(self.item_user[u]) * len(self.item_user[v])))

    def recommend(self, target, k):
        interacted_items = self.user_item[target]
        for u in interacted_items:
            for v, w in sorted(self.item_item[u].items(), key=operator.itemgetter(1), reverse=True)[0:k]:
                if v not in interacted_items:
                    if v not in self.rank:
                        self.rank[v] = 0
                    self.rank[v] += w
        print self.rank
        return self.rank

if __name__ == '__main__':
    itemcf = ItemCF()
    itemcf.read_from_file('/tmp/try')
    itemcf.train()
    itemcf.recommend(3, 2)
