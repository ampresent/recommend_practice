__author__ = 'wuyihao'
from operator import itemgetter
import math

class UserIIF:
    def __init__(self):
        self.user_item = {}
        self.item_user = {}
        self.user_user = {}
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
        for item, v in self.item_user.iteritems():
            for user in v:
                if user not in self.user_user:
                    self.user_user[user] = dict()
                for user2 in v:
                    if user2 != user:
                        if user2 not in self.user_user[user]:
                            # The W matrix
                            self.user_user[user][user2] = 0
                        self.user_user[user][user2] += 1.0 / math.log(1 + len(self.item_user[item]))
        for u, vw in self.user_user:
            for v, w in vw:
                self.user_user[u][v] /= math.sqrt(len(self.user_item[u])*len(self.user_item[v]))

    def recommend(self, target, k):
        interacted_items = self.user_item[target]
        # Only refer to top K related user, k is not the bigger the better
        for v, w in sorted(interacted_items(), key=itemgetter(1), reverse=True)[0:k]:
            for item in self.user_item[v]:
                # Don't recommend what's already known to me
                if item not in interacted_items:
                    if item not in self.rank:
                        self.rank[item] = 0
                    self.rank[item] += w
        self.rank = sorted(self.rank)
        return self.rank

if __name__ == '__main__':
    useriif = UserIIF()
    useriif.read_from_file('/tmp/try')
    useriif.train()
    useriif.recommend(3, 2)
