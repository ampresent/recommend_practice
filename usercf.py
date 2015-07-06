__author__ = 'wuyihao'
from operator import itemgetter
import math

class UserCF:
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
        for item, v in self.item_user.iteritems():
            for user in v:
                if user not in self.N:
                    self.N[user] = 0
                self.N[user] += 1
                if user not in self.user_user:
                    self.user_user[user] = dict()
                for user2 in v:
                    if user2 not in self.user_user[user]:
                        self.user_user[user][user2] = 0
                    self.user_user[user][user2] += 1
        for u in self.user_item.keys():
            if u in self.user_user:
                for v in self.user_item.keys():
                    if v in self.user_user[u]:
                        self.user_user[u][v] /= math.sqrt(self.N[u]*self.N[v])

    def recommend(self, target, K):
        interacted_items = self.user_item[target]
        for v, wuv in sorted(self.user_user[target].items(), key=itemgetter(1), reverse=True)[0:K]:
            for item in self.user_item[v]:
                if item not in interacted_items:
                    if item not in self.rank:
                        self.rank[item] = 0
                    self.rank[item] += wuv
        print self.rank
        self.rank = sorted(self.rank)
        return sorted(self.rank)

if __name__ == '__main__':
    usercf = UserCF()
    usercf.read_from_file('/tmp/try')
    usercf.train()
    usercf.recommend(3, 2)
