import operator
import Queue
"""
A must be bidirected
"""

def node_permutation(A):
    p = []
    # O(E)
    count = {key: len(value) for key, value in A.iteritems()}
    # Duplicate
    count0 = dict(count)
    n = len(A)
    for i in range(n):
        # O(n)
        min_count = min(count.values())
        # O(n)
        u_set = filter(lambda f: f[1] == min_count, count.iteritems())
        # O(nlgn)
        degree_set = {u: count0[u] for u, _ in u_set}
        # O(n)
        v, max_count0 = max(degree_set.iteritems(), key=operator.itemgetter(1))
        # O(1)
        p.append(v)
        count.pop(v)
        # O(nlgn)
        for u in count.iterkeys():
            if v in A[u]:
                count[u] -= 1
    return p

"""
A : adjacent matrix
doesn't sacrifice time complexity,
because replace list with binomial tree
(however,array could be even better, not so handy in python)

d : seeds set
"""
def lower_bound(A, d, c):
    lb = dict.fromkeys(A.iterkeys(), 0)
    for seed, weight in d.iteritems():
        lb[seed] = c * weight
    old_layer = set(d.iterkeys())
    visited = set(old_layer)
    # Introduce old_layer and new_layer to enable
    # transition from one WHOLE layer to another
    while len(old_layer) > 0:
        new_layer = set()

        for u in old_layer:
            for v in A[u]:
                if v not in visited:
                    # TODO A[u][v] or A[v][u]?????
                    lb[v] += (1-c)*A[u][v]*lb[u]
                    new_layer.add(v)
        for u in new_layer:
            visited.add(u)

        print new_layer
        old_layer = new_layer
    return lb


def top_k(A, K, lb, theta):
    lower_bound(A, d, c)
    # K dummy nodes
    relevance = dict.fromkeys(A.keys()[0:K], 0)
    for i in range(len(A)):
        u = max(lb.iteritems(), key=operator.itemgetter(1))[0]
        lb.pop(u)
        ub = upper_bound(u)
        if ub < theta:
            return Va
        else:
            relevance_u = exact(u, d)
            if relevance_u > theta:
                v = min(relevance.iteritems(), key=operator.itemgetter(1))[0]
                # Replace v with u, whose relevance is greater
                relevance_u.pop(v)
                relevance[u] = relevance_u
                # Refresh theta
                theta = min(relevance.iteritems(), key=operator.itemgetter(1))[1]


if __name__ == '__main__':
    A = {1: {2: 0.6, 3: 0.4}, 2: {3: 0.3, 4: 0.7}, 3: {1: 0.8, 4: 0.2}, 4: {}}
    d = {1: 0.9, 4: 0.1}
    c = 0.2
    lb = lower_bound(A, d, c)
    print lb
