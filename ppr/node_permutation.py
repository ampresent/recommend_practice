# TODO : 5. R^-1     6. g    7. exact(with fx)

import operator
import numpy
from scipy.sparse import *
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

# Shared by exact and upper_bound
exact_id1 = 0

def exact(u):
    # Shared by exact and upper_bound
    global exact_id1
    for i in range(n):

        exact_i =
        yield exact_i
        exact_id1 = exact_i

def upper_bound(u, lbu, sum_lb, n):
    # Shared by exact and upper_bound
    global exact_id1
    ubi = 1 - sum_lb + lbu
    yield ubi

    # Upper bound _ (i-1)
    ub_id1 = ubi
    for i in range(1, n):
        ub_i = lbu - exact_id1 + ub_id1
        yield ub_i
        ub_id1 = ub_i


def top_k(A, K, theta):
    lower_bound(A, d, c)
    sum_lb = sum(lb.itervalues())
    n = len(A)
    # K dummy nodes, After all, only keeps K of all
    relevance = dict.fromkeys(A.keys()[0:K], 0)
    for i in range(len(A)):
        u, lbu = max(lb.iteritems(), key=operator.itemgetter(1))
        lb.pop(u)
        ub = upper_bound(u, lbu, sum_lb, n)
        if ub < theta:
            return relevance
        else:
            relevance_u = exact(u, d)
            if relevance_u > theta:
                v = min(relevance.iteritems(), key=operator.itemgetter(1))[0]
                # Replace v with u, whose relevance is greater
                relevance_u.pop(v)
                relevance[u] = relevance_u
                # Refresh theta
                theta = min(relevance.iteritems(), key=operator.itemgetter(1))[1]

# A : fake sparse
# P : fake
# c : constant
# n : constant
# w : coo sparse
def w(A, P, c, n):
    # W has to be dense, so use numpy.array
    # O(|Q|)
    w = dict()
    for i in A:
        pi = P[i]
        for j in A[pi]:
            w[i][j] -= A[pi][P[j]]
    return coo_matrix(w)

# w : coo sparse
# n : constant
# q :
# r :
def q(w, n):
    # TODO is it slow? And it has side effects???
    qd = numpy.array(w.transpose())
    q = numpy.zeros((n, n))
    r = numpy.zeros((n, n))
    for i in range(n):
        for j in range(1, i):
            # TODO is it very slow???
            qd[i] -= w[i].dot(q[j]) * q[j]
        norm_qdi = numpy.linalg.norm(qd[i])
        q[i] = qd[i] /  numpy.linalg.norm(qd[i])
        r[i][i] = norm_qdi
        for j in range(i+1, n):
            r[i][j] = w[j].dot(q[i])

'''
# Reverse of R
def Rr(r):
    n = len(r)
    rr = numpy.zeros((n, n))
    for i in xrange(n-1, -1, -1):
        rr[i][i] = 1.0 / r[i][i]
        for j in xrange(n-1, i, -1):
            for
'''

# q: dense, p: fake, d: sparse
def g(q, p, d, n):
    # p reverse
    pr = numpy.zeros(n)
    for i in xrange(n):
        pr[p[i]] = i

    g = numpy.zeros(n)
    for i in xrange(n):
        g[i] = q[pr[i]] * d[i]

if __name__ == '__main__':
    '''
    A = {1: {2: 0.6, 3: 0.4}, 2: {3: 0.3, 4: 0.7}, 3: {1: 0.8, 4: 0.2}, 4: {}}
    d = {1: 0.9, 4: 0.1}
    c = 0.2
    lb = lower_bound(A, d, c)
    print lb
    '''
