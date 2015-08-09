# TODO 1. Test Top-k function   2. result decode with p   3. Test With dataset

import operator
import numpy
from scipy import sparse
from scipy.sparse import linalg
"""
A must be bidirected
"""

def P(A):
    p = []
    # O(E)
    count = {key: A.getrow(key).nnz for key in xrange(A.shape[0])}
    # Duplicate
    count0 = dict(count)
    for i in xrange(A.shape[0]):
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
            # TODO getrow could be slow
            if v in A.getrow(u).indices:
                count[u] -= 1
    pt = [0 for _ in xrange(len(p))]
    for i in xrange(len(p)):
        pt[p[i]] = i
    return p, pt

"""
A : adjacent matrix
doesn't sacrifice time complexity,
because replace list with binomial tree
(however,array could be even better, not so handy in python)

d : seeds set
"""
def lower_bound(A, d, c):
    lb = dict(zip(d.row, d.data * c))
    old_layer = set(d.row)
    visited = set(old_layer)
    # Introduce old_layer and new_layer to enable
    # transition from one WHOLE layer to another
    while len(old_layer) > 0:
        new_layer = set()

        for u in old_layer:
            A_relevant = A.getrow(u)
            for v, w in zip(A_relevant.indices, A_relevant.data):
                if v not in visited:
                    # TODO A[u][v] or A[v][u]?????
                    if v not in lb:
                        lb[v] = 0
                    lb[v] += (1-c) * w * lb[u]
                    new_layer.add(v)
        for u in new_layer:
            visited.add(u)

        old_layer = new_layer
    return lb

# Shared by exact and upper_bound
exact_id1 = None
ub_id1 = None

def AD(A, pt):
    # TODO A.row , no need to tolist()
    id = map(lambda i: pt[i], A.row)
    jd = map(lambda j: pt[j], A.col)
    xd = A.data
    return sparse.coo_matrix((xd, (id, jd)))

def W(Ad, c):
    return sparse.eye(A.shape[0]) - (1-c)*Ad

def QR(w, pt, d):
    w = w.tocoo()
    #print sparse.coo_matrix((d.data, (map(lambda x: pt[x], d.row), [0]*d.nnz))).todense()
    import spqr_wrapper
    # TODO if we pass w as csc_matrix , we could avoid 2 transforms
    Z_data, Z_row, Z_col,\
    R_data, R_row, R_col =\
    spqr_wrapper.qr(w.data.tolist(),
                    w.row.tolist(),
                    w.col.tolist(),
                    w.shape[0], w.shape[1],
                    d.data.tolist(), d.row.tolist())
    g = sparse.coo_matrix((Z_data, (Z_row, Z_col)))
    R = sparse.coo_matrix((R_data, (R_row, R_col)))
    return g, R

def exact(R_rev, c, g, u):
    # Shared by exact and upper_bound
    global exact_id1, ub_id1
    exact_i = (c * R_rev.getrow(u) * g).data[0]
    exact_id1 = exact_i
    return exact_i

def upper_bound(u, lbu, sum_lb, n):
    # Shared by exact and upper_bound
    global exact_id1, ub_id1
    ubi = 0
    if ub_id1 == None:
        ubi = 1 - sum_lb + lbu
    else:
        # Upper bound _ (i-1)
        ubi = lbu - exact_id1 + ub_id1
    ub_id1 = ubi
    return ubi


def top_k(lb, g, R, n, K, theta):
    sum_lb = sum(lb.itervalues())
    # K dummy nodes, After all, only keeps K of all
    relevance = dict.fromkeys(xrange(K), 0)
    R_rev = linalg.inv(R)
    for i in xrange(n):
        u, lbu = max(lb.iteritems(), key=operator.itemgetter(1))
        lb.pop(u)
        ubu = upper_bound(u, lbu, sum_lb, n)
        print lbu, ubu
        if ubu < theta:
            return relevance
        else:
            relevance_u = exact(R_rev, c, g, u)
            if relevance_u > theta:
                v = min(relevance.iteritems(), key=operator.itemgetter(1))[0]
                # Replace v with u, whose relevance is greater
                relevance.pop(v)
                relevance[u] = relevance_u
                # Refresh theta
                theta = min(relevance.iteritems(), key=operator.itemgetter(1))[1]
    return relevance

'''
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
'''

'''
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

# Reverse of R
def Rr(r):
    n = len(r)
    rr = numpy.zeros((n, n))
    for i in xrange(n-1, -1, -1):
        rr[i][i] = 1.0 / r[i][i]
        for j in xrange(n-1, i, -1):
            for
'''

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
'''

if __name__ == '__main__':
    A = sparse.coo_matrix([
        [0, 0.6, 0.4, 0],
        [0, 0, 0.3, 0.7],
        [0.8, 0, 0, 0.2],
        [0, 0, 0, 0]]
    )
    d = sparse.coo_matrix(([0.9, 0.1], ([0, 3], [0, 0])))
    #d = [0.9, 0, 0, 0.1]
    c = 0.2
    p, pt = P(A)
    d.row = numpy.array(map(lambda x: pt[x], d.row))
    Ad = AD(A, pt)
    w = W(Ad, c)
    g, R = QR(w, pt, d)
    lb = lower_bound(Ad, d, c)
    res = top_k(lb, g, R, Ad.shape[0], 3, 0.0)
    print res
