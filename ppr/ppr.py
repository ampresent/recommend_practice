# TODO 1. Test Top-k function   2. result decode with p   3. Test With dataset.
# TODO PO: Potential Optimization   NC: Not clear   m:nonzeroes(numedges)   Convention:'All assesments talk about a sole statement'
# TODO All the todo, may refer to similar ones, PLEASE / to FIND ALL OCCURRANCE
# TODO Remove getrow and getcol
# TODO Reevanluating P's time complexity

import itertools
import heapq
import operator
import numpy
from scipy import sparse
from scipy.sparse import linalg

"""
A must be bidirected
"""
def P(A):
    visited = set()
    p = []
    n = A.shape[0]
    A_csc = A.tocsc()
    # O(m) PO csr_matrix can reduce
    count = []
    hash_count = dict()
    hash_count_orig = dict()
    for x, group in itertools.groupby(A.row):
        l = len(list(group))
        count.append((l,-l,x))
        hash_count[x] = l
        hash_count_orig[x] = l
    heapq.heapify(count)
    while count:
    # O(n+m) REMOVE A ROW, one node could be removed more than once
        v = heapq.heappop(count)[2]
    # O(1)
        if v not in visited:
    # O(n)
            visited.add(v)
    # O(1)
            p.append(v)
            for ui in xrange(A_csc.indptr[v], A_csc.indptr[v+1]):
                u = A_csc.indices[ui]
                hash_count[u] -= 1
                heapq.heappush(count, (hash_count[u], hash_count_orig[u], u))
    # O(n)
    pt = [0] * n
    # O(n)
    for i, pi in enumerate(p):
        pt[pi] = i
    return p, pt

"""
A : adjacent matrix
doesn't sacrifice time complexity,
because replace list with binomial tree
(however,array could be even better, not so handy in python)

d : seeds set
"""
def lower_bound(A, d, c):
    # O(n)
    lb = dict(itertools.izip(d.row, d.data * c))
    # O(n)
    old_layer = set(d.row)
    visited = set(old_layer)
    # Introduce old_layer and new_layer to enable
    # transition from one WHOLE layer to another
    while old_layer:
        new_layer = set()
    # O(n)
        for u in old_layer:
    # O(m)
            A_relevant = A.getrow(u)
    # O(m)
            for v, w in itertools.izip(A_relevant.indices, A_relevant.data):
                if v not in visited:
    # O(n)
                    lb.setdefault(v, 0)
                    lb[v] += (1-c) * w * lb[u]
                    new_layer.add(v)
        for u in new_layer:
    # O(n)
            visited.add(u)
    # O(n)
        old_layer = new_layer
    return lb

# Shared by exact and upper_bound
exact_id1 = None
ub_id1 = None

def AD(A, pt):
    # TODO A.row , no need to tolist()
    # O(n)
    id = map(lambda i: pt[i], A.row)
    jd = map(lambda j: pt[j], A.col)
    # O(1)
    xd = A.data
    return sparse.coo_matrix((xd, (id, jd)))

def W(Ad, c):
    # O(m+n)
    return sparse.eye(Ad.shape[0]) - (1-c)*Ad

def QR(w, pt, d):
    w = w.tocoo()
    #print sparse.coo_matrix((d.data, (map(lambda x: pt[x], d.row), [0]*d.nnz))).todense()
    import spqr_wrapper
    # NC ASSUME it's optical
    # PO if we pass w as csc_matrix , we could avoid 2 transforms
    # g: O(|Q|)
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
    # O(n)
    sum_lb = sum(lb.itervalues())
    heaplb = zip(lb.values(), lb.keys())
    # O(n)
    heapq.heapify(heaplb)
    # K dummy nodes, After all, only keeps K of all
    # O(K)
    relevance = [(0, k) for k in xrange(K)]
    heapq.heapify(relevance)
    # NC ASSUME it's optical
    R_rev = linalg.inv(R)
    for i in xrange(n):
        # O(n)
        lbu, u = heapq.heappop(heaplb)
        # O(n)
        ubu = upper_bound(u, lbu, sum_lb, n)
        if ubu < theta:
            return list(relevance)
        else:
            # O(|Q|+|F|)
            # TODO When calc exactness, F=cPtR-1 not F=cR-1
            relevance_u = exact(R_rev, c, g, u)
            if relevance_u > theta:
                # Replace v with u, whose relevance is greater
                # O(n)
                v = heapq.heappop(relevance)
                heapq.heappush(relevance,(relevance_u, u))
                # O(n)
                # Refresh theta
                theta = relevance[0][0]
    return relevance


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
