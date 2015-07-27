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

def lower_bound(A, d, c):
    visited = set()
    lb = {}
    queue = Queue.Queue()
    for seed, weight in d.iteritems():
        visited.add(seed)
        lb[seed] = c * weight
        for v in A[seed]:
            if v not in visited:
                visited.add(v)
                queue.put(v)

    old_layer = []
    while queue.not_empty():
        new_layer = []
        while queue.not_empty():
            v = queue.get()
            if v not in lb:
                lb[v] = 0
            # The bisection
            for u in old_layer:
                if u in A[v]:
                    lb[v] += A[u][v]*lb[u]
            lb[v] *= 1 - c

            for vv in A[u]:
                if vv not in visited:
                    visited.add(vv)
                    new_layer.append(vv)

        for u in new_layer:
            queue.put(u)
        old_layer = new_layer


def top_k(A):
    Va = A.iterkeys()[0:K]
    lower_bound(A,)
    for i in range(len(A)):

