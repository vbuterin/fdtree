# FDTree (Fixed Depth Tree), Vitalik Buterin, Mar 2015
#
# This python script implements a novel tree design which tries to
# achieve hard depth limits, similar to red black trees et al, but
# through a different strategy: instead of giving each node a fixed
# maximum number of children and trying to force the depth to grow
# logarithmically, we force a constant depth, and instead have the
# width of each node grow. If a tree is set up with depth d, and has
# N nodes, then we have the invariant that each node will have at most
# 2 * N**(1/d) children. Hence, we guarantee insert, lookup and delete
# in O(N**(1/d)) time.
#
# When the size is approximately known to at least
# within a few orders of magnitude, this allows for fairly similar
# performance to traditional O(log(N)) tries: for example, setting
# d=12, 1000000 nodes gives ~3*12 = 36 node accesses (times a small
# constant) in order to perform an operation; a binary trie would
# require ~2*20 = 40 node accesses also times a small constant.
# Increasing the dataset to a gigabyte, the binary tree becomes 60
# accesses and the fixed-depth tree 67 accesses, and decreasing to
# a kilobyte the binary tree becomes 20 accesses and the fixed-depth
# tree 21 accesses. Of course, in reality, the performance of each
# algorithm depends entirely on details such as the relative speed
# of reading and writing, the actual runtime of the recursive
# function, etc.
#
# The way the tree works is that every node maintains (i) a key
# (used for the same purposes as in any other tree), (ii) a counter
# of all descendants, and (iii) pointers to its children (or the
# value if the node is a leaf). Children are always in sorted order
# of keys. Inserting a node involves recursively descending and
# ultimately modifying or inserting a leaf, in the natural way. To
# ensure node width invariants are satisfied, we add a "split"
# operation by which a node with too few children given its current
# number of total descendants can split its biggest child into
# two, and a "join" operation by which a node with too many children
# can combine two together. Split and join are both O(N**(1/d)).

# Nodes are represented as follows:
#
# leaf node: [k, 1, v]
# other nodes: [k, G, children]
#
# FDTree is released as free software under the WTFPL:
# http://www.wtfpl.net/about/


# Unfortunately with this kind of tree it's not possible to have
# zero nodes. So we start off with one node at key 0. If you try
# to remove all nodes you will get an error. Also, note that you
# must pass the depth as an argument

def newtree(d):
    if d == 0:
        return [0, 1, 0]
    else:
        return [0, 1, [newtree(d-1)]]


# Split a node in half
def split(t):
    left_children = t[2][:len(t[2])/2]
    left_G = sum([c[1] for c in left_children])
    left_key = t[0]
    right_children = t[2][len(t[2])/2:]
    right_G = sum([c[1] for c in right_children])
    right_key = right_children[0][0]
    return [left_key, left_G, left_children], \
           [right_key, right_G, right_children]


# Join two adjacent nodes together
def join(t1, t2):
    children = t1[2] + t2[2]
    G = t1[1] + t2[1]
    key = t1[0]
    return [key, G, children]


# If a node has been increased and it has too few children given its total
# number of descendants, then we find the biggest child (in terms of its
# number of children) and split it up
def rebalance_increase(t, d):
    if d <= 1 or t[1] <= len(t[2]) ** d:
        return t
    # Determine index of largest child
    maxkey, maxlen = 0, 0
    for i, child in enumerate(t[2]):
        if len(child[2]) > maxlen:
            maxkey, maxlen = i, len(child[2])
    # Split it up
    new_children = t[2][:maxkey] + list(split(t[2][maxkey])) + t[2][maxkey+1:]
    return [t[0], t[1], new_children]


# If a node has been decreased and it has too many children given its total
# number of descendants, then we find the smallest pair of adjacent children
# (in terms of their combined number of children) and join them together
def rebalance_decrease(t, d):
    # We also take the opportunity to remove nodes with zero children
    c = [u for u in t[2] if len(u[2])]
    if d <= 1 or t[1] > (len(t[2]) - 1) ** d or len(c) <= 1:
        return [c[0][0] if len(c) else t[0], t[1], c]
    # Determine index of the first of the smallest-child pair
    minkey, minlen = 0, 2**999
    for i in range(len(c) - 1):
        L = len(c[i][2]) + len(c[i+1][2])
        if L < minlen:
            minkey, minlen = i, L
    # Join it and its neighbor together
    new_children = c[:minkey] + [join(c[minkey], c[minkey + 1])] + c[minkey+2:]
    return [new_children[0][0] if len(new_children) else t[0], t[1], new_children]


# Get the child of t that would contain key k
def get_subindex(t, k):
    i = 0
    while i < len(t[2]) and t[2][i][0] <= k:
        i += 1
    return i-1 if i > 0 else i


# Add a key/value pair to a tree, at a particular depth (when calling the
# function, use the depth of the tree)
def add(t, k, v, d):
    # Base case: insert the child
    if d == 1:
        i = 0
        while i < len(t[2]) and t[2][i][0] < k:
            i += 1
        if i < len(t[2]) and k == t[2][i][0]:
            # Key matches existing child, so just replace the value
            new_children = t[2][:i] + [[k, 1, v]] + t[2][i+1:]
            o = [t[0], t[1], new_children]
        else:
            # Key does not match, so add new node
            new_children = t[2][:i] + [[k, 1, v]] + t[2][i:]
            o = [min(k, t[0]), t[1] + 1, new_children]
        assert o[0] == new_children[0][0], o
        return o
    # Recursive case
    else:
        i = get_subindex(t, k)
        new_children = t[2][:i] + [add(t[2][i], k, v, d-1)] + t[2][i+1:]
        new_G = t[1] - t[2][i][1] + new_children[i][1]
        new_key = new_children[0][0]
        o = [new_key, new_G, new_children]
        return rebalance_increase(o, d)


# Remove a key from a tree
def remove(t, k, d):
    # Base case: filter away the child
    if d == 1:
        new_children = [u for u in t[2] if u[0] != k]
        new_G = len(new_children)
        new_key = new_children[0][0] if len(new_children) else t[0]
        return [new_key, new_G, new_children]
    # Recursive case
    else:
        i = get_subindex(t, k)
        if i == len(t[2]):
            return t
        new_children = t[2][:i] + [remove(t[2][i], k, d-1)] + t[2][i+1:]
        new_G = t[1] - t[2][i][1] + new_children[i][1]
        new_key = new_children[0][0]
        o = [new_key, new_G, new_children]
        return rebalance_decrease(o, d)


# Get the value associated with a key
def get(t, k, d):
    if d == 1:
        for key, _, value in t[2]:
            if k == key:
                return value
        return None
    else:
        i = get_subindex(t, k)
        return get(t[2][i], k, d-1)


def check_invariants(t, d):
    if d == 0:
        assert len(t) == 3
        assert t[1] == 1
    else:
        # check G-value correctness
        children_G_sum = sum([u[1] for u in t[2]])
        assert t[1] == children_G_sum
        # check that it has children
        assert len(t[2]) > 0, t
        # check key correctness
        for i in range(len(t[2])-1):
            assert t[2][i][0] < t[2][i+1][0]
        assert t[0] == t[2][0][0], (t, d)
        # check size correctness
        mincount = ((len(t[2]) - 1) / 2) ** d
        maxcount = (len(t[2]) * 2) ** d
        assert mincount <= t[1] <= maxcount, (t, mincount, t[1], maxcount)
        for u in t[2]:
            check_invariants(u, d-1)


def test(depth, valuecount):
    import random
    vals = list(range(1, valuecount + 1))
    random.shuffle(vals)
    t = newtree(depth)
    print 'adding'
    for v in vals:
        t = add(t, v, v**2, depth)
        print t[1]
        check_invariants(t, depth)
    print 'getting'
    for v in vals:
        assert get(t, v, depth) == v**2
    random.shuffle(vals)
    print 'removing'
    for v in vals:
        t = remove(t, v, depth)
        print t[1]
        check_invariants(t, depth)
        assert get(t, v, depth) is None
