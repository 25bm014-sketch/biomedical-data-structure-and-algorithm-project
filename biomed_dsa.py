"""Data Structures and Algorithms for Biomedical Applications (mini project).

Module 1 : AVL tree versus BST for patient registration       (Unit 3, T4)
Module 2 : Topological ordering of clinical workflows        (Unit 4, T10)
Module 3 : Recursive max-min of a sampled ECG signal          (Unit 5, T16)

Usage:  python3 biomed_dsa.py            run demonstrations and all test cases
        python3 biomed_dsa.py --bench    run the measured benchmarks
Only the Python standard library is used. The core algorithms are written
from first principles (no library sorting, no min()/max() in Module 3's
recursive routine).
"""
import math
import random
import sys
import time

sys.setrecursionlimit(10000)

# =====================================================================
# MODULE 1 : BST and AVL tree for patient registration
# =====================================================================
STATS = {"cmp": 0, "rot": 0}   # comparison and rotation counters
LOG = []                       # rotation log used for the worked example


class Node:
    __slots__ = ("pid", "name", "left", "right", "height")

    def __init__(self, pid, name):
        self.pid, self.name = pid, name
        self.left = self.right = None
        self.height = 1


def reset_stats():
    STATS["cmp"] = 0
    STATS["rot"] = 0
    del LOG[:]


# ---------------- plain binary search tree ----------------
def bst_insert(root, pid, name):
    """Iterative insert; one comparison is counted per node visited."""
    new = Node(pid, name)
    if root is None:
        return new
    cur = root
    while True:
        STATS["cmp"] += 1
        if pid < cur.pid:
            if cur.left is None:
                cur.left = new
                break
            cur = cur.left
        elif pid > cur.pid:
            if cur.right is None:
                cur.right = new
                break
            cur = cur.right
        else:
            break                      # duplicate identifier: ignored
    return root


def min_node(n):
    while n.left is not None:
        n = n.left


    return n


def bst_delete(n, pid):
    if n is None:
        return None
    if pid < n.pid:
        n.left = bst_delete(n.left, pid)
    elif pid > n.pid:
        n.right = bst_delete(n.right, pid)
    else:
        if n.left is None:
             return n.right
        if n.right is None:
             return n.left
        s = min_node(n.right)          # in-order successor
        n.pid, n.name = s.pid, s.name
        n.right = bst_delete(n.right, s.pid)
    return n


def search(root, pid):
    """Return (node or None, number of nodes examined); same for BST/AVL."""
    steps, cur = 0, root
    while cur is not None:
        steps += 1
        if pid == cur.pid:
            return cur, steps
        cur = cur.left if pid < cur.pid else cur.right
    return None, steps


# ---------------- AVL tree ----------------
def h(n):
    return n.height if n else 0


def update(n):
    n.height = 1 + max(h(n.left), h(n.right))


def balance(n):
    return h(n.left) - h(n.right) if n else 0


def rot_right(y):
    x = y.left
    y.left, x.right = x.right, y
    update(y)
    update(x)
    STATS["rot"] += 1
    return x


def rot_left(x):
    y = x.right
    x.right, y.left = y.left, x
    update(x)
    update(y)
    STATS["rot"] += 1
    return y


def rebalance(n):
    update(n)
    b = balance(n)
    if b > 1:
        if balance(n.left) < 0:                    # Left-Right case
             LOG.append("LR at %d" % n.pid)
             n.left = rot_left(n.left)
        else:                                      # Left-Left case
             LOG.append("LL at %d" % n.pid)
        return rot_right(n)
    if b < -1:
        if balance(n.right) > 0:                   # Right-Left case
             LOG.append("RL at %d" % n.pid)
             n.right = rot_right(n.right)
        else:                                      # Right-Right case
             LOG.append("RR at %d" % n.pid)
        return rot_left(n)
    return n


def avl_insert(n, pid, name):
    if n is None:
        return Node(pid, name)
    STATS["cmp"] += 1
    if pid < n.pid:
        n.left = avl_insert(n.left, pid, name)
    elif pid > n.pid:
        n.right = avl_insert(n.right, pid, name)
    else:
        return n                                   # duplicate ignored


    return rebalance(n)


def avl_delete(n, pid):
    if n is None:
        return None
    if pid < n.pid:
        n.left = avl_delete(n.left, pid)
    elif pid > n.pid:
        n.right = avl_delete(n.right, pid)
    else:
        if n.left is None:
            return n.right
        if n.right is None:
            return n.left
        s = min_node(n.right)
        n.pid, n.name = s.pid, s.name
        n.right = avl_delete(n.right, s.pid)
    return rebalance(n)


# ---------------- helpers used by tests ----------------
def true_height(n):
    """Height computed by recursion (does not trust the stored field)."""
    return 0 if n is None else 1 + max(true_height(n.left), true_height(n.right))


def is_avl(n):
    if n is None:
        return True
    ok = abs(true_height(n.left) - true_height(n.right)) <= 1
    return ok and n.height == true_height(n) and is_avl(n.left) and is_avl(n.right)


def inorder(n, out):
    if n is not None:
        inorder(n.left, out)
        out.append(n.pid)
        inorder(n.right, out)
    return out


def preorder(n, out):
    if n is not None:
        out.append(n.pid)
        preorder(n.left, out)
        preorder(n.right, out)
    return out


def build(keys, kind):
    root = None
    for k in keys:
        name = "Patient-%d" % k
        root = bst_insert(root, k, name) if kind == "bst" else avl_insert(root, k, name)
    return root


# =====================================================================
# MODULE 2 : topological ordering of a clinical workflow (Kahn)
# =====================================================================
class ClinicalDAG:
    def __init__(self):
        self.tasks, self.index, self.adj, self.edges = [], {}, [], set()

    def add_task(self, name):
        if name not in self.index:
            self.index[name] = len(self.tasks)
            self.tasks.append(name)
            self.adj.append([])
        return self.index[name]

    def add_dependency(self, before, after):
        """Task `before` must be completed before task `after` starts."""
        u, v = self.add_task(before), self.add_task(after)
        if (u, v) not in self.edges:
            self.edges.add((u, v))
            self.adj[u].append(v)

    def indegrees(self):
        deg = [0] * len(self.tasks)
        for u in range(len(self.tasks)):
            for v in self.adj[u]:
                deg[v] += 1
        return deg


def kahn(dag, trace=None):
    """Return (order, is_dag). Queue is a list with a head index."""
    n = len(dag.tasks)
    deg = dag.indegrees()
    queue = [v for v in range(n) if deg[v] == 0]
    head, order = 0, []


    while head < len(queue):
        u = queue[head]
        head += 1
        order.append(u)
        released = []
        for v in dag.adj[u]:
            deg[v] -= 1
            if deg[v] == 0:
                queue.append(v)
                released.append(v)
        if trace is not None:
            trace.append((u, released, list(deg)))
    return order, len(order) == n


def is_valid_order(dag, order):
    """Check every dependency (u, v): u appears before v, each task once."""
    if sorted(order) != list(range(len(dag.tasks))):
        return False
    pos = {t: i for i, t in enumerate(order)}
    return all(pos[u] < pos[v] for (u, v) in dag.edges)


def unresolved_tasks(dag):
    """Tasks that Kahn's algorithm cannot release (on or behind a cycle)."""
    order, ok = kahn(dag)
    done = set(order)
    return [dag.tasks[v] for v in range(len(dag.tasks)) if v not in done]


def sample_workflow():
    g = ClinicalDAG()
    for t in ["Patient Registration", "Initial Assessment", "Vital Signs Recording",
              "Laboratory Testing", "Imaging", "Diagnosis", "Treatment Planning",
              "Treatment Administration", "Discharge"]:
        g.add_task(t)
    g.add_dependency("Patient Registration", "Initial Assessment")
    g.add_dependency("Patient Registration", "Vital Signs Recording")
    g.add_dependency("Initial Assessment", "Laboratory Testing")
    g.add_dependency("Initial Assessment", "Imaging")
    g.add_dependency("Vital Signs Recording", "Diagnosis")
    g.add_dependency("Laboratory Testing", "Diagnosis")
    g.add_dependency("Imaging", "Diagnosis")
    g.add_dependency("Diagnosis", "Treatment Planning")
    g.add_dependency("Treatment Planning", "Treatment Administration")
    g.add_dependency("Treatment Administration", "Discharge")
    return g


# =====================================================================
# MODULE 3 : recursive max-min of an ECG sample array
# =====================================================================
def max_min(a, lo, hi, st, depth=1):
    """Divide and conquer. Returns (minimum, maximum) of a[lo..hi].
    st counts element comparisons, calls and the deepest recursion level."""
    st["calls"] += 1
    if depth > st["depth"]:
        st["depth"] = depth
    if lo == hi:                                   # base case 1: one sample
        return a[lo], a[lo]
    if hi == lo + 1:                               # base case 2: two samples
        st["cmp"] += 1
        return (a[lo], a[hi]) if a[lo] < a[hi] else (a[hi], a[lo])
    mid = (lo + hi) // 2                           # divide
    mn1, mx1 = max_min(a, lo, mid, st, depth + 1)
    mn2, mx2 = max_min(a, mid + 1, hi, st, depth + 1)
    st["cmp"] += 2                                 # combine
    return (mn1 if mn1 < mn2 else mn2), (mx1 if mx1 > mx2 else mx2)


def ecg_extrema(samples):
    """Validate the input and return (min, max, stats)."""
    if len(samples) == 0:
        raise ValueError("empty ECG record")
    for x in samples:
        if x != x:
            raise ValueError("NaN sample in ECG record")
    st = {"cmp": 0, "calls": 0, "depth": 0}
    mn, mx = max_min(samples, 0, len(samples) - 1, st)
    return mn, mx, st


def linear_extrema(samples):
    """Reference method: one pass, two comparisons per element."""
    cmp = 0
    mn = mx = samples[0]
    for x in samples[1:]:
        cmp += 1
        if x < mn:
            mn = x
        cmp += 1
        if x > mx:
            mx = x


    return mn, mx, cmp


def synthetic_ecg(n, fs=250, bpm=72, seed=1):
    """Illustrative ECG-like waveform (sum of Gaussians) in millivolts.
    It is synthetic test data, not a patient recording."""
    rnd = random.Random(seed)
    period = 60.0 / bpm
    waves = [(-0.20, 0.025, 0.12), (-0.04, 0.010, -0.10), (0.0, 0.012, 1.10),
              (0.04, 0.010, -0.25), (0.28, 0.045, 0.30)]   # P Q R S T
    out = []
    for i in range(n):
        t = (i / fs) % period
        v = 0.0
        for c, w, amp in waves:
             for centre in (c + period * 0.35, c + period * 0.35 - period):
                 v += amp * math.exp(-((t - centre) ** 2) / (2 * w * w))
        out.append(round(v + rnd.uniform(-0.01, 0.01), 3))
    return out


ECG16 = [0.02, 0.05, 0.10, 0.04, -0.05, -0.12, 1.20, -0.30,
         0.00, 0.08, 0.22, 0.30, 0.18, 0.05, 0.02, 0.01]


# =====================================================================
# Demonstrations
# =====================================================================
def demo_module1():
    print("MODULE 1 DEMONSTRATION")
    for name, keys in [("LL", [30, 20, 10]), ("RR", [10, 20, 30]),
                       ("LR", [30, 10, 20]), ("RL", [10, 30, 20])]:
        reset_stats()
        t = build(keys, "avl")
        print(" %s case, insert %s -> root=%d, pre-order=%s, log=%s"
              % (name, keys, t.pid, preorder(t, []), LOG))
    seq = [45, 85, 50, 75, 20, 65, 30]
    reset_stats()
    b = build(seq, "bst")
    bst_cmp = STATS["cmp"]
    reset_stats()
    a = build(seq, "avl")
    print(" Worked example, insert order %s" % seq)
    print("    BST: height=%d, pre-order=%s, comparisons=%d"
          % (true_height(b), preorder(b, []), bst_cmp))
    print("    AVL: height=%d, pre-order=%s, comparisons=%d, rotations=%d"
          % (true_height(a), preorder(a, []), STATS["cmp"], STATS["rot"]))
    print("    AVL rotation log: %s" % LOG)
    node, steps = search(a, 65)
    print("    search(65) in AVL -> %s in %d steps" % (node.name, steps))
    node, steps = search(b, 65)
    print("    search(65) in BST -> %s in %d steps" % (node.name, steps))
    node, steps = search(a, 99)
    print("    search(99) in AVL -> %s after %d steps" % (node, steps))
    a = avl_delete(a, 50)
    print("    after deleting 50: in-order=%s, AVL=%s" % (inorder(a, []), is_avl(a)))


def demo_module2():
    print("\nMODULE 2 DEMONSTRATION")
    g = sample_workflow()
    print(" Indegrees:")
    for i, d in enumerate(g.indegrees()):
        print("     %d %-26s indegree=%d" % (i, g.tasks[i], d))
    tr = []
    order, ok = kahn(g, tr)
    print(" Kahn trace (task removed -> tasks released):")
    for u, rel, _ in tr:
        print("     remove %-26s -> released %s" % (g.tasks[u], [g.tasks[v] for v in rel]))
    print(" Is DAG: %s" % ok)
    print(" Valid order:")
    for i, v in enumerate(order, 1):
        print("     %d. %s" % (i, g.tasks[v]))
    print(" Order verified against all %d dependencies: %s"
          % (len(g.edges), is_valid_order(g, order)))


def demo_module3():
    print("\nMODULE 3 DEMONSTRATION")
    print(" Illustrative 16-sample ECG segment (mV): %s" % ECG16)
    mn, mx, st = ecg_extrema(ECG16)
    print(" Recursive result: min=%.2f mV, max=%.2f mV" % (mn, mx))
    print(" Comparisons=%d, calls=%d, recursion depth=%d"
          % (st["cmp"], st["calls"], st["depth"]))
    mn2, mx2, c2 = linear_extrema(ECG16)
    print(" Linear scan:       min=%.2f mV, max=%.2f mV, comparisons=%d" % (mn2, mx2, c2))
    sig = synthetic_ecg(512)
    mn, mx, st = ecg_extrema(sig)
    print(" Synthetic 2.048 s record, 512 samples at 250 Hz: min=%.3f mV, max=%.3f mV"
          % (mn, mx))
    print(" Comparisons=%d (3n/2-2 = %d for n = 512), depth=%d"
          % (st["cmp"], 3 * 512 // 2 - 2, st["depth"]))
    print(" Note: the numbers are extrema only; no diagnosis is made.")



# =====================================================================
# Test cases
# =====================================================================
RESULTS = []


def check(tid, ok, text):
    RESULTS.append(ok)
    print(" [%s] %-6s %s" % ("PASS" if ok else "FAIL", tid, text))


def self_test():
    print("\n=========== MODULE 1 TEST CASES ===========")
    keys = list(range(1, 16))
    reset_stats()
    b, a = build(keys, "bst"), build(keys, "avl")
    print(" T1: 15 ascending IDs -> BST height=%d, AVL height=%d"
          % (true_height(b), true_height(a)))
    check("M1-T1", true_height(b) == 15 and true_height(a) == 4 and is_avl(a),
          "AVL stays balanced, BST degenerates to a chain")

    shapes = {}
    for case, ks in [("LL", [30, 20, 10]), ("RR", [10, 20, 30]),
                     ("LR", [30, 10, 20]), ("RL", [10, 30, 20])]:
        shapes[case] = preorder(build(ks, "avl"), [])
    print(" T2: pre-order after single/double rotations: %s" % shapes)
    check("M1-T2", all(s == [20, 10, 30] for s in shapes.values()),
          "LL, RR, LR and RL each give root 20 with children 10 and 30")

    a = build(range(1, 32), "avl")
    found_before = search(a, 17)[0] is not None
    for k in range(1, 32, 2):
        a = avl_delete(a, k)
    ids = inorder(a, [])
    print(" T3: 31 IDs, delete all odd IDs -> %d left, height=%d"
          % (len(ids), true_height(a)))
    check("M1-T3", found_before and search(a, 17)[0] is None and len(ids) == 15
          and ids == sorted(ids) and is_avl(a),
          "deletion keeps order and balance; deleted ID not found")

    a = build([5, 3, 8, 3, 5], "avl")
    check("M1-T4", inorder(a, []) == [3, 5, 8] and search(a, 4)[0] is None,
          "duplicate IDs ignored; absent ID reported as not found")

    print("\n=========== MODULE 2 TEST CASES ===========")
    g = sample_workflow()
    order, ok = kahn(g)
    print(" T1: 9-task clinical workflow -> %s" % [g.tasks[v] for v in order])
    check("M2-T1", ok and len(order) == 9 and is_valid_order(g, order)
          and g.tasks[order[0]] == "Patient Registration"
          and g.tasks[order[-1]] == "Discharge",
          "valid dependency order from registration to discharge")

    c = sample_workflow()
    c.add_dependency("Treatment Planning", "Diagnosis")      # creates a cycle
    order, ok = kahn(c)
    print(" T2: cycle Diagnosis <-> Treatment Planning -> ordered=%d of %d, "
          "unresolved=%s" % (len(order), len(c.tasks), unresolved_tasks(c)))
    check("M2-T2", (not ok) and len(order) == 5 and len(unresolved_tasks(c)) == 4,
          "cycle detected; ordering reports the unresolved tasks")

    d = ClinicalDAG()
    d.add_dependency("Triage", "Consultation")
    d.add_dependency("Blood Sample", "Blood Report")
    d.add_task("Billing")
    order, ok = kahn(d)
    e = ClinicalDAG()
    print(" T3: two chains and an isolated task -> %s; empty graph -> %s"
          % ([d.tasks[v] for v in order], kahn(e)))
    check("M2-T3", ok and is_valid_order(d, order) and kahn(e) == ([], True),
          "several independent tasks and the empty graph are handled")

    print("\n=========== MODULE 3 TEST CASES ===========")
    mn, mx, st = ecg_extrema(ECG16)
    print(" T1: 16-sample segment -> min=%.2f, max=%.2f, comparisons=%d"
          % (mn, mx, st["cmp"]))
    check("M3-T1", mn == -0.30 and mx == 1.20 and st["cmp"] == 22 and st["depth"] == 4,
          "known extrema; 3n/2-2 = 22 comparisons for n = 16")

    cases = [[0.5], [0.4, -0.2], [0.1] * 9, [-0.8, -0.3, -0.5, -0.9, -0.1],
              [0.0, 0.2, 0.9, 0.3, -0.4, 0.1, 0.0]]
    good = all(ecg_extrema(c)[:2] == (min(c), max(c)) for c in cases)
    err = False
    try:
         ecg_extrema([])
    except ValueError:
         err = True
    print(" T2: n=1, n=2, all-equal, all-negative, odd length, empty input -> %s"
           % ("handled" if good and err else "ERROR"))
    check("M3-T2", good and err, "boundary inputs handled; empty record rejected")


    rnd = random.Random(2026)
    ok_all, bound_ok = True, True
    for n in range(1, 201):
        data = [round(rnd.uniform(-1.0, 1.5), 3) for _ in range(n)]
        mn, mx, st = ecg_extrema(data)
        ok_all = ok_all and (mn, mx) == (min(data), max(data))
        if n > 1:
            bound_ok = bound_ok and st["cmp"] <= 2 * n - 3
        if n > 1 and n & (n - 1) == 0:
            bound_ok = bound_ok and st["cmp"] == 3 * n // 2 - 2
    print(" T3: 200 seeded random records (n = 1..200) vs built-in min/max "
          "-> %s" % ("all agree" if ok_all else "MISMATCH"))
    check("M3-T3", ok_all and bound_ok,
          "agrees with built-in min/max; comparisons <= 2n-3, = 3n/2-2 if n is 2^k")

    print("\n=========== SUMMARY: %d of %d test cases passed ==========="
          % (sum(RESULTS), len(RESULTS)))


# =====================================================================
# Measured benchmarks (times depend on the computer used)
# =====================================================================
def bench():
    print("BENCHMARKS (Python %s)" % sys.version.split()[0])
    print("\nModule 1: BST versus AVL, 1000 searches over inserted keys")
    rnd = random.Random(2026)
    sets = [("1000 ascending", list(range(1, 1001))),
             ("1000 random", rnd.sample(range(1, 100000), 1000)),
             ("2000 ascending", list(range(1, 2001)))]
    for label, ks in sets:
        for kind in ("bst", "avl"):
             reset_stats()
             t0 = time.perf_counter()
             tree = build(ks, kind)
             dt = (time.perf_counter() - t0) * 1000
             ic, rot = STATS["cmp"], STATS["rot"]
             probes = [ks[i * len(ks) // 1000] for i in range(1000)]
             steps = sum(search(tree, k)[1] for k in probes) / 1000.0
             print(" %-15s %s height=%-5d insert-cmp=%-8d rotations=%-5d "
                   "insert=%8.2f ms avg-search-steps=%.1f"
                   % (label, kind.upper(), true_height(tree), ic, rot, dt, steps))
    print("\nModule 2: Kahn's algorithm on layered random DAGs (edges go forward)")
    rnd = random.Random(7)
    for n in (1000, 5000, 20000):
        g = ClinicalDAG()
        for i in range(n):
             g.add_task("T%d" % i)
        for v in range(1, n):
             for _ in range(2):
                 g.add_dependency("T%d" % rnd.randrange(max(0, v - 50), v), "T%d" % v)
        t0 = time.perf_counter()
        order, ok = kahn(g)
        dt = (time.perf_counter() - t0) * 1000
        print(" V=%-6d E=%-6d ordered=%-6d valid=%s time=%.2f ms"
               % (n, len(g.edges), len(order), is_valid_order(g, order), dt))
    print("\nModule 3: recursive max-min versus linear scan versus built-in")
    for n in (1024, 16384, 131072):
        sig = synthetic_ecg(n, seed=3)
        t0 = time.perf_counter()
        mn, mx, st = ecg_extrema(sig)
        t_rec = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        lmn, lmx, lc = linear_extrema(sig)
        t_lin = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        bmn, bmx = min(sig), max(sig)
        t_bi = (time.perf_counter() - t0) * 1000
        print(" n=%-7d rec: cmp=%-7d depth=%-3d %8.2f ms | linear: cmp=%-7d "
               "%8.2f ms | built-in: %6.2f ms | agree=%s"
               % (n, st["cmp"], st["depth"], t_rec, lc, t_lin, t_bi,
                  (mn, mx) == (lmn, lmx) == (bmn, bmx)))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        demo_module1()
        demo_module2()
        demo_module3()
        self_test()
        sys.exit(0 if all(RESULTS) else 1)
