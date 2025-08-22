import random

from actions import find_sum_subsets_ids, decode_action_ids

def brute_sum_subsets_ids(table_ids, target):
    n = len(table_ids)
    results = []
    for mask in range(1, 1 << n):
        s = 0
        sub = []
        for i in range(n):
            if (mask >> i) & 1:
                cid = table_ids[i]
                s += (cid // 4) + 1
                sub.append(cid)
        if s == target:
            results.append(sorted(sub))
    # dedup
    results.sort()
    dedup = []
    for r in results:
        if len(dedup) == 0 or dedup[-1] != r:
            dedup.append(r)
    return dedup

def test_find_sum_subsets_ids_matches_bruteforce():
    random.seed(0)
    for _ in range(50):
        # sample random table of up to 8 cards
        table = random.sample(range(40), k=random.randint(0, 8))
        target = random.randint(1, 10)
        fast = find_sum_subsets_ids(table, target)
        fast_sorted = sorted([sorted(x) for x in fast])
        brute = brute_sum_subsets_ids(table, target)
        assert fast_sorted == brute


def test_find_sum_subsets_ids_stress_small():
    # table di 6 carte (caso worst-case ragionevole); target 6
    table = [0, 1, 2, 3, 4, 5]
    subs = find_sum_subsets_ids(table, 6)
    for sub in subs:
        s = sum(((cid // 4) + 1) for cid in sub)
        assert s == 6
