# Taken from scripts/tp4.sh of github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104 (the Python program inside the function pair_links),
# Copyright 2026 Mia's AI Lab (MiaAI-Lab), licensed under the Apache License, Version 2.0.
# Changes: extracted into a standalone script (the shell function wrapper and its comment block were left out).

import ipaddress, sys
nodes, default = {}, {}
for line in sys.stdin:
    f = line.split()
    if len(f) == 3 and f[1] == "default":
        default[int(f[0])] = f[2]
    elif len(f) == 5:
        nodes.setdefault(int(f[0]), []).append((f[1], f[2], ipaddress.ip_interface(f[3]), f[4]))
ranks = [int(x) for x in sys.argv[1:]]
linked = lambda e, others: any(e[2].network == o[2].network for o in others)
links = {r: set() for r in ranks}
for r in ranks:
    used = set()
    for s in ranks:
        if s == r:
            continue
        mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
        if not mine:
            if r < s:
                print("missing", r, s)
            continue
        links[r].add(s)
        used.update(id(e) for e in mine)
        print("toward", r, s, ",".join(dict.fromkeys(e[1] for e in mine)))
        if r == 0:
            print("peer", s, min(o[2].ip for o in nodes[s] if linked(o, mine)))
    hcas, gids = [], set()
    for e in nodes.get(r, []):
        if id(e) in used and e[1] not in hcas:
            hcas.append(e[1]); gids.add(e[3])
    print("rank", r, default.get(r) or "-", ",".join(hcas) or "-", gids.pop() if len(gids) == 1 else "-")
def splits(n):
    for r in ranks:
        for s in links[r]:
            mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
            theirs = [o for o in nodes.get(s, []) if linked(o, mine)]
            nets = [ipaddress.ip_interface(f"{e[2].ip}/{n}").network for e in mine]
            if len(set(nets)) != len(nets) or any(sum(o[2].ip in net for o in theirs) != 1 for net in nets):
                return False
    return True
own = max((e[2].network.prefixlen for r in ranks for e in nodes.get(r, [])), default=32)
pre = next((n for n in range(own, 31) if splits(n)), None)
if pre is not None:
    print("prefix", pre)
if len(ranks) > 3 and all(len(links[r]) == 2 for r in ranks):
    order = [ranks[0], min(links[ranks[0]])]
    while len(order) <= len(ranks):
        nxt = (links[order[-1]] - {order[-2]}).pop()
        if nxt == order[0]:
            break
        order.append(nxt)
    if len(order) == len(ranks):
        print("ring", *order)

