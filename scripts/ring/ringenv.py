#!/usr/bin/env python3
"""Generate per-rank NCCL ring settings (env file + graph xml) for four Sparks cabled as a ring without a switch.
Derived from the ring discovery, graph files and NCCL ring settings in MiaAI-Lab's GLM-5.3-Flash-DGX-Sparks-1104 recipe
(github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104, scripts/tp4.sh), Copyright 2026 Mia's AI Lab (MiaAI-Lab), Apache License 2.0.
Changes: rewritten in Python and adapted to write one env file and one NCCL graph file per rank (the shell original is in the file named above).
usage: ringenv.py OUTDIR host0 host1 host2 host3   (hosts in ring order, ssh-reachable; rank i = host i)"""
import subprocess, sys, os, pathlib, ipaddress, json, socket
HERE = pathlib.Path(__file__).resolve().parent
out = pathlib.Path(sys.argv[1]); hosts = sys.argv[2:6]; out.mkdir(parents=True, exist_ok=True)
CH = int(os.environ.get("NCCL_CHANNELS", "4"))
inv_fn = (HERE / "node_inventory.fn.sh").read_text()
lines = []
for r, h in enumerate(hosts):
    # the host this script runs on is inventoried locally (a box usually has no ssh key to itself)
    cmd = ["bash", "-s"] if h in (socket.gethostname(), socket.gethostname().split(".")[0]) else ["ssh", "-o", "BatchMode=yes", h, "bash -s"]
    t = subprocess.run(cmd, input=inv_fn + "\nnode_inventory\n",
                       capture_output=True, text=True, check=True).stdout
    for l in t.splitlines():
        lines.append(f"{r} {l}")
pl = subprocess.run([sys.executable, str(HERE / "pair_links.py"), "0", "1", "2", "3"], input="\n".join(lines) + "\n",
                    capture_output=True, text=True, check=True).stdout
nodes_hcas, toward, gid, prefix, ring = {}, {}, {}, None, None
for l in pl.splitlines():
    f = l.split()
    if f[0] == "rank": nodes_hcas[int(f[1])] = f[3].split(","); gid[int(f[1])] = f[4]
    elif f[0] == "toward": toward[(int(f[1]), int(f[2]))] = f[3].split(",")
    elif f[0] == "prefix": prefix = f[1]
    elif f[0] == "ring": ring = [int(x) for x in f[1:]]
if ring != [0, 1, 2, 3]:
    sys.exit(f"hosts are not in ring order (found ring {ring}); pair_links output:\n{pl}")
# all RoCE devices of the node in uverbs order (NCCL numbering): from inventory column 2
alldev = {}
for l in lines:
    f = l.split()
    if len(f) == 5: alldev.setdefault(int(f[0]), []).append(f[2])
env = {}
for r in range(4):
    prev = (r + 3) % 4
    idx_all = alldev[r]
    tw = toward[(r, prev)]
    ch = ""
    for k in range(CH):
        d = tw[k % len(tw)]; i = idx_all.index(d)
        ch += f'    <channel><net dev="{i}"/><gpu dev="0" rank="{r}"/><net dev="{i}"/></channel>\n'
    xml = '<graphs version="1">\n'
    for gid_, pat in ((0, 4), (1, 3)):
        xml += f'  <graph id="{gid_}" pattern="{pat}" crossnic="1" nchannels="{CH}" speedintra="12" speedinter="12" latencyinter="0" typeintra="LOC" typeinter="P2C" samechannels="1">\n{ch}  </graph>\n'
    for gid_, pat in ((4, 6), (2, 3), (3, 5)):
        xml += f'  <graph id="{gid_}" pattern="{pat}" crossnic="0" nchannels="0" speedintra="0" speedinter="0" latencyinter="0" typeintra="LOC" typeinter="LOC" samechannels="0"/>\n'
    xml += "</graphs>\n"
    (out / f"nccl-graph-rank{r}.xml").write_text(xml)
    e = {"NCCL_IB_HCA": ",".join(nodes_hcas[r]), "NCCL_ALGO": "Ring", "NCCL_CROSS_NIC": "1",
         "NCCL_IB_SUBNET_AWARE_ROUTING": "1", "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_IB_MERGE_NICS": "0",
         "NCCL_RUNTIME_CONNECT": "1", "NCCL_CUMEM_ENABLE": "1", "NCCL_CUMEM_HOST_ENABLE": "0",
         "NCCL_MIN_NCHANNELS": str(CH), "NCCL_MAX_NCHANNELS": str(CH)}
    if gid[r] != "-": e["NCCL_IB_GID_INDEX"] = gid[r]
    if prefix: e["NCCL_IB_SUBNET_PREFIX_LEN"] = prefix
    env[r] = e
    (out / f"rank{r}.env").write_text("".join(f"{k}={v}\n" for k, v in e.items()))
print(json.dumps({"ring": hosts, "prefix": prefix, "env0": env[0]}, indent=1))
