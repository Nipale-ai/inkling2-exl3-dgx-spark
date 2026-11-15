import os, sys, time, torch
sys.path.insert(0, "/work/src/repo/overlay") if False else None
from exllamav3.modules.inkling2 import Inkling2Cfg
from exllamav3.modules.inkling2_b12x import B12xMoEShard
from types import SimpleNamespace
rank = int(os.environ.get("PROBE_RANK", "0"))
cfg = SimpleNamespace(i2=Inkling2Cfg())
dev = torch.device("cuda:0")
t0 = time.time()
m = B12xMoEShard("/work/model", cfg, [5], dev, rank, 4, groups=2, sizes=(8,))
torch.cuda.synchronize(); print(f"init+compile {time.time()-t0:.1f}s, K={m.K}, experts {m.lo}-{m.hi}", flush=True)
torch.manual_seed(0)
for T in (1, 8, 64, 200):
    x = (torch.randn(T, 4608, device=dev) * 0.5).half()
    t0 = time.time(); y = m.forward(5, None, x, out_dtype=torch.float16); torch.cuda.synchronize(); t1 = time.time() - t0
    t0 = time.time()
    for _ in range(5): y = m.forward(5, None, x, out_dtype=torch.float16)
    torch.cuda.synchronize(); t2 = (time.time() - t0) / 5
    print(f"T={T}: first {t1:.2f}s steady {t2*1000:.2f} ms, finite {bool(torch.isfinite(y).all())}, absmean {y.float().abs().mean():.4f}, nonzero {int((y!=0).sum())}", flush=True)
print("PROBE OK")
