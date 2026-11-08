src = open("/home/saturn/gwgi/gwgifu0h/00_libs/paper_checks/paper_checks.py").read().split("rows = []; log =")[0]
exec(src)
import os
out = open(f"{OUT}/frames.txt", "w")
for name, ed, pre0, post0 in ERUPTIONS:
    G = Grid(f"{MASKS}/ASHMASK_{name}_strict.tif")
    fa = G.files(pre0); fb = G.files(post0)
    raw_glob = [os.path.basename(p) for p in glob.glob(f"{BASE}/{pre0[:4]}/{pre0}*155_0045*")]
    print(name, "pre frames", len(fa), "post frames", len(fb), "| glob order pre:", [x[11:17] for x in raw_glob], file=out, flush=True)
    for a, b in itertools.product(fa, fb):
        dh, nm = dh_field(G, a, b); s = stats(G, dh, G.read(b))
        if not s:
            print("  ", os.path.basename(a)[8:25], os.path.basename(b)[8:25], "too few px", file=out); continue
        cr, cm, nt = tile_boot(G, dh, G.read(b), n=500)
        print(f"  pre {a.split('/')[-3][:17]} post {b.split('/')[-3][:17]}: ash {s['ash']:+.2f} clean {s['clean']:+.2f} D_raw {s['D_raw']:+.2f} [{cr[0]:+.2f},{cr[1]:+.2f}] D_elev {s['D_elev']:+.2f} [{cm[0]:+.2f},{cm[1]:+.2f}] n_ash {s['n_ash']} matched {s['matched_frac']:.2f} NMAD {nm:.2f}", file=out, flush=True)
    G.cache.clear()
