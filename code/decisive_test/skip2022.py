# POST-HOC diagnostic, not part of PROTOCOL.md: is the 2022 result caused by the 2022 epoch itself?
s = open("/home/saturn/gwgi/gwgifu0h/00_libs/paper_checks/decisive/decisive.py").read().split("rows = []; lodo = []")[0]
s = s.replace("list(range(2023, 2026)) if SMOKE else list(range(2012, 2026))", "[2021, 2022, 2023]")
s = s.replace("{'smoke' if SMOKE else 'verdicts'}.txt", "skip2022.txt").replace('f"{OUT}/scenes.csv"', 'f"{OUT}/skip2022_scenes.csv"')
exec(s)
for name in ("2022_Nov", "2023_PAROXYSM", "2020_Dec", "2019_Oct"):
    ash, clean = load_mask(name, "strict")
    P(f"\n{name}")
    for a, b in ((2021, 2022), (2022, 2023), (2021, 2023)):
        x = compute(dh_of(comp("A", a), comp("A", b)), ash, clean, boot=1000)
        P(f"  {a}->{b}: D {x['D']:+.2f} [{x['D_lo']:+.2f},{x['D_hi']:+.2f}] raw {x['D_raw']:+.2f} ash {x['ash']:+.2f} clean {x['clean']:+.2f} n_ash {x['n_ash']:,}")
ash, clean = load_mask("2022_Nov", "strict")
P("\nEach 2022 date against the 2021 composite (2022 mask):")
for d in EPOCH[2022]:
    x = compute(dh_of(comp("A", 2021), A[d]), ash, clean)
    P(f"  {d} kept={kept.get(d,'NA')}: ash {x['ash']:+.2f} clean {x['clean']:+.2f} D {x['D']:+.2f} n_ash {x['n_ash']:,}" if x else f"  {d}: too few pixels")
P("Each 2023 date against the 2021 composite (2022 mask):")
for d in EPOCH[2023]:
    x = compute(dh_of(comp("A", 2021), A[d]), ash, clean)
    P(f"  {d} kept={kept.get(d,'NA')}: ash {x['ash']:+.2f} clean {x['clean']:+.2f} D {x['D']:+.2f}" if x else f"  {d}: too few")
