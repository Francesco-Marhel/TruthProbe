# -*- coding: utf-8 -*-
"""
swiglu.py  --  Inside the eroding FFN: is it the GATE (context switching) or the
VALUE (content) that writes anti-truth in the transition band?

Background (measured in ffn_erosion.py): the FFN flips from writing PRO-truth at
layer 15 (gap +0.73 on the fixed axis) to ANTI-truth at 16-18 (-0.69/-0.26/-0.28),
it pulls BOTH classes toward the false side (true more), it rotates the readable
direction, and its logit-lens content is format/template junk. Question: WHERE
inside the SwiGLU does that live?

The FFN computes  f = W_down( g (*) u )  with  g = silu(W_gate x),  u = W_up x,
elementwise in the shared 8960-neuron basis. The projection of f on the truth
axis v1 is  w . (g(*)u)  with  w = W_down^T v1  -- the RESIDUAL axis pulled back
through down_proj. No axis is ever fit in the 8960-dim space (no p>>n fitting).
With the symmetric intra-pair split (Dg = g_t - g_f, gbar = (g_t+g_f)/2):

    w.(g_t(*)u_t - g_f(*)u_f)  =  w.(Dg (*) ubar)  +  w.(gbar (*) Du)      [EXACT]
                                   ^^^^ GATE term       ^^^^ VALUE term

The gate term is the part of the class gap carried by WHICH neurons open
(context switching); the value term is the part carried by WHAT they transport.
The identity is algebraic (the mixed term cancels); the script verifies it
numerically on the real data before printing anything.

    python swiglu.py attrib      # exact gate/value attribution, per layer, held-out
    python swiglu.py framing     # 2x2 moral/prudential design (polarity-style factor test)
    python swiglu.py gatefreeze  # CAUSAL: freeze gate variation vs value variation in band

Conventions as in the repo: held-out CV over pairs / leave-facts-out for quads;
fp32; 'block b' readout = hidden_states[b+1]. Reuses truth_probe.py.
"""
import argparse
import torch
import truth_probe as T


# =====================================================================
#  FRAMING 2x2: same fact, moral vs prudential justification, true/false
#  Cells: MT moral-true, MF moral-false, PT prudential-true, PF prudential-false.
#  Falsity always sits in the justification clause; syntax is kept parallel.
# =====================================================================
FRAMING_QUADS = [
    ("Stealing is wrong because it harms other people.",
     "Stealing is wrong because it helps other people.",
     "Stealing is risky because it can lead to prison.",
     "Stealing is risky because it leads to rewards."),
    ("Lying is wrong because it destroys trust.",
     "Lying is wrong because it builds trust.",
     "Lying is risky because it can ruin your reputation.",
     "Lying is risky because it improves your reputation."),
    ("Cheating on exams is wrong because it is unfair to other students.",
     "Cheating on exams is wrong because it helps other students.",
     "Cheating on exams is risky because it can get you expelled.",
     "Cheating on exams is risky because it earns you praise from teachers."),
    ("Bullying is wrong because it hurts vulnerable people.",
     "Bullying is wrong because it protects vulnerable people.",
     "Bullying is risky because it can get you suspended.",
     "Bullying is risky because it gets you rewarded by teachers."),
    ("Vandalism is wrong because it destroys other people's property.",
     "Vandalism is wrong because it protects other people's property.",
     "Vandalism is risky because it can lead to fines.",
     "Vandalism is risky because it leads to cash prizes."),
    ("Bribery is wrong because it corrupts public institutions.",
     "Bribery is wrong because it strengthens public institutions.",
     "Bribery is risky because it can lead to prosecution.",
     "Bribery is risky because it leads to official praise."),
    ("Fraud is wrong because it robs people of their savings.",
     "Fraud is wrong because it grows people's savings.",
     "Fraud is risky because it can lead to prison.",
     "Fraud is risky because it is rewarded by the courts."),
    ("Plagiarism is wrong because it steals another person's work.",
     "Plagiarism is wrong because it honors another person's work.",
     "Plagiarism is risky because it can end your career.",
     "Plagiarism is risky because it is rewarded by universities."),
    ("Drunk driving is wrong because it endangers innocent lives.",
     "Drunk driving is wrong because it protects innocent lives.",
     "Drunk driving is risky because it can cost you your license.",
     "Drunk driving is risky because it earns you a safe-driver discount."),
    ("Tax evasion is wrong because it cheats the whole community.",
     "Tax evasion is wrong because it funds the whole community.",
     "Tax evasion is risky because it can lead to heavy fines.",
     "Tax evasion is risky because it leads to government rewards."),
    ("Animal cruelty is wrong because it causes needless suffering.",
     "Animal cruelty is wrong because it prevents needless suffering.",
     "Animal cruelty is risky because it can lead to criminal charges.",
     "Animal cruelty is risky because it is praised by the law."),
    ("Polluting rivers is wrong because it poisons the water people drink.",
     "Polluting rivers is wrong because it purifies the water people drink.",
     "Polluting rivers is risky because it can lead to heavy fines.",
     "Polluting rivers is risky because it earns environmental awards."),
    ("Breaking promises is wrong because it betrays people who trust you.",
     "Breaking promises is wrong because it honors people who trust you.",
     "Breaking promises is risky because it makes people stop relying on you.",
     "Breaking promises is risky because it makes people rely on you more."),
    ("Spreading rumors is wrong because it damages innocent reputations.",
     "Spreading rumors is wrong because it repairs innocent reputations.",
     "Spreading rumors is risky because it can get you sued.",
     "Spreading rumors is risky because it protects you from lawsuits."),
    ("Shoplifting is wrong because it harms small business owners.",
     "Shoplifting is wrong because it supports small business owners.",
     "Shoplifting is risky because it can get you arrested.",
     "Shoplifting is risky because it gets you store discounts."),
    ("Blackmail is wrong because it exploits people's fears.",
     "Blackmail is wrong because it calms people's fears.",
     "Blackmail is risky because it can lead to prison.",
     "Blackmail is risky because it is rewarded by judges."),
    ("Arson is wrong because it endangers entire neighborhoods.",
     "Arson is wrong because it protects entire neighborhoods.",
     "Arson is risky because it carries long prison sentences.",
     "Arson is risky because it carries large cash rewards."),
    ("Kidnapping is wrong because it terrorizes families.",
     "Kidnapping is wrong because it comforts families.",
     "Kidnapping is risky because it carries life sentences.",
     "Kidnapping is risky because it carries public honors."),
    ("Perjury is wrong because it corrupts the justice system.",
     "Perjury is wrong because it strengthens the justice system.",
     "Perjury is risky because it can lead to criminal charges.",
     "Perjury is risky because it is encouraged by the courts."),
    ("Insider trading is wrong because it cheats ordinary investors.",
     "Insider trading is wrong because it enriches ordinary investors.",
     "Insider trading is risky because it can lead to prosecution.",
     "Insider trading is risky because it is celebrated by regulators."),
    ("Counterfeiting money is wrong because it undermines the whole economy.",
     "Counterfeiting money is wrong because it strengthens the whole economy.",
     "Counterfeiting money is risky because it can lead to federal prison.",
     "Counterfeiting money is risky because it earns rewards from the central bank."),
    ("Poaching is wrong because it drives species toward extinction.",
     "Poaching is wrong because it saves species from extinction.",
     "Poaching is risky because it can lead to heavy penalties.",
     "Poaching is risky because it earns conservation awards."),
    ("Hacking private accounts is wrong because it violates people's privacy.",
     "Hacking private accounts is wrong because it protects people's privacy.",
     "Hacking private accounts is risky because it can lead to criminal charges.",
     "Hacking private accounts is risky because it is rewarded by the police."),
    ("Dumping toxic waste is wrong because it poisons entire communities.",
     "Dumping toxic waste is wrong because it heals entire communities.",
     "Dumping toxic waste is risky because it can lead to massive fines.",
     "Dumping toxic waste is risky because it earns government subsidies."),
]


def build_framing_items():
    items, fact, is_true, is_prud = [], [], [], []
    for fi, (mt, mf, pt, pf) in enumerate(FRAMING_QUADS):
        for text, tr, pr in [(mt, 1, 0), (mf, 0, 0), (pt, 1, 1), (pf, 0, 1)]:
            items.append((tr, text)); fact.append(fi); is_true.append(tr); is_prud.append(pr)
    return items, torch.tensor(fact), torch.tensor(is_true), torch.tensor(is_prud)


# =====================================================================
#  pure helpers (unit-testable)
# =====================================================================
def gate_value_split(g_t, u_t, g_f, u_f, w):
    """EXACT decomposition of the pair gap on axis w in the expanded basis.
    Returns (gate_term, value_term, total) per pair; total == gate+value exactly."""
    Dg, Du = g_t - g_f, u_t - u_f
    gbar, ubar = (g_t + g_f) / 2, (u_t + u_f) / 2
    gate = (Dg * ubar) @ w
    value = (gbar * Du) @ w
    total = (g_t * u_t - g_f * u_f) @ w
    return gate, value, total


def cells_framing(fact, is_true, is_prud, facts_subset):
    fs = set(int(f) for f in facts_subset)
    sel = [i for i in range(len(fact)) if int(fact[i]) in fs]
    def cell(pr, tr):
        return torch.tensor([i for i in sel
                             if int(is_prud[i]) == pr and int(is_true[i]) == tr])
    return dict(MT=cell(0, 1), MF=cell(0, 0), PT=cell(1, 1), PF=cell(1, 0),
                ALL=torch.tensor(sel))


# =====================================================================
#  extraction: residual + last-token g, u per band layer
# =====================================================================
def collect_gu(model, tok, items, dev, layers_wanted):
    """Returns H_resid [N, L+1, d], G {l: [N, d_i]}, U {l: [N, d_i]} (last token)."""
    layers = model.model.layers
    buf = {}

    def mk_hook(name):
        def hook(_m, _i, out):
            buf[name] = out[0, -1, :].detach().float().cpu()
        return hook

    handles = []
    for i in layers_wanted:
        handles.append(layers[i].mlp.act_fn.register_forward_hook(mk_hook(f"g{i}")))
        handles.append(layers[i].mlp.up_proj.register_forward_hook(mk_hook(f"u{i}")))

    H_resid = []
    G = {i: [] for i in layers_wanted}
    U = {i: [] for i in layers_wanted}
    try:
        for n, (_, txt) in enumerate(items):
            buf.clear()
            ids = tok(txt, return_tensors="pt").to(dev)
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states
            H_resid.append(torch.stack([h[0, -1, :].detach().float().cpu() for h in hs], 0))
            for i in layers_wanted:
                G[i].append(buf[f"g{i}"]); U[i].append(buf[f"u{i}"])
            if (n + 1) % 10 == 0 or n + 1 == len(items):
                print(f"\r[extract] {n+1}/{len(items)} sentences", end="", flush=True)
        print()
    finally:
        for h in handles:
            h.remove()
    return (torch.stack(H_resid, 0),
            {i: torch.stack(v, 0) for i, v in G.items()},
            {i: torch.stack(v, 0) for i, v in U.items()})


def load_all(a, dataset_items=None):
    dev, _ = T.resolve_device_dtype("auto", "auto")
    dt = torch.float32
    if dataset_items is None:
        pairs = list(T.DEFAULT_PAIRS) if a.dataset == "builtin" else \
            T.load_pairs(a.dataset, a.max_pairs, a.seed, a.rev_counterfact,
                         a.rev_truthfulqa, a.file_counterfact, a.file_truthfulqa)
        items, pidx = T.pairs_to_items(pairs)
        print(f"[data] {len(items)} sentences = {len(pidx)} pairs")
    else:
        items, pidx = dataset_items, None
    print(f"[model] {a.model} on {dev} ({dt})")
    tok, model = T.load_model(a.model, dt, dev)
    return dev, tok, model, items, pidx


# =====================================================================
#  TASK attrib -- exact gate/value attribution of the FFN's axis gap
# =====================================================================
def cmd_attrib(a):
    scan = list(range(a.scan_start, a.scan_end + 1))
    print("[task attrib] gate vs value: EXACT split of the FFN's truth-axis gap")
    print(f"[note] w = W_down^T v1 (residual axis pulled back; NO fitting in 8960-dim)")
    print(f"[note] v1 from intact residual @block {a.axis_block}, train folds only")
    dev, tok, model, items, pidx = load_all(a)
    H_resid, G, U = collect_gu(model, tok, items, dev, scan)

    # exactness check on real data (identity gate + value == total)
    Lb0 = scan[0]
    w0 = (model.model.layers[Lb0].mlp.down_proj.weight.detach().float().cpu().T
          @ torch.randn(H_resid.shape[2]))
    it0, if0 = pidx[0]
    g1, v1_, t1 = gate_value_split(G[Lb0][it0], U[Lb0][it0], G[Lb0][if0], U[Lb0][if0], w0)
    err = abs(float(g1 + v1_ - t1)) / max(abs(float(t1)), 1e-6)
    print(f"[identity check] |gate+value-total|/|total| = {err:.2e}  (must be ~0)")
    if err > 1e-3:
        print("  ABORT: split identity violated."); return

    Hax = H_resid[:, a.axis_block + 1, :]
    print(f"\n{'layer':>5} | {'gate term':>10} {'value term':>11} {'total':>9} | "
          f"{'gate share':>10}")
    print("-" * 60)
    for Lb in scan:
        Wd = model.model.layers[Lb].mlp.down_proj.weight.detach().float().cpu()  # [d, d_i]
        gate_ms, val_ms, tot_ms = [], [], []
        for tr, te in T.kfold_pairs(len(pidx), a.folds, a.seed):
            ax = T.fit_axis(Hax, [pidx[p] for p in tr])
            w = Wd.T @ ax["v1"]                              # [d_i]
            gs, vs, ts = [], [], []
            for p in te:
                it, iff = pidx[p]
                gt, vt, tt = gate_value_split(G[Lb][it], U[Lb][it],
                                              G[Lb][iff], U[Lb][iff], w)
                gs.append(float(gt)); vs.append(float(vt)); ts.append(float(tt))
            gate_ms.append(sum(gs) / len(gs)); val_ms.append(sum(vs) / len(vs))
            tot_ms.append(sum(ts) / len(ts))
        gm = sum(gate_ms) / len(gate_ms); vm = sum(val_ms) / len(val_ms)
        tm = sum(tot_ms) / len(tot_ms)
        share = gm / tm if abs(tm) > 1e-8 else float("nan")
        band = "<- band" if a.band_start <= Lb <= a.band_end else ""
        print(f"{Lb:>5} | {gm:>+10.3f} {vm:>+11.3f} {tm:>+9.3f} | {share:>10.2f} {band}")

    print("\n=== reading guide ===")
    print("  total = FFN's mean intra-pair gap on the truth axis (matches contrib's gap_ffn).")
    print("  In the band (total < 0): gate share >> value share -> the anti-truth writing")
    print("  is CONTEXT SWITCHING (which neurons open differ by class); value share")
    print("  dominant -> the content itself is anti-truth. Around the peak (total > 0):")
    print("  the same split tells you how the PRO-truth writing is built at 13-15.")
    print("  gate share > 1 with value share < 0 (or vice versa) = the two terms FIGHT.")


# =====================================================================
#  TASK framing -- the 2x2 moral/prudential factor test (polarity-style)
# =====================================================================
def cmd_framing(a):
    print("[task framing] is 'truth' contaminated by justification framing?")
    print("[design] 2x2: moral/prudential x true/false, leave-facts-out CV")
    items, fact, is_true, is_prud = build_framing_items()
    print(f"[data] {len(items)} sentences = {int(fact.max())+1} facts x 4 cells")
    dev, tok, model, _, _ = load_all(a, dataset_items=items)
    H = T.collect(model, tok, items, dev, "last")
    layers = list(range(H.shape[1]))
    n_facts = int(fact.max()) + 1

    rows = {}
    for L in layers:
        Hl = H[:, L, :].float()
        acc = dict(cosMP=[], mm=[], mp=[], gm=[], gp=[], bias=[])
        for tr_f, te_f in T.kfold_facts(n_facts, a.folds, a.seed):
            c = cells_framing(fact, is_true, is_prud, tr_f)
            ct = cells_framing(fact, is_true, is_prud, te_f)
            if min(len(ct["MT"]), len(ct["MF"]), len(ct["PT"]), len(ct["PF"])) == 0:
                continue
            tM = T.unit(Hl[c["MT"]].mean(0) - Hl[c["MF"]].mean(0))
            tP = T.unit(Hl[c["PT"]].mean(0) - Hl[c["PF"]].mean(0))
            tG = T.unit(Hl[torch.cat([c["MT"], c["PT"]])].mean(0)
                        - Hl[torch.cat([c["MF"], c["PF"]])].mean(0))
            acc["cosMP"].append(float(torch.dot(tM, tP)))
            im = torch.cat([ct["MT"], ct["MF"]]); ip = torch.cat([ct["PT"], ct["PF"]])
            acc["mm"].append(T.auc_score(Hl[im] @ tM, is_true[im].long()))
            acc["mp"].append(T.auc_score(Hl[ip] @ tM, is_true[ip].long()))
            acc["gm"].append(T.auc_score(Hl[im] @ tG, is_true[im].long()))
            acc["gp"].append(T.auc_score(Hl[ip] @ tG, is_true[ip].long()))
            # framing bias among TRUE statements, on the general axis, in sigma units
            pm = Hl[ct["MT"]] @ tG; pp = Hl[ct["PT"]] @ tG
            pooled = float(torch.sqrt((pm.var() + pp.var()) / 2).clamp_min(1e-8))
            acc["bias"].append(float(pm.mean() - pp.mean()) / pooled)
        rows[L] = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}

    print(f"\n{'layer':>5} | {'cos(tM,tP)':>10} | {'M->M':>6} {'M->P':>6} | "
          f"{'tG->M':>6} {'tG->P':>6} | {'bias MT-PT (sigma)':>18}")
    for L in layers:
        r = rows[L]
        print(f"{L:>5} | {r['cosMP']:>+10.3f} | {r['mm']:>6.3f} {r['mp']:>6.3f} | "
              f"{r['gm']:>6.3f} {r['gp']:>6.3f} | {r['bias']:>+18.2f}")

    best = max(layers, key=lambda L: rows[L]["mm"] if rows[L]["mm"] == rows[L]["mm"] else -1)
    r = rows[best]
    print(f"\nverdict @layer {best} (chosen on moral->moral; prudential stays OOD):")
    print(f"  cos(t_moral, t_prudential) : {r['cosMP']:+.3f}   "
          "(near +1 = same truth axis; low/negative = framing rotates it, like polarity did)")
    print(f"  moral axis -> moral        : {r['mm']:.3f}")
    print(f"  moral axis -> prudential   : {r['mp']:.3f}   (<0.5 = flip, framing factor is real)")
    print(f"  general tG -> M / P        : {r['gm']:.3f} / {r['gp']:.3f}")
    print(f"  bias MT vs PT on tG        : {r['bias']:+.2f} sigma   "
          "(|bias| >~ 0.5: the axis systematically prefers one framing among TRUE statements")
    print("   -> the 'truth' reading is framing-contaminated: your prison-vs-wrong effect, measured.)")

    # unsupervised recovery of a framing direction (like recovery of t_P)
    print("\n[recovery-style check] does the SVD of MIXED-framing pairs put framing in a top PC?")
    res = {}
    for L in layers:
        Hl = H[:, L, :].float()
        cg1, cfk, pck = [], [], []
        for s in range(a.splits):
            fl = list(range(n_facts))
            import random as _r; _r.Random(a.seed + s).shuffle(fl)
            sub = fl[: int(n_facts * 0.8)]
            c = cells_framing(fact, is_true, is_prud, sub)
            by = {}
            for i in c["ALL"].tolist():
                by.setdefault(int(fact[i]), {})[(int(is_prud[i]), int(is_true[i]))] = i
            it, iff = [], []
            for f_, cc in by.items():
                if (0, 1) in cc and (0, 0) in cc: it.append(cc[(0, 1)]); iff.append(cc[(0, 0)])
                if (1, 1) in cc and (1, 0) in cc: it.append(cc[(1, 1)]); iff.append(cc[(1, 0)])
            D = Hl[torch.tensor(it)] - Hl[torch.tensor(iff)]
            _, _, Vh = torch.linalg.svd(D, full_matrices=False)
            V = Vh[: a.topk]
            tM = Hl[c["MT"]].mean(0) - Hl[c["MF"]].mean(0)
            tPd = Hl[c["PT"]].mean(0) - Hl[c["PF"]].mean(0)
            tG = T.unit(tM + tPd)
            tF = tM - tPd; tF = T.unit(tF - torch.dot(tF, tG) * tG)
            cg1.append(abs(float(torch.dot(T.unit(V[0]), tG))))
            coss = [abs(float(torch.dot(T.unit(V[j]), tF))) for j in range(V.shape[0])]
            cfk.append(max(coss)); pck.append(int(torch.tensor(coss).argmax()) + 1)
        res[L] = dict(cg1=sum(cg1) / len(cg1), cfk=sum(cfk) / len(cfk),
                      pc=round(sum(pck) / len(pck), 1))
    bestL = max(layers, key=lambda L: res[L]["cfk"])
    r = res[bestL]
    print(f"  @layer {bestL}: best |cos(SVD_k, t_framing)| = {r['cfk']:.3f} in PC ~{r['pc']}   "
          f"cos(SVD_1, tG) = {r['cg1']:.3f}")
    print("  -> high cos = the framing direction exists and the unsupervised SVD finds it")
    print("     without labels, exactly as it recovered Buerger's t_P.")


# =====================================================================
#  TASK gatefreeze -- CAUSAL: kill gate variation vs value variation in band
# =====================================================================
def cmd_gatefreeze(a):
    band = list(range(a.band_start, a.band_end + 1))
    ro = a.readout
    print("[task gatefreeze] causal: is the erosion carried by gate or value VARIATION?")
    print(f"[note] band {band}, readout block {ro}. Intervention: replace last-token g")
    print("       (or u) with the INTRA-PAIR MEAN cached from the intact run -- kills the")
    print("       class-driven variation of that stream, keeps the other. FFN stays on.")
    dev, tok, model, items, pidx = load_all(a)

    # pass 0: intact -- cache g,u and read intact/frozen reference states
    H_resid, G, U = collect_gu(model, tok, items, dev, band)
    H_int_ro = H_resid[:, ro + 1, :]
    H_int_entry = H_resid[:, a.band_start, :]
    # intra-pair mean targets
    Gbar, Ubar = {}, {}
    for Lb in band:
        gb = torch.empty_like(G[Lb]); ub = torch.empty_like(U[Lb])
        for it, iff in pidx:
            gb[it] = gb[iff] = (G[Lb][it] + G[Lb][iff]) / 2
            ub[it] = ub[iff] = (U[Lb][it] + U[Lb][iff]) / 2
        Gbar[Lb], Ubar[Lb] = gb, ub

    # patched forward machinery
    holder = {"idx": None, "kind": None}
    originals = {}

    def make_fwd(mlp, Lb):
        def fwd(x):
            g = mlp.act_fn(mlp.gate_proj(x)); u = mlp.up_proj(x)
            if holder["kind"] == "gate":
                g = g.clone()
                g[0, -1, :] = Gbar[Lb][holder["idx"]].to(g.device).to(g.dtype)
            elif holder["kind"] == "value":
                u = u.clone()
                u[0, -1, :] = Ubar[Lb][holder["idx"]].to(u.device).to(u.dtype)
            return mlp.down_proj(g * u)
        return fwd

    for Lb in band:
        mlp = model.model.layers[Lb].mlp
        originals[Lb] = mlp.forward
        mlp.forward = make_fwd(mlp, Lb)

    def run_condition(kind, tag):
        holder["kind"] = kind
        H = []
        for n, (_, txt) in enumerate(items):
            holder["idx"] = n
            ids = tok(txt, return_tensors="pt").to(dev)
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states
            H.append(hs[ro + 1][0, -1, :].detach().float().cpu())
            if (n + 1) % 20 == 0 or n + 1 == len(items):
                print(f"\r  [{tag:>12}] {n+1}/{len(items)}", end="", flush=True)
        print()
        return torch.stack(H, 0)

    try:
        print("\n[interventions]")
        H_gf = run_condition("gate", "gate-freeze")
        H_vf = run_condition("value", "value-freeze")
    finally:
        for Lb, f in originals.items():
            model.model.layers[Lb].mlp.forward = f

    def refit(Hs):
        aucs = []
        for tr, te in T.kfold_pairs(len(pidx), a.folds, a.seed):
            ax = T.fit_axis(Hs, [pidx[p] for p in tr])
            I, Y = [], []
            for p in te:
                it, iff = pidx[p]; I += [it, iff]; Y += [1, 0]
            aucs.append(T.auc_score(T.project_fields(Hs[I], ax)["Re"], torch.tensor(Y)))
        return sum(aucs) / len(aucs)

    r_int = refit(H_int_ro); r_frz = refit(H_int_entry)
    r_gf = refit(H_gf); r_vf = refit(H_vf)

    print(f"\n=== held-out truth AUC @block {ro}, band {band} ===")
    print(f"  intact @readout            : {r_int:.3f}")
    print(f"  intact @entry (=frozen ref): {r_frz:.3f}")
    print(f"  GATE-freeze  (Dg killed)   : {r_gf:.3f}   ({r_gf-r_int:+.3f} vs intact)")
    print(f"  VALUE-freeze (Du killed)   : {r_vf:.3f}   ({r_vf-r_int:+.3f} vs intact)")
    print("\n=== causal reading ===")
    print("  reference: full FFN-off recovered to ~0.815 (ffn_erosion ablate).")
    if r_gf - r_int > 0.03 and r_gf - r_vf > 0.03:
        print("  -> killing GATE variation recovers the signal: the erosion is CONTEXT")
        print("     SWITCHING -- which neurons open differs by class in a way that writes")
        print("     anti-truth. The content stream is not the problem. H-context confirmed.")
    elif r_vf - r_int > 0.03 and r_vf - r_gf > 0.03:
        print("  -> killing VALUE variation recovers the signal: the transported content")
        print("     is the eroder; the gating pattern is innocent. H-content.")
    elif max(r_gf, r_vf) - r_int <= 0.03:
        print("  -> neither freeze recovers what full FFN-off recovers: the erosion needs")
        print("     BOTH streams (or acts off the last token / through earlier positions).")
        print("     Honest partial negative; compare against attrib's split for coherence.")
    else:
        print("  -> both freezes recover comparably: gate and value variation each carry")
        print("     part of the erosion; the attrib split quantifies the shares.")


# =====================================================================
#  CLI
# =====================================================================
def add_common(p):
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--dataset", default="counterfact",
                   choices=["builtin", "counterfact", "truthfulqa", "mix"])
    p.add_argument("--max-pairs", type=int, default=250)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--band-start", type=int, default=16)
    p.add_argument("--band-end", type=int, default=18)
    p.add_argument("--rev-counterfact", default="c945b082ca08d0a8f3ba227fb78404a09614c36e")
    p.add_argument("--rev-truthfulqa", default="741b8276f2d1982aa3d5b832d3ee81ed3b896490")
    p.add_argument("--file-counterfact", default=None)
    p.add_argument("--file-truthfulqa", default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)

    pa = sub.add_parser("attrib", help="exact gate/value split of the FFN's truth-axis gap")
    add_common(pa)
    pa.add_argument("--axis-block", type=int, default=15)
    pa.add_argument("--scan-start", type=int, default=12)
    pa.add_argument("--scan-end", type=int, default=19)
    pa.set_defaults(func=cmd_attrib)

    pf = sub.add_parser("framing", help="2x2 moral/prudential factor test (built-in quads)")
    add_common(pf)
    pf.add_argument("--splits", type=int, default=8)
    pf.add_argument("--topk", type=int, default=6)
    pf.set_defaults(func=cmd_framing, folds=6)

    pg = sub.add_parser("gatefreeze", help="causal: freeze gate vs value variation in band")
    add_common(pg)
    pg.add_argument("--readout", type=int, default=18)
    pg.set_defaults(func=cmd_gatefreeze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
