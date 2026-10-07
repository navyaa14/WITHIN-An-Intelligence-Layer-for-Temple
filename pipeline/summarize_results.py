"""Print the real results in plain language, so you can read them before anything goes into the pitch.
    python summarize_results.py --out ./out
"""
import argparse, json, os


def pct(v):
    return f"{100 * v:.1f}%"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./out"); a = ap.parse_args()
    load = lambda n: json.load(open(os.path.join(a.out, n))) if os.path.exists(os.path.join(a.out, n)) else None
    R, T, P, Z = (load(n) for n in ("refed_results.json", "tufts_results.json", "physionet_results.json", "personalization_results.json"))
    line = "-" * 72
    if R:
        print(line, "\nFEEL  (REFED, fNIRS only, new people never seen in training)")
        for m, r in R["models"].items():
            for d in ("valence", "arousal"):
                x = r[d]
                print(f"  {m:12s} {d:8s} balanced acc {pct(x['bacc'])} ± {pct(x['bacc_std'])}   chance {pct(x.get('chance_bacc', 1 / 3))}   macro-F1 {x['f1_macro']:.2f}")
        vh = R.get("subject_video_holdout")
        if vh:
            print("  new people AND new videos (stimulus check):")
            for m, r in vh["models"].items():
                print("   ", m, "  ".join(f"{d} {pct(r[d]['bacc'])} (chance {pct(r[d].get('chance_bacc', 1 / 3))})" for d in ("valence", "arousal")))
    else:
        print(line, "\nFEEL: no refed_results.json")
    if T:
        print(line, "\nTHINK  (Tufts, 4 workload levels, official subject buckets)")
        for m, x in T["models"].items():
            print(f"  {m:12s} balanced acc {pct(x['bacc'])} ± {pct(x['bacc_std'])}   chance {pct(x.get('chance_bacc', .25))}   macro-F1 {x['f1_macro']:.2f}")
    else:
        print(line, "\nTHINK: no tufts_results.json (automatic Tufts download/training did not complete)")
    if P:
        print(line, "\nTEMPLE RELEVANCE  (PhysioNet, same n-back task)")
        t = P["tcd"]["two_back"]; print(f"  TCD blood-flow velocity, 2-back: {t['mean']:+.1f}% ± {t['sem']:.1f}, up in {t['n_positive']} of {t['n']}")
        f = P.get("fnirs")
        if f:
            x = f["two_back"]; print(f"  fNIRS prefrontal HbO, 2-back: {x['mean']:+.2f} SD ± {x['sem']:.2f}, up in {x['n_positive']} of {x['n']}"
                                     + (f", p = {x['p_wilcoxon_greater']:.3g}" if "p_wilcoxon_greater" in x else ""))
            if f.get("failed"):
                print("  fNIRS recordings that failed:", f["failed"])
    if Z:
        print(line, "\nPERSONALIZATION  (does a person's own labelled data help?)")
        tasks = [(f"Feel {d}", Z["refed"][d]) for d in ("arousal", "valence") if "refed" in Z] + ([("Think load", Z["tufts"]["load"])] if "tufts" in Z else [])
        for name, c in tasks:
            for r in c["fine_tuned"]:
                if r["k"] == 0:
                    print(f"  {name:12s} k=0  {pct(r['bacc'])} (generic model; chance {pct(c['chance_bacc'])})"); continue
                tag = "RELIABLE GAIN" if r.get("significant") else "not reliable"
                ci = r.get("ci95"); ci = f"CI {ci[0] * 100:+.1f}..{ci[1] * 100:+.1f}" if ci else ""
                print(f"  {name:12s} k={r['k']}  {pct(r['bacc'])}  gain {r['gain_vs_generic'] * 100:+.1f} pts  {ci}  {r['n_improved']}/{r['n_people']} improved  -> {tag}")
        print("  ", Z.get("significance_rule", ""))
    print(line, "\nOnly numbers printed above may go into the pitch. Send the four JSON files in", a.out, "for a check.")


if __name__ == "__main__":
    main()
