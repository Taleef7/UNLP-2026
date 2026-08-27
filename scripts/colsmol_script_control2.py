#!/usr/bin/env python3
"""C2 REBUILD — the 2x2 the first version collapsed to its diagonal.

The first C2 (`colsmol_script_control.py`) rendered pages in Cyrillic vs Latin and reported that
transliteration more than doubled ColSmol's Doc@1 (+0.240), concluding a *visual* script-coverage
deficit. But it transliterated the QUERY at the same time:

    qtexts = [unidecode(q["text"]) if latin else q["text"] for q in qs]     # <-- the confound

So the "Latin" condition changed BOTH the page image AND the query text. SmolVLM's *text* tower
handles Latin far better than Cyrillic, so the entire gain may be query-side and have nothing to do
with reading glyphs off a page. As written, the experiment cannot distinguish its own two
hypotheses.

The fix is the full factorial:

                        query: Cyrillic        query: Latin
    page: Cyrillic          (a)                    (b)
    page: Latin             (c)                    (d)

  * (c - a) = the PAGE/visual script effect, with the query held in Cyrillic   <- the claim we made
  * (b - a) = the QUERY/text-tower script effect, with the page held in Cyrillic
  * (d - a) = what the old experiment actually measured (both moved at once)
  * interaction = (d - c) - (b - a)

Also fixed here:
  * D2 CONTENT MATCHING — `unidecode` expands Cyrillic (щ -> shch), and the old renderer capped at a
    fixed line count, so the Latin page held strictly LESS of the document. We now truncate the
    SOURCE text to a fixed character budget first and verify both renders retain it, reporting the
    retained fraction for each condition.
  * D4 CHANCE BASELINE — the old `1/n_docs` ignores that first-unique-doc ranking over PAGES makes
    multi-page documents likelier to rank first. Under random scores, P(doc d first) =
    n_pages(d)/n_pages_total, so chance for a query with gold g is n_pages(g)/n_pages_total. The old
    figure (0.024) understated chance, which is how 0.039 got called "at or below chance."
  * D3 ARTIFACTS — every cell dumps per-query hits so the paired bootstrap and any later test have
    a serialized basis (the ColQwen 0.920 number previously existed only as a line of stdout).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_eval import load_unlp_corpus, load_unlp_queries, paired_bootstrap, REPO

FONT = os.environ.get("DEJAVU_TTF", "")


def render(text, latin, W=800, H=1000, size=16):
    """Identical renderer for both scripts. `text` is ALREADY truncated at the source level."""
    from PIL import Image, ImageDraw, ImageFont
    from unidecode import unidecode
    if latin:
        text = unidecode(text)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, size)
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > W - 40:
            lines.append(cur); cur = w
        else:
            cur = t
    lines.append(cur)
    max_lines = (H - 40) // (size + 6)
    kept = lines[:max_lines]
    y = 20
    for ln in kept:
        d.text((20, y), ln, fill="black", font=f)
        y += size + 6
    retained = sum(len(l) for l in kept) / max(1, sum(len(l) for l in lines))
    return img, retained


def load_model(name):
    from colpali_engine.models import (ColIdefics3, ColIdefics3Processor,
                                       ColQwen2_5, ColQwen2_5_Processor)
    if "colSmol" in name:
        m = ColIdefics3.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        p = ColIdefics3Processor.from_pretrained(name)
    else:
        m = ColQwen2_5.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        p = ColQwen2_5_Processor.from_pretrained(name)
    return m, p


def embed_images(model, proc, images, bs=8):
    out = []
    for i in range(0, len(images), bs):
        b = proc.process_images(images[i:i + bs]).to(model.device)
        with torch.no_grad():
            out.extend(list(torch.unbind(model(**b).cpu().float())))
    return out


def embed_queries(model, proc, texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        b = proc.process_queries(texts[i:i + bs]).to(model.device)
        with torch.no_grad():
            out.extend(list(torch.unbind(model(**b).cpu().float())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vidore/colSmol-500M")
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--src-chars", type=int, default=1100,
                    help="source-text budget per page, applied BEFORE transliteration")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "outputs/colsmol2"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    units, doc_domain, _ = load_unlp_corpus(os.path.join(REPO, "data/extracted_text"))
    queries = load_unlp_queries(os.path.join(REPO, "data/dev_questions.csv"))
    docs_present = {u[0] for u in units}
    # random subset, not a prefix (questions cluster by Doc_ID in file order)
    rng = np.random.default_rng(42)
    elig = [q for q in queries if q["gold_doc"] in docs_present]
    if args.max_queries and len(elig) > args.max_queries:
        idx = sorted(rng.choice(len(elig), size=args.max_queries, replace=False))
        qs = [elig[i] for i in idx]
    else:
        qs = elig
    unit_docs = [u[0] for u in units]

    # ---- correct, page-count-weighted chance baseline (D4) ----
    from collections import Counter
    pages_per_doc = Counter(unit_docs)
    n_pages_total = len(units)
    chance = float(np.mean([pages_per_doc[q["gold_doc"]] / n_pages_total for q in qs]))
    naive_chance = 1.0 / len(docs_present)
    print(f"[data] {len(units)} pages, {len(qs)} queries, {len(docs_present)} docs", file=sys.stderr)
    print(f"[chance] page-count-weighted Doc@1 chance = {chance:.4f}  "
          f"(the old 1/n_docs figure was {naive_chance:.4f} — an UNDERSTATEMENT)", file=sys.stderr)

    from unidecode import unidecode
    model, proc = load_model(args.model)
    print(f"[model] {args.model}", file=sys.stderr)

    src = [u[2][: args.src_chars] for u in units]        # truncate SOURCE, then transliterate
    imgs, retain = {}, {}
    for ps in ("cyr", "lat"):
        rr = [render(t, ps == "lat") for t in src]
        imgs[ps] = [x[0] for x in rr]
        retain[ps] = float(np.mean([x[1] for x in rr]))
    print(f"[render] source retained: cyr={retain['cyr']:.3f}  lat={retain['lat']:.3f}",
          file=sys.stderr)

    demb = {ps: embed_images(model, proc, imgs[ps]) for ps in ("cyr", "lat")}
    qemb = {
        "cyr": embed_queries(model, proc, [q["text"] for q in qs]),
        "lat": embed_queries(model, proc, [unidecode(q["text"]) for q in qs]),
    }

    results, hits = {}, {}
    for ps in ("cyr", "lat"):
        for qsc in ("cyr", "lat"):
            t0 = time.time()
            scores = proc.score_multi_vector(qemb[qsc], demb[ps]).numpy()
            d1 = d10 = 0.0
            per_q = {}
            for i, q in enumerate(qs):
                order = np.argsort(-scores[i])
                seen, dd = set(), []
                for j in order:
                    if unit_docs[j] not in seen:
                        seen.add(unit_docs[j]); dd.append(unit_docs[j])
                h1 = 1.0 if dd[:1] == [q["gold_doc"]] else 0.0
                h10 = 1.0 if q["gold_doc"] in dd[:10] else 0.0
                per_q[q["qid"]] = (h1, h10)
                d1 += h1; d10 += h10
            n = len(qs)
            cell = f"page-{ps}_query-{qsc}"
            hits[cell] = per_q
            results[cell] = {"Doc@1": d1 / n, "Doc@10": d10 / n, "N": n}
            print(f"  {cell:<24} Doc@1={d1/n:.4f}  Doc@10={d10/n:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)

    # ---- the decomposition the first version could not do ----
    qids = sorted(hits["page-cyr_query-cyr"])
    def col(cell, mi=0):
        return [hits[cell][q][mi] for q in qids]

    a = "page-cyr_query-cyr"; b = "page-cyr_query-lat"
    c = "page-lat_query-cyr"; d = "page-lat_query-lat"
    eff = {}
    for name, (x, y) in {
        "PAGE effect (c-a): page->Latin, query held Cyrillic": (c, a),
        "QUERY effect (b-a): query->Latin, page held Cyrillic": (b, a),
        "BOTH (d-a): what the old experiment measured": (d, a),
    }.items():
        m, lo, hi = paired_bootstrap(col(x), col(y))
        eff[name] = {"delta": m, "ci": [lo, hi]}
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"\n{name}\n   ΔDoc@1 = {m:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")

    inter = [ (hits[d][q][0]-hits[c][q][0]) - (hits[b][q][0]-hits[a][q][0]) for q in qids ]
    m, lo, hi = paired_bootstrap(inter, [0.0]*len(inter))
    print(f"\ninteraction (d-c)-(b-a) = {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    tag = args.model.split("/")[-1]
    json.dump({"model": args.model, "chance_doc1_pageweighted": chance,
               "chance_doc1_naive_1_over_ndocs": naive_chance,
               "source_retained": retain, "results": results, "effects": eff,
               "interaction": {"delta": m, "ci": [lo, hi]},
               "per_query_hits": {k: {q: list(v) for q, v in h.items()} for k, h in hits.items()}},
              open(os.path.join(args.out_dir, f"{tag}_2x2.json"), "w"), indent=2)
    print(f"\n[saved] {args.out_dir}/{tag}_2x2.json")


if __name__ == "__main__":
    main()
