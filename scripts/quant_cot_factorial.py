#!/usr/bin/env python3
"""C3 — Precision x prompt-mode factorial: does quantization disproportionately break CoT?

Our paper claimed chain-of-thought "collapses to near-random" under 4-bit quantization. That claim
is confounded (quantization vs. the prompt itself) and, crucially, is CONTRADICTED by the UNLP 2026
2nd-place system (Trokhymovych et al., arXiv:2604.22095), which ran the SAME MamayLM-12B at the SAME
4-bit quantization with a Ukrainian light-reasoning prompt and saw NO collapse -- because they
LoRA-fine-tuned first.

So the claim must be stated CONDITIONALLY (off-the-shelf / no-fine-tune regime), and the
quantization effect must be isolated from the prompt effect. This is the 2x2:

                     direct prompt        CoT prompt
    INT8  (ref)          a                    b
    NF4   (4-bit)        c                    d

  * (c - a) = quantization cost on DIRECT answering
  * (d - b) = quantization cost on CoT
  * INTERACTION = (d - b) - (c - a)  <- this is the scientific claim: does low precision hurt
    long-form reasoning MORE than short-form answering?

Note: a true bf16 anchor for a 12B model does not fit in 24 GB (A10), so INT8 is the
high-precision reference (near-lossless per the quantization literature). State as a limitation.

Context comes from the same dense retrieval the system uses (top-k pages), so this measures the
answering stage in situ, not in a vacuum.
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import stamp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_eval import (load_unlp_corpus, load_unlp_queries, bge_encode,
                            run_dense, paired_bootstrap, REPO)

OPTS = ["A", "B", "C", "D", "E", "F"]

DIRECT_TMPL = (
    "Контекст:\n{ctx}\n\n"
    "Питання: {q}\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}\nE) {E}\nF) {F}\n\n"
    "Відповідь (лише одна літера):"
)

COT_TMPL = (
    "Контекст:\n{ctx}\n\n"
    "Питання: {q}\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}\nE) {E}\nF) {F}\n\n"
    "Поміркуй крок за кроком, потім дай відповідь у форматі 'Відповідь: X'.\n"
    "Міркування:"
)


def build_ctx(q, ranked, page_text, k=2, max_chars=1500):
    _, units = ranked[q["qid"]]
    parts = []
    for (doc, pg) in units[:k]:
        t = page_text.get((doc, pg), "")[:max_chars]
        parts.append(f"[{doc} c.{pg}]\n{t}")
    return "\n\n".join(parts)


# Cyrillic homoglyphs the model may emit instead of the Latin option letters.
HOMOGLYPH = {"А": "A", "В": "B", "С": "C", "Е": "E", "Ф": "F", "Д": "D"}


def parse_answer(text, rule="last"):
    """SYMMETRIC answer extraction — the SAME rule for every arm of the factorial.

    The first version used different rules per arm (`direct` took the FIRST standalone letter in an
    8-token window; `cot` took the LAST letter in a 256-token window). That is a parser difference
    aligned exactly with the factor whose interaction we are testing, so any interaction estimate was
    confounded by the extraction rule.

    `rule` is applied identically to both arms; we report BOTH rules as a sensitivity check.
    """
    t = "".join(HOMOGLYPH.get(ch, ch) for ch in text.strip())
    m = re.search(r"Відповідь\s*:?\s*\**\s*([A-Fa-f])", t)      # explicit answer slot, either arm
    if m:
        return m.group(1).upper()
    cands = re.findall(r"\b([A-Fa-f])\b", t)
    if not cands:
        return None
    return (cands[-1] if rule == "last" else cands[0]).upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--precisions", default="int8,nf4")
    ap.add_argument("--modes", default="direct,cot")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/c3_quant_cot.json"))
    args = ap.parse_args()

    units, doc_domain, _ = load_unlp_corpus(os.path.join(REPO, "data/extracted_text"))
    queries = load_unlp_queries(os.path.join(REPO, "data/dev_questions.csv"))[: args.n]
    page_text = {(u[0], u[1]): u[2] for u in units}

    # retrieve context with the same dense retriever the system uses
    cache = os.path.join(REPO, "outputs/retrieval_eval/unlp_embs.npz")
    z = np.load(cache)
    page_embs = z["page"]
    q_embs = bge_encode([q["text"] for q in queries], max_length=256)
    ranked, _ = run_dense(queries, units, page_embs, q_embs)
    print(f"[data] {len(queries)} questions, context = top-2 dense pages", file=sys.stderr)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # RESUME. Reload any cells a previous (killed) run already completed, so a restart pays only for
    # what is actually missing. `per_question` is the item-level record, so raw[] rebuilds exactly.
    results, raw = {}, {}
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out)).get("results", {})
            for k, v in prev.items():
                if isinstance(v, dict) and "per_question" in v:
                    results[k], raw[k] = v, v["per_question"]
            if results:
                print(f"[resume] {len(results)} cell(s) already done: {sorted(results)}",
                      file=sys.stderr)
        except Exception as e:
            print(f"[resume] ignoring unreadable {args.out}: {e}", file=sys.stderr)

    todo = [(p, m) for p in args.precisions.split(",") for m in args.modes.split(",")
            if f"{p}_{m}" not in results]
    if not todo:
        print("[resume] every cell already present; going straight to the analysis",
              file=sys.stderr)

    for prec in args.precisions.split(","):
        if not any(p == prec for p, _ in todo):
            print(f"[skip] {prec}: all its cells are done", file=sys.stderr)
            continue
        if prec == "nf4":
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)
        elif prec == "int8":
            qc = BitsAndBytesConfig(load_in_8bit=True)
        else:
            qc = None
        print(f"[load] {args.model} @ {prec}", file=sys.stderr)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=qc, device_map="cuda",
            dtype=torch.bfloat16).eval()

        for mode in args.modes.split(","):
            if (prec, mode) not in todo:
                continue
            # cot_trunc reproduces the ORIGINAL system's bug: a CoT prompt run under the
            # 3-4 token generation cap of pipeline_shared.score_mcq (max_tokens=3/4,
            # stop=["\n",".", " "]) -- the model can never reach its answer letter.
            tmpl = DIRECT_TMPL if mode == "direct" else COT_TMPL
            maxnew = {"direct": 8, "cot": 256, "cot_trunc": 4}[mode]
            hits_last = hits_first = 0
            per_q, per_q_first, gen_texts = {}, {}, {}
            t0 = time.time()
            parse_fail, gen_lens = 0, []
            for q in queries:
                ctx = build_ctx(q, ranked, page_text)
                prompt = tmpl.format(ctx=ctx, q=q["text"],
                                     **{o: q.get(o, "") for o in OPTS})
                msgs = [{"role": "user", "content": prompt}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=maxnew, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                new = out[0][ids.shape[-1]:]
                gen_lens.append(int(new.shape[-1]))
                txt = tok.decode(new, skip_special_tokens=True)
                # SAME rule for every arm; both rules recorded as a sensitivity check
                p_last = parse_answer(txt, "last")
                p_first = parse_answer(txt, "first")
                if p_last is None:
                    parse_fail += 1
                ok = 1.0 if p_last == q["gold_answer"] else 0.0
                okf = 1.0 if p_first == q["gold_answer"] else 0.0
                per_q[q["qid"]] = ok
                per_q_first[q["qid"]] = okf
                gen_texts[q["qid"]] = txt[:300]
                hits_last += ok
                hits_first += okf
            n = len(queries)
            acc = hits_last / n
            key = f"{prec}_{mode}"
            results[key] = {"answer_acc": acc, "n": n,
                            "answer_acc_firstletter_rule": hits_first / n,
                            "parse_fail_rate": parse_fail / n,
                            "mean_gen_tokens": float(np.mean(gen_lens)),
                            "runaway_rate": float(np.mean([g >= maxnew for g in gen_lens])),
                            "secs": round(time.time() - t0, 1),
                            # E1: per-item data, so "identical to the item" is checkable rather
                            # than inferred from two equal means (119/150 == 119/150 says nothing
                            # about WHICH 150).
                            "per_question": per_q,
                            "per_question_firstletter": per_q_first,
                            "sample_generations": dict(list(gen_texts.items())[:5])}
            raw[key] = per_q
            r = results[key]
            print(f"{key:<14} acc={acc:.4f}  parse_fail={r['parse_fail_rate']:.3f}  "
                  f"gen_tok={r['mean_gen_tokens']:.0f}  runaway={r['runaway_rate']:.3f}  "
                  f"({r['secs']}s)", flush=True)
            # CHECKPOINT AFTER EVERY CELL. A cell costs minutes of 12B generation, and the whole run
            # was previously a single dump at the end -- so a wall-clock timeout or an OOM lost all
            # of it. (We already lost an 18-language MIRACL run to exactly this.) With this, a
            # killed run resumes from the last completed cell instead of from zero.
            json.dump(stamp({"model": args.model, "results": results}, "quant_cot_factorial.py"), open(args.out, "w"), indent=2)
        del model
        torch.cuda.empty_cache()

    # --- E6: the baselines the claim "collapses to near-random" needs but never had ---
    golds = [q["gold_answer"] for q in queries]
    n_opts = [sum(1 for o in OPTS if str(q.get(o, "")).strip()) for q in queries]
    always_a = float(np.mean([g == "A" for g in golds]))
    uniform = float(np.mean([1.0 / k for k in n_opts if k]))
    results["_baselines"] = {"always_A": always_a, "uniform_random": uniform,
                             "mean_n_options": float(np.mean(n_opts))}
    print(f"\n=== BASELINES (N={len(queries)}) ===")
    print(f"  always-'A' (the parser's silent fallback) : {always_a:.4f}")
    print(f"  uniform random over live options          : {uniform:.4f}")

    # --- the interaction: does low precision hurt CoT more than direct? ---
    def acc(k):
        return results[k]["answer_acc"]
    try:
        a, b = acc("int8_direct"), acc("int8_cot")
        c, d = acc("nf4_direct"), acc("nf4_cot")
        qids = sorted(raw["int8_direct"])

        # E1: equal MEANS are not identical ITEMS. Report the actual item-level churn.
        disc = [(raw["int8_direct"][q], raw["nf4_direct"][q]) for q in qids]
        n01 = sum(1 for x, y in disc if x == 0 and y == 1)
        n10 = sum(1 for x, y in disc if x == 1 and y == 0)
        md, lod, hid = paired_bootstrap([raw["nf4_direct"][q] for q in qids],
                                        [raw["int8_direct"][q] for q in qids])
        print("\n=== FACTORIAL ===")
        print(f"  INT8 direct={a:.4f}   INT8 cot={b:.4f}")
        print(f"  NF4  direct={c:.4f}   NF4  cot={d:.4f}")
        print(f"\n  quantization cost on DIRECT : {md:+.4f}  95% CI [{lod:+.4f}, {hid:+.4f}]")
        print(f"     item-level churn: {n01} items INT8-wrong->NF4-right, "
              f"{n10} INT8-right->NF4-wrong  ({n01+n10} of {len(qids)} items differ)")
        if n01 + n10 > 0:
            print(f"     >> the two cells are NOT item-identical; equal accuracy is a coincidence "
                  f"of equal COUNTS. Do not write 'identical to the item'.")
        else:
            print(f"     >> the two cells ARE item-identical (0 discordant pairs).")

        mc, loc, hic = paired_bootstrap([raw["nf4_cot"][q] for q in qids],
                                        [raw["int8_cot"][q] for q in qids])
        print(f"  quantization cost on CoT    : {mc:+.4f}  95% CI [{loc:+.4f}, {hic:+.4f}]")

        dd = [(raw["nf4_cot"][q]-raw["int8_cot"][q]) - (raw["nf4_direct"][q]-raw["int8_direct"][q])
              for q in qids]
        m, lo, hi = paired_bootstrap(dd, [0.0]*len(dd))
        print(f"  INTERACTION (cot - direct)  : {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] "
              f"{'(significant)' if (lo>0 or hi<0) else '(NOT significant)'}")
        # B6: an accept-the-null needs a resolvable-effect statement.
        sd = float(np.std(dd, ddof=1))
        mde = 2.802 * sd / np.sqrt(len(dd))          # (z_.975 + z_.80) * SE
        print(f"     MDE @80% power for this interaction at n={len(dd)}: {mde:.4f}. "
              f"We can only call the interaction NULL relative to effects of that size or larger.")
        results["interaction"] = {
            "quant_cost_direct": md, "quant_cost_direct_ci": [lod, hid],
            "direct_discordant_pairs": {"n01": n01, "n10": n10, "total_items": len(qids)},
            "quant_cost_cot": mc, "quant_cost_cot_ci": [loc, hic],
            "interaction": m, "ci": [lo, hi], "mde80": float(mde)}
    except KeyError:
        pass

    json.dump(stamp({"model": args.model, "results": results}, "quant_cot_factorial.py"), open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
