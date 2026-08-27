#!/usr/bin/env python3
"""Stamp every output artifact with the code that produced it, and refuse to read a stale one.

WHY THIS EXISTS. On 2026-07-13 `outputs/retrieval_eval/per_query_hits.json` (written 07-12 22:56) was
still being read by `holm_correction.py` AFTER `retrieval_eval.py` had been fixed (07-13 10:37, commit
848fcce, "fix 3 scoring/design bugs"). Holm family FAM1 was therefore computed from TWO CODE VERSIONS
AT ONCE: its lexical half from pre-fix data on disk, its native half from post-fix data regenerated
that evening. Nothing warned. The file simply sat there looking current.

It changed a reported number (Holm survivors 7/30 -> 6/30). The conclusions happened to survive. Next
time they might not, and we would not know.

A JSON file has no memory of the program that wrote it. So we give it one:

    json.dump(stamp(payload), open(path, "w"))     # writing
    d = load_checked(path)                          # reading -- warns/raises if written by other code

`check_fresh()` compares the recorded commit against HEAD. It does NOT hard-fail by default, because
a legitimately-unchanged artifact from an earlier commit is fine; it fails loudly only when asked
(strict=True), and always prints, so a stale read can never again be silent.
"""
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", REPO, *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def sha16(path):
    """Content hash of an input artifact. Content, not mtime: a file can be rewritten identically."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def provenance(script=None, inputs=None):
    """
    `inputs`: paths to the artifacts this one was DERIVED FROM. We record a content hash of each.

    WHY (2026-07-14). The first version of this module checked only "is the artifact older than the
    script that wrote it". That is necessary and it is NOT sufficient, and we found out the hard way:

        outputs/power_analysis.json was FRESH with respect to power_analysis.py -- and STALE with
        respect to outputs/retrieval_eval/per_query_hits.json, which it reads and which had been
        regenerated under the C10 fix. Its MDE (3.77pp), its churn (5.3%), and every TOST verdict in
        docs/results/C1_hybrid_fusion.md had been computed from the pre-fix hits. The provenance audit
        reported 9/9 OK throughout, because power_analysis.json was not in its target list AND because
        the check it ran could not have caught this even if it had been.

    An artifact is stale if the code that made it changed, OR if anything it was made FROM changed.
    The second edge is the one that bites, because it crosses files and nobody is watching it.
    """
    commit = _git("rev-parse", "HEAD")
    # Dirtiness is scoped to `scripts/`, NOT the whole tree. The question this flag answers is "can
    # this artifact be reproduced from this commit's CODE?" -- and that depends on the source, not on
    # whether some other output file happens to be uncommitted. (Whole-tree dirtiness would also make
    # every artifact self-report dirty the moment the first one in a batch was written, which is a
    # false alarm, and false alarms are how checks get ignored.)
    dirty = bool(_git("status", "--porcelain", "--", "scripts"))
    p = {
        "git_commit": commit or "unknown",
        "git_dirty_scripts": dirty,  # True => scripts/ differs from `commit`; the artifact cannot be
                                     # reproduced from `commit` alone. Say so rather than lie.
        "script": script or os.path.basename(sys.argv[0]),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if inputs:
        p["inputs"] = {os.path.relpath(f, REPO): sha16(f) for f in inputs if os.path.exists(f)}
        missing = [f for f in inputs if not os.path.exists(f)]
        if missing:
            p["inputs_missing"] = [os.path.relpath(f, REPO) for f in missing]
    return p


def stamp(payload, script=None, inputs=None):
    """Return `payload` with a `_provenance` block attached. Payload must be a dict."""
    if not isinstance(payload, dict):
        raise TypeError("stamp() needs a dict payload")
    out = dict(payload)
    out["_provenance"] = provenance(script, inputs)
    return out


def sidecar_path(path):
    return path + ".prov.json"


def write_sidecar(path, script=None, inputs=None):
    """For artifacts whose top-level keys ARE the data (metrics.json is keyed by language;
    per_query_hits.json by variant), injecting a `_provenance` key would masquerade as a language or
    a variant and would trip holm_correction's membership check. Those get a sidecar instead."""
    json.dump(provenance(script, inputs), open(sidecar_path(path), "w"), indent=2)


def read_provenance(path, d=None):
    """Embedded stamp if present, else the sidecar, else None."""
    if d is None:
        try:
            d = json.load(open(path))
        except Exception:
            d = None
    if isinstance(d, dict) and "_provenance" in d:
        return d["_provenance"]
    sc = sidecar_path(path)
    if os.path.exists(sc):
        try:
            return json.load(open(sc))
        except Exception:
            return None
    return None


def _script_last_changed(script):
    """Commit in which the producing script was last modified."""
    if not script:
        return ""
    base = script.split()[0]                       # tolerate "foo.py (retro-stamped ...)"
    for cand in (os.path.join("scripts", base), base):
        c = _git("log", "-1", "--format=%H", "--", cand)
        if c:
            return c
    return ""


def _has_commit(sha):
    """Is this commit object reachable in the repository we are running in?"""
    try:
        subprocess.check_call(["git", "-C", REPO, "cat-file", "-e", f"{sha}^{{commit}}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def staleness(path, d=None):
    """Is this artifact older than the last change to the script that wrote it?

    That -- not "commit != HEAD" -- is the question. An artifact written three commits ago is fine if
    nothing touched its producer since. An artifact written BEFORE its producer was fixed is exactly
    the bug that made Holm family FAM1 span two code versions, and it is what we must catch.

    Returns (status, message) with status in {OK, REPLAYED, DIRTY, STALE, STALE_INPUT, UNSTAMPED}.
    """
    p = read_provenance(path, d)
    name = os.path.relpath(path, REPO)
    if not p:
        return "UNSTAMPED", (f"{name}: no provenance. It may predate a fix to its producer -- "
                             f"exactly how FAM1 ended up spanning two code versions. Regenerate it.")

    # --- INPUT staleness: was this artifact derived from a version of its inputs that no longer
    # exists? This is the edge that produced the power_analysis.json bug (see provenance() above).
    # It crosses files, so neither the producing script's mtime nor its commit can reveal it. ---
    for rel, want in (p.get("inputs") or {}).items():
        f = os.path.join(REPO, rel)
        if not os.path.exists(f):
            return "STALE_INPUT", f"{name}: input {rel} is GONE (was {want}). REGENERATE."
        got = sha16(f)
        if got != want:
            return "STALE_INPUT", (f"{name}: input {rel} has CHANGED since this was written "
                                   f"({want} -> {got}). The artifact was derived from a version of "
                                   f"that file which no longer exists. REGENERATE before using.")

    art_commit = p.get("git_commit", "")
    script_commit = _script_last_changed(p.get("script", ""))

    # --- A history this artifact was not written in. The commit lineage check below is only
    # meaningful when the recorded commit is reachable here; in a squashed or re-created history
    # (e.g. the curated public release, which is one commit) it is not, and every artifact would
    # report STALE for a reason that has nothing to do with staleness. Say what is actually true:
    # the content-hash check above still ran and passed, the lineage check cannot run. ---
    if art_commit and art_commit != "unknown" and not _has_commit(art_commit):
        return "REPLAYED", (f"{name}: written at {art_commit[:8]}, a commit not present in this "
                            f"repository (squashed or re-created history). Inputs verified by "
                            f"content hash; commit lineage not checkable here.")

    if script_commit and art_commit and script_commit != art_commit:
        # STALE iff the script's last change is NOT an ancestor of the artifact's commit,
        # i.e. the script was modified after this artifact was written.
        try:
            subprocess.check_call(["git", "-C", REPO, "merge-base", "--is-ancestor",
                                   script_commit, art_commit],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return "STALE", (f"{name}: written at {art_commit[:8]}, but its producer "
                             f"({p.get('script')}) was last changed in {script_commit[:8]}, which is "
                             f"NOT an ancestor of it. The artifact predates a change to the code that "
                             f"made it. REGENERATE before using.")
    if p.get("git_dirty_scripts", p.get("git_dirty")):
        return "DIRTY", (f"{name}: written while scripts/ was uncommitted (at {art_commit[:8]}) -- "
                         f"not reproducible from that commit alone.")
    return "OK", f"{name}: ok ({art_commit[:8]})"


def check_fresh(path, d=None, strict=False):
    """Warn (or raise) if an artifact predates a change to the script that produced it."""
    d = d if d is not None else json.load(open(path))
    status, msg = staleness(path, d)
    if status == "OK":
        return d
    print(f"[provenance:{status}] {msg}", file=sys.stderr)
    if strict and status in ("STALE", "STALE_INPUT", "UNSTAMPED"):
        raise SystemExit(f"[provenance] refusing to proceed on a {status} artifact: {path}")
    return d


def load_checked(path, strict=False):
    return check_fresh(path, strict=strict)


if __name__ == "__main__":
    # Audit every JSON artifact the documents are generated from.
    targets = [
        "outputs/miracl2/metrics.json", "outputs/miracl2/nfr_decomposition.json",
        "outputs/miracl2/pool_sensitivity.json", "outputs/miracl2/fusion_weight.json",
        "outputs/miracl2/risk_signal_v2.json", "outputs/c3_quant_cot_v2.json",
        "outputs/holm_correction.json", "outputs/retrieval_eval/per_query_hits.json",
        "outputs/retrieval_eval/c1d_per_query_hits.json",
        # Added 2026-07-14. These three back C1 and C4 and were UNSTAMPED -- they were not in this
        # list, so the audit reported "9/9 OK" while three artifacts it never looked at sat behind two
        # documents. An audit that does not enumerate its own targets is a green light with no wiring.
        "outputs/c1_nested_cv.json", "outputs/power_analysis.json",
        "outputs/answer_budget_fertility.json",
        # Added 2026-07-14. The four-cell stop-sequence experiment: this is the artifact behind the
        # empty-string finding and behind BOTH budget nulls, so it must be hash-pinned to the traces it
        # was computed from -- if a cell is ever re-run, this must go STALE_INPUT rather than sit there
        # looking authoritative.
        "outputs/e2e_stop_bug.json",
        # Added 2026-07-17. The C2 §5.2 cluster-robust bootstrap: recomputes the four 2x2 effects with
        # the correct resampling unit (gold documents, not i.i.d. queries). Hash-pinned to the two
        # *_2x2.json it re-reads and to dev_questions.csv (the qid->gold_doc join) -- a rerun of the 2x2
        # must force STALE_INPUT here.
        "outputs/colsmol2/c2_cluster_bootstrap.json",
    ]
    head = _git("rev-parse", "HEAD")
    print(f"HEAD = {head[:8]}  (scripts/ dirty={bool(_git('status','--porcelain','--','scripts'))})\n")
    bad, untracked = 0, []
    for t in targets:
        f = os.path.join(REPO, t)
        if not os.path.exists(f):
            print(f"  MISSING    {t}")
            bad += 1
            continue
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  UNREADABLE {t}: {e}")
            bad += 1
            continue
        status, msg = staleness(f, d)
        p = read_provenance(f, d) or {}
        n_in = len(p.get("inputs") or {})
        if n_in == 0:
            untracked.append(t)
        print(f"  {status:11s} {msg}" + (f"  [{n_in} input(s) tracked]" if n_in else ""))
        if status not in ("OK", "REPLAYED"):
            bad += 1

    print(f"\n  {len(targets) - bad}/{len(targets)} OK")
    if untracked:
        # Not a failure: a leaf artifact (one computed from a corpus or a model, not from another
        # artifact) legitimately has no upstream to track. But say which, so "no inputs declared" is
        # a visible choice rather than an invisible gap -- that gap is what hid the stale
        # power_analysis.json for a day.
        print(f"  NOTE: {len(untracked)} artifact(s) declare no inputs (leaf, or not yet wired):")
        for t in untracked:
            print(f"        - {t}")
    sys.exit(1 if bad else 0)
