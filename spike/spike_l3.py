"""Phase 0c — the cross-document spike.

The one question this answers: does windowed concat-scan catch split
payloads that per-chunk scanning misses -- including when the two halves
land far apart in a retrieved set?

Four methodology fixes are built in (see plan Issues A-D). Without them
the numbers look decisive and aren't:

  A. Keep only pairs where NEITHER half is detectable alone -- a naive
     sentence split often leaves one half a complete injection, which
     inflates every number with pairs that were never "split" attacks.
  B. Check whether split halves are even distinguishable from clean text
     BEFORE building any tier -- if there's no signal, sorting by score
     (Tier 2) is sorting noise. Runs first because it's the cheapest
     thing that can kill the most downstream work.
  C. Filler chunks are real ~200-token document passages, not short
     prompts -- short filler would fit in one window and never test the
     512-token wall this whole cascade exists to work around.
  D. Compare methods at equal false-positive rate, not a fixed threshold
     -- more windows means more chances to fire, which inflates recall
     for free unless every method is calibrated to the same FPR first.
"""
import functools
import random
import re
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 -- force every print
                                                # to hit the file immediately;
                                                # buffered output is exactly what
                                                # made an earlier run look "stuck"
                                                # for 26+ minutes when it wasn't

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from detect import Detector  # noqa: E402

random.seed(0)
np.random.seed(0)

SEP = "\n\n"
WINDOW_BUDGET = 480  # a little under 512 to leave room for special tokens
OVERLAP = 2
DETECTABLE_ALONE = 0.5   # a half scoring above this was never a "hidden" half
SET_SIZE = 20

QUICK = "--quick" in sys.argv
if QUICK:
    # A directional smoke test, same standard as the 2-sentence model checks:
    # near, mid, far distance only; fewer pairs.
    #
    # TARGET_FPR=0.05 (not 0.01) is the real fix from the first --quick run:
    # a "top 1%" cutoff needs hundreds of clean samples to estimate reliably --
    # with only 20 you're just taking whichever single sample scored highest,
    # which is noise, not a real threshold. "Top 5%" is stable at this sample
    # size. Phase 0 is a go/no-go gate, not the final number -- Phase 9's
    # ablation is where the rigorous, fixed-seed evaluation belongs.
    DISTANCES = [1, 10, 19]
    N_CLEAN_SETS = 40
    TARGET_FPR = 0.05
    MAX_PAIRS = 20
else:
    DISTANCES = [1, 3, 5, 10, 19]
    N_CLEAN_SETS = 60
    TARGET_FPR = 0.01
    MAX_PAIRS = None
MAX_BATCH = 24           # cap on any single score_batch call -- every real call
                         # site in the actual library scores ~20 chunks at once;
                         # a spike script handing the detector 200+ texts in one
                         # call is testing something that will never happen in
                         # production, and it's what OOM'd the first run of this
                         # script (268 texts, one padded to 512 tokens, in one batch)


def score_in_batches(det: Detector, texts: list[str], batch_size: int = MAX_BATCH) -> list[float]:
    """score_batch, but capped -- never hand the detector more than batch_size
    texts at once, regardless of how large the input list is."""
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(det.score_batch(texts[i:i + batch_size]))
    return out


# ---------------------------------------------------------------------------
# Fix A: build genuine split pairs from real injections
# ---------------------------------------------------------------------------

def split_at_sentence(text: str):
    """Cut a real injection at a sentence boundary. We choose the cut
    point, never the wording -- that's what keeps this from being
    circular the way the August plan was."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) < 2:
        return None
    mid = len(parts) // 2
    a, b = " ".join(parts[:mid]), " ".join(parts[mid:])
    if not a or not b:
        return None
    return a, b


def build_pairs(det: Detector) -> list[tuple[str, str]]:
    print("Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    injections = [r["text"] for r in ds if r["label"] == 1]
    print(f"  {len(injections)} injection examples")

    candidates = [p for p in (split_at_sentence(t) for t in injections) if p]
    print(f"  {len(candidates)} split into two non-empty halves")

    # Fix A: keep only pairs where NEITHER half is detectable alone
    all_halves = [h for pair in candidates for h in pair]
    scores = score_in_batches(det, all_halves)
    pairs = []
    for i, (a, b) in enumerate(candidates):
        sa, sb = scores[2 * i], scores[2 * i + 1]
        if sa < DETECTABLE_ALONE and sb < DETECTABLE_ALONE:
            pairs.append((a, b))

    print(f"  {len(pairs)} pairs survive the 'neither half detectable alone' filter")
    if len(pairs) < 20:
        print("  WARNING: fewer than 20 usable pairs -- statistics below are thin")
    return pairs


# ---------------------------------------------------------------------------
# Fix C: realistic, document-length filler
# ---------------------------------------------------------------------------

def build_filler(det: Detector, n: int = 400) -> list[str]:
    print("Loading squad passages for realistic filler...")
    sq = load_dataset("rajpurkar/squad", split="train")
    passages = list({r["context"] for r in sq.select(range(min(5000, len(sq))))})
    random.shuffle(passages)
    passages = passages[:n]

    lens = [det.n_tokens(p) for p in passages]
    avg = sum(lens) / len(lens)
    print(f"  {len(passages)} passages, avg {avg:.0f} tokens each")
    return passages


def assert_realistic(chunks: list[str], det: Detector):
    total = sum(det.n_tokens(c) for c in chunks)
    assert 2000 <= total <= 6000, (
        f"unrealistic set: {total} tokens -- filler is too short/long to "
        f"actually test the 512-token wall"
    )
    return total


# ---------------------------------------------------------------------------
# Fix B: is there any signal for Tier 2 to sort on?
# ---------------------------------------------------------------------------

def auc_check(det: Detector, pairs, filler) -> float:
    halves = [h for pair in pairs for h in pair]
    clean = random.sample(filler, min(200, len(filler)))

    halves_scores = score_in_batches(det, halves)
    clean_scores = score_in_batches(det, clean)

    y = [0] * len(clean_scores) + [1] * len(halves_scores)
    scores = clean_scores + halves_scores
    return roc_auc_score(y, scores)


# ---------------------------------------------------------------------------
# The cascade primitives (mirrors what setscan.py will implement for real)
# ---------------------------------------------------------------------------

def pack_windows(order: list[int], lens: list[int]) -> list[list[int]]:
    out, i = [], 0
    n = len(order)
    while i < n:
        j, total = i, 0
        while j < n and total + lens[order[j]] <= WINDOW_BUDGET:
            total += lens[order[j]]
            j += 1
        j = max(j, i + 1)
        out.append(order[i:j])
        if j >= n:
            break
        i = max(i + 1, j - OVERLAP)
    return out


def suspicion_order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def method_scores(chunks: list[str], det: Detector, lens: list[int]):
    """Returns (per_chunk_max, tier1_max, tier2_max) for one 20-chunk set."""
    per_chunk = det.score_batch(chunks)
    per_chunk_max = max(per_chunk)

    wins1 = pack_windows(list(range(len(chunks))), lens)
    texts1 = [SEP.join(chunks[i] for i in w) for w in wins1]
    tier1_max = max(det.score_batch(texts1))

    order = suspicion_order(per_chunk)  # trust held flat -- isolates the variable
    wins2 = pack_windows(order, lens)
    texts2 = [SEP.join(chunks[i] for i in w) for w in wins2]
    tier2_max = max(det.score_batch(texts2))

    return per_chunk_max, tier1_max, tier2_max


# ---------------------------------------------------------------------------
# Fix D: calibrate each method to equal FPR before comparing recall
# ---------------------------------------------------------------------------

def calibrate_thresholds(det: Detector, filler: list[str]):
    print(f"Calibrating thresholds on {N_CLEAN_SETS} clean sets (target FPR={TARGET_FPR})...")
    pc_scores, t1_scores, t2_scores = [], [], []
    t_start = time.time()
    for i in range(N_CLEAN_SETS):
        chunks = random.sample(filler, SET_SIZE)
        lens = [det.n_tokens(c) for c in chunks]
        pc, t1, t2 = method_scores(chunks, det, lens)
        pc_scores.append(pc)
        t1_scores.append(t1)
        t2_scores.append(t2)

        if (i + 1) % 10 == 0 or i + 1 == N_CLEAN_SETS:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (N_CLEAN_SETS - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{N_CLEAN_SETS}] {elapsed:.0f}s elapsed, "
                  f"~{eta:.0f}s remaining in this step")

    thresholds = {
        "per_chunk": float(np.quantile(pc_scores, 1 - TARGET_FPR)),
        "tier1": float(np.quantile(t1_scores, 1 - TARGET_FPR)),
        "tier2": float(np.quantile(t2_scores, 1 - TARGET_FPR)),
    }
    print(f"  thresholds: {thresholds}")
    return thresholds


# ---------------------------------------------------------------------------
# The main experiment: recall by distance, at the calibrated thresholds
# ---------------------------------------------------------------------------

def plant(a: str, b: str, filler: list[str], d: int) -> list[str]:
    chunks = random.sample(filler, SET_SIZE)
    i = random.randrange(0, SET_SIZE - d)
    chunks[i], chunks[i + d] = a, b
    return chunks


def run_distance_experiment(det: Detector, pairs, filler, thresholds):
    results = {}
    total_iters = len(DISTANCES) * len(pairs)
    done = 0
    t_start = time.time()
    for d in DISTANCES:
        hits = {"per_chunk": 0, "tier1": 0, "tier2": 0}
        n = len(pairs)
        for k, (a, b) in enumerate(pairs):
            chunks = plant(a, b, filler, d)
            lens = [det.n_tokens(c) for c in chunks]
            pc, t1, t2 = method_scores(chunks, det, lens)
            hits["per_chunk"] += pc >= thresholds["per_chunk"]
            hits["tier1"] += t1 >= thresholds["tier1"]
            hits["tier2"] += t2 >= thresholds["tier2"]
            done += 1

            if done % 15 == 0 or k + 1 == n:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (total_iters - done) / rate if rate > 0 else 0
                print(f"  d={d:2d} [{k+1}/{n} pairs]  overall [{done}/{total_iters}]  "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining total")

        results[str(d)] = {k: v / n for k, v in hits.items()}
        print(f"  >>> d={d:2d} DONE  per_chunk={results[str(d)]['per_chunk']:.2f}  "
              f"tier1={results[str(d)]['tier1']:.2f}  tier2={results[str(d)]['tier2']:.2f}")
    return results


# ---------------------------------------------------------------------------

def main():
    import json
    import time

    t0 = time.time()
    print("Loading detector...")
    det = Detector()

    pairs = build_pairs(det)
    if QUICK and MAX_PAIRS and len(pairs) > MAX_PAIRS:
        pairs = random.sample(pairs, MAX_PAIRS)
        print(f"  --quick: using a random subsample of {MAX_PAIRS} pairs")
    filler = build_filler(det)

    print("\nAsserting filler sets are realistic (2000-6000 tokens per 20-chunk set)...")
    sample_set = random.sample(filler, SET_SIZE)
    total = assert_realistic(sample_set, det)
    print(f"  OK: sample 20-chunk set = {total} tokens")

    print("\n--- Fix B: is there any signal for Tier 2 to sort on? ---")
    auc = auc_check(det, pairs, filler)
    print(f"AUC (split halves vs clean filler): {auc:.3f}")
    if auc > 0.70:
        signal_verdict = "viable -- halves score measurably above clean text"
    elif auc > 0.55:
        signal_verdict = "weak -- Tier 2 helps a little, Tier 3 carries more load"
    else:
        signal_verdict = "no signal -- Tier 2 as designed will not help"
    print(f"  -> {signal_verdict}")

    print()
    thresholds = calibrate_thresholds(det, filler)

    print(f"\n--- Distance experiment ({len(pairs)} pairs x {len(DISTANCES)} distances) ---")
    distance_results = run_distance_experiment(det, pairs, filler, thresholds)

    out = {
        "phase": "0c",
        "date": "2026-09-09",
        "n_pairs": len(pairs),
        "n_filler_passages": len(filler),
        "sample_set_tokens": total,
        "auc_halves_vs_clean": auc,
        "auc_verdict": signal_verdict,
        "thresholds_at_fpr": {"target_fpr": TARGET_FPR, **thresholds},
        "recall_by_distance": distance_results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = Path(__file__).parent.parent / "results" / "spike_l3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
