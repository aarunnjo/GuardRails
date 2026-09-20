"""Phase 8 -- evaluation + ablation harness.

The empirical question this answers: does each v1 component actually earn
its place? Two components are checked here (both real, reproducible claims,
not restated intuition):

  1. L2 alone (the ONNX detector) -- recall at a fixed FPR on real
     injections. This is the baseline everything else sits on top of.
  2. L0 normalize's marginal contribution -- recall at the SAME fixed FPR
     on the same injections after a homoglyph/zero-width obfuscation pass,
     with and without normalize.py running first. This is normalize's
     actual job (evade-by-invisible-character), so it's the right thing to
     ablate it against -- plain unobfuscated injections don't exercise it
     at all and would make normalize look useless when it isn't.

L1 (trust) and L3 (cut) are not re-litigated here: L1's effect is
deterministic algebra over a tier + incident count (see tests/test_trust.py
and trust.py directly), not something an aggregate recall/FPR table can
represent without conflating multiple thresholds into one number. L3's
result already lives in results/spike_l3.json (Phase 0c) -- see the
"Cross-document verdict" in the plan for why it isn't re-run here.

Usage:
    python eval/harness.py            # quick mode -- ~2 min, network + CPU
    python eval/harness.py --ablate   # same, alias (ablation IS the harness)
    python eval/harness.py --full     # larger samples, ~10-15 min
"""
import functools
import json
import random
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "fireguard"))
from fireguard.backends.onnx_detector import OnnxDetector
from fireguard.normalize import normalize_text

random.seed(0)
np.random.seed(0)

FULL = "--full" in sys.argv
TARGET_FPR = 0.05
MAX_BATCH = 24

if FULL:
    N_ATTACK = 200
    N_CLEAN = 400
else:
    N_ATTACK = 60
    N_CLEAN = 150

# A small, deliberate obfuscation: fold Latin letters to their Cyrillic
# lookalikes (the exact confusables normalize.py folds back) and interleave
# zero-width spaces. This is not a strong evasion by real adversarial
# standards -- it's the simplest possible thing normalize.py is designed to
# defeat, which is exactly what makes it a fair ablation target.
OBFUSCATE_MAP = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х"}


def obfuscate(text: str) -> str:
    out = []
    for ch in text:
        out.append(OBFUSCATE_MAP.get(ch, ch))
        out.append("\u200b")  # zero-width space after every character
    return "".join(out)


def score_in_batches(det: OnnxDetector, texts: list[str], batch_size: int = MAX_BATCH) -> list[float]:
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(det.score_batch(texts[i : i + batch_size]))
    return out


def load_data():
    print("Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections", split="train")
    attacks = [r["text"] for r in ds if r["label"] == 1]
    benign_prompts = [r["text"] for r in ds if r["label"] == 0]
    print(f"  {len(attacks)} injections, {len(benign_prompts)} benign prompts")

    print("Loading squad passages for document-length clean filler...")
    sq = load_dataset("rajpurkar/squad", split="train")
    passages = list({r["context"] for r in sq.select(range(min(5000, len(sq))))})
    print(f"  {len(passages)} unique passages")

    random.shuffle(attacks)
    random.shuffle(benign_prompts)
    random.shuffle(passages)

    attacks = attacks[:N_ATTACK]
    clean = (benign_prompts[: N_CLEAN // 2]) + (passages[: N_CLEAN - N_CLEAN // 2])
    random.shuffle(clean)
    return attacks, clean


def calibrate_threshold(scores: list[float], target_fpr: float) -> float:
    return float(np.quantile(scores, 1 - target_fpr))


def recall_at_threshold(scores: list[float], threshold: float) -> float:
    return sum(s >= threshold for s in scores) / len(scores)


def run_variant(name: str, det: OnnxDetector, attacks: list[str], clean: list[str], transform) -> dict:
    t0 = time.time()
    clean_t = [transform(c) for c in clean]
    attacks_t = [transform(a) for a in attacks]

    clean_scores = score_in_batches(det, clean_t)
    attack_scores = score_in_batches(det, attacks_t)

    threshold = calibrate_threshold(clean_scores, TARGET_FPR)
    recall = recall_at_threshold(attack_scores, threshold)
    achieved_fpr = recall_at_threshold(clean_scores, threshold)

    result = {
        "threshold": round(threshold, 4),
        "recall": round(recall, 4),
        "achieved_fpr": round(achieved_fpr, 4),
        "n_attack": len(attacks_t),
        "n_clean": len(clean_t),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"  {name:32s} recall={result['recall']:.3f}  fpr={result['achieved_fpr']:.3f}  "
          f"threshold={result['threshold']:.3f}  ({result['elapsed_s']:.1f}s)")
    return result


def main():
    t0 = time.time()
    print(f"Mode: {'full' if FULL else 'quick'} (n_attack={N_ATTACK}, n_clean={N_CLEAN}, "
          f"target_fpr={TARGET_FPR})\n")

    print("Loading detector...")
    det = OnnxDetector()

    attacks, clean = load_data()

    print(f"\n--- Ablation 1: L2 alone, plain injections vs. clean (n={len(attacks)}/{len(clean)}) ---")
    plain = run_variant("detector_only (plain)", det, attacks, clean, lambda t: t)

    print("\n--- Ablation 2: does L0 normalize matter under obfuscation? ---")
    obf_attacks = [obfuscate(a) for a in attacks]
    obf_clean = [obfuscate(c) for c in clean]

    no_normalize = run_variant("detector_only (obfuscated)", det, obf_attacks, obf_clean, lambda t: t)
    with_normalize = run_variant("detector + normalize (obfuscated)", det, obf_attacks, obf_clean, normalize_text)

    out = {
        "phase": 8,
        "mode": "full" if FULL else "quick",
        "target_fpr": TARGET_FPR,
        "detector_only_plain": plain,
        "detector_only_obfuscated": no_normalize,
        "detector_plus_normalize_obfuscated": with_normalize,
        "normalize_recall_gain": round(with_normalize["recall"] - no_normalize["recall"], 4),
        "elapsed_s": round(time.time() - t0, 1),
    }

    print("\n=== Summary ===")
    print(f"Plain injections, no obfuscation:        recall={plain['recall']:.3f}")
    print(f"Obfuscated, WITHOUT normalize:            recall={no_normalize['recall']:.3f}")
    print(f"Obfuscated, WITH normalize:                recall={with_normalize['recall']:.3f}")
    print(f"normalize's recall gain under obfuscation: {out['normalize_recall_gain']:+.3f}")
    if out["normalize_recall_gain"] <= 0:
        print("  -> normalize did NOT help on this run. Per the plan's own rule "
              "(\"if a component doesn't help, cut it and record that\"), that's a "
              "finding to act on, not to explain away.")

    out_path = Path(__file__).parent.parent / "results" / "eval_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path} ({out['elapsed_s']}s total)")


if __name__ == "__main__":
    main()
