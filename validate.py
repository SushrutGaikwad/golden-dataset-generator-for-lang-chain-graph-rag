"""Validate the golden dataset for completeness, schema correctness, and quality."""

import json
from collections import Counter

from loguru import logger

from config import OUTPUT_FILE, QUESTION_TYPE_TARGETS

VALID_QUESTION_TYPES = {
    "factual",
    "conceptual",
    "procedural",
    "comparative",
    "multi_hop",
    "edge_case",
}


def validate() -> None:
    """Run all validation checks on the golden dataset."""
    logger.info(f"Loading dataset from {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    pairs = dataset["qa_pairs"]
    metadata = dataset["metadata"]
    total = len(pairs)
    logger.info(f"Metadata: {metadata}")
    logger.info(f"Total QA pairs: {total}")

    errors: list[str] = []
    warnings: list[str] = []

    # --- 1. Schema check ---
    required_fields = {
        "question",
        "answer",
        "source_files",
        "supporting_passages",
        "question_type",
    }
    for i, pair in enumerate(pairs):
        missing = required_fields - set(pair.keys())
        if missing:
            errors.append(f"Pair {i + 1}: missing fields {missing}")

    # --- 2. Question type validity ---
    type_counts = Counter()
    for i, pair in enumerate(pairs):
        qt = pair.get("question_type", "")
        if qt not in VALID_QUESTION_TYPES:
            errors.append(f"Pair {i + 1}: invalid question_type '{qt}'")
        type_counts[qt] += 1

    # --- 3. Distribution check ---
    logger.info("Question type distribution:")
    for qt in sorted(VALID_QUESTION_TYPES):
        count = type_counts.get(qt, 0)
        pct = count / total * 100 if total > 0 else 0
        target_pct = QUESTION_TYPE_TARGETS.get(qt, 0) * 100
        status = "OK" if pct >= target_pct * 0.5 else "LOW"
        logger.info(
            f"  {qt:12s}: {count:3d} ({pct:5.1f}%) | target: {target_pct:.0f}% [{status}]"
        )
        if status == "LOW":
            warnings.append(
                f"{qt} is underrepresented ({pct:.1f}% vs {target_pct:.0f}% target)"
            )

    # --- 4. Source file checks ---
    all_sources: set[str] = set()
    multi_source_count = 0
    for i, pair in enumerate(pairs):
        sources = pair.get("source_files", [])
        if not sources:
            errors.append(f"Pair {i + 1}: empty source_files")
        all_sources.update(sources)
        if len(sources) >= 2:
            multi_source_count += 1

    logger.info(f"Unique source files referenced: {len(all_sources)}")
    logger.info(
        f"Multi-source pairs: {multi_source_count}/{total} ({multi_source_count / total * 100:.0f}%)"
    )

    # --- 5. Content length stats ---
    q_lengths = [len(p["question"]) for p in pairs]
    a_lengths = [len(p["answer"]) for p in pairs]
    p_counts = [len(p["supporting_passages"]) for p in pairs]

    logger.info(
        f"Question length: min={min(q_lengths)}, max={max(q_lengths)}, avg={sum(q_lengths) // len(q_lengths)}"
    )
    logger.info(
        f"Answer length:   min={min(a_lengths)}, max={max(a_lengths)}, avg={sum(a_lengths) // len(a_lengths)}"
    )
    logger.info(
        f"Passages/pair:   min={min(p_counts)}, max={max(p_counts)}, avg={sum(p_counts) / len(p_counts):.1f}"
    )

    # --- 6. Duplicate check ---
    questions = [p["question"] for p in pairs]
    dupes = [q for q, count in Counter(questions).items() if count > 1]
    if dupes:
        errors.append(f"Found {len(dupes)} duplicate questions")
        for d in dupes:
            logger.warning(f"  Duplicate: {d[:80]}...")

    # --- 7. Sample pairs for manual review ---
    logger.info("\n=== SAMPLE PAIRS FOR MANUAL REVIEW ===")
    import random

    random.seed(42)
    samples = random.sample(pairs, min(5, total))
    for i, pair in enumerate(samples, 1):
        logger.info(f"\n--- Sample {i} [{pair['question_type']}] ---")
        logger.info(f"Q: {pair['question']}")
        logger.info(f"A: {pair['answer'][:200]}...")
        logger.info(f"Sources: {pair['source_files']}")
        logger.info(f"Passages: {len(pair['supporting_passages'])} passage(s)")

    # --- Summary ---
    logger.info("\n=== VALIDATION SUMMARY ===")
    if errors:
        for e in errors:
            logger.error(f"ERROR: {e}")
    else:
        logger.success("No errors found")

    if warnings:
        for w in warnings:
            logger.warning(f"WARNING: {w}")
    else:
        logger.success("No warnings")

    logger.info(f"Total: {total} pairs, {len(errors)} errors, {len(warnings)} warnings")


if __name__ == "__main__":
    validate()
