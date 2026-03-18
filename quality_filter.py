"""Filter and validate QA pairs for quality, then write final output."""

import json
from pathlib import Path

from loguru import logger

from config import OUTPUT_FILE, OUTPUT_DIR, QUESTION_TYPE_TARGETS, TARGET_QA_PAIRS


# --- Quality Checks ---

MIN_QUESTION_LENGTH: int = 20
MIN_ANSWER_LENGTH: int = 50
MAX_ANSWER_LENGTH: int = 2000
MIN_PASSAGE_LENGTH: int = 20


def _check_pair(pair: dict) -> list[str]:
    """Run quality checks on a single QA pair. Returns list of issues found."""
    issues: list[str] = []

    # Question length
    if len(pair.get("question", "")) < MIN_QUESTION_LENGTH:
        issues.append(f"question too short ({len(pair.get('question', ''))} chars)")

    # Answer length
    answer_len = len(pair.get("answer", ""))
    if answer_len < MIN_ANSWER_LENGTH:
        issues.append(f"answer too short ({answer_len} chars)")
    if answer_len > MAX_ANSWER_LENGTH:
        issues.append(f"answer too long ({answer_len} chars)")

    # Source files present
    if not pair.get("source_files") or len(pair["source_files"]) == 0:
        issues.append("missing source_files")

    # Supporting passages present and non-trivial
    passages = pair.get("supporting_passages", [])
    if not passages or len(passages) == 0:
        issues.append("missing supporting_passages")
    else:
        for i, passage in enumerate(passages):
            if len(passage) < MIN_PASSAGE_LENGTH:
                issues.append(
                    f"supporting_passage[{i}] too short ({len(passage)} chars)"
                )

    # Question ends with question mark
    question = pair.get("question", "").strip()
    if question and not question.endswith("?"):
        issues.append("question does not end with '?'")

    # Multi-hop and comparative should ideally reference multiple sources
    qtype = pair.get("question_type", "")
    if qtype in ("multi_hop", "comparative") and len(pair.get("source_files", [])) < 2:
        issues.append(
            f"{qtype} question references only {len(pair.get('source_files', []))} source file(s)"
        )

    return issues


class QualityFilter:
    """Validates QA pairs, logs issues, and writes the final dataset."""

    def __init__(self) -> None:
        """Initialize counters."""
        self.passed: list[dict] = []
        self.failed: list[dict] = []

    def filter(self, pairs: list[dict]) -> list[dict]:
        """Run quality checks on all pairs.

        Returns:
            List of pairs that passed all checks.
        """
        self.passed = []
        self.failed = []

        for i, pair in enumerate(pairs):
            issues = _check_pair(pair)
            if issues:
                logger.warning(
                    f"Pair {i + 1} FAILED: {pair['question'][:60]}... | Issues: {issues}"
                )
                pair["_issues"] = issues
                self.failed.append(pair)
            else:
                self.passed.append(pair)

        logger.info(
            f"Quality filter: {len(self.passed)} passed, {len(self.failed)} failed "
            f"out of {len(pairs)} total"
        )
        return self.passed

    def log_distribution(self, pairs: list[dict]) -> None:
        """Log the question type distribution and compare to targets."""
        type_counts: dict[str, int] = {}
        for pair in pairs:
            qt = pair["question_type"]
            type_counts[qt] = type_counts.get(qt, 0) + 1

        total = len(pairs)
        logger.info(f"Final dataset: {total} QA pairs")
        logger.info("Question type distribution:")

        for qtype, target_pct in QUESTION_TYPE_TARGETS.items():
            actual_count = type_counts.get(qtype, 0)
            actual_pct = actual_count / total if total > 0 else 0
            target_count = int(target_pct * TARGET_QA_PAIRS)
            status = "OK" if actual_count >= target_count * 0.5 else "LOW"
            logger.info(
                f"  {qtype:12s}: {actual_count:3d} ({actual_pct:.0%}) "
                f"| target: ~{target_count} ({target_pct:.0%}) [{status}]"
            )

    def save(self, pairs: list[dict], output_path: Path = OUTPUT_FILE) -> Path:
        """Write the final dataset to JSON.

        Returns:
            Path to the written file.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        dataset = {
            "metadata": {
                "total_pairs": len(pairs),
                "generator_model": "gpt-5.4-2026-03-05",
                "reasoning_effort": "high",
                "question_types": list(QUESTION_TYPE_TARGETS.keys()),
            },
            "qa_pairs": pairs,
        }

        output_path.write_text(
            json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Dataset saved to {output_path.resolve()}")
        return output_path
