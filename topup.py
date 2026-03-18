"""Top-up script to generate additional comparative QA pairs and merge into dataset."""

import json
import random

from loguru import logger

from config import OUTPUT_FILE
from doc_loader import DocLoader
from qa_generator import QAGenerator
from quality_filter import QualityFilter


def main() -> None:
    """Generate targeted comparative pairs and merge with existing dataset."""
    random.seed(43)  # different seed from main run

    # Load existing dataset
    logger.info("Loading existing dataset...")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    existing_pairs = dataset["qa_pairs"]
    logger.info(f"Existing dataset has {len(existing_pairs)} pairs")

    # Load documents
    loader = DocLoader()
    documents = loader.load_all()

    # Generate targeted comparative pairs
    logger.info("Generating targeted comparative pairs...")
    generator = QAGenerator()
    new_pairs = generator.generate_targeted(
        documents=documents,
        num_pairs=12,  # request a few extra to account for filtering
        target_types=["comparative"],
        min_sources=2,
    )
    logger.info(f"Generated {len(new_pairs)} targeted comparative pairs")

    # Quality filter the new pairs
    qf = QualityFilter()
    filtered_new = qf.filter(new_pairs)
    logger.info(f"After quality filter: {len(filtered_new)} pairs")

    # Merge
    merged = existing_pairs + filtered_new
    logger.info(f"Merged dataset: {len(merged)} total pairs")

    # Update and save
    dataset["qa_pairs"] = merged
    dataset["metadata"]["total_pairs"] = len(merged)
    dataset["metadata"]["topup_applied"] = True
    dataset["metadata"]["topup_count"] = len(filtered_new)

    OUTPUT_FILE.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Updated dataset saved to {OUTPUT_FILE}")

    # Log final distribution
    qf.log_distribution(merged)


if __name__ == "__main__":
    main()
