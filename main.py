"""Main script to generate the golden evaluation dataset."""

import random

from loguru import logger

from config import TARGET_QA_PAIRS
from doc_loader import DocLoader
from qa_generator import QAGenerator
from quality_filter import QualityFilter


def main() -> None:
    """Run the full golden dataset generation pipeline."""
    random.seed(42)

    # Step 1: Load documents
    logger.info("=== Step 1: Loading documents ===")
    loader = DocLoader()
    documents = loader.load_all()

    # Step 2: Generate QA pairs
    logger.info("=== Step 2: Generating QA pairs ===")
    generator = QAGenerator()
    raw_pairs = generator.generate(documents)
    logger.info(f"Raw pairs generated: {len(raw_pairs)}")

    # Step 3: Quality filtering
    logger.info("=== Step 3: Quality filtering ===")
    qf = QualityFilter()
    filtered_pairs = qf.filter(raw_pairs)

    # Step 4: Log distribution and save
    logger.info("=== Step 4: Saving dataset ===")
    qf.log_distribution(filtered_pairs)
    output_path = qf.save(filtered_pairs)

    logger.info(
        f"Done! {len(filtered_pairs)} QA pairs saved to {output_path}. "
        f"(Target was {TARGET_QA_PAIRS})"
    )


if __name__ == "__main__":
    main()
