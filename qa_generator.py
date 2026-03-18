"""Generate golden QA pairs from documentation using OpenAI GPT-5.4."""

import json
import random
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from loguru import logger

from config import (
    MODEL_NAME,
    REASONING_EFFORT,
    TARGET_QA_PAIRS,
    QUESTION_TYPE_TARGETS,
)
from doc_loader import Document

load_dotenv()

# --- Pydantic Schemas ---

VALID_QUESTION_TYPES = [
    "factual",
    "conceptual",
    "procedural",
    "comparative",
    "multi_hop",
    "edge_case",
]


class QAPair(BaseModel):
    """Schema for a single QA pair."""

    question: str
    answer: str
    source_files: list[str]
    supporting_passages: list[str]
    question_type: str


class QABatchResponse(BaseModel):
    """Schema for a batch of QA pairs returned by the model."""

    qa_pairs: list[QAPair]


# --- Prompt Construction ---

SYSTEM_PROMPT = """You are an expert technical writer creating a golden evaluation dataset \
for a RAG (Retrieval-Augmented Generation) system built on LangChain and LangGraph documentation.

Your task: generate high-quality question-answer pairs from the provided documentation excerpts.

RULES:
1. Every answer MUST be fully supported by the provided documentation. Do not hallucinate.
2. supporting_passages must be EXACT quotes copied from the source documents.
3. source_files must list the relative file path(s) provided in the document headers.
4. question_type must be one of: factual, conceptual, procedural, comparative, multi_hop, edge_case.
5. For multi_hop and comparative questions, use information from MULTIPLE documents when possible.
6. Questions should be specific and unambiguous.
7. Answers should be detailed (2-5 sentences) and self-contained.

QUESTION TYPE DEFINITIONS:
- factual: Direct fact lookup from a single passage (e.g., "What is the default value of X?")
- conceptual: Requires understanding/explaining a concept (e.g., "Explain how state management works in LangGraph")
- procedural: Asks how to do something step by step (e.g., "How do you add memory to a LangGraph agent?")
- comparative: Requires comparing two or more things (e.g., "What is the difference between X and Y?")
- multi_hop: Answer requires chaining information from multiple sections or files
- edge_case: Targets boundary conditions, caveats, or limitations mentioned in the docs

Respond with ONLY valid JSON matching this schema:
{
  "qa_pairs": [
    {
      "question": "...",
      "answer": "...",
      "source_files": ["path/to/file.mdx"],
      "supporting_passages": ["exact quote from doc..."],
      "question_type": "factual"
    }
  ]
}"""


def _build_user_prompt(
    documents: list[Document],
    num_pairs: int,
    type_guidance: str,
) -> str:
    """Build the user prompt containing document content and generation instructions."""
    doc_sections = []
    for doc in documents:
        # Truncate very large docs to avoid token limits
        content = doc.content[:12000] if len(doc.content) > 12000 else doc.content
        doc_sections.append(
            f"--- DOCUMENT: {doc.relative_path} (library: {doc.library}) ---\n{content}\n"
        )

    docs_block = "\n".join(doc_sections)

    return f"""Here are the documentation files:

{docs_block}

Generate exactly {num_pairs} question-answer pairs from these documents.

{type_guidance}

Remember:
- supporting_passages must be EXACT quotes from the documents above
- For comparative and multi_hop types, try to reference multiple documents
- Vary the difficulty and specificity of questions
- Respond with ONLY valid JSON"""


def _compute_type_guidance(current_counts: dict[str, int], remaining: int) -> str:
    """Compute which question types are underrepresented and guide the model."""
    total_so_far = sum(current_counts.values())
    if total_so_far == 0:
        return (
            "Distribute question types roughly evenly across: "
            "factual, conceptual, procedural, comparative, multi_hop, edge_case."
        )

    target_total = total_so_far + remaining
    guidance_parts = []
    for qtype, target_pct in QUESTION_TYPE_TARGETS.items():
        current = current_counts.get(qtype, 0)
        target_count = int(target_pct * target_total)
        deficit = max(0, target_count - current)
        if deficit > 0:
            guidance_parts.append(f"{qtype}: need ~{deficit} more")

    if guidance_parts:
        return (
            "To balance the dataset, prioritize these underrepresented types:\n"
            + "\n".join(f"  - {part}" for part in guidance_parts)
        )
    return "Distribute question types as you see fit to maintain balance."


# --- Main Generator Class ---


class QAGenerator:
    """Generates QA pairs from documents using OpenAI GPT-5.4."""

    def __init__(self) -> None:
        """Initialize the OpenAI client."""
        self.client = OpenAI()
        self.all_pairs: list[dict] = []
        self.type_counts: dict[str, int] = {qt: 0 for qt in VALID_QUESTION_TYPES}

    def _create_batches(
        self, documents: list[Document], batch_size: int = 8
    ) -> list[list[Document]]:
        """Split documents into batches for processing.

        Each batch mixes libraries to enable cross-document questions.
        """
        shuffled = documents.copy()
        random.shuffle(shuffled)
        batches = [
            shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)
        ]
        logger.info(
            f"Created {len(batches)} document batches (batch_size={batch_size})"
        )
        return batches

    def _pairs_per_batch(self, num_batches: int) -> list[int]:
        """Distribute target QA pairs across batches."""
        base = TARGET_QA_PAIRS // num_batches
        remainder = TARGET_QA_PAIRS % num_batches
        counts = [base + (1 if i < remainder else 0) for i in range(num_batches)]
        return counts

    def _call_openai(self, user_prompt: str) -> str | None:
        """Make a single API call to GPT-5.4 with high reasoning effort."""
        try:
            response = self.client.responses.create(
                model=MODEL_NAME,
                reasoning={"effort": REASONING_EFFORT},
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.output_text
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return None

    def _parse_response(self, raw_response: str) -> list[dict]:
        """Parse and validate the model response into QA pairs."""
        # Strip markdown code fences if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            logger.debug(f"Raw response (first 500 chars): {cleaned[:500]}")
            return []

        try:
            batch = QABatchResponse(**parsed)
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            return []

        valid_pairs = []
        for pair in batch.qa_pairs:
            if pair.question_type not in VALID_QUESTION_TYPES:
                logger.warning(
                    f"Invalid question_type '{pair.question_type}', skipping pair"
                )
                continue
            valid_pairs.append(pair.model_dump())

        return valid_pairs

    def generate(self, documents: list[Document]) -> list[dict]:
        """Run the full QA generation pipeline across document batches.

        Returns:
            List of validated QA pair dictionaries.
        """
        batches = self._create_batches(documents)
        pairs_distribution = self._pairs_per_batch(len(batches))

        for i, (batch, num_pairs) in enumerate(zip(batches, pairs_distribution)):
            remaining = TARGET_QA_PAIRS - len(self.all_pairs)
            if remaining <= 0:
                logger.info("Target reached, stopping generation.")
                break

            # Adjust num_pairs if we're close to target
            num_pairs = min(num_pairs, remaining)

            doc_names = [d.relative_path for d in batch]
            logger.info(
                f"Batch {i + 1}/{len(batches)}: {len(batch)} docs, "
                f"requesting {num_pairs} pairs"
            )
            logger.debug(f"  Docs: {doc_names}")

            type_guidance = _compute_type_guidance(self.type_counts, remaining)
            user_prompt = _build_user_prompt(batch, num_pairs, type_guidance)

            raw_response = self._call_openai(user_prompt)
            if raw_response is None:
                logger.warning(f"Batch {i + 1} failed, skipping")
                continue

            pairs = self._parse_response(raw_response)
            logger.info(f"Batch {i + 1} returned {len(pairs)} valid pairs")

            # Update tracking
            for pair in pairs:
                self.type_counts[pair["question_type"]] = (
                    self.type_counts.get(pair["question_type"], 0) + 1
                )
            self.all_pairs.extend(pairs)

            logger.info(
                f"Running total: {len(self.all_pairs)} pairs | "
                f"Distribution: {dict(self.type_counts)}"
            )

        return self.all_pairs
