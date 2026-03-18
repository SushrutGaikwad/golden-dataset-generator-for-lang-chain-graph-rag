"""Load and parse markdown documentation files from disk."""

from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from config import DOCS_ROOT, SUPPORTED_EXTENSIONS, MIN_DOC_LENGTH_CHARS


@dataclass
class Document:
    """A single loaded documentation file."""

    relative_path: str  # e.g. "langchain/guides/rag.mdx"
    content: str
    char_count: int
    library: str  # "langchain" or "langgraph"


class DocLoader:
    """Recursively loads markdown files from the documentation directory."""

    def __init__(self, docs_root: Path = DOCS_ROOT) -> None:
        """Initialize with path to documentation root."""
        self.docs_root = docs_root
        self._validate_root()

    def _validate_root(self) -> None:
        """Check that the docs root exists and contains expected subdirectories."""
        if not self.docs_root.exists():
            raise FileNotFoundError(f"Docs root not found: {self.docs_root.resolve()}")

        subdirs = [d.name for d in self.docs_root.iterdir() if d.is_dir()]
        logger.info(f"Found subdirectories in docs root: {subdirs}")

        expected = {"langchain", "langgraph"}
        found = expected.intersection(subdirs)
        if not found:
            logger.warning(
                f"Expected subdirectories {expected}, but found {subdirs}. "
                "Proceeding anyway."
            )

    def _detect_library(self, file_path: Path) -> str:
        """Determine which library a file belongs to based on its path."""
        relative = file_path.relative_to(self.docs_root)
        top_level = relative.parts[0] if relative.parts else "unknown"
        return top_level

    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if a file should be loaded based on extension."""
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def load_all(self) -> list[Document]:
        """Load all valid documentation files recursively.

        Returns:
            List of Document objects, sorted by relative path.
        """
        documents: list[Document] = []
        skipped_small = 0
        skipped_extension = 0

        for file_path in sorted(self.docs_root.rglob("*")):
            if not file_path.is_file():
                continue

            if not self._is_valid_file(file_path):
                skipped_extension += 1
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

            if len(content) < MIN_DOC_LENGTH_CHARS:
                skipped_small += 1
                continue

            relative_path = str(file_path.relative_to(self.docs_root)).replace(
                "\\", "/"
            )
            library = self._detect_library(file_path)

            documents.append(
                Document(
                    relative_path=relative_path,
                    content=content,
                    char_count=len(content),
                    library=library,
                )
            )

        logger.info(
            f"Loaded {len(documents)} documents "
            f"(skipped {skipped_small} too-small, {skipped_extension} non-markdown)"
        )

        # Log per-library breakdown
        libs = {}
        for doc in documents:
            libs[doc.library] = libs.get(doc.library, 0) + 1
        for lib, count in sorted(libs.items()):
            logger.info(f"  {lib}: {count} files")

        return documents
