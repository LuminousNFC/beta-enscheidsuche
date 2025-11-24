#!/usr/bin/env python3
"""
Generate embeddings for legal documents with chunking strategy.

EMBEDDING CONFIGURATION (for reproducibility across machines):
- Chunking: chunk_size=2800 chars, chunk_overlap=400 chars (~14%)
- Model: text-embedding-3-small (1536 dimensions)
- Strategy: embed all chunks, then average embeddings per document
- Input: complete HTML content from .html files, no truncation
- Skip/resume: --skip-existing flag to avoid re-embedding
- API: respects 8192 token limit per chunk (2800 chars ≈ 700 tokens)
- Parallelism: 4-6 workers max per machine (CPU/RAM/API stability)
- Encoding: JSON UTF-8, ensure_ascii=True, SHA256 for hash/dedup
- Output: *_embedded.json with 1536-dim embedding vector
- Logs: embedding_<name>.log for audit
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    sys.exit(1)

# =============================================================================
# CONFIGURATION CONSTANTS - DO NOT MODIFY FOR CROSS-MACHINE CONSISTENCY
# =============================================================================
CHUNK_SIZE = 2800          # characters per chunk
CHUNK_OVERLAP = 400        # overlap between chunks (~14%)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
MAX_WORKERS = 4            # conservative default, can be 4-6
MAX_RETRIES = 3
RETRY_DELAY = 2            # seconds

# Sentence boundary pattern for smart chunking
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])')


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, removing tags."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self.current_skip = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.skip_tags:
            self.current_skip = True

    def handle_endtag(self, tag):
        if tag.lower() in self.skip_tags:
            self.current_skip = False

    def handle_data(self, data):
        if not self.current_skip:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self):
        return ' '.join(self.text_parts)


def extract_text_from_html(html_content: str) -> str:
    """Extract plain text from HTML content."""
    parser = HTMLTextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text()
    except Exception:
        # Fallback: simple regex removal of tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def setup_logging(log_name: str, log_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_path = log_dir / f"embedding_{log_name}.log"

    logger = logging.getLogger("embeddings")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_path}")
    return logger


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks with overlap, preferring sentence boundaries.

    Args:
        text: Full text to chunk
        chunk_size: Maximum characters per chunk (default: 2800)
        chunk_overlap: Overlap between chunks (default: 400)

    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find sentence boundary near the end
        search_start = max(start + chunk_size - 200, start)
        search_end = min(start + chunk_size + 100, len(text))
        search_text = text[search_start:search_end]

        # Find the last sentence boundary in search window
        matches = list(SENTENCE_BOUNDARY.finditer(search_text))
        if matches:
            # Use the last match position
            boundary_pos = search_start + matches[-1].start()
            if boundary_pos > start:
                end = boundary_pos

        chunks.append(text[start:end])
        start = end - chunk_overlap

    return chunks


def get_embedding(client: OpenAI, text: str, logger: logging.Logger) -> Optional[list[float]]:
    """
    Get embedding for a single text chunk with retry logic.

    Args:
        client: OpenAI client
        text: Text to embed
        logger: Logger instance

    Returns:
        Embedding vector (1536 dims) or None on failure
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def embed_document(client: OpenAI, content: str, logger: logging.Logger) -> Optional[list[float]]:
    """
    Embed a full document by chunking and averaging embeddings.

    Strategy:
    1. Split content into chunks (2800 chars, 400 overlap)
    2. Get embedding for each chunk
    3. Average all chunk embeddings to get document embedding

    Args:
        client: OpenAI client
        content: Full document content (text extracted from HTML)
        logger: Logger instance

    Returns:
        Document embedding (1536 dims) or None on failure
    """
    if not content:
        return None

    chunks = chunk_text(content)
    if not chunks:
        return None

    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(client, chunk, logger)
        if embedding:
            embeddings.append(embedding)
        else:
            logger.error(f"Failed to embed chunk {i + 1}/{len(chunks)}")

    if not embeddings:
        return None

    # Average embeddings
    avg_embedding = np.mean(embeddings, axis=0).tolist()

    # Verify dimensions
    assert len(avg_embedding) == EMBEDDING_DIMS, \
        f"Expected {EMBEDDING_DIMS} dims, got {len(avg_embedding)}"

    return avg_embedding


def load_html_content(json_path: Path, logger: logging.Logger) -> Optional[str]:
    """
    Load HTML content from the corresponding .html file.

    Args:
        json_path: Path to the JSON file
        logger: Logger instance

    Returns:
        Extracted text content or None if file not found
    """
    # Try .html extension
    html_path = json_path.with_suffix('.html')
    if html_path.exists():
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return extract_text_from_html(html_content)
        except Exception as e:
            logger.warning(f"Error reading HTML {html_path.name}: {e}")

    # Try .pdf (skip for now, would need PDF extraction)
    pdf_path = json_path.with_suffix('.pdf')
    if pdf_path.exists():
        logger.warning(f"PDF file found but not supported: {pdf_path.name}")
        return None

    return None


def process_file(input_path: Path, output_path: Path, client: OpenAI,
                 logger: logging.Logger, skip_existing: bool) -> dict:
    """
    Process a single JSON file and add embeddings.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output *_embedded.json file
        client: OpenAI client
        logger: Logger instance
        skip_existing: Skip if output already exists

    Returns:
        Status dict with success/error info
    """
    result = {
        "input": str(input_path),
        "output": str(output_path),
        "status": "unknown",
        "documents": 0,
        "embedded": 0,
        "skipped": 0,
        "errors": 0,
        "chunks": 0
    }

    if skip_existing and output_path.exists():
        result["status"] = "skipped"
        return result

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load HTML content
        content = load_html_content(input_path, logger)

        if not content:
            result["status"] = "no_content"
            result["skipped"] = 1
            return result

        result["documents"] = 1

        # Check if already has embedding
        if skip_existing and data.get("embedding"):
            result["skipped"] = 1
            result["status"] = "skipped"
            return result

        # Add content hash for deduplication
        data["content_hash"] = compute_hash(content)

        # Count chunks for logging
        chunks = chunk_text(content)
        result["chunks"] = len(chunks)

        # Generate embedding
        embedding = embed_document(client, content, logger)
        if embedding:
            data["embedding"] = embedding
            result["embedded"] = 1
            result["status"] = "success"
        else:
            result["errors"] = 1
            result["status"] = "error"
            logger.error(f"Failed to embed document: {input_path.name}")

        # Write output with ensure_ascii=True for consistency
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"Error processing {input_path.name}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for legal documents with chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSON file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have embeddings"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS}, max: 6)"
    )
    parser.add_argument(
        "--log-name",
        default="default",
        help="Name for log file (default: 'default')"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Validate workers
    if args.workers > 6:
        print("Warning: More than 6 workers not recommended. Using 6.")
        args.workers = 6

    # Setup output directory
    output_dir = args.output or (args.input if args.input.is_dir() else args.input.parent)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging in output directory
    logger = setup_logging(args.log_name, output_dir)

    # Log configuration
    logger.info("=" * 60)
    logger.info("EMBEDDING CONFIGURATION")
    logger.info(f"  Chunk size: {CHUNK_SIZE} chars")
    logger.info(f"  Chunk overlap: {CHUNK_OVERLAP} chars ({CHUNK_OVERLAP/CHUNK_SIZE*100:.1f}%)")
    logger.info(f"  Model: {EMBEDDING_MODEL}")
    logger.info(f"  Dimensions: {EMBEDDING_DIMS}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Skip existing: {args.skip_existing}")
    logger.info("=" * 60)

    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Use --api-key or set environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Collect input files
    if args.input.is_file():
        input_files = [args.input]
    elif args.input.is_dir():
        input_files = list(args.input.glob("*.json"))
        # Exclude special files and already embedded files
        exclude = {'hitlist.json', 'status.json'}
        input_files = [f for f in input_files
                      if f.name not in exclude
                      and not f.name.endswith("_embedded.json")]
    else:
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    if not input_files:
        logger.error("No JSON files found to process")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} file(s) to process")

    # Process files with progress tracking
    results = []
    processed = 0
    total = len(input_files)

    def process_task(input_file):
        output_file = output_dir / f"{input_file.stem}_embedded.json"
        return process_file(input_file, output_file, client, logger, args.skip_existing)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_task, f): f for f in input_files}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            processed += 1

            if processed % 100 == 0 or processed == total:
                logger.info(f"Progress: {processed}/{total} ({100*processed/total:.1f}%)")

    # Summary
    total_embedded = sum(r["embedded"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_chunks = sum(r["chunks"] for r in results)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Files processed: {len(results)}")
    logger.info(f"  Documents embedded: {total_embedded}")
    logger.info(f"  Documents skipped: {total_skipped}")
    logger.info(f"  Total chunks: {total_chunks}")
    logger.info(f"  Errors: {total_errors}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
