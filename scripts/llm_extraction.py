"""
LLM Extraction Module

Contains all LLM-based extraction functions and prompt management.
Prompts are loaded from config/prompts.yaml for easy editing.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import yaml

# =============================================================================
# Logging Setup
# =============================================================================

_logger = None
_log_file_path = None


def get_extraction_logger(output_dir: Optional[str] = None) -> logging.Logger:
    """
    Get or create the extraction logger.

    Args:
        output_dir: Directory to save log file. If provided and file handler
                   doesn't exist yet, creates a new log file there.

    Returns:
        Logger instance
    """
    global _logger, _log_file_path

    # Create logger if it doesn't exist
    if _logger is None:
        _logger = logging.getLogger("textbook_extraction")
        _logger.setLevel(logging.DEBUG)
        _logger.handlers = []  # Clear any existing handlers

        # Console handler - only warnings and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)
        _logger.addHandler(console_handler)

    # Add file handler if output_dir provided and we don't have one yet
    if output_dir and _log_file_path is None:
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _log_file_path = log_dir / "extraction.log"

        print(f"[DEBUG] Creating log file at: {_log_file_path}")

        file_handler = logging.FileHandler(_log_file_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        _logger.addHandler(file_handler)

        # Log session start
        _logger.info("=" * 60)
        _logger.info(f"Extraction session started")

    return _logger


def get_log_file_path() -> Optional[Path]:
    """Get the current log file path, if logging to file is enabled."""
    return _log_file_path


def reset_logger():
    """Reset the logger (useful for testing or changing output directories)."""
    global _logger, _log_file_path
    if _logger:
        _logger.handlers = []
    _logger = None
    _log_file_path = None


# =============================================================================
# Model Context Limits (for chunking)
# =============================================================================

# Characters per token ratio (conservative estimate for English text)
CHARS_PER_TOKEN = 3.5

# Default max output tokens (fallback if API query fails)
DEFAULT_MAX_OUTPUT_TOKENS = 64000

# Cache for model max tokens
_model_max_tokens_cache: dict = {}


def stream_message(
    client,
    model_id: str,
    messages: list[dict],
    max_tokens: int = None,
    on_token: callable = None
) -> tuple[str, dict]:
    """
    Stream a message from the Anthropic API.

    Args:
        client: Anthropic client
        model_id: Model ID to use
        messages: List of message dicts
        max_tokens: Max output tokens (defaults to model max)
        on_token: Optional callback called with each token chunk for progress

    Returns:
        Tuple of (response_text, usage_dict)
        usage_dict contains: input_tokens, output_tokens, stop_reason
    """
    if max_tokens is None:
        max_tokens = get_model_max_tokens(model_id)

    response_text = ""
    input_tokens = 0
    output_tokens = 0
    stop_reason = None

    with client.messages.stream(
        model=model_id,
        max_tokens=max_tokens,
        messages=messages
    ) as stream:
        for event in stream:
            # Handle different event types
            if hasattr(event, 'type'):
                if event.type == 'message_start':
                    if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                        input_tokens = event.message.usage.input_tokens
                elif event.type == 'content_block_delta':
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        chunk = event.delta.text
                        response_text += chunk
                        if on_token:
                            on_token(chunk)
                elif event.type == 'message_delta':
                    if hasattr(event, 'usage'):
                        output_tokens = event.usage.output_tokens
                    if hasattr(event, 'delta') and hasattr(event.delta, 'stop_reason'):
                        stop_reason = event.delta.stop_reason

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason
    }

    return response_text, usage

# Maximum chapter size in characters before chunking is required.
# This is based on OUTPUT constraints, not input context limits.
#
# Empirical data from extraction:
# - 37k chars input → 16k output tokens (TRUNCATED for dense chapters)
# - 25k chars input → ~10k output tokens (safe)
#
# Conservative limit: 25,000 chars per chunk ensures output stays under 16k tokens
# even for question-dense chapters like Tumors and Trauma
MAX_CHUNK_CHARS = 25000


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return int(len(text) / CHARS_PER_TOKEN)


def get_max_chapter_chars(model_id: str) -> int:
    """
    Get maximum chapter size in characters before chunking.

    Based on OUTPUT token limits (16k), not input context limits.
    More text = more questions = more output JSON.
    """
    # Could adjust based on model in future, but output limit is the constraint
    return MAX_CHUNK_CHARS

# =============================================================================
# Two-Pass Extraction Functions
# =============================================================================


def extract_line_ranges_llm(
    client,
    chapter_num: int,
    chapter_text: str,
    model_id: str
) -> list[dict]:
    """
    First pass: Extract just line ranges for each Q&A pair.

    Output is small (~50 bytes per question), so no token limit issues.

    Args:
        client: Anthropic client
        chapter_num: Chapter number for logging
        chapter_text: Line-numbered chapter text with [LINE:NNNN] markers
        model_id: Model to use

    Returns:
        List of dicts with keys:
        - question_id: The question number (e.g., "1", "1a")
        - question_start: First line number
        - question_end: Last line number of question
        - answer_start: First line of answer (0 if none)
        - answer_end: Last line of answer (0 if none)
        - correct_letter: A/B/C/D/E or ""
        - image_files: List of image filenames
    """
    logger = get_extraction_logger()
    log_prefix = f"Chapter {chapter_num} (pass 1)"

    prompt = get_prompt("extract_line_ranges", chapter_text=chapter_text)

    try:
        logger.debug(f"{log_prefix}: Calling API (streaming) with model {model_id}")

        response_text, usage = stream_message(
            client,
            model_id,
            messages=[{"role": "user", "content": prompt}]
        )

        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        stop_reason = usage["stop_reason"]

        logger.info(
            f"{log_prefix}: API response - "
            f"input={input_tokens:,}, output={output_tokens:,}, stop={stop_reason}"
        )

        if stop_reason == "max_tokens":
            logger.warning(f"{log_prefix}: Response may be truncated (max_tokens reached)")

        # Extract JSON from markdown code blocks if present
        if "```" in response_text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if match:
                response_text = match.group(1)

        # Fix leading zeros on numbers (LLM copies format from [LINE:0941] → 0941)
        # JSON doesn't allow leading zeros on numbers, so convert 0941 → 941
        response_text = re.sub(r':\s*0+(\d+)', r': \1', response_text)

        line_ranges = json.loads(response_text)
        logger.info(f"{log_prefix}: Found {len(line_ranges)} Q&A pairs")

        return line_ranges

    except json.JSONDecodeError as e:
        logger.error(f"{log_prefix}: JSON parse error - {e}")
        logger.error(f"{log_prefix}: Response preview: {response_text[:500]!r}")
        return []
    except Exception as e:
        logger.error(f"{log_prefix}: API error - {type(e).__name__}: {e}")
        return []


def format_qa_pair_llm(
    client,
    question_id: str,
    question_text: str,
    answer_text: str,
    model_id: str,
    chapter_num: int,
    max_retries: int = 5
) -> dict:
    """
    Second pass: Format a single Q&A pair into structured JSON.

    Each call is small, so no token limit issues.
    Includes retry logic with exponential backoff for rate limits.

    Args:
        client: Anthropic client
        question_id: The question ID (e.g., "1", "1a")
        question_text: The question text (extracted by line range)
        answer_text: The answer text (extracted by line range)
        model_id: Model to use
        chapter_num: Chapter number for logging
        max_retries: Maximum retry attempts for rate limit errors

    Returns:
        Dict with structured Q&A data
    """
    import time
    import anthropic

    logger = get_extraction_logger()
    log_prefix = f"Ch{chapter_num} Q{question_id}"

    prompt = get_prompt("format_qa_pair",
                       question_id=question_id,
                       question_text=question_text,
                       answer_text=answer_text)

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=get_model_max_tokens(model_id),
                messages=[{"role": "user", "content": prompt}]
            )

            logger.debug(
                f"{log_prefix}: in={response.usage.input_tokens}, "
                f"out={response.usage.output_tokens}"
            )

            response_text = response.content[0].text

            # Extract JSON from markdown code blocks if present
            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            return json.loads(response_text)

        except anthropic.RateLimitError as e:
            if attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                wait_time = 2 ** attempt
                logger.warning(f"{log_prefix}: Rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"{log_prefix}: Rate limit exceeded after {max_retries} retries")
                return {
                    "id": question_id,
                    "text": question_text,
                    "choices": {},
                    "has_image": False,
                    "correct_answer": "",
                    "explanation": answer_text,
                    "error": f"Rate limit exceeded: {e}"
                }

        except json.JSONDecodeError as e:
            logger.error(f"{log_prefix}: JSON parse error - {e}")
            return {
                "id": question_id,
                "text": question_text,
                "choices": {},
                "has_image": False,
                "correct_answer": "",
                "explanation": answer_text,
                "error": str(e)
            }

        except Exception as e:
            logger.error(f"{log_prefix}: API error - {type(e).__name__}: {e}")
            return {
                "id": question_id,
                "text": question_text,
                "choices": {},
                "has_image": False,
                "correct_answer": "",
                "explanation": answer_text,
                "error": str(e)
            }


def extract_qa_pairs_two_pass(
    client,
    chapter_num: int,
    chapter_text: str,
    model_id: str,
    lines_with_images: list[str]
) -> dict:
    """
    Two-pass extraction for large chapters.

    Pass 1: Get line ranges for each Q&A pair (minimal output)
    Pass 2: Format each Q&A pair individually (parallel calls)

    Args:
        client: Anthropic client
        chapter_num: Chapter number
        chapter_text: Line-numbered chapter text with [LINE:NNNN] markers
        model_id: Model to use
        lines_with_images: The lines array with [IMAGE:] markers inserted

    Returns:
        Dict with 'chapter' and 'questions' keys
    """
    from pdf_extraction import extract_lines_by_range

    logger = get_extraction_logger()
    logger.info(f"Chapter {chapter_num}: Using two-pass extraction")

    # First pass: get line ranges
    line_ranges = extract_line_ranges_llm(client, chapter_num, chapter_text, model_id)

    if not line_ranges:
        logger.error(f"Chapter {chapter_num}: First pass failed, no line ranges extracted")
        return {"chapter": chapter_num, "questions": [], "error": "first_pass_failed"}

    logger.info(f"Chapter {chapter_num}: Formatting {len(line_ranges)} Q&A pairs (pass 2)")

    # Second pass: format each pair
    questions = []
    for i, lr in enumerate(line_ranges):
        q_id = lr.get("question_id", str(i + 1))

        # Extract text by line ranges
        q_start = lr.get("question_start", 0)
        q_end = lr.get("question_end", 0)
        a_start = lr.get("answer_start", 0)
        a_end = lr.get("answer_end", 0)

        q_text = extract_lines_by_range(lines_with_images, q_start, q_end) if q_start > 0 else ""
        a_text = extract_lines_by_range(lines_with_images, a_start, a_end) if a_start > 0 else ""

        # Format the Q&A pair
        formatted = format_qa_pair_llm(
            client, q_id, q_text, a_text, model_id, chapter_num
        )

        # Add image files from line ranges
        formatted["image_files"] = lr.get("image_files", [])

        # Preserve correct_letter from first pass as fallback
        if not formatted.get("correct_answer") and lr.get("correct_letter"):
            formatted["correct_answer"] = lr["correct_letter"]

        questions.append(formatted)

        # Progress log every 10 questions
        if (i + 1) % 10 == 0:
            logger.info(f"Chapter {chapter_num}: Formatted {i + 1}/{len(line_ranges)} questions")

    logger.info(f"Chapter {chapter_num}: Completed two-pass extraction with {len(questions)} questions")

    return {"chapter": chapter_num, "questions": questions}


# =============================================================================
# Prompt Loading
# =============================================================================

_cached_prompts = None

def get_prompts_path() -> Path:
    """Get path to prompts.yaml file."""
    return Path(__file__).parent / "config" / "prompts.yaml"


def load_prompts() -> dict:
    """Load prompts from YAML file. Caches after first load."""
    global _cached_prompts

    if _cached_prompts is not None:
        return _cached_prompts

    prompts_path = get_prompts_path()
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_path) as f:
        _cached_prompts = yaml.safe_load(f)

    return _cached_prompts


def get_prompt(name: str, **kwargs) -> str:
    """
    Get a formatted prompt by name.

    Args:
        name: Prompt name (e.g., 'identify_chapters', 'extract_qa_pairs')
        **kwargs: Variables to substitute into the prompt template

    Returns:
        Formatted prompt string
    """
    prompts = load_prompts()

    if name not in prompts:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(prompts.keys())}")

    prompt_template = prompts[name]["prompt"]

    # Use format_map to handle missing keys gracefully
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing variable for prompt '{name}': {e}")


def reload_prompts():
    """Force reload prompts from disk (useful after editing prompts.yaml)."""
    global _cached_prompts
    _cached_prompts = None
    return load_prompts()


def save_prompts(prompts: dict):
    """Save prompts to YAML file and update cache."""
    global _cached_prompts
    prompts_path = get_prompts_path()

    with open(prompts_path, 'w') as f:
        yaml.dump(prompts, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    _cached_prompts = prompts


# =============================================================================
# Anthropic Client & Model Management
# =============================================================================

# Fallback Claude models (used if API fetch fails)
FALLBACK_MODELS = {
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-haiku-4-5-20251218": "Claude Haiku 4.5",
    "claude-3-5-sonnet-20241022": "Claude Sonnet 3.5",
    "claude-3-5-haiku-20241022": "Claude Haiku 3.5",
}
DEFAULT_MODEL_ID = "claude-haiku-4-5-20251218"
DEFAULT_MODEL_NAME = "Claude Haiku 4.5"

_cached_models = None


def get_anthropic_client():
    """Get Anthropic client, loading API key from .env if needed."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic
    return anthropic.Anthropic()


def fetch_available_models() -> dict:
    """
    Fetch available models from the Anthropic API.
    Returns dict mapping display_name to model_id.
    Falls back to static list if API call fails.
    """
    global _cached_models

    if _cached_models is not None:
        return _cached_models

    try:
        client = get_anthropic_client()
        if not client:
            _cached_models = {v: k for k, v in FALLBACK_MODELS.items()}
            return _cached_models

        response = client.models.list(limit=100)

        models = {}
        for model in response.data:
            display_name = getattr(model, 'display_name', None) or model.id
            models[display_name] = model.id

        if models:
            _cached_models = models
            return _cached_models

    except Exception as e:
        print(f"Failed to fetch models from API: {e}")

    _cached_models = {v: k for k, v in FALLBACK_MODELS.items()}
    return _cached_models


def get_model_options() -> list[str]:
    """Get list of model display names for dropdown."""
    models = fetch_available_models()
    return list(models.keys())


def get_model_id(display_name: str) -> str:
    """Get model ID from display name."""
    models = fetch_available_models()
    return models.get(display_name, DEFAULT_MODEL_ID)


def get_model_max_tokens(model_id: str) -> int:
    """
    Get the maximum output tokens for a model from the API.
    Caches the result to avoid repeated API calls.
    Falls back to DEFAULT_MAX_OUTPUT_TOKENS if API call fails.
    """
    global _model_max_tokens_cache

    if model_id in _model_max_tokens_cache:
        return _model_max_tokens_cache[model_id]

    try:
        client = get_anthropic_client()
        if client:
            model_info = client.models.retrieve(model_id)
            max_tokens = getattr(model_info, 'max_tokens', None)
            if max_tokens:
                _model_max_tokens_cache[model_id] = max_tokens
                return max_tokens
    except Exception as e:
        print(f"Failed to get max tokens for {model_id}: {e}")

    # Fallback to default
    _model_max_tokens_cache[model_id] = DEFAULT_MAX_OUTPUT_TOKENS
    return DEFAULT_MAX_OUTPUT_TOKENS


# =============================================================================
# LLM Extraction Functions
# =============================================================================

def identify_chapters_llm(client, pages: list[dict], model_id: str) -> list[dict]:
    """Use Claude to identify chapter boundaries."""
    from pdf_extraction import create_page_index

    logger = get_extraction_logger()
    logger.info(f"Identifying chapters from {len(pages)} pages using {model_id}")

    page_index = create_page_index(pages)
    prompt = get_prompt("identify_chapters", page_index=page_index)

    try:
        response_text, usage = stream_message(
            client,
            model_id,
            messages=[{"role": "user", "content": prompt}]
        )

        logger.info(
            f"Chapter identification: input={usage['input_tokens']:,}, "
            f"output={usage['output_tokens']:,}, stop={usage['stop_reason']}"
        )

        if "```" in response_text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if match:
                response_text = match.group(1)

        chapters = json.loads(response_text)
        logger.info(f"Identified {len(chapters)} chapters")
        return chapters

    except json.JSONDecodeError as e:
        logger.error(f"Chapter identification JSON parse error: {e}")
        raise
    except Exception as e:
        logger.error(f"Chapter identification API error: {type(e).__name__}: {e}")
        raise


def chunk_chapter_text(chapter_text: str, max_chars: int, overlap_chars: int = 2000) -> list[str]:
    """
    Split chapter text into overlapping chunks that fit within model limits.

    Uses overlapping chunks to avoid losing questions at boundaries.
    Duplicate questions are removed during merge.

    Args:
        chapter_text: Full chapter text
        max_chars: Maximum characters per chunk
        overlap_chars: Characters to overlap between chunks (default 2000, ~2-3 questions worth)

    Returns:
        List of text chunks (with overlap)
    """
    if len(chapter_text) <= max_chars:
        return [chapter_text]

    chunks = []
    start = 0

    while start < len(chapter_text):
        end = start + max_chars

        # If this is the last chunk, take everything remaining
        if end >= len(chapter_text):
            chunks.append(chapter_text[start:])
            break

        # Find a good split point (paragraph boundary) near the end
        search_start = start + int(max_chars * 0.8)
        search_end = end

        # Look for double newline (paragraph break)
        para_break = chapter_text.rfind('\n\n', search_start, search_end)

        if para_break > search_start:
            split_point = para_break + 2
        else:
            # Fall back to single newline
            line_break = chapter_text.rfind('\n', search_start, search_end)
            if line_break > search_start:
                split_point = line_break + 1
            else:
                split_point = end

        chunks.append(chapter_text[start:split_point])

        # Move start back by overlap amount for next chunk
        # This ensures overlap_chars of content appears in both chunks
        start = split_point - overlap_chars

        # Make sure we don't go backwards
        if start < 0:
            start = 0

    return chunks


def extract_qa_pairs_llm(
    client,
    chapter_num: int,
    chapter_text: str,
    model_id: str,
    lines_with_images: list[str] = None
) -> dict:
    """
    Use Claude to extract Q&A pairs from a single chapter.

    Supports two modes:
    1. Two-pass mode (when lines_with_images is provided):
       - First pass: Extract line ranges (minimal output)
       - Second pass: Format each Q&A pair individually
       - Used for large chapters to avoid output token limits

    2. Single-pass mode (legacy, when no lines provided):
       - Direct extraction with chunking for large chapters
       - Falls back to this mode for backward compatibility

    Args:
        client: Anthropic client
        chapter_num: Chapter number
        chapter_text: Chapter text (may be line-numbered if two-pass)
        model_id: Model to use
        lines_with_images: If provided, enables two-pass extraction

    Returns:
        Dict with 'chapter' and 'questions' keys
    """
    logger = get_extraction_logger()

    # Check if two-pass mode is enabled
    if lines_with_images is not None and "[LINE:" in chapter_text:
        return extract_qa_pairs_two_pass(
            client, chapter_num, chapter_text, model_id, lines_with_images
        )

    # Legacy single-pass mode
    chapter_chars = len(chapter_text)
    estimated_tokens = estimate_tokens(chapter_text)
    max_chars = get_max_chapter_chars(model_id)

    logger.info(f"Chapter {chapter_num}: {chapter_chars:,} chars (~{estimated_tokens:,} tokens)")

    # Check if chunking is needed
    if chapter_chars > max_chars:
        logger.warning(
            f"Chapter {chapter_num} exceeds limit ({chapter_chars:,} > {max_chars:,} chars). "
            f"Splitting into chunks."
        )
        return extract_qa_pairs_chunked(client, chapter_num, chapter_text, model_id, max_chars)

    # Single extraction for normal-sized chapters
    return extract_qa_pairs_single(client, chapter_num, chapter_text, model_id)


def extract_qa_pairs_single(client, chapter_num: int, chapter_text: str, model_id: str,
                            chunk_info: str = "") -> dict:
    """
    Extract Q&A pairs from a single piece of text (chapter or chunk).

    Args:
        client: Anthropic client
        chapter_num: Chapter number
        chapter_text: Text to extract from
        model_id: Model to use
        chunk_info: Optional string describing which chunk this is (for logging)

    Returns:
        Dict with 'chapter' and 'questions' keys
    """
    logger = get_extraction_logger()
    log_prefix = f"Chapter {chapter_num}{chunk_info}"

    prompt = get_prompt("extract_qa_pairs",
                       chapter_num=chapter_num,
                       chapter_text=chapter_text)

    try:
        logger.debug(f"{log_prefix}: Calling API (streaming) with model {model_id}")

        response_text, usage = stream_message(
            client,
            model_id,
            messages=[{"role": "user", "content": prompt}]
        )

        # Log token usage
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        stop_reason = usage["stop_reason"]

        logger.info(
            f"{log_prefix}: API response - "
            f"input={input_tokens:,}, output={output_tokens:,}, stop={stop_reason}"
        )

        # Check for truncation
        if stop_reason == "max_tokens":
            logger.error(
                f"{log_prefix}: Response truncated at max_tokens ({get_model_max_tokens(model_id)}). "
                f"Output may be incomplete!"
            )

        # Extract JSON from markdown code blocks if present
        if "```" in response_text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if match:
                response_text = match.group(1)

        # Parse JSON
        try:
            result = json.loads(response_text)
            question_count = len(result.get("questions", []))
            logger.info(f"{log_prefix}: Extracted {question_count} questions")
            return result

        except json.JSONDecodeError as e:
            # Log the error with context
            logger.error(
                f"{log_prefix}: JSON parse error - {e}. "
                f"Response length: {len(response_text)} chars. "
                f"First 200 chars: {response_text[:200]!r}"
            )
            # Log last 200 chars to see if truncated
            logger.error(f"{log_prefix}: Last 200 chars: {response_text[-200:]!r}")

            return {"chapter": chapter_num, "questions": [], "error": str(e)}

    except Exception as e:
        # Log API errors
        logger.error(f"{log_prefix}: API error - {type(e).__name__}: {e}")
        return {"chapter": chapter_num, "questions": [], "error": str(e)}


def extract_qa_pairs_chunked(client, chapter_num: int, chapter_text: str,
                              model_id: str, max_chars: int) -> dict:
    """
    Extract Q&A pairs from a large chapter by processing in overlapping chunks.

    Args:
        client: Anthropic client
        chapter_num: Chapter number
        chapter_text: Full chapter text
        model_id: Model to use
        max_chars: Maximum characters per chunk

    Returns:
        Merged dict with 'chapter' and 'questions' keys
    """
    logger = get_extraction_logger()

    chunks = chunk_chapter_text(chapter_text, max_chars)
    logger.info(f"Chapter {chapter_num}: Split into {len(chunks)} overlapping chunks (2k char overlap)")

    all_questions = []
    errors = []

    for i, chunk in enumerate(chunks, 1):
        chunk_info = f" (chunk {i}/{len(chunks)})"
        logger.info(f"Chapter {chapter_num}{chunk_info}: {len(chunk):,} chars")

        result = extract_qa_pairs_single(client, chapter_num, chunk, model_id, chunk_info)

        if "error" in result:
            errors.append(f"Chunk {i}: {result['error']}")
        else:
            chunk_questions = result.get("questions", [])
            all_questions.extend(chunk_questions)

    # Deduplicate questions by ID (overlap causes intentional duplicates)
    seen_ids = set()
    unique_questions = []
    duplicates_removed = 0
    for q in all_questions:
        q_id = q.get("id", "")
        if q_id not in seen_ids:
            seen_ids.add(q_id)
            unique_questions.append(q)
        else:
            duplicates_removed += 1

    logger.info(
        f"Chapter {chapter_num}: Merged {len(unique_questions)} unique questions "
        f"from {len(chunks)} chunks ({duplicates_removed} duplicates removed)"
    )

    result = {"chapter": chapter_num, "questions": unique_questions}
    if errors:
        result["chunk_errors"] = errors
        logger.warning(f"Chapter {chapter_num}: {len(errors)} chunk(s) had errors")

    return result


def process_chapter_extraction(
    client,
    ch_num: int,
    ch_key: str,
    ch_text: str,
    model_id: str,
    lines_with_images: list[str] = None
) -> tuple[str, list[dict]]:
    """
    Process a single chapter extraction (for parallel execution).

    Args:
        client: Anthropic client
        ch_num: Chapter number
        ch_key: Chapter key (e.g., "ch6")
        ch_text: Chapter text (may be line-numbered for two-pass)
        model_id: Model ID to use
        lines_with_images: If provided, enables two-pass extraction

    Returns:
        Tuple of (ch_key, list of formatted questions)
    """
    result = extract_qa_pairs_llm(
        client, ch_num, ch_text, model_id,
        lines_with_images=lines_with_images
    )

    questions = []
    for q in result.get("questions", []):
        q_data = {
            "full_id": f"ch{ch_num}_{q['id']}",
            "local_id": q["id"],
            "text": q.get("text", ""),
            "choices": q.get("choices", {}),
            "has_image": q.get("has_image", False),
            "image_group": q.get("image_group"),
            "correct_answer": q.get("correct_answer", ""),
            "explanation": q.get("explanation", "")
        }
        # Preserve image_files from two-pass extraction
        if q.get("image_files"):
            q_data["image_files"] = q["image_files"]
        questions.append(q_data)

    return (ch_key, questions)


def match_images_to_questions_llm(client, images: list[dict], chapters: list[dict],
                                   questions: dict, model_id: str) -> dict:
    """
    Use Claude to intelligently match images to questions based on flanking text context.
    Returns dict mapping image filename to question full_id.
    """
    logger = get_extraction_logger()
    logger.info(f"Matching {len(images)} images to questions using {model_id}")

    assignments = {}

    for i, ch in enumerate(chapters):
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"
        start_page = ch["start_page"]
        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else 9999

        ch_questions = questions.get(ch_key, [])
        if not ch_questions:
            continue

        ch_images = [img for img in images if start_page <= img["page"] < end_page]
        if not ch_images:
            continue

        logger.debug(f"Chapter {ch_num}: {len(ch_images)} images, {len(ch_questions)} questions")

        # Build question summary for the prompt
        questions_text = []
        for q in ch_questions:
            q_summary = f"- {q['full_id']}: {q['text'][:150]}..."
            if q.get("has_image"):
                q_summary += " [NEEDS IMAGE]"
            questions_text.append(q_summary)

        # Build image context for the prompt
        images_text = []
        for img in ch_images:
            ctx_before = img.get('context_before', '')
            ctx_after = img.get('context_after', '')
            img_info = f"- {img['filename']} (page {img['page']})\n"
            img_info += f"  Text BEFORE image: \"...{ctx_before[-300:]}\"\n"
            img_info += f"  Text AFTER image: \"{ctx_after[:300]}...\""
            images_text.append(img_info)

        prompt = get_prompt("match_images_to_questions",
                           chapter_num=ch_num,
                           questions_text="\n".join(questions_text),
                           images_text="\n".join(images_text))

        try:
            response_text, usage = stream_message(
                client,
                model_id,
                messages=[{"role": "user", "content": prompt}]
            )

            logger.info(
                f"Image matching Ch{ch_num}: input={usage['input_tokens']:,}, "
                f"output={usage['output_tokens']:,}"
            )

            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            ch_assignments = json.loads(response_text)
            matched_count = sum(1 for q_id in ch_assignments.values() if q_id and q_id != "(none)")
            logger.info(f"Chapter {ch_num}: Matched {matched_count}/{len(ch_images)} images")

            for img_file, q_id in ch_assignments.items():
                if q_id and q_id != "(none)":
                    assignments[img_file] = q_id

        except json.JSONDecodeError as e:
            logger.error(f"Image matching Ch{ch_num}: JSON parse error - {e}")
            continue
        except Exception as e:
            logger.error(f"Image matching Ch{ch_num}: API error - {type(e).__name__}: {e}")
            continue

    logger.info(f"Total image assignments: {len(assignments)}")
    return assignments


def match_images_to_questions_simple(images: list[dict], chapters: list[dict],
                                      questions: dict) -> dict:
    """
    Simple fallback: match images to questions based on page proximity.
    Used when LLM matching is not available.
    """
    assignments = {}

    for i, ch in enumerate(chapters):
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"
        start_page = ch["start_page"]
        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else 9999

        ch_questions = [q for q in questions.get(ch_key, []) if q.get("has_image")]
        ch_images = [img for img in images if start_page <= img["page"] < end_page]

        for img, q in zip(ch_images, ch_questions):
            if q["full_id"] not in assignments.values():
                assignments[img["filename"]] = q["full_id"]

    return assignments


def postprocess_questions_llm(client, questions: dict, model_id: str) -> dict:
    """
    Post-process extracted questions to link context to sub-questions.
    """
    for ch_key, ch_questions in questions.items():
        if not ch_questions:
            continue

        questions_json = json.dumps(ch_questions, indent=2)
        prompt = get_prompt("postprocess_questions", questions_json=questions_json)

        try:
            response_text, usage = stream_message(
                client,
                model_id,
                messages=[{"role": "user", "content": prompt}]
            )

            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            updated_questions = json.loads(response_text)
            questions[ch_key] = updated_questions

        except Exception as e:
            print(f"Post-processing failed for {ch_key}: {e}")
            for q in ch_questions:
                if "is_context_only" not in q:
                    has_choices = bool(q.get("choices"))
                    has_letter = any(c.isalpha() for c in q.get("local_id", ""))
                    q["is_context_only"] = not has_choices and not has_letter
                if "context" not in q:
                    q["context"] = ""
                if "context_question_id" not in q:
                    q["context_question_id"] = ""

    return questions


def add_page_numbers_to_questions(questions: dict, pages: list[dict], chapters: list[dict]) -> dict:
    """
    Add question_page and answer_page fields to all questions.

    Searches pages.json for question text and explanation text to determine
    which PDF pages they appear on.

    Args:
        questions: Dict of chapter_key -> list of question dicts
        pages: List of page dicts from pages.json
        chapters: List of chapter dicts with start_page info

    Returns:
        Updated questions dict with page numbers added
    """
    from pdf_extraction import detect_question_pages

    for i, ch in enumerate(chapters):
        ch_key = f"ch{ch['chapter_number']}"
        ch_start = ch["start_page"]
        ch_end = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else None

        ch_questions = questions.get(ch_key, [])
        if ch_questions:
            detect_question_pages(ch_questions, pages, ch_start, ch_end)

    return questions


def associate_context_llm(client, questions: dict, image_assignments: dict,
                          model_id: str) -> tuple[dict, dict, dict]:
    """
    Use LLM to identify context relationships and merge context into sub-questions.

    Returns:
        Tuple of (updated_questions, updated_image_assignments, stats)
    """
    logger = get_extraction_logger()
    total_questions = sum(len(qs) for qs in questions.values())
    logger.info(f"Associating context for {total_questions} questions using {model_id}")

    if image_assignments is None:
        image_assignments = {}

    updated_assignments = dict(image_assignments)

    stats = {
        "context_questions_found": 0,
        "sub_questions_updated": 0,
        "images_copied": 0
    }

    for ch_key, ch_questions in questions.items():
        if not ch_questions:
            continue

        # Build a summary of questions for the LLM
        questions_summary = []
        for q in ch_questions:
            has_choices = bool(q.get("choices"))
            summary = {
                "full_id": q["full_id"],
                "local_id": q["local_id"],
                "text_preview": q["text"][:200] + "..." if len(q["text"]) > 200 else q["text"],
                "has_choices": has_choices
            }
            questions_summary.append(summary)

        prompt = get_prompt("associate_context",
                           questions_summary=json.dumps(questions_summary, indent=2))

        try:
            response_text, usage = stream_message(
                client,
                model_id,
                messages=[{"role": "user", "content": prompt}]
            )

            logger.info(
                f"Context association {ch_key}: input={usage['input_tokens']:,}, "
                f"output={usage['output_tokens']:,}"
            )

            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            result = json.loads(response_text)
            mappings = result.get("context_mappings", [])
            logger.debug(f"{ch_key}: Found {len(mappings)} context mappings")

            question_by_id = {q["full_id"]: q for q in ch_questions}

            for mapping in mappings:
                context_id = mapping.get("context_id")
                sub_ids = mapping.get("sub_question_ids", [])

                if not context_id or context_id not in question_by_id:
                    continue

                context_q = question_by_id[context_id]
                context_text = context_q.get("text", "").strip()

                context_q["is_context_only"] = True
                stats["context_questions_found"] += 1

                # Count context images (they stay assigned to context question,
                # sub-questions inherit them via context_from field)
                context_images = [
                    img_file for img_file, assigned_to in image_assignments.items()
                    if assigned_to == context_id
                ]
                stats["images_copied"] += len(context_images)

                for sub_id in sub_ids:
                    if sub_id not in question_by_id:
                        continue

                    sub_q = question_by_id[sub_id]

                    if sub_q.get("context_merged"):
                        continue

                    original_text = sub_q.get("text", "").strip()
                    sub_q["text"] = f"{context_text} {original_text}"
                    sub_q["context_merged"] = True
                    sub_q["context_from"] = context_id
                    sub_q["is_context_only"] = False
                    stats["sub_questions_updated"] += 1
                    # Images stay assigned to context question - sub-questions
                    # inherit them via context_from lookup in display code

        except json.JSONDecodeError as e:
            logger.error(f"Context association {ch_key}: JSON parse error - {e}")
            continue
        except Exception as e:
            logger.error(f"Context association {ch_key}: API error - {type(e).__name__}: {e}")
            continue

    logger.info(
        f"Context association complete: {stats['context_questions_found']} context questions, "
        f"{stats['sub_questions_updated']} sub-questions updated"
    )

    # Ensure all questions have is_context_only set
    for ch_key, ch_questions in questions.items():
        for q in ch_questions:
            if "is_context_only" not in q:
                q["is_context_only"] = False

    return questions, updated_assignments, stats
