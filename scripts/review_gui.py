#!/usr/bin/env python3
"""
Textbook Q&A Extractor - Web Interface

A Streamlit-based GUI for extracting Q&A pairs from textbooks using LLM.

Workflow:
1. Select source PDF
2. Extract chapters and preview text
3. Extract questions per chapter
4. QC questions with progress tracking
5. Export to Anki deck

Usage:
    streamlit run scripts/review_gui.py
"""

import streamlit as st
import json
import os
import copy
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF

# Page config
st.set_page_config(
    page_title="Textbook Q&A Extractor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
SOURCE_DIR = "source"
BASE_OUTPUT_DIR = "output"

# Fallback Claude models (used if API fetch fails)
# Maps model_id -> display_name
FALLBACK_MODELS = {
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-3-5-sonnet-20241022": "Claude Sonnet 3.5",
    "claude-3-5-haiku-20241022": "Claude Haiku 3.5",
}
DEFAULT_MODEL_ID = "claude-sonnet-4-20250514"
DEFAULT_MODEL_NAME = "Claude Sonnet 4"  # Display name for default

# Cache for fetched models
_cached_models = None


# =============================================================================
# Dynamic Path Functions (per-PDF output directories)
# =============================================================================

def get_pdf_slug(pdf_name: str) -> str:
    """Convert PDF filename to a safe directory slug."""
    # Remove .pdf extension and convert to safe directory name
    slug = Path(pdf_name).stem
    # Replace spaces and special chars with underscores
    slug = re.sub(r'[^\w\-]', '_', slug)
    # Remove multiple consecutive underscores
    slug = re.sub(r'_+', '_', slug)
    return slug.strip('_')


def get_output_dir() -> str:
    """Get output directory for current PDF."""
    if "current_pdf" in st.session_state and st.session_state.current_pdf:
        slug = get_pdf_slug(st.session_state.current_pdf)
        return f"{BASE_OUTPUT_DIR}/{slug}"
    return BASE_OUTPUT_DIR


def get_images_dir() -> str:
    """Get images directory for current PDF."""
    return f"{get_output_dir()}/images"


def get_chapters_file() -> str:
    return f"{get_output_dir()}/chapters.json"


def get_chapter_text_file() -> str:
    return f"{get_output_dir()}/chapter_text.json"


def get_questions_file() -> str:
    return f"{get_output_dir()}/questions_by_chapter.json"


def get_questions_merged_file() -> str:
    return f"{get_output_dir()}/questions_merged.json"


def get_image_assignments_merged_file() -> str:
    return f"{get_output_dir()}/image_assignments_merged.json"


def get_images_file() -> str:
    return f"{get_output_dir()}/images.json"


def get_image_assignments_file() -> str:
    return f"{get_output_dir()}/image_assignments.json"


def get_qc_progress_file() -> str:
    return f"{get_output_dir()}/qc_progress.json"


def get_settings_file() -> str:
    return f"{get_output_dir()}/settings.json"


def get_pages_file() -> str:
    return f"{get_output_dir()}/pages.json"


def get_available_textbooks() -> list[str]:
    """Get list of textbooks that have output data."""
    textbooks = []
    if os.path.exists(BASE_OUTPUT_DIR):
        for item in os.listdir(BASE_OUTPUT_DIR):
            item_path = os.path.join(BASE_OUTPUT_DIR, item)
            # Check if it's a directory with a chapters.json file
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "chapters.json")):
                textbooks.append(item)
    return sorted(textbooks)


# =============================================================================
# PDF Extraction Functions (from llm_pipeline.py)
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract raw text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def extract_images_from_pdf(pdf_path: str, output_dir: str = None) -> list[dict]:
    """
    Extract images from PDF with page numbers, positions, and flanking text context.
    Returns list of image metadata including text before/after each image.

    Flanking text is extracted across page boundaries - if an image is at the top
    of a page, context_before will include text from the bottom of the previous page.
    """
    if output_dir is None:
        output_dir = get_images_dir()
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    # First pass: collect all text blocks from all pages
    all_text_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "") + " "
                if block_text.strip():
                    all_text_blocks.append({
                        "text": block_text.strip(),
                        "page": page_num,
                        "y0": block.get("bbox", [0, 0, 0, 0])[1],
                        "y1": block.get("bbox", [0, 0, 0, 0])[3],
                    })

    # Sort all text blocks by page, then y position
    all_text_blocks.sort(key=lambda x: (x["page"], x["y0"]))

    # Second pass: extract images and find flanking text (across page boundaries)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Get image position on page
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    y_position = rect.y0
                    y_bottom = rect.y1
                    x_position = rect.x0
                else:
                    y_position = 0
                    y_bottom = 0
                    x_position = 0

                # Extract flanking text context across page boundaries
                text_before = []
                text_after = []

                for tb in all_text_blocks:
                    # Text is BEFORE image if:
                    # - It's on a previous page, OR
                    # - It's on the same page and ends before the image starts
                    if tb["page"] < page_num:
                        text_before.append(tb["text"])
                    elif tb["page"] == page_num and tb["y1"] < y_position:
                        text_before.append(tb["text"])
                    # Text is AFTER image if:
                    # - It's on a later page, OR
                    # - It's on the same page and starts after the image ends
                    elif tb["page"] > page_num:
                        text_after.append(tb["text"])
                    elif tb["page"] == page_num and tb["y0"] > y_bottom:
                        text_after.append(tb["text"])

                # Keep last 500 chars before and first 500 chars after
                context_before = " ".join(text_before)[-500:] if text_before else ""
                context_after = " ".join(text_after)[:500] if text_after else ""

                # Create filename with page and position info
                filename = f"p{page_num + 1:03d}_y{int(y_position):04d}_x{int(x_position):04d}_{img_idx}.{image_ext}"
                filepath = os.path.join(output_dir, filename)

                # Save image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                images.append({
                    "filename": filename,
                    "filepath": filepath,
                    "page": page_num + 1,
                    "y_position": y_position,
                    "x_position": x_position,
                    "width": rect.width if img_rects else 0,
                    "height": rect.height if img_rects else 0,
                    "context_before": context_before,
                    "context_after": context_after
                })

            except Exception as e:
                # Skip problematic images
                continue

    doc.close()

    # Sort by page and y-position
    images.sort(key=lambda x: (x["page"], x["y_position"]))

    return images


def assign_chapters_to_images(images: list[dict], chapters: list[dict]) -> list[dict]:
    """
    Assign chapter numbers to images based on page ranges.
    This should be called after chapter detection to group images by chapter.
    """
    if not chapters:
        return images

    # Build page range lookup
    chapter_ranges = []
    for i, ch in enumerate(chapters):
        start_page = ch["start_page"]
        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else 9999
        chapter_ranges.append({
            "chapter_number": ch["chapter_number"],
            "chapter_key": f"ch{ch['chapter_number']}",
            "start_page": start_page,
            "end_page": end_page
        })

    # Assign chapter to each image
    for img in images:
        img_page = img["page"]
        img["chapter"] = None
        img["chapter_key"] = None

        for ch_range in chapter_ranges:
            if ch_range["start_page"] <= img_page < ch_range["end_page"]:
                img["chapter"] = ch_range["chapter_number"]
                img["chapter_key"] = ch_range["chapter_key"]
                break

    return images


def match_images_to_questions_llm(client, images: list[dict], chapters: list[dict], questions: dict) -> dict:
    """
    Use Claude to intelligently match images to questions based on flanking text context.
    Returns dict mapping image filename to question full_id.
    """
    assignments = {}

    # Process chapter by chapter
    for i, ch in enumerate(chapters):
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"
        start_page = ch["start_page"]
        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else 9999

        # Get questions for this chapter
        ch_questions = questions.get(ch_key, [])
        if not ch_questions:
            continue

        # Get images in this chapter's page range
        ch_images = [img for img in images if start_page <= img["page"] < end_page]
        if not ch_images:
            continue

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
            # IMPORTANT: For context_before, use the LAST 300 chars (question number is at the end)
            # For context_after, use the FIRST 300 chars (choices and next question at the start)
            ctx_before = img.get('context_before', '')
            ctx_after = img.get('context_after', '')
            img_info = f"- {img['filename']} (page {img['page']})\n"
            img_info += f"  Text BEFORE image: \"...{ctx_before[-300:]}\"\n"
            img_info += f"  Text AFTER image: \"{ctx_after[:300]}...\""
            images_text.append(img_info)

        # Ask Claude to match images to questions
        prompt = f"""Match images to questions for Chapter {ch_num}.

TEXTBOOK STRUCTURE (critical for correct matching):
In this textbook, each question appears in this order:
1. Question text ending with the question number (e.g., "...needle placement? 2a")
2. IMAGE (if the question has one)
3. Answer choices (A, B, C, D)
4. Next question...

Therefore: The question number at the END of "Text BEFORE image" tells you which question the image belongs to.

QUESTIONS:
{chr(10).join(questions_text)}

IMAGES (with surrounding text context):
{chr(10).join(images_text)}

MATCHING RULES:
1. Find the LAST question number mentioned in "Text BEFORE image" - that's the question this image belongs to
2. Example: If text before ends with "...rotator interval approach? 2a" → image belongs to ch{ch_num}_2a
3. Example: If text before ends with "...mixture for the arthrogram? 2b" → image belongs to ch{ch_num}_2b
4. The text AFTER typically shows the answer choices, then the NEXT question
5. Each question has its OWN image - do NOT share images across questions
6. Decorative images or images not matching any question → assign to "(none)"

Return ONLY a JSON object mapping image filenames to question IDs:
{{
  "image_filename.jpeg": "ch{ch_num}_2a",
  "another_image.jpeg": "(none)"
}}"""

        try:
            response = client.messages.create(
                model=get_selected_model_id(),
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Extract JSON from response
            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            ch_assignments = json.loads(response_text)

            # Add to overall assignments (skip "(none)" assignments)
            for img_file, q_id in ch_assignments.items():
                if q_id and q_id != "(none)":
                    assignments[img_file] = q_id

        except Exception as e:
            # If LLM matching fails, fall back to simple matching for this chapter
            print(f"LLM matching failed for chapter {ch_num}: {e}")
            continue

    return assignments


def match_images_to_questions_simple(images: list[dict], chapters: list[dict], questions: dict) -> dict:
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

        # Simple assignment: first image to first question needing one, etc.
        for img, q in zip(ch_images, ch_questions):
            if q["full_id"] not in assignments.values():
                assignments[img["filename"]] = q["full_id"]

    return assignments


def get_questions_sharing_image(q_id: str, questions: dict) -> list[str]:
    """Get all question IDs that share the same image_group as the given question."""
    # Find the question and its image_group
    for ch_key, qs in questions.items():
        for q in qs:
            if q["full_id"] == q_id:
                image_group = q.get("image_group")
                if not image_group:
                    return [q_id]
                # Find all questions in same chapter with same image_group
                shared = [qq["full_id"] for qq in qs if qq.get("image_group") == image_group]
                return shared
    return [q_id]


def create_page_index(pages: list[dict]) -> str:
    """Create condensed index of pages for chapter identification."""
    index_parts = []
    for p in pages:
        preview = p["text"][:300].replace("\n", " ").strip()
        index_parts.append(f"[PAGE {p['page']}] {preview}")
    return "\n".join(index_parts)


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

    # Return cached models if available
    if _cached_models is not None:
        return _cached_models

    try:
        client = get_anthropic_client()
        if not client:
            _cached_models = {v: k for k, v in FALLBACK_MODELS.items()}
            return _cached_models

        # Fetch models from API
        import anthropic
        response = client.models.list(limit=100)

        models = {}
        for model in response.data:
            # Use display_name if available, otherwise format the ID
            display_name = getattr(model, 'display_name', None) or model.id
            models[display_name] = model.id

        if models:
            _cached_models = models
            return _cached_models

    except Exception as e:
        # Log error but don't fail - use fallback
        print(f"Failed to fetch models from API: {e}")

    # Fallback to static list
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


def identify_chapters_llm(client, pages: list[dict]) -> list[dict]:
    """Use Claude to identify chapter boundaries."""
    page_index = create_page_index(pages)

    response = client.messages.create(
        model=get_selected_model_id(),
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Analyze this textbook page index and identify all chapters that contain QUESTIONS sections.

This is a medical textbook where each chapter has:
- A chapter header (e.g., "Chapter 1: Title" or "1 Title")
- A QUESTIONS section with numbered questions
- An ANSWERS section with explanations

For each chapter with questions, provide:
- chapter_number: The chapter number (integer)
- title: The chapter title
- start_page: The page number where the chapter starts (from the [PAGE X] markers)
- has_questions: true (only include chapters that have questions)

Look for patterns like "QUESTIONS" or numbered questions (1., 2., etc.) to identify which chapters have Q&A content.

Return ONLY a JSON array, no other text:
[
  {{"chapter_number": 1, "title": "Chapter Title", "start_page": 14, "has_questions": true}},
  ...
]

PAGE INDEX:
{page_index}"""
        }]
    )

    response_text = response.content[0].text
    if "```" in response_text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    return json.loads(response_text)


def extract_chapter_text(pages: list[dict], start_page: int, end_page: Optional[int]) -> str:
    """Extract text for a specific chapter given page range."""
    chapter_pages = []
    for p in pages:
        if p["page"] >= start_page:
            if end_page is None or p["page"] < end_page:
                chapter_pages.append(f"[PAGE {p['page']}]\n{p['text']}")
    return "\n\n".join(chapter_pages)


def extract_qa_pairs_llm(client, chapter_num: int, chapter_text: str, model_id: str = None) -> dict:
    """Use Claude to extract Q&A pairs from a single chapter.

    Args:
        client: Anthropic client
        chapter_num: Chapter number
        chapter_text: Full text of the chapter
        model_id: Optional model ID override (for parallel execution)
    """
    # Use provided model_id or get from session state
    model = model_id or get_selected_model_id()

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": f"""You are analyzing Chapter {chapter_num} of a medical textbook to extract all questions and their corresponding answers.

TASK:
1. Find all questions in the QUESTIONS section
2. Find all answers in the ANSWERS section
3. Match each question to its answer
4. Identify the correct answer choice (A, B, C, D, or E)

IMPORTANT - MULTI-PART QUESTIONS:
Some questions follow a multi-part format:
- A "context question" (e.g., "1") contains a clinical scenario/case and possibly an image, but NO answer choices
- Sub-questions (e.g., "1a", "1b", "1c") contain the actual questions WITH answer choices

You MUST extract BOTH:
1. The context question (e.g., "1") - extract it with empty choices {{}}, empty correct_answer "", and empty explanation ""
2. All sub-questions (e.g., "1a", "1b", "1c") - extract normally with their choices, answers, and explanations

Do NOT skip context questions just because they have no answer choices. Extract them as-is.

ADDITIONAL RULES:
- Questions may have sub-parts like 2a, 2b, 2c - treat each as a separate question
- Question IDs should match exactly as they appear (e.g., "1", "1a", "1b", "2a", "2b", "3")
- For images: If a question references an image (e.g., "image below", "figure", "radiograph shown"), mark has_image: true
- Use image_group to indicate which questions share the same image (e.g., "1" for questions 1, 1a, 1b, 1c sharing one image)
- Extract the full question text and all answer choices (if present)

Return ONLY a JSON object in this exact format:
{{
  "chapter": {chapter_num},
  "questions": [
    {{
      "id": "1",
      "text": "Clinical scenario/context text here (no answer choices)",
      "choices": {{}},
      "has_image": true,
      "image_group": "1",
      "correct_answer": "",
      "explanation": ""
    }},
    {{
      "id": "1a",
      "text": "First sub-question text",
      "choices": {{
        "A": "Choice A text",
        "B": "Choice B text",
        "C": "Choice C text",
        "D": "Choice D text"
      }},
      "has_image": false,
      "image_group": "1",
      "correct_answer": "B",
      "explanation": "Brief explanation from the answer section"
    }}
  ]
}}

CHAPTER TEXT:
{chapter_text}"""
        }]
    )

    response_text = response.content[0].text
    if "```" in response_text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        return {"chapter": chapter_num, "questions": [], "error": str(e)}


def process_chapter_extraction(client, ch_num: int, ch_key: str, ch_text: str, model_id: str) -> tuple[str, list[dict]]:
    """
    Process a single chapter extraction (for parallel execution).

    Returns:
        Tuple of (ch_key, list of formatted questions)
    """
    result = extract_qa_pairs_llm(client, ch_num, ch_text, model_id)

    questions = []
    for q in result.get("questions", []):
        questions.append({
            "full_id": f"ch{ch_num}_{q['id']}",
            "local_id": q["id"],
            "text": q.get("text", ""),
            "choices": q.get("choices", {}),
            "has_image": q.get("has_image", False),
            "image_group": q.get("image_group"),
            "correct_answer": q.get("correct_answer", ""),
            "explanation": q.get("explanation", "")
        })

    return (ch_key, questions)


def postprocess_questions_llm(client, questions: dict, model_id: str = None) -> dict:
    """
    Post-process extracted questions to link context to sub-questions.

    This function:
    1. Identifies "context-only" questions (no choices, just setup text)
    2. Tags them with is_context_only: true so they're excluded from Anki
    3. Links context text to sub-questions that share the same image_group
    4. Adds inherited context as a "context" field in sub-questions

    Args:
        client: Anthropic client
        questions: Dict mapping ch_key -> list of question dicts
        model_id: Optional model ID override

    Returns:
        Updated questions dict with context linking applied
    """
    model = model_id or get_selected_model_id()

    # Process each chapter
    for ch_key, ch_questions in questions.items():
        if not ch_questions:
            continue

        # Build JSON representation for the LLM
        questions_json = json.dumps(ch_questions, indent=2)

        prompt = f"""Analyze this list of extracted questions and identify context relationships.

TASK:
1. Identify "context-only" entries - these have descriptive text but NO answer choices (empty choices dict)
2. For each context-only entry, find its related sub-questions by matching the image_group
3. Return the updated questions with context properly linked

RULES:
- A question is "context-only" if it has no choices (choices is empty {{}}) AND its ID is just a number (e.g., "5") not a letter suffix (e.g., "5a")
- Sub-questions share the same image_group as their context question
- Example: If question "5" has image_group="5" and no choices, and questions "5a", "5b", "5c" also have image_group="5", then 5a/5b/5c are sub-questions of context "5"

FOR EACH QUESTION, add these fields:
- "is_context_only": true/false - true if this is just context (no choices, no correct answer)
- "context": "" - for sub-questions, copy the context question's text here. Leave empty for standalone questions.
- "context_question_id": "" - for sub-questions, the local_id of their context question (e.g., "5"). Leave empty otherwise.

QUESTIONS TO PROCESS:
{questions_json}

Return ONLY the updated JSON array with all original fields preserved plus the new fields added."""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Extract JSON from response
            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            updated_questions = json.loads(response_text)
            questions[ch_key] = updated_questions

        except Exception as e:
            # If post-processing fails, add default fields to existing questions
            print(f"Post-processing failed for {ch_key}: {e}")
            for q in ch_questions:
                if "is_context_only" not in q:
                    # Heuristic: no choices and no letter suffix = context only
                    has_choices = bool(q.get("choices"))
                    has_letter = any(c.isalpha() for c in q.get("local_id", ""))
                    q["is_context_only"] = not has_choices and not has_letter
                if "context" not in q:
                    q["context"] = ""
                if "context_question_id" not in q:
                    q["context_question_id"] = ""

    return questions


def associate_context_llm(client, questions: dict, image_assignments: dict, model_id: str = None) -> tuple[dict, dict, dict]:
    """
    Use LLM to identify context relationships and merge context into sub-questions.

    This function:
    1. Sends questions to LLM to identify context-only questions and their sub-questions
    2. Prepends context text to sub-questions
    3. Copies image assignments from context questions to sub-questions
    4. Marks context-only questions for exclusion from Anki export

    Args:
        client: Anthropic client
        questions: Dict mapping ch_key -> list of question dicts
        image_assignments: Dict mapping image filename -> question full_id
        model_id: Optional model ID override

    Returns:
        Tuple of (updated_questions, updated_image_assignments, stats)
    """
    model = model_id or get_selected_model_id()

    if image_assignments is None:
        image_assignments = {}

    updated_assignments = dict(image_assignments)

    stats = {
        "context_questions_found": 0,
        "sub_questions_updated": 0,
        "images_copied": 0
    }

    # Process each chapter
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

        prompt = f"""Analyze these questions from a medical textbook and identify CONTEXT relationships.

CONTEXT PATTERN:
Some questions follow this pattern:
- A "context question" (e.g., ID "1") contains a clinical scenario but NO answer choices
- Sub-questions (e.g., "1a", "1b", "1c") contain the actual questions WITH answer choices
- The sub-questions all relate to the context question

YOUR TASK:
1. Identify which questions are "context-only" (have text but NO answer choices)
2. For each context question, identify which sub-questions belong to it
3. Sub-questions typically share the same base number (1a, 1b, 1c all belong to context 1)

QUESTIONS:
{json.dumps(questions_summary, indent=2)}

Return a JSON object with this structure:
{{
  "context_mappings": [
    {{
      "context_id": "ch1_1",
      "sub_question_ids": ["ch1_1a", "ch1_1b", "ch1_1c"]
    }}
  ]
}}

If there are no context relationships, return: {{"context_mappings": []}}

Return ONLY the JSON, no other text."""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Extract JSON from response
            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            result = json.loads(response_text)
            mappings = result.get("context_mappings", [])

            # Build lookup tables
            question_by_id = {q["full_id"]: q for q in ch_questions}

            # Apply the mappings
            for mapping in mappings:
                context_id = mapping.get("context_id")
                sub_ids = mapping.get("sub_question_ids", [])

                if not context_id or context_id not in question_by_id:
                    continue

                context_q = question_by_id[context_id]
                context_text = context_q.get("text", "").strip()

                # Mark context question
                context_q["is_context_only"] = True
                stats["context_questions_found"] += 1

                # Find images assigned to the context question
                context_images = [
                    img_file for img_file, assigned_to in image_assignments.items()
                    if assigned_to == context_id
                ]

                # Update each sub-question
                for sub_id in sub_ids:
                    if sub_id not in question_by_id:
                        continue

                    sub_q = question_by_id[sub_id]

                    # Skip if already merged
                    if sub_q.get("context_merged"):
                        continue

                    # Prepend context text
                    original_text = sub_q.get("text", "").strip()
                    sub_q["text"] = f"{context_text} {original_text}"
                    sub_q["context_merged"] = True
                    sub_q["context_from"] = context_id
                    sub_q["is_context_only"] = False
                    stats["sub_questions_updated"] += 1

                    # Copy image assignments
                    for img_file in context_images:
                        updated_assignments[img_file] = sub_id
                        stats["images_copied"] += 1

        except Exception as e:
            print(f"LLM context association failed for {ch_key}: {e}")
            # Continue to next chapter on error
            continue

    # Ensure all questions have is_context_only set
    for ch_key, ch_questions in questions.items():
        for q in ch_questions:
            if "is_context_only" not in q:
                q["is_context_only"] = False

    return questions, updated_assignments, stats


# =============================================================================
# State Management
# =============================================================================

def init_session_state():
    """Initialize session state variables and auto-load saved data."""
    # Track if this is a fresh initialization
    is_fresh_init = "initialized" not in st.session_state

    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "pages" not in st.session_state:
        st.session_state.pages = None
    if "chapters" not in st.session_state:
        st.session_state.chapters = None
    if "chapter_texts" not in st.session_state:
        st.session_state.chapter_texts = {}
    if "questions" not in st.session_state:
        st.session_state.questions = {}
    if "questions_merged" not in st.session_state:
        st.session_state.questions_merged = {}
    if "images" not in st.session_state:
        st.session_state.images = []
    if "image_assignments" not in st.session_state:
        st.session_state.image_assignments = {}
    if "image_assignments_merged" not in st.session_state:
        st.session_state.image_assignments_merged = {}
    if "qc_progress" not in st.session_state:
        st.session_state.qc_progress = {"reviewed": {}, "corrections": {}, "metadata": {}}
    if "current_step" not in st.session_state:
        st.session_state.current_step = "source"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL_NAME
    if "qc_selected_idx" not in st.session_state:
        st.session_state.qc_selected_idx = 0

    # Auto-load saved data on first initialization is deferred until PDF is selected
    if is_fresh_init:
        st.session_state.initialized = True


def load_qc_progress() -> dict:
    """Load QC progress from file."""
    qc_file = get_qc_progress_file()
    if os.path.exists(qc_file):
        with open(qc_file) as f:
            return json.load(f)
    return {"reviewed": {}, "corrections": {}, "metadata": {}}


def save_qc_progress():
    """Save QC progress to file."""
    st.session_state.qc_progress["metadata"]["last_saved"] = datetime.now().isoformat()
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_qc_progress_file(), "w") as f:
        json.dump(st.session_state.qc_progress, f, indent=2)


def save_chapters():
    """Save chapters to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_chapters_file(), "w") as f:
        json.dump(st.session_state.chapters, f, indent=2)
    with open(get_chapter_text_file(), "w") as f:
        json.dump(st.session_state.chapter_texts, f, indent=2)


def save_questions():
    """Save questions to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_questions_file(), "w") as f:
        json.dump(st.session_state.questions, f, indent=2)


def save_images():
    """Save image metadata to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_images_file(), "w") as f:
        json.dump(st.session_state.images, f, indent=2)


def save_pages():
    """Save raw PDF pages to file."""
    if st.session_state.pages:
        os.makedirs(get_output_dir(), exist_ok=True)
        with open(get_pages_file(), "w") as f:
            json.dump(st.session_state.pages, f, indent=2)


def save_image_assignments():
    """Save image-to-question assignments to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_image_assignments_file(), "w") as f:
        json.dump(st.session_state.image_assignments, f, indent=2)


def save_questions_merged():
    """Save merged questions (with context associated) to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_questions_merged_file(), "w") as f:
        json.dump(st.session_state.questions_merged, f, indent=2)


def save_image_assignments_merged():
    """Save merged image-to-question assignments to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_image_assignments_merged_file(), "w") as f:
        json.dump(st.session_state.image_assignments_merged, f, indent=2)


def save_settings():
    """Save user settings to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    settings = {
        "selected_model": st.session_state.selected_model,
        "current_step": st.session_state.current_step,
        "qc_selected_idx": st.session_state.qc_selected_idx,
        "current_pdf": st.session_state.get("current_pdf", ""),
        "last_saved": datetime.now().isoformat()
    }
    with open(get_settings_file(), "w") as f:
        json.dump(settings, f, indent=2)


def load_settings():
    """Load user settings from file."""
    settings_file = get_settings_file()
    if os.path.exists(settings_file):
        with open(settings_file) as f:
            settings = json.load(f)
            if "selected_model" in settings:
                st.session_state.selected_model = settings["selected_model"]
            if "current_step" in settings:
                st.session_state.current_step = settings["current_step"]
            if "qc_selected_idx" in settings:
                st.session_state.qc_selected_idx = settings["qc_selected_idx"]


def load_saved_data():
    """Load previously saved data if available."""
    pages_file = get_pages_file()
    chapters_file = get_chapters_file()
    chapter_text_file = get_chapter_text_file()
    questions_file = get_questions_file()
    questions_merged_file = get_questions_merged_file()
    images_file = get_images_file()
    assignments_file = get_image_assignments_file()
    assignments_merged_file = get_image_assignments_merged_file()

    if os.path.exists(pages_file):
        with open(pages_file) as f:
            st.session_state.pages = json.load(f)
    if os.path.exists(chapters_file):
        with open(chapters_file) as f:
            st.session_state.chapters = json.load(f)
    if os.path.exists(chapter_text_file):
        with open(chapter_text_file) as f:
            st.session_state.chapter_texts = json.load(f)
    if os.path.exists(questions_file):
        with open(questions_file) as f:
            st.session_state.questions = json.load(f)
    if os.path.exists(questions_merged_file):
        with open(questions_merged_file) as f:
            st.session_state.questions_merged = json.load(f)
    if os.path.exists(images_file):
        with open(images_file) as f:
            st.session_state.images = json.load(f)
    if os.path.exists(assignments_file):
        with open(assignments_file) as f:
            st.session_state.image_assignments = json.load(f)
    if os.path.exists(assignments_merged_file):
        with open(assignments_merged_file) as f:
            st.session_state.image_assignments_merged = json.load(f)

    # Assign chapters to images if chapters exist but images don't have chapter info
    if st.session_state.chapters and st.session_state.images:
        needs_chapter_update = any(
            img.get("chapter") is None for img in st.session_state.images
        )
        if needs_chapter_update:
            st.session_state.images = assign_chapters_to_images(
                st.session_state.images, st.session_state.chapters
            )
            save_images()


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.title("📚 Textbook Q&A Extractor")

    # Show current textbook
    if st.session_state.current_pdf:
        textbook_name = get_pdf_slug(st.session_state.current_pdf).replace('_', ' ')
        st.sidebar.caption(f"Working on: **{textbook_name}**")

    st.sidebar.markdown("---")

    steps = [
        ("source", "1. Select Source"),
        ("chapters", "2. Extract Chapters"),
        ("questions", "3. Extract Questions"),
        ("context", "4. Associate Context"),
        ("qc", "5. QC Questions"),
        ("export", "6. Export")
    ]

    for step_id, step_name in steps:
        if st.sidebar.button(step_name, key=f"nav_{step_id}"):
            st.session_state.current_step = step_id
            save_settings()

    st.sidebar.markdown("---")

    # Status summary
    st.sidebar.subheader("Status")
    if st.session_state.chapters:
        st.sidebar.success(f"Chapters: {len(st.session_state.chapters)}")
    else:
        st.sidebar.info("Chapters: Not extracted")

    q_count = sum(len(qs) for qs in st.session_state.questions.values())
    if q_count > 0:
        st.sidebar.success(f"Questions: {q_count}")
    else:
        st.sidebar.info("Questions: Not extracted")

    img_count = len(st.session_state.images)
    if img_count > 0:
        assigned = len(st.session_state.image_assignments)
        st.sidebar.success(f"Images: {img_count} ({assigned} assigned)")
    else:
        st.sidebar.info("Images: Not extracted")

    merged_count = sum(len(qs) for qs in st.session_state.questions_merged.values())
    if merged_count > 0:
        st.sidebar.success(f"Context: Associated")
    else:
        st.sidebar.info("Context: Not associated")

    reviewed = len(st.session_state.qc_progress.get("reviewed", {}))
    if reviewed > 0:
        st.sidebar.success(f"QC'd: {reviewed}/{q_count}")

    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Settings")
    model_options = get_model_options()
    current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    selected = st.sidebar.selectbox(
        "Claude Model:",
        model_options,
        index=current_idx,
        key="model_selector"
    )
    if selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
        save_settings()


def get_selected_model_id() -> str:
    """Get the currently selected Claude model ID."""
    return get_model_id(st.session_state.selected_model)


def clear_session_data():
    """Clear all session data for switching to a new PDF."""
    st.session_state.pages = None
    st.session_state.chapters = None
    st.session_state.chapter_texts = {}
    st.session_state.questions = {}
    st.session_state.questions_merged = {}
    st.session_state.images = []
    st.session_state.image_assignments = {}
    st.session_state.image_assignments_merged = {}
    st.session_state.qc_progress = {"reviewed": {}, "corrections": {}, "metadata": {}}
    st.session_state.qc_selected_idx = 0


def render_source_step():
    """Render source PDF selection step."""
    st.header("Step 1: Select Source PDF")

    # Find available PDFs
    pdf_files = list(Path(SOURCE_DIR).glob("*.pdf")) if os.path.exists(SOURCE_DIR) else []

    if not pdf_files:
        st.warning(f"No PDF files found in '{SOURCE_DIR}/' directory. Please add a PDF file.")
        return

    pdf_options = [f.name for f in pdf_files]

    # Check for existing textbooks with saved data
    available_textbooks = get_available_textbooks()

    # Show existing textbooks section if any exist
    if available_textbooks:
        st.subheader("Load Existing Textbook")
        st.caption("These textbooks have saved progress:")

        textbook_col1, textbook_col2 = st.columns([3, 1])
        with textbook_col1:
            selected_textbook = st.selectbox(
                "Select saved textbook:",
                available_textbooks,
                format_func=lambda x: x.replace('_', ' '),
                key="textbook_selector"
            )
        with textbook_col2:
            if st.button("Load Textbook", type="primary"):
                # Find corresponding PDF
                for pdf_name in pdf_options:
                    if get_pdf_slug(pdf_name) == selected_textbook:
                        st.session_state.current_pdf = pdf_name
                        break
                else:
                    # No matching PDF found, use slug as placeholder
                    st.session_state.current_pdf = selected_textbook + ".pdf"

                clear_session_data()
                load_saved_data()
                load_settings()
                st.session_state.qc_progress = load_qc_progress()
                st.success(f"Loaded: {selected_textbook}")
                st.rerun()

        st.markdown("---")
        st.subheader("Start New Textbook")

    # PDF selection
    selected_pdf = st.selectbox("Select PDF file:", pdf_options)

    if selected_pdf:
        pdf_path = f"{SOURCE_DIR}/{selected_pdf}"
        output_slug = get_pdf_slug(selected_pdf)
        st.info(f"Selected: {pdf_path}")
        st.caption(f"Output folder: output/{output_slug}/")

        # Check if this PDF has existing data
        has_existing_data = output_slug in available_textbooks

        col1, col2 = st.columns(2)

        with col1:
            btn_label = "Load PDF (Fresh Start)" if has_existing_data else "Load PDF"
            if st.button(btn_label, type="primary"):
                # Set current PDF first so paths are correct
                st.session_state.current_pdf = selected_pdf
                clear_session_data()

                with st.spinner("Extracting text from PDF..."):
                    st.session_state.pages = extract_text_from_pdf(pdf_path)
                    st.session_state.pdf_path = pdf_path
                    save_pages()

                with st.spinner("Extracting images from PDF..."):
                    st.session_state.images = extract_images_from_pdf(pdf_path)
                    save_images()

                st.success(f"Loaded {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
                st.rerun()

        with col2:
            if has_existing_data:
                if st.button("Load Existing Progress"):
                    st.session_state.current_pdf = selected_pdf
                    clear_session_data()
                    load_saved_data()
                    load_settings()
                    st.session_state.qc_progress = load_qc_progress()
                    st.success("Loaded previous session data")
                    st.rerun()

        if st.session_state.pages:
            st.success(f"PDF loaded: {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
            st.markdown("**Next:** Go to 'Extract Chapters' to identify chapter boundaries.")


def render_chapters_step():
    """Render chapter extraction step."""
    st.header("Step 2: Extract Chapters")

    # Check if we have pages or existing chapters
    has_pages = st.session_state.pages is not None
    has_chapters = st.session_state.chapters is not None

    if not has_pages and not has_chapters:
        st.warning("Please load a PDF first (Step 1)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please configure your .env file.")
        return

    # Only show extraction controls if we have pages to extract from
    if has_pages:
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            # Inline model selector for this step
            model_options = get_model_options()
            current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
            selected_model = st.selectbox("Model:", model_options, index=current_idx, key="chapters_model")
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                save_settings()

        with col2:
            btn_label = "Re-extract Chapters" if has_chapters else "Extract Chapters"
            if st.button(btn_label, type="primary"):
                with st.spinner(f"Using {st.session_state.selected_model} to identify chapters..."):
                    chapters = identify_chapters_llm(client, st.session_state.pages)
                    st.session_state.chapters = chapters

                    # Extract text for each chapter
                    for i, ch in enumerate(chapters):
                        start_page = ch["start_page"]
                        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else None
                        ch_key = f"ch{ch['chapter_number']}"
                        st.session_state.chapter_texts[ch_key] = extract_chapter_text(
                            st.session_state.pages, start_page, end_page
                        )

                    save_chapters()

                    # Assign chapter numbers to images
                    if st.session_state.images:
                        st.session_state.images = assign_chapters_to_images(
                            st.session_state.images, chapters
                        )
                        save_images()

                st.success(f"Found {len(chapters)} chapters")
                st.rerun()
    elif has_chapters:
        st.info("Chapters already extracted. Go to Step 1 to reload PDF if you need to re-extract.")

    if st.session_state.chapters:
        st.subheader("Extracted Chapters")

        # Chapter list
        for ch in st.session_state.chapters:
            st.markdown(f"**Chapter {ch['chapter_number']}:** {ch['title']} (page {ch['start_page']})")

        st.markdown("---")

        # Chapter preview
        st.subheader("Preview Chapter Text")

        chapter_options = [f"Ch{ch['chapter_number']}: {ch['title'][:40]}..."
                         for ch in st.session_state.chapters]
        selected_ch_idx = st.selectbox("Select chapter to preview:",
                                        range(len(chapter_options)),
                                        format_func=lambda x: chapter_options[x])

        if selected_ch_idx is not None:
            ch = st.session_state.chapters[selected_ch_idx]
            ch_key = f"ch{ch['chapter_number']}"
            ch_text = st.session_state.chapter_texts.get(ch_key, "")

            st.text_area("Chapter text:", ch_text, height=400, disabled=True)
            st.caption(f"Text length: {len(ch_text):,} characters")


def render_questions_step():
    """Render question extraction step."""
    st.header("Step 3: Extract Questions")

    if not st.session_state.chapters:
        st.warning("Please extract chapters first (Step 2)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please configure your .env file.")
        return

    # Chapter selection
    chapter_options = [f"Ch{ch['chapter_number']}: {ch['title'][:40]}..."
                      for ch in st.session_state.chapters]
    selected_ch_idx = st.selectbox("Select chapter:",
                                    range(len(chapter_options)),
                                    format_func=lambda x: chapter_options[x])

    if selected_ch_idx is not None:
        ch = st.session_state.chapters[selected_ch_idx]
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"

        # Model selector for question extraction
        model_col, btn_col1, btn_col2 = st.columns([2, 2, 2])

        with model_col:
            model_options = get_model_options()
            current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
            selected_model = st.selectbox("Model:", model_options, index=current_idx, key="questions_model")
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                save_settings()

        with btn_col1:
            if st.button(f"Extract Chapter {ch_num}", type="primary"):
                with st.spinner(f"Using {st.session_state.selected_model} to extract Q&A from Chapter {ch_num}..."):
                    ch_text = st.session_state.chapter_texts.get(ch_key, "")
                    result = extract_qa_pairs_llm(client, ch_num, ch_text)

                    # Convert to GUI format
                    questions = []
                    for q in result.get("questions", []):
                        questions.append({
                            "full_id": f"ch{ch_num}_{q['id']}",
                            "local_id": q["id"],
                            "text": q.get("text", ""),
                            "choices": q.get("choices", {}),
                            "has_image": q.get("has_image", False),
                            "image_group": q.get("image_group"),
                            "correct_answer": q.get("correct_answer", ""),
                            "explanation": q.get("explanation", "")
                        })

                    st.session_state.questions[ch_key] = questions
                    save_questions()

                    # Auto-match images for this chapter using LLM
                    if st.session_state.images and st.session_state.chapters:
                        with st.spinner("Matching images to questions..."):
                            st.session_state.image_assignments = match_images_to_questions_llm(
                                client,
                                st.session_state.images,
                                st.session_state.chapters,
                                st.session_state.questions
                            )
                            save_image_assignments()

                st.success(f"Extracted {len(questions)} questions from Chapter {ch_num}")
                st.rerun()

        with btn_col2:
            if st.button("Extract ALL Chapters"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Capture model ID before parallel execution (session state not thread-safe)
                model_id = get_selected_model_id()
                total_chapters = len(st.session_state.chapters)

                # Prepare chapter data for parallel processing
                chapter_tasks = []
                for ch in st.session_state.chapters:
                    ch_num = ch["chapter_number"]
                    ch_key = f"ch{ch_num}"
                    ch_text = st.session_state.chapter_texts.get(ch_key, "")
                    chapter_tasks.append((ch_num, ch_key, ch_text))

                status_text.text(f"Processing {total_chapters} chapters in parallel...")

                # Process chapters in parallel using ThreadPoolExecutor
                completed = 0
                with ThreadPoolExecutor(max_workers=min(total_chapters, 5)) as executor:
                    # Submit all chapter extraction tasks
                    future_to_chapter = {
                        executor.submit(
                            process_chapter_extraction,
                            client, ch_num, ch_key, ch_text, model_id
                        ): ch_key
                        for ch_num, ch_key, ch_text in chapter_tasks
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_chapter):
                        ch_key = future_to_chapter[future]
                        try:
                            result_key, questions = future.result()
                            st.session_state.questions[result_key] = questions
                            completed += 1
                            progress_bar.progress(completed / total_chapters)
                            status_text.text(f"Completed {completed}/{total_chapters} chapters...")
                        except Exception as e:
                            st.warning(f"Error processing {ch_key}: {e}")
                            completed += 1
                            progress_bar.progress(completed / total_chapters)

                save_questions()

                # Auto-match images to questions using LLM
                if st.session_state.images:
                    status_text.text("Matching images to questions (using Claude)...")
                    st.session_state.image_assignments = match_images_to_questions_llm(
                        client,
                        st.session_state.images,
                        st.session_state.chapters,
                        st.session_state.questions
                    )
                    save_image_assignments()

                status_text.text("Done!")
                st.success(f"Extracted questions from all {total_chapters} chapters")
                st.rerun()

        # Note about context association
        if st.session_state.questions:
            st.markdown("---")
            st.info("**Next step:** Go to **Step 4: Associate Context** to link context and images from parent questions to sub-questions.")

        # Preview extracted questions
        if ch_key in st.session_state.questions:
            st.markdown("---")
            st.subheader(f"Questions in Chapter {ch_num}")

            questions = st.session_state.questions[ch_key]

            # Count different types
            total = len(questions)
            context_only_count = sum(1 for q in questions if q.get("is_context_only"))
            merged_count = sum(1 for q in questions if q.get("context_merged"))
            actual_questions = total - context_only_count

            st.info(f"Total: {actual_questions} questions" +
                   (f" + {context_only_count} context-only" if context_only_count > 0 else "") +
                   (f" ({merged_count} with merged context)" if merged_count > 0 else ""))

            # Question list
            for q in questions:
                # Check if this question has directly assigned images
                q_images = [img for img in st.session_state.images
                           if st.session_state.image_assignments.get(img["filename"]) == q["full_id"]]

                # Build status indicators
                indicators = []

                # Context status
                if q.get("is_context_only"):
                    indicators.append("[CTX-ONLY]")
                elif q.get("context_merged"):
                    indicators.append("[+CTX]")

                # Image status
                if q_images:
                    indicators.append(f"[{len(q_images)} img]")
                elif q.get("has_image"):
                    indicators.append("[needs img]")

                indicator_str = " ".join(indicators)
                if indicator_str:
                    indicator_str = " " + indicator_str

                # Truncate text for display
                display_text = q['text'][:70] + "..." if len(q['text']) > 70 else q['text']

                with st.expander(f"Q{q['local_id']}{indicator_str}: {display_text}"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Show context-only warning
                        if q.get("is_context_only"):
                            st.warning("**CONTEXT ONLY** - This provides context for sub-questions and will NOT become an Anki card")

                        # Show merged context indicator
                        if q.get("context_merged"):
                            st.success(f"Context merged from Q{q.get('context_from', '?').split('_')[-1]}")

                        st.markdown(f"**Question:** {q['text']}")

                        if q.get("choices"):
                            st.markdown("**Choices:**")
                            for letter, choice in q.get("choices", {}).items():
                                st.markdown(f"- {letter}: {choice}")
                            st.markdown(f"**Correct Answer:** {q.get('correct_answer', 'N/A')}")
                            st.markdown(f"**Explanation:** {q.get('explanation', 'N/A')}")

                    with col2:
                        if q_images:
                            for img in q_images:
                                if os.path.exists(img["filepath"]):
                                    st.image(img["filepath"], caption=f"Page {img['page']}", width=200)
                        elif q.get("has_image"):
                            st.warning("Needs image assignment")


def question_sort_key(q_id: str) -> tuple:
    """Sort key for question IDs."""
    match = re.match(r'ch(\d+)_(\d+)([a-z]?)', q_id, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)), match.group(3) or "")
    return (999, 999, q_id)


def get_images_for_question(q_id: str) -> list[dict]:
    """
    Get all images for a question, including shared images from image_group.

    For multi-part questions (e.g., 5a, 5b, 5c sharing context "5"):
    - First checks for directly assigned images
    - Then checks if question belongs to an image_group
    - If in a group, also returns images assigned to the group's base question

    Example: If ch1_5a has image_group="5", and an image is assigned to ch1_5,
             then calling get_images_for_question("ch1_5a") returns that image.
    """
    images = []
    directly_assigned = set()

    # First, get directly assigned images
    for img in st.session_state.images:
        assigned_to = st.session_state.image_assignments.get(img["filename"])
        if assigned_to == q_id:
            images.append(img)
            directly_assigned.add(img["filename"])

    # If we found direct images, return them
    if images:
        return images

    # Otherwise, check for image_group inheritance
    # Find the question and its image_group
    for ch_key, qs in st.session_state.questions.items():
        for q in qs:
            if q["full_id"] == q_id:
                image_group = q.get("image_group")
                if not image_group:
                    return images  # No group, return whatever we have

                # Extract chapter prefix from q_id (e.g., "ch1" from "ch1_5a")
                ch_prefix = q_id.split("_")[0]

                # Build the base question ID from the image_group
                # e.g., if q_id="ch1_5a" and image_group="5", base_id="ch1_5"
                base_id = f"{ch_prefix}_{image_group}"

                # Look for images assigned to the base question
                for img in st.session_state.images:
                    assigned_to = st.session_state.image_assignments.get(img["filename"])
                    if assigned_to == base_id and img["filename"] not in directly_assigned:
                        images.append(img)

                return images

    return images


def get_all_question_options() -> list[str]:
    """Get list of all question IDs for reassignment dropdown."""
    options = ["(none)"]
    for ch_key in sorted(st.session_state.questions.keys(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 0):
        for q in st.session_state.questions[ch_key]:
            options.append(q["full_id"])
    return options


def render_context_step():
    """Render context association step."""
    st.header("Step 4: Associate Context")

    if not st.session_state.questions:
        st.warning("Please extract questions first (Step 3)")
        return

    st.markdown("""
    This step identifies **context-only questions** (clinical scenarios without answer choices)
    and associates their text and images with the related sub-questions.

    **Example:**
    - Q1 contains a clinical scenario and image (no answer choices)
    - Q1a, Q1b, Q1c are the actual questions with choices
    - After association, Q1's text is prepended to Q1a/Q1b/Q1c and images are linked
    """)

    # Count context-only questions (questions without choices)
    context_only_count = 0
    total_questions = 0
    for ch_key, ch_questions in st.session_state.questions.items():
        for q in ch_questions:
            total_questions += 1
            if not q.get("choices"):
                context_only_count += 1

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Questions", total_questions)
    with col2:
        st.metric("Potential Context Questions", context_only_count)

    st.markdown("---")

    # Show current merged state
    merged_count = sum(len(qs) for qs in st.session_state.questions_merged.values())
    if merged_count > 0:
        st.success(f"Context already associated. {merged_count} merged questions saved.")
        if st.button("View Merged Questions Preview"):
            st.subheader("Merged Questions Preview")
            for ch_key in sorted(st.session_state.questions_merged.keys()):
                with st.expander(f"Chapter {ch_key}", expanded=False):
                    for q in st.session_state.questions_merged[ch_key]:
                        is_context = q.get("is_context_only", False)
                        has_merged = q.get("context_merged", False)
                        status = "🔵 Context-only (excluded)" if is_context else ("🟢 Has merged context" if has_merged else "⚪ Regular")
                        st.markdown(f"**{q['full_id']}** {status}")
                        if has_merged:
                            st.caption(f"Context from: {q.get('context_from', 'N/A')}")
                        st.text(q.get("text", "")[:200] + "..." if len(q.get("text", "")) > 200 else q.get("text", ""))
                        st.markdown("---")

    st.markdown("---")

    # Model selector and Associate Context button
    col1, col2 = st.columns([2, 3])

    with col1:
        model_options = get_model_options()
        current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        selected_model = st.selectbox("Model:", model_options, index=current_idx, key="context_model")
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            save_settings()

    with col2:
        st.write("")  # Spacer for alignment
        if st.button("🔗 Associate Context", type="primary"):
            client = get_anthropic_client()
            if not client:
                st.error("ANTHROPIC_API_KEY not set. Please set the environment variable.")
            else:
                with st.spinner("Analyzing questions and associating context..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text(f"Using {st.session_state.selected_model} for context analysis...")
                    progress_bar.progress(20)

                    # Make a deep copy of questions to avoid modifying the original
                    questions_copy = copy.deepcopy(st.session_state.questions)
                    assignments_copy = copy.deepcopy(st.session_state.image_assignments)

                    # Get model ID for the selected model
                    model_id = get_model_id(st.session_state.selected_model)

                    updated_questions, updated_assignments, stats = associate_context_llm(
                        client,
                        questions_copy,
                        assignments_copy,
                        model_id=model_id
                    )

                    progress_bar.progress(80)
                    status_text.text("Saving merged data...")

                    # Save to merged state (separate from original)
                    st.session_state.questions_merged = updated_questions
                    st.session_state.image_assignments_merged = updated_assignments
                    save_questions_merged()
                    save_image_assignments_merged()

                    progress_bar.progress(100)
                    status_text.text("Done!")

                    st.success(
                        f"Context association complete!\n\n"
                        f"- Context questions found: {stats['context_questions_found']}\n"
                        f"- Sub-questions updated: {stats['sub_questions_updated']}\n"
                        f"- Images copied: {stats['images_copied']}"
                    )

                    st.rerun()

    # Option to clear merged data
    if merged_count > 0:
        st.markdown("---")
        if st.button("🗑️ Clear Merged Data", type="secondary"):
            st.session_state.questions_merged = {}
            st.session_state.image_assignments_merged = {}
            # Delete files if they exist
            merged_file = get_questions_merged_file()
            assignments_merged_file = get_image_assignments_merged_file()
            if os.path.exists(merged_file):
                os.remove(merged_file)
            if os.path.exists(assignments_merged_file):
                os.remove(assignments_merged_file)
            st.success("Merged data cleared. Original questions remain intact.")
            st.rerun()


def render_qc_step():
    """Render QC review step."""
    st.header("Step 5: QC Questions")

    if not st.session_state.questions:
        st.warning("Please extract questions first (Step 3)")
        return

    # Flatten all questions for QC
    all_questions = []
    for ch_key, questions in st.session_state.questions.items():
        for q in questions:
            all_questions.append((ch_key, q))

    all_questions.sort(key=lambda x: question_sort_key(x[1]["full_id"]))

    if not all_questions:
        st.warning("No questions to review")
        return

    # Progress
    reviewed = st.session_state.qc_progress.get("reviewed", {})
    total = len(all_questions)
    reviewed_count = len(reviewed)

    st.progress(reviewed_count / total if total > 0 else 0)
    st.caption(f"Progress: {reviewed_count}/{total} questions reviewed")

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_option = st.radio("Show:", ["All", "Unreviewed only", "Reviewed only"], horizontal=True)
    with col2:
        chapter_filter = st.selectbox("Filter by chapter:",
                                       ["All chapters"] + list(st.session_state.questions.keys()))
    with col3:
        hide_context = st.checkbox("Hide context-only entries", value=True)

    # Filter questions
    filtered_questions = []
    for ch_key, q in all_questions:
        q_id = q["full_id"]
        is_reviewed = q_id in reviewed

        # Apply filters
        if filter_option == "Unreviewed only" and is_reviewed:
            continue
        if filter_option == "Reviewed only" and not is_reviewed:
            continue
        if chapter_filter != "All chapters" and ch_key != chapter_filter:
            continue
        if hide_context and q.get("is_context_only"):
            continue

        filtered_questions.append((ch_key, q))

    st.caption(f"Showing {len(filtered_questions)} questions")

    # Question selector with session state for navigation
    if filtered_questions:
        def format_question_option(q):
            prefix = ""
            if q.get("is_context_only"):
                prefix = "[CTX] "
            elif q.get("context"):
                prefix = "[+ctx] "
            return f"{prefix}{q['full_id']}: {q['text'][:50]}..."

        question_options = [format_question_option(q) for _, q in filtered_questions]

        # Ensure selected index is within bounds
        if st.session_state.qc_selected_idx >= len(filtered_questions):
            st.session_state.qc_selected_idx = 0

        # Navigation buttons at the top
        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
        with nav_col1:
            if st.button("← Previous", disabled=(st.session_state.qc_selected_idx <= 0)):
                st.session_state.qc_selected_idx -= 1
                save_settings()
                st.rerun()
        with nav_col2:
            selected_idx = st.selectbox(
                "Select question:",
                range(len(question_options)),
                index=st.session_state.qc_selected_idx,
                format_func=lambda x: question_options[x],
                key="qc_question_selector"
            )
            if selected_idx != st.session_state.qc_selected_idx:
                st.session_state.qc_selected_idx = selected_idx
                save_settings()
        with nav_col3:
            if st.button("Next →", disabled=(st.session_state.qc_selected_idx >= len(filtered_questions) - 1)):
                st.session_state.qc_selected_idx += 1
                save_settings()
                st.rerun()

        selected_idx = st.session_state.qc_selected_idx
        if selected_idx is not None and selected_idx < len(filtered_questions):
            ch_key, q = filtered_questions[selected_idx]
            q_id = q["full_id"]

            st.markdown("---")

            # Two column layout: question on left, images on right
            left_col, right_col = st.columns([1, 1])

            with left_col:
                # Display question
                st.subheader(f"Question {q['local_id']}")

                # Check if this is a context-only question
                if q.get("is_context_only"):
                    st.warning("**CONTEXT ONLY** - This entry provides context for sub-questions and will not be exported to Anki.")
                    st.markdown(f"**Context Text:**\n\n{q['text']}")
                else:
                    # Show inherited context if this is a sub-question
                    if q.get("context"):
                        st.info(f"**Context (from Q{q.get('context_question_id', '?')}):**\n\n{q['context']}")

                    st.markdown(f"**Question:** {q['text']}")

                    if q.get("choices"):
                        st.markdown("**Choices:**")
                        for letter, choice in q.get("choices", {}).items():
                            if letter == q.get("correct_answer"):
                                st.markdown(f"- **{letter}: {choice}** ✓")
                            else:
                                st.markdown(f"- {letter}: {choice}")
                    else:
                        st.caption("No answer choices")

                    st.markdown(f"**Explanation:** {q.get('explanation', 'N/A')}")

                # Show current status
                if q_id in reviewed:
                    status = reviewed[q_id].get("status", "unknown")
                    if status == "approved":
                        st.success(f"Status: Approved")
                    elif status == "flagged":
                        st.warning(f"Status: Flagged")

            with right_col:
                # Display images assigned to this question
                assigned_images = get_images_for_question(q_id)

                # Get chapter page range for showing available images
                ch_num = int(ch_key[2:])
                ch_start = next((c["start_page"] for c in st.session_state.chapters if c["chapter_number"] == ch_num), 1)
                ch_end = 9999
                for i, c in enumerate(st.session_state.chapters):
                    if c["chapter_number"] == ch_num and i + 1 < len(st.session_state.chapters):
                        ch_end = st.session_state.chapters[i + 1]["start_page"]

                if assigned_images:
                    st.subheader("Assigned Image(s)")
                    for img in assigned_images:
                        filepath = img["filepath"]
                        if os.path.exists(filepath):
                            st.image(filepath, caption=f"Page {img['page']} - {img['filename']}", use_column_width=True)

                            # Image actions
                            img_col1, img_col2 = st.columns(2)
                            with img_col1:
                                if st.button("✓ Image Correct", key=f"img_ok_{img['filename']}", type="primary"):
                                    st.success("Image confirmed!")
                            with img_col2:
                                if st.button("✗ Remove Image", key=f"img_remove_{img['filename']}"):
                                    st.session_state.image_assignments.pop(img["filename"], None)
                                    save_image_assignments()
                                    st.rerun()
                        else:
                            st.warning(f"Image not found: {filepath}")

                    # Option to add more images
                    with st.expander("Assign different/additional image"):
                        unassigned = [img for img in st.session_state.images
                                      if img["filename"] not in st.session_state.image_assignments
                                      and ch_start <= img["page"] < ch_end]
                        if unassigned:
                            for img in unassigned[:5]:
                                filepath = img["filepath"]
                                if os.path.exists(filepath):
                                    st.image(filepath, caption=f"Page {img['page']}", width=150)
                                    if st.button(f"Add this image", key=f"add_{img['filename']}"):
                                        st.session_state.image_assignments[img["filename"]] = q_id
                                        save_image_assignments()
                                        st.rerun()
                        else:
                            st.caption("No more unassigned images in this chapter")

                else:
                    # No image currently assigned - show option to add one
                    if q.get("has_image"):
                        st.subheader("Image Required")
                        st.warning("This question needs an image but none assigned yet.")
                    else:
                        st.subheader("No Image Assigned")
                        st.caption("No image currently linked to this question")

                    # Show unassigned images from same chapter for manual linking
                    with st.expander("Manually assign an image" if not q.get("has_image") else "Select from chapter images", expanded=q.get("has_image", False)):
                        unassigned = [img for img in st.session_state.images
                                      if img["filename"] not in st.session_state.image_assignments
                                      and ch_start <= img["page"] < ch_end]

                        if unassigned:
                            for img in unassigned[:6]:
                                filepath = img["filepath"]
                                if os.path.exists(filepath):
                                    st.image(filepath, caption=f"Page {img['page']}", width=180)
                                    if st.button(f"Assign", key=f"assign_{img['filename']}"):
                                        st.session_state.image_assignments[img["filename"]] = q_id
                                        save_image_assignments()
                                        st.success("Assigned!")
                                        st.rerun()
                        else:
                            st.caption("No unassigned images in this chapter")

            # QC actions
            st.markdown("---")

            # Check if currently approved
            is_approved = q_id in reviewed and reviewed[q_id].get("status") == "approved"
            is_flagged = q_id in reviewed and reviewed[q_id].get("status") == "flagged"

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Approve & Next button (auto-advances)
                if st.button("✓ Approve & Next", type="primary", disabled=is_approved):
                    reviewed[q_id] = {"status": "approved", "timestamp": datetime.now().isoformat()}
                    st.session_state.qc_progress["reviewed"] = reviewed
                    save_qc_progress()
                    # Auto-advance to next question
                    if st.session_state.qc_selected_idx < len(filtered_questions) - 1:
                        st.session_state.qc_selected_idx += 1
                        save_settings()
                    st.rerun()

            with col2:
                if st.button("✗ Flag Issue", disabled=is_flagged):
                    reviewed[q_id] = {"status": "flagged", "timestamp": datetime.now().isoformat()}
                    st.session_state.qc_progress["reviewed"] = reviewed
                    save_qc_progress()
                    st.rerun()

            with col3:
                # Unapprove button (only shown if already reviewed)
                if is_approved or is_flagged:
                    if st.button("↩ Unapprove"):
                        reviewed.pop(q_id, None)
                        st.session_state.qc_progress["reviewed"] = reviewed
                        save_qc_progress()
                        st.rerun()

            with col4:
                # Show current status
                if is_approved:
                    st.success("✓ Approved")
                elif is_flagged:
                    st.warning("✗ Flagged")


def render_export_step():
    """Render export step."""
    st.header("Step 6: Export to Anki")

    if not st.session_state.questions:
        st.warning("Please extract questions first (Step 3)")
        return

    # Count questions
    total = sum(len(qs) for qs in st.session_state.questions.values())
    reviewed = st.session_state.qc_progress.get("reviewed", {})
    approved = sum(1 for r in reviewed.values() if r.get("status") == "approved")

    st.info(f"Total questions: {total}")
    st.info(f"Approved (QC'd): {approved}")

    st.markdown("---")

    export_option = st.radio("Export:", [
        "All questions",
        "Only approved (QC'd) questions"
    ])

    if st.button("Export to Anki Deck", type="primary"):
        st.warning("Anki export functionality coming soon!")
        st.markdown("""
        **Planned features:**
        - Generate .apkg file for direct Anki import
        - Include question, choices, correct answer, and explanation
        - Tag cards by chapter
        - Optional: include associated images
        """)


# =============================================================================
# Main
# =============================================================================

def main():
    init_session_state()
    render_sidebar()

    step = st.session_state.current_step

    if step == "source":
        render_source_step()
    elif step == "chapters":
        render_chapters_step()
    elif step == "questions":
        render_questions_step()
    elif step == "context":
        render_context_step()
    elif step == "qc":
        render_qc_step()
    elif step == "export":
        render_export_step()


if __name__ == "__main__":
    main()
