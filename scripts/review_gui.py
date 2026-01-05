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
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
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
OUTPUT_DIR = "output"
IMAGES_DIR = "output/images"

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

# Files
CHAPTERS_FILE = f"{OUTPUT_DIR}/chapters.json"
CHAPTER_TEXT_FILE = f"{OUTPUT_DIR}/chapter_text.json"
QUESTIONS_FILE = f"{OUTPUT_DIR}/questions_by_chapter.json"
IMAGES_FILE = f"{OUTPUT_DIR}/images.json"
IMAGE_ASSIGNMENTS_FILE = f"{OUTPUT_DIR}/image_assignments.json"
QC_PROGRESS_FILE = f"{OUTPUT_DIR}/qc_progress.json"
SETTINGS_FILE = f"{OUTPUT_DIR}/settings.json"


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


def extract_images_from_pdf(pdf_path: str, output_dir: str = IMAGES_DIR) -> list[dict]:
    """
    Extract images from PDF with page numbers, positions, and flanking text context.
    Returns list of image metadata including text before/after each image.

    Flanking text is extracted across page boundaries - if an image is at the top
    of a page, context_before will include text from the bottom of the previous page.
    """
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


def extract_qa_pairs_llm(client, chapter_num: int, chapter_text: str) -> dict:
    """Use Claude to extract Q&A pairs from a single chapter."""
    response = client.messages.create(
        model=get_selected_model_id(),
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": f"""You are analyzing Chapter {chapter_num} of a medical textbook to extract all questions and their corresponding answers.

TASK:
1. Find all questions in the QUESTIONS section
2. Find all answers in the ANSWERS section
3. Match each question to its answer
4. Identify the correct answer choice (A, B, C, D, or E)

IMPORTANT:
- Questions may have sub-parts like 2a, 2b, 2c - treat each as a separate question
- Question IDs should match exactly as they appear in the ANSWERS section (e.g., "1", "2a", "2b", "3")
- For images: If a question or ANY of its sub-parts reference an image (e.g., "image below", "figure", "radiograph shown"), mark ALL related sub-questions with has_image: true. For example, if questions 2a, 2b, 2c all refer to the same image shown before 2a, then 2a, 2b, AND 2c should ALL have has_image: true
- Use image_group to indicate which questions share the same image (e.g., "2" for 2a, 2b, 2c sharing one image)
- Extract the full question text and all answer choices

Return ONLY a JSON object in this exact format:
{{
  "chapter": {chapter_num},
  "questions": [
    {{
      "id": "1",
      "text": "Full question text here",
      "choices": {{
        "A": "Choice A text",
        "B": "Choice B text",
        "C": "Choice C text",
        "D": "Choice D text"
      }},
      "has_image": true,
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


# =============================================================================
# State Management
# =============================================================================

def init_session_state():
    """Initialize session state variables and auto-load saved data."""
    # Track if this is a fresh initialization
    is_fresh_init = "initialized" not in st.session_state

    if "pages" not in st.session_state:
        st.session_state.pages = None
    if "chapters" not in st.session_state:
        st.session_state.chapters = None
    if "chapter_texts" not in st.session_state:
        st.session_state.chapter_texts = {}
    if "questions" not in st.session_state:
        st.session_state.questions = {}
    if "images" not in st.session_state:
        st.session_state.images = []
    if "image_assignments" not in st.session_state:
        st.session_state.image_assignments = {}
    if "qc_progress" not in st.session_state:
        st.session_state.qc_progress = load_qc_progress()
    if "current_step" not in st.session_state:
        st.session_state.current_step = "source"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL_NAME
    if "qc_selected_idx" not in st.session_state:
        st.session_state.qc_selected_idx = 0

    # Auto-load saved data on first initialization
    if is_fresh_init:
        st.session_state.initialized = True
        load_saved_data()
        load_settings()


def load_qc_progress() -> dict:
    """Load QC progress from file."""
    if os.path.exists(QC_PROGRESS_FILE):
        with open(QC_PROGRESS_FILE) as f:
            return json.load(f)
    return {"reviewed": {}, "corrections": {}, "metadata": {}}


def save_qc_progress():
    """Save QC progress to file."""
    st.session_state.qc_progress["metadata"]["last_saved"] = datetime.now().isoformat()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(QC_PROGRESS_FILE, "w") as f:
        json.dump(st.session_state.qc_progress, f, indent=2)


def save_chapters():
    """Save chapters to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CHAPTERS_FILE, "w") as f:
        json.dump(st.session_state.chapters, f, indent=2)
    with open(CHAPTER_TEXT_FILE, "w") as f:
        json.dump(st.session_state.chapter_texts, f, indent=2)


def save_questions():
    """Save questions to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(st.session_state.questions, f, indent=2)


def save_images():
    """Save image metadata to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(IMAGES_FILE, "w") as f:
        json.dump(st.session_state.images, f, indent=2)


def save_image_assignments():
    """Save image-to-question assignments to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(IMAGE_ASSIGNMENTS_FILE, "w") as f:
        json.dump(st.session_state.image_assignments, f, indent=2)


def save_settings():
    """Save user settings to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    settings = {
        "selected_model": st.session_state.selected_model,
        "current_step": st.session_state.current_step,
        "qc_selected_idx": st.session_state.qc_selected_idx,
        "last_saved": datetime.now().isoformat()
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def load_settings():
    """Load user settings from file."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)
            if "selected_model" in settings:
                st.session_state.selected_model = settings["selected_model"]
            if "current_step" in settings:
                st.session_state.current_step = settings["current_step"]
            if "qc_selected_idx" in settings:
                st.session_state.qc_selected_idx = settings["qc_selected_idx"]


def load_saved_data():
    """Load previously saved data if available."""
    if os.path.exists(CHAPTERS_FILE):
        with open(CHAPTERS_FILE) as f:
            st.session_state.chapters = json.load(f)
    if os.path.exists(CHAPTER_TEXT_FILE):
        with open(CHAPTER_TEXT_FILE) as f:
            st.session_state.chapter_texts = json.load(f)
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE) as f:
            st.session_state.questions = json.load(f)
    if os.path.exists(IMAGES_FILE):
        with open(IMAGES_FILE) as f:
            st.session_state.images = json.load(f)
    if os.path.exists(IMAGE_ASSIGNMENTS_FILE):
        with open(IMAGE_ASSIGNMENTS_FILE) as f:
            st.session_state.image_assignments = json.load(f)

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
    st.sidebar.markdown("---")

    steps = [
        ("source", "1. Select Source"),
        ("chapters", "2. Extract Chapters"),
        ("questions", "3. Extract Questions"),
        ("qc", "4. QC Questions"),
        ("export", "5. Export")
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


def render_source_step():
    """Render source PDF selection step."""
    st.header("Step 1: Select Source PDF")

    # Find available PDFs
    pdf_files = list(Path(SOURCE_DIR).glob("*.pdf")) if os.path.exists(SOURCE_DIR) else []

    if not pdf_files:
        st.warning(f"No PDF files found in '{SOURCE_DIR}/' directory. Please add a PDF file.")
        return

    pdf_options = [f.name for f in pdf_files]
    selected_pdf = st.selectbox("Select PDF file:", pdf_options)

    if selected_pdf:
        pdf_path = f"{SOURCE_DIR}/{selected_pdf}"
        st.info(f"Selected: {pdf_path}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Load PDF", type="primary"):
                with st.spinner("Extracting text from PDF..."):
                    st.session_state.pages = extract_text_from_pdf(pdf_path)
                    st.session_state.pdf_path = pdf_path

                with st.spinner("Extracting images from PDF..."):
                    st.session_state.images = extract_images_from_pdf(pdf_path)
                    save_images()

                st.success(f"Loaded {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
                st.rerun()

        with col2:
            if st.button("Load Previous Session"):
                load_saved_data()
                st.success("Loaded previous session data")
                st.rerun()

        if st.session_state.pages:
            st.success(f"PDF loaded: {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
            st.markdown("**Next:** Go to 'Extract Chapters' to identify chapter boundaries.")


def render_chapters_step():
    """Render chapter extraction step."""
    st.header("Step 2: Extract Chapters")

    if not st.session_state.pages:
        st.warning("Please load a PDF first (Step 1)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please configure your .env file.")
        return

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
        if st.button("Extract Chapters", type="primary"):
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

                for i, ch in enumerate(st.session_state.chapters):
                    ch_num = ch["chapter_number"]
                    ch_key = f"ch{ch_num}"
                    status_text.text(f"Processing Chapter {ch_num}...")

                    ch_text = st.session_state.chapter_texts.get(ch_key, "")
                    result = extract_qa_pairs_llm(client, ch_num, ch_text)

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
                    progress_bar.progress((i + 1) / len(st.session_state.chapters))

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
                st.success(f"Extracted questions from all {len(st.session_state.chapters)} chapters")
                st.rerun()

        # Preview extracted questions
        if ch_key in st.session_state.questions:
            st.markdown("---")
            st.subheader(f"Questions in Chapter {ch_num}")

            questions = st.session_state.questions[ch_key]
            st.info(f"Total: {len(questions)} questions")

            # Question list
            for q in questions:
                # Check if this question has directly assigned images
                q_images = [img for img in st.session_state.images
                           if st.session_state.image_assignments.get(img["filename"]) == q["full_id"]]

                # Show indicator: 📷(n) if has images, 📷? if needs but none assigned
                if q_images:
                    img_indicator = f" [{len(q_images)} img]"
                elif q.get("has_image"):
                    img_indicator = " [needs img]"
                else:
                    img_indicator = ""

                with st.expander(f"Q{q['local_id']}{img_indicator}: {q['text'][:70]}..."):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Question:** {q['text']}")
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
    """Get all images directly assigned to a specific question."""
    # Return only images explicitly assigned to this question ID
    # Do NOT share images across questions with the same image_group,
    # as each question typically has its own distinct image
    images = []
    for img in st.session_state.images:
        assigned_to = st.session_state.image_assignments.get(img["filename"])
        if assigned_to == q_id:
            images.append(img)
    return images


def get_all_question_options() -> list[str]:
    """Get list of all question IDs for reassignment dropdown."""
    options = ["(none)"]
    for ch_key in sorted(st.session_state.questions.keys(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 0):
        for q in st.session_state.questions[ch_key]:
            options.append(q["full_id"])
    return options


def render_qc_step():
    """Render QC review step."""
    st.header("Step 4: QC Questions")

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
    col1, col2 = st.columns(2)
    with col1:
        filter_option = st.radio("Show:", ["All", "Unreviewed only", "Reviewed only"], horizontal=True)
    with col2:
        chapter_filter = st.selectbox("Filter by chapter:",
                                       ["All chapters"] + list(st.session_state.questions.keys()))

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

        filtered_questions.append((ch_key, q))

    st.caption(f"Showing {len(filtered_questions)} questions")

    # Question selector with session state for navigation
    if filtered_questions:
        question_options = [f"{q['full_id']}: {q['text'][:50]}..." for _, q in filtered_questions]

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
                st.markdown(f"**{q['text']}**")

                st.markdown("**Choices:**")
                for letter, choice in q.get("choices", {}).items():
                    if letter == q.get("correct_answer"):
                        st.markdown(f"- **{letter}: {choice}** ✓")
                    else:
                        st.markdown(f"- {letter}: {choice}")

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
    st.header("Step 5: Export to Anki")

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
    elif step == "qc":
        render_qc_step()
    elif step == "export":
        render_export_step()


if __name__ == "__main__":
    main()
