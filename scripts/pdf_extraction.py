"""
PDF Extraction Module

Contains functions for extracting text and images from PDF files.
"""

import os
from typing import Optional

import fitz  # PyMuPDF


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


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> list[dict]:
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


def create_page_index(pages: list[dict]) -> str:
    """Create condensed index of pages for chapter identification."""
    index_parts = []
    for p in pages:
        preview = p["text"][:300].replace("\n", " ").strip()
        index_parts.append(f"[PAGE {p['page']}] {preview}")
    return "\n".join(index_parts)


def extract_chapter_text(pages: list[dict], start_page: int, end_page: Optional[int]) -> str:
    """Extract text for a specific chapter given page range."""
    chapter_pages = []
    for p in pages:
        if p["page"] >= start_page:
            if end_page is None or p["page"] < end_page:
                chapter_pages.append(f"[PAGE {p['page']}]\n{p['text']}")
    return "\n\n".join(chapter_pages)


def get_questions_sharing_image(q_id: str, questions: dict) -> list[str]:
    """Get all question IDs that share the same image_group as the given question."""
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


def find_page_for_text(search_text: str, pages: list[dict], start_page: int = 1, end_page: Optional[int] = None) -> Optional[int]:
    """
    Search pages for text and return the page number where it's found.

    Args:
        search_text: Text to search for (first 80 chars used for matching)
        pages: List of page dicts with 'page' and 'text' keys
        start_page: Minimum page number to search (inclusive)
        end_page: Maximum page number to search (exclusive), None for no limit

    Returns:
        Page number (1-indexed) where text was found, or None if not found
    """
    if not search_text or not pages:
        return None

    # Normalize search text - take first 80 chars, lowercase, collapse whitespace
    search_normalized = " ".join(search_text[:80].lower().split())

    for page_dict in pages:
        page_num = page_dict["page"]

        # Check page range
        if page_num < start_page:
            continue
        if end_page is not None and page_num >= end_page:
            continue

        # Normalize page text for comparison
        page_text_normalized = " ".join(page_dict["text"].lower().split())

        if search_normalized in page_text_normalized:
            return page_num

    return None


def find_pages_for_text(search_text: str, pages: list[dict], start_page: int = 1, end_page: Optional[int] = None) -> list[int]:
    """
    Search pages for text and return all page numbers where it appears.
    Detects multi-page content by checking both the start and end of the text.

    Args:
        search_text: Full text to search for
        pages: List of page dicts with 'page' and 'text' keys
        start_page: Minimum page number to search (inclusive)
        end_page: Maximum page number to search (exclusive), None for no limit

    Returns:
        List of page numbers (1-indexed) where text appears, or empty list if not found
    """
    if not search_text or not pages:
        return []

    # Find page where text starts (first 80 chars)
    start_text = " ".join(search_text[:80].lower().split())
    # Find page where text ends (last 80 chars)
    end_text = " ".join(search_text[-80:].lower().split()) if len(search_text) > 80 else start_text

    first_page = None
    last_page = None

    for page_dict in pages:
        page_num = page_dict["page"]

        # Check page range
        if page_num < start_page:
            continue
        if end_page is not None and page_num >= end_page:
            continue

        page_text_normalized = " ".join(page_dict["text"].lower().split())

        # Check for start of text
        if first_page is None and start_text in page_text_normalized:
            first_page = page_num

        # Check for end of text
        if end_text in page_text_normalized:
            last_page = page_num

    if first_page is None:
        return []

    # If we found both start and end, return all pages in range
    if last_page is None:
        last_page = first_page

    return list(range(first_page, last_page + 1))


def detect_question_pages(questions: list[dict], pages: list[dict], chapter_start: int, chapter_end: Optional[int] = None) -> list[dict]:
    """
    Detect page numbers for questions by searching pages.json.

    Adds 'question_pages' and 'answer_pages' fields (lists) to each question.
    Also adds legacy 'question_page' and 'answer_page' (first page) for compatibility.

    Args:
        questions: List of question dicts for a single chapter
        pages: Full pages list from pages.json
        chapter_start: Start page of the chapter
        chapter_end: End page of the chapter (exclusive)

    Returns:
        Updated questions list with page numbers added
    """
    for q in questions:
        q_text = q.get("text", "")
        choices = q.get("choices", {})

        # Find where question text starts
        first_page = find_page_for_text(q_text, pages, chapter_start, chapter_end)

        # Find where last choice ends (to detect multi-page questions)
        last_page = first_page
        if choices:
            last_choice_letter = sorted(choices.keys())[-1]  # e.g., 'D' or 'E'
            last_choice_text = choices[last_choice_letter]
            # Search for the last choice text to find where question ends
            choice_page = find_page_for_text(last_choice_text, pages, chapter_start, chapter_end)
            if choice_page and first_page:
                last_page = max(first_page, choice_page)

        # Build page range
        if first_page and last_page:
            q_pages = list(range(first_page, last_page + 1))
        elif first_page:
            q_pages = [first_page]
        else:
            q_pages = []

        q["question_pages"] = q_pages
        q["question_page"] = q_pages[0] if q_pages else None

        # Find pages for answer/explanation text
        explanation = q.get("explanation", "")
        a_pages = find_pages_for_text(explanation, pages, chapter_start, chapter_end)
        q["answer_pages"] = a_pages
        q["answer_page"] = a_pages[0] if a_pages else None

    return questions


def render_pdf_page(pdf_path: str, page_num: int, zoom: float = 1.5) -> Optional[bytes]:
    """
    Render a PDF page as PNG bytes for display in Streamlit.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (1-indexed)
        zoom: Zoom factor for rendering (1.5 = 150% size)

    Returns:
        PNG image bytes, or None if rendering fails
    """
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None

        page = doc[page_num - 1]  # Convert to 0-indexed
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes
    except Exception:
        return None
