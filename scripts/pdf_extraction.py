"""
PDF Extraction Module

Contains functions for extracting text and images from PDF files.
"""

import os
from typing import Optional

import fitz  # PyMuPDF


# Module-level cache for position data between extract_text_with_lines and insert_image_markers
_position_cache: dict = {}


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


def extract_text_with_positions(pdf_path: str) -> list[dict]:
    """
    Extract text lines from PDF with their actual y-positions.

    Returns a list of dicts, each containing:
    - page: 1-indexed page number
    - y_position: vertical position on page (PDF points, top of line)
    - y_bottom: bottom of the text line
    - text: the text content
    - type: "text"

    This allows accurate interleaving with images based on actual positions.
    """
    doc = fitz.open(pdf_path)
    all_items = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Get text as dictionary with position info
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks (images)
                continue

            # Process each line in the block
            for line in block.get("lines", []):
                bbox = line.get("bbox", [0, 0, 0, 0])
                y_top = bbox[1]
                y_bottom = bbox[3]

                # Concatenate all spans in the line
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")

                if line_text.strip():  # Skip empty lines
                    all_items.append({
                        "page": page_num + 1,
                        "y_position": y_top,
                        "y_bottom": y_bottom,
                        "text": line_text,
                        "type": "text"
                    })

    doc.close()

    # Sort by page, then by y_position
    all_items.sort(key=lambda x: (x["page"], x["y_position"]))

    return all_items


def extract_text_with_lines(pdf_path: str) -> tuple[list[dict], list[str]]:
    """
    Extract text with line numbers and build a global line array.

    Uses position-aware extraction to get accurate y-coordinates for each line,
    enabling proper interleaving with images.

    Returns:
        Tuple of:
        - pages: List of {"page": N, "text": "...", "start_line": N, "end_line": N}
        - lines: Global list of all lines (0-indexed), each with position metadata
        - lines_with_positions: List of dicts with text and y_position for each line
    """
    doc = fitz.open(pdf_path)
    pages = []
    all_lines = []
    lines_with_positions = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_dict = page.get_text("dict")

        page_lines = []
        page_lines_with_pos = []

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks
                continue

            for line in block.get("lines", []):
                bbox = line.get("bbox", [0, 0, 0, 0])
                y_top = bbox[1]
                y_bottom = bbox[3]

                # Concatenate all spans in the line
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")

                # Include all lines, even empty ones (preserve structure)
                page_lines.append(line_text)
                page_lines_with_pos.append({
                    "text": line_text,
                    "y_position": y_top,
                    "y_bottom": y_bottom,
                    "page": page_num + 1
                })

        # Sort by y_position within the page
        page_lines_with_pos.sort(key=lambda x: x["y_position"])
        page_lines = [item["text"] for item in page_lines_with_pos]

        start_line = len(all_lines)
        all_lines.extend(page_lines)
        lines_with_positions.extend(page_lines_with_pos)
        end_line = len(all_lines)

        # Reconstruct full text for compatibility
        full_text = "\n".join(page_lines)

        pages.append({
            "page": page_num + 1,
            "text": full_text,
            "start_line": start_line,
            "end_line": end_line
        })

    doc.close()

    # Store position data in module-level cache for insert_image_markers
    _position_cache["lines_with_positions"] = lines_with_positions

    return pages, all_lines


def insert_image_markers(
    lines: list[str],
    images: list[dict],
    pages: list[dict]
) -> list[str]:
    """
    Insert [IMAGE: filename.jpg] markers at correct positions in the text.

    Uses actual y-positions from both text lines and images to determine
    exact placement. Images are inserted AFTER the last text line that
    appears above them on the page.

    Args:
        lines: Global list of all text lines
        images: Image metadata with page, y_position, filename
        pages: Page metadata with start_line, end_line

    Returns:
        New list of lines with image markers inserted
    """
    lines_with_positions = _position_cache.get("lines_with_positions", [])

    if not lines_with_positions:
        raise RuntimeError(
            "No position data available. Call extract_text_with_lines() first, "
            "or use insert_image_markers_for_page() for single-page operations."
        )

    if len(lines_with_positions) != len(lines):
        raise RuntimeError(
            f"Position cache mismatch: {len(lines_with_positions)} positions vs {len(lines)} lines. "
            "Call extract_text_with_lines() again to refresh the cache."
        )

    # Build a combined list of text lines and images with positions
    combined_items = []

    # Add text lines with their positions
    for idx, line_data in enumerate(lines_with_positions):
        combined_items.append({
            "type": "text",
            "page": line_data["page"],
            "y_position": line_data["y_position"],
            "content": lines[idx] if idx < len(lines) else "",
            "index": idx
        })

    # Add images with their positions
    for img in images:
        combined_items.append({
            "type": "image",
            "page": img["page"],
            "y_position": img.get("y_position", 0),
            "content": f"[IMAGE: {img['filename']}]",
            "filename": img["filename"]
        })

    # Sort by page, then by y_position
    # For items at the same position, text comes before images
    combined_items.sort(key=lambda x: (x["page"], x["y_position"], 0 if x["type"] == "text" else 1))

    # Build result list
    result = []
    for item in combined_items:
        result.append(item["content"])

    return result


def extract_page_text_with_positions(pdf_path: str, page_num: int) -> tuple[list[str], list[dict]]:
    """
    Extract text lines with positions for a single page.

    Use this for previews or when you only need one page.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (1-indexed)

    Returns:
        Tuple of:
        - lines: List of text strings
        - lines_with_positions: List of dicts with text, y_position, page
    """
    doc = fitz.open(pdf_path)

    if page_num < 1 or page_num > len(doc):
        doc.close()
        return [], []

    page = doc[page_num - 1]  # Convert to 0-indexed
    text_dict = page.get_text("dict")

    lines_with_pos = []

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip non-text blocks
            continue

        for line in block.get("lines", []):
            bbox = line.get("bbox", [0, 0, 0, 0])
            y_top = bbox[1]
            y_bottom = bbox[3]

            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")

            lines_with_pos.append({
                "text": line_text,
                "y_position": y_top,
                "y_bottom": y_bottom,
                "page": page_num
            })

    doc.close()

    # Sort by y_position
    lines_with_pos.sort(key=lambda x: x["y_position"])
    lines = [item["text"] for item in lines_with_pos]

    return lines, lines_with_pos


def insert_image_markers_for_page(
    lines: list[str],
    lines_with_positions: list[dict],
    images: list[dict]
) -> list[str]:
    """
    Insert image markers for a single page using position data.

    This is the preferred method for previews - pass the position data directly.

    Args:
        lines: List of text lines
        lines_with_positions: Position data for each line
        images: Image metadata with page, y_position, filename

    Returns:
        List of lines with image markers inserted at correct positions
    """
    if len(lines) != len(lines_with_positions):
        raise ValueError("lines and lines_with_positions must have same length")

    # Build combined list
    combined_items = []

    for idx, line_data in enumerate(lines_with_positions):
        combined_items.append({
            "type": "text",
            "page": line_data["page"],
            "y_position": line_data["y_position"],
            "content": lines[idx]
        })

    for img in images:
        combined_items.append({
            "type": "image",
            "page": img["page"],
            "y_position": img.get("y_position", 0),
            "content": f"[IMAGE: {img['filename']}]"
        })

    # Sort by page, then y_position (text before images at same position)
    combined_items.sort(key=lambda x: (x["page"], x["y_position"], 0 if x["type"] == "text" else 1))

    return [item["content"] for item in combined_items]


def build_chapter_text_with_lines(
    lines: list[str],
    pages: list[dict],
    start_page: int,
    end_page: Optional[int]
) -> tuple[str, dict]:
    """
    Build chapter text preserving GLOBAL line numbers for LLM consumption.

    Output format (preserves global line numbers):
    [LINE:0450] First line of chapter text
    [LINE:0451] Second line of chapter text
    [IMAGE: p014_y0321_x0172_0.jpeg]
    [LINE:0452] Third line...

    This ensures the LLM returns global line numbers that can be used
    directly without any mapping/translation.

    Args:
        lines: Global line array (may include [IMAGE:] markers)
        pages: Page metadata with start_line, end_line
        start_page: First page of chapter (1-indexed)
        end_page: Last page of chapter (exclusive), or None for end

    Returns:
        Tuple of:
        - numbered_text: The chapter text with GLOBAL line numbers
        - line_mapping: Dict mapping global line numbers to array indices
    """
    # Find line range for chapter
    start_line = None
    end_line = None

    for p in pages:
        if p["page"] >= start_page:
            if end_page is None or p["page"] < end_page:
                if start_line is None:
                    start_line = p["start_line"]
                end_line = p["end_line"]

    if start_line is None:
        return "", {}

    # Account for image markers inserted before start_line
    # We need to scan from the beginning to count inserted markers
    actual_start = 0
    actual_end = len(lines)

    # Find actual indices by scanning (image markers shift positions)
    current_orig_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("[IMAGE:"):
            continue
        if current_orig_idx == start_line:
            actual_start = i
        if current_orig_idx == end_line:
            actual_end = i
            break
        current_orig_idx += 1

    # Build output preserving GLOBAL line numbers
    output_lines = []
    global_line_num = start_line + 1  # 1-indexed global line number
    line_mapping = {}

    for i in range(actual_start, min(actual_end, len(lines))):
        line = lines[i]

        if line.startswith("[IMAGE:"):
            # Image markers don't get line numbers
            output_lines.append(line)
        else:
            output_lines.append(f"[LINE:{global_line_num:04d}] {line}")
            line_mapping[global_line_num] = i
            global_line_num += 1

    return "\n".join(output_lines), line_mapping


def extract_lines_by_range(lines: list[str], start: int, end: int) -> str:
    """
    Extract text from lines array by line number range.

    Handles the fact that image markers don't have line numbers.
    Finds the actual indices that correspond to the given line numbers.

    Args:
        lines: The lines array (may include [IMAGE:] markers)
        start: Starting line number (1-indexed, from [LINE:NNNN])
        end: Ending line number (inclusive)

    Returns:
        The extracted text (without line number prefixes)
    """
    if start <= 0 or end < start:
        return ""

    result = []
    current_line_num = 0

    for line in lines:
        if line.startswith("[IMAGE:"):
            # Include image markers that fall within the range
            if current_line_num >= start and current_line_num <= end:
                result.append(line)
            continue

        current_line_num += 1

        if current_line_num >= start and current_line_num <= end:
            # Remove the [LINE:NNNN] prefix if present
            if line.startswith("[LINE:"):
                # Find the closing bracket
                bracket_end = line.find("]")
                if bracket_end > 0:
                    line = line[bracket_end + 2:]  # +2 to skip "] "
            result.append(line)

        if current_line_num > end:
            break

    return "\n".join(result)


def extract_lines_by_range_mapped(
    lines: list[str],
    start: int,
    end: int,
    line_mapping: dict
) -> str:
    """
    Extract text from lines array using a line mapping for translation.

    This is used when the line numbers are chapter-relative (from LLM output)
    and need to be translated to global array indices.

    Args:
        lines: The global lines array (may include [IMAGE:] markers)
        start: Starting line number (1-indexed, chapter-relative)
        end: Ending line number (inclusive, chapter-relative)
        line_mapping: Dict mapping chapter line numbers to global array indices

    Returns:
        The extracted text (without line number prefixes)
    """
    if start <= 0 or end < start or not line_mapping:
        return ""

    # Find the global indices for start and end
    start_idx = line_mapping.get(start)
    end_idx = line_mapping.get(end)

    if start_idx is None or end_idx is None:
        # Try to find closest valid indices
        valid_keys = sorted(line_mapping.keys())
        if not valid_keys:
            return ""

        # Find closest start
        start_idx = None
        for k in valid_keys:
            if k >= start:
                start_idx = line_mapping[k]
                break
        if start_idx is None:
            start_idx = line_mapping[valid_keys[-1]]

        # Find closest end
        end_idx = None
        for k in reversed(valid_keys):
            if k <= end:
                end_idx = line_mapping[k]
                break
        if end_idx is None:
            end_idx = line_mapping[valid_keys[0]]

    # Extract lines from start_idx to end_idx (inclusive)
    result = []
    for i in range(start_idx, min(end_idx + 1, len(lines))):
        line = lines[i]

        # Remove the [LINE:NNNN] prefix if present
        if line.startswith("[LINE:"):
            bracket_end = line.find("]")
            if bracket_end > 0:
                line = line[bracket_end + 2:]  # +2 to skip "] "

        result.append(line)

    # Also include any images between start_idx and end_idx
    # (They should already be in the range, but let's ensure we don't skip them)

    return "\n".join(result)


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
