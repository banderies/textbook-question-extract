#!/usr/bin/env python3
"""
Image Extraction and Matching Pipeline

This script demonstrates:
1. Extracting images from PDF with positional metadata
2. Creating a searchable manifest/index
3. Two matching strategies:
   - Position-based (fast, free)
   - Vision-based (accurate, uses Claude API)

Usage:
    # Extract images from PDF
    python image_pipeline.py extract input/book.pdf --output images/

    # Match images to questions using position
    python image_pipeline.py match-position images/ questions.json

    # Match images to questions using vision API
    python image_pipeline.py match-vision images/ questions.json

Dependencies:
    pip install pymupdf anthropic
"""

import json
import os
import sys
import base64
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("Warning: PyMuPDF not installed. Run: pip install pymupdf")

try:
    import anthropic
except ImportError:
    anthropic = None
    print("Warning: anthropic not installed. Run: pip install anthropic")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ImageMetadata:
    """Metadata for an extracted image."""
    filename: str
    page: int              # 1-indexed page number
    y_position: float      # Y coordinate on page (top = 0)
    x_position: float      # X coordinate on page
    width: float
    height: float
    xref: int              # PDF internal reference ID

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageMatch:
    """Result of matching an image to a question."""
    image_filename: str
    question_id: str
    confidence: str        # high, medium, low
    method: str            # position, vision
    reasoning: Optional[str] = None


# =============================================================================
# STEP 1: IMAGE EXTRACTION
# =============================================================================

def extract_images_from_pdf(
    pdf_path: str,
    output_dir: str,
    min_width: int = 50,
    min_height: int = 50
) -> list[ImageMetadata]:
    """
    Extract all images from a PDF with positional metadata.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        min_width: Minimum image width to extract (filters tiny icons)
        min_height: Minimum image height to extract

    Returns:
        List of ImageMetadata for each extracted image
    """
    if fitz is None:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    manifest = []

    print(f"Processing {len(doc)} pages...")

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1  # 1-indexed

        # Get all images on this page
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]  # PDF cross-reference ID

            try:
                # Get image position on page
                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                rect = rects[0]  # Use first occurrence

                # Filter small images (icons, decorations)
                if rect.width < min_width or rect.height < min_height:
                    continue

                # Extract image bytes
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]  # jpg, png, etc.

                # Create filename encoding position
                # Format: p{page}_y{y-pos}_x{x-pos}_{xref}.{ext}
                filename = f"p{page_num:03d}_y{int(rect.y0):04d}_x{int(rect.x0):04d}_{xref}.{ext}"
                filepath = os.path.join(output_dir, filename)

                # Save image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                # Create metadata entry
                metadata = ImageMetadata(
                    filename=filename,
                    page=page_num,
                    y_position=rect.y0,
                    x_position=rect.x0,
                    width=rect.width,
                    height=rect.height,
                    xref=xref
                )
                manifest.append(metadata)

            except Exception as e:
                print(f"  Warning: Could not extract image {xref} from page {page_num}: {e}")

        # Progress indicator
        if page_num % 20 == 0:
            print(f"  Processed page {page_num}/{len(doc)}")

    doc.close()

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump([m.to_dict() for m in manifest], f, indent=2)

    print(f"\nExtracted {len(manifest)} images to {output_dir}")
    print(f"Manifest saved to {manifest_path}")

    return manifest


# =============================================================================
# STEP 2: POSITION-BASED MATCHING (Fast, No API)
# =============================================================================

def load_manifest(images_dir: str) -> list[dict]:
    """Load the image manifest from a directory."""
    manifest_path = os.path.join(images_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        return json.load(f)


def match_by_position(
    manifest: list[dict],
    questions: list[dict],
    page_tolerance: int = 1
) -> list[ImageMatch]:
    """
    Match images to questions based on page proximity.

    This is a fast heuristic that works well when:
    - Questions reference "the image below/above"
    - Images appear on the same or adjacent page as the question

    Args:
        manifest: List of image metadata dicts
        questions: List of question dicts with 'page' or 'figure_ref' field
        page_tolerance: How many pages away to search (default: 1)

    Returns:
        List of ImageMatch results
    """
    matches = []

    # Index images by page for fast lookup
    images_by_page: dict[int, list[dict]] = {}
    for img in manifest:
        page = img["page"]
        if page not in images_by_page:
            images_by_page[page] = []
        images_by_page[page].append(img)

    for q in questions:
        q_id = q.get("question_id", q.get("id", "unknown"))

        # Try to determine which page(s) to search
        search_pages = set()

        # Method 1: Explicit page reference in question
        if "page" in q:
            base_page = q["page"]
            for offset in range(-page_tolerance, page_tolerance + 1):
                search_pages.add(base_page + offset)

        # Method 2: Figure reference (e.g., "Figure 37" might be on page ~37)
        if "figure_ref" in q and q["figure_ref"]:
            fig_ref = q["figure_ref"]
            # Extract number from figure reference
            nums = re.findall(r'\d+', str(fig_ref))
            if nums:
                fig_num = int(nums[0])
                # Figure numbers often correlate loosely with page numbers
                for offset in range(-5, 6):
                    search_pages.add(fig_num + offset)

        # Find images on those pages
        candidate_images = []
        for page in search_pages:
            if page in images_by_page:
                candidate_images.extend(images_by_page[page])

        # Sort by Y position (top to bottom)
        candidate_images.sort(key=lambda x: (x["page"], x["y_position"]))

        # For now, associate all candidates (you might want to be more selective)
        for img in candidate_images:
            confidence = "medium" if len(candidate_images) <= 3 else "low"

            match = ImageMatch(
                image_filename=img["filename"],
                question_id=q_id,
                confidence=confidence,
                method="position",
                reasoning=f"Image on page {img['page']}, question references page {search_pages}"
            )
            matches.append(match)

    return matches


# =============================================================================
# STEP 3: VISION-BASED MATCHING (Accurate, Uses API)
# =============================================================================

def match_by_vision(
    images_dir: str,
    manifest: list[dict],
    questions: list[dict],
    model: str = "claude-sonnet-4-20250514",  # Sonnet is good for vision, cheaper than Opus
    max_images: int = 50,  # Limit for cost control
    batch_size: int = 5    # Questions per API call
) -> list[ImageMatch]:
    """
    Match images to questions using Claude's vision capability.

    This analyzes the actual image content to find the best match,
    handling cases where position-based matching fails.

    Args:
        images_dir: Directory containing extracted images
        manifest: Image manifest list
        questions: List of question dicts
        model: Claude model to use (sonnet recommended for vision)
        max_images: Maximum images to process (cost control)
        batch_size: How many questions to send per image

    Returns:
        List of ImageMatch results
    """
    if anthropic is None:
        raise ImportError("anthropic required: pip install anthropic")

    client = anthropic.Anthropic()
    matches = []

    # Process subset of images
    images_to_process = manifest[:max_images]
    print(f"Processing {len(images_to_process)} images with vision API...")

    for i, img_meta in enumerate(images_to_process):
        img_path = os.path.join(images_dir, img_meta["filename"])

        if not os.path.exists(img_path):
            print(f"  Warning: Image not found: {img_path}")
            continue

        # Load and encode image
        with open(img_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        ext = Path(img_path).suffix.lower()
        media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")

        # Find candidate questions (nearby pages)
        img_page = img_meta["page"]
        candidate_questions = [
            q for q in questions
            if abs(q.get("page", img_page) - img_page) <= 3
        ][:batch_size]

        if not candidate_questions:
            # Fall back to first N questions if no page info
            candidate_questions = questions[:batch_size]

        # Format questions for prompt
        q_text = "\n".join([
            f"{j+1}. [ID: {q.get('question_id', j)}] {q.get('question', q.get('text', ''))[:200]}"
            for j, q in enumerate(candidate_questions)
        ])

        # Build multimodal message
        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": f"""Analyze this medical/educational image and determine which question it belongs to.

CANDIDATE QUESTIONS:
{q_text}

Return JSON:
{{
    "matched_question_index": 1,  // 1-indexed, or 0 if no match
    "matched_question_id": "the ID value",
    "confidence": "high/medium/low",
    "image_description": "Brief description of what the image shows",
    "reasoning": "Why this image matches the question"
}}

If the image doesn't clearly match any question, set matched_question_index to 0."""
            }
        ]

        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": message_content}],
                system="You are an expert at analyzing medical images. Return only valid JSON."
            )

            response_text = response.content[0].text

            # Parse JSON
            if "```" in response_text:
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    response_text = json_match.group(1)

            result = json.loads(response_text)

            if result.get("matched_question_index", 0) > 0:
                match = ImageMatch(
                    image_filename=img_meta["filename"],
                    question_id=result.get("matched_question_id", "unknown"),
                    confidence=result.get("confidence", "medium"),
                    method="vision",
                    reasoning=result.get("reasoning")
                )
                matches.append(match)
                print(f"  {img_meta['filename']} → Q{result.get('matched_question_id')} ({result.get('confidence')})")
            else:
                print(f"  {img_meta['filename']} → No match found")

        except Exception as e:
            print(f"  Error processing {img_meta['filename']}: {e}")

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(images_to_process)} images")

    return matches


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "extract":
        if len(sys.argv) < 3:
            print("Usage: python image_pipeline.py extract <pdf_path> [--output <dir>]")
            sys.exit(1)

        pdf_path = sys.argv[2]
        output_dir = "images"

        if "--output" in sys.argv:
            idx = sys.argv.index("--output")
            output_dir = sys.argv[idx + 1]

        extract_images_from_pdf(pdf_path, output_dir)

    elif command == "match-position":
        if len(sys.argv) < 4:
            print("Usage: python image_pipeline.py match-position <images_dir> <questions.json>")
            sys.exit(1)

        images_dir = sys.argv[2]
        questions_file = sys.argv[3]

        manifest = load_manifest(images_dir)
        with open(questions_file) as f:
            questions = json.load(f)

        matches = match_by_position(manifest, questions)

        output_file = "image_matches_position.json"
        with open(output_file, "w") as f:
            json.dump([asdict(m) for m in matches], f, indent=2)

        print(f"Found {len(matches)} matches, saved to {output_file}")

    elif command == "match-vision":
        if len(sys.argv) < 4:
            print("Usage: python image_pipeline.py match-vision <images_dir> <questions.json>")
            sys.exit(1)

        images_dir = sys.argv[2]
        questions_file = sys.argv[3]

        manifest = load_manifest(images_dir)
        with open(questions_file) as f:
            questions = json.load(f)

        matches = match_by_vision(images_dir, manifest, questions)

        output_file = "image_matches_vision.json"
        with open(output_file, "w") as f:
            json.dump([asdict(m) for m in matches], f, indent=2)

        print(f"Found {len(matches)} matches, saved to {output_file}")

    else:
        print(f"Unknown command: {command}")
        print("Commands: extract, match-position, match-vision")
        sys.exit(1)


if __name__ == "__main__":
    main()
