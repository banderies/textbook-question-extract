#!/usr/bin/env python3
"""
Image-to-Question Linker v2 - Improved Accuracy

This version uses a block-based approach:
1. Parse markdown into "question blocks" (from one question to the next)
2. All images within a block belong to that question
3. Handle sub-questions (2a, 2b, 2c) as part of the same group
4. Optionally verify with Claude Vision API

Usage:
    python link_images_v2.py <markdown_file> <manifest_json>
    python link_images_v2.py <markdown_file> <manifest_json> --verify  # Use Vision API

Output:
    - question_image_map.json: Accurate question → images mapping
    - image_question_map.json: Reverse mapping (image → question)
"""

import json
import re
import sys
import os
import base64
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# Optional: Claude Vision for verification
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class Question:
    """A parsed question from the markdown."""
    id: str                    # e.g., "1", "2a", "2b"
    base_id: str               # e.g., "2" for "2a", "2b", "2c"
    line_start: int            # Line where question starts
    line_end: int              # Line where question ends (before next question)
    text: str                  # Question text (first line)
    has_choices: bool          # Whether A/B/C/D choices were found
    image_markers: list[int] = field(default_factory=list)  # Line numbers of images


@dataclass
class ImageInfo:
    """Information about an extracted image."""
    filename: str
    page: int
    y_position: float
    marker_index: int          # Order in the markdown
    line_number: int           # Line in markdown
    question_id: Optional[str] = None
    confidence: str = "unknown"


def parse_questions(markdown_path: str) -> tuple[list[Question], dict[int, int]]:
    """
    Parse the markdown to identify all questions and their boundaries.

    Returns:
        - List of Question objects
        - Dict mapping line_number → marker_index for image markers
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    questions = []
    image_marker_lines = {}  # line_number → marker_index

    # Patterns for question detection
    # Pattern 1: "- 1 ", "- 2a ", "- 37 " (standard format)
    question_pattern_1 = re.compile(r'^-\s+(\d+[a-z]?)\s+(.+)', re.IGNORECASE)
    # Pattern 2: "5. The patient..." or "10. 2c Which..." (line number artifact)
    # The line number prefix is an artifact from PDF, real question ID follows
    question_pattern_2 = re.compile(r'^\d+\.\s+(\d+[a-z]?)\s+(.+)', re.IGNORECASE)
    # Pattern 3: Just "5. The patient..." where 5 IS the question number
    question_pattern_3 = re.compile(r'^(\d+)\.\s+(?![\d])(.*)', re.IGNORECASE)
    # Also matches standalone sub-question markers like "2b" on its own line
    subquestion_pattern = re.compile(r'^(\d+[a-z])\s*$')
    # Image marker
    image_pattern = re.compile(r'<!--\s*image\s*-->')
    # Choice pattern (A., B., C., D.)
    choice_pattern = re.compile(r'^-\s+[A-E]\.\s+')

    marker_index = 0
    current_question = None
    in_questions_section = False
    in_answers_section = False

    for line_num, line in enumerate(lines, start=1):
        # Track sections
        if 'Q U E S T I O N S' in line:
            in_questions_section = True
            in_answers_section = False
            continue
        if 'A N S W E R S' in line:
            in_questions_section = False
            in_answers_section = True
            # Close any open question
            if current_question:
                current_question.line_end = line_num - 1
                questions.append(current_question)
                current_question = None
            continue

        # Track image markers (everywhere)
        if image_pattern.search(line):
            image_marker_lines[line_num] = marker_index
            if current_question and in_questions_section:
                current_question.image_markers.append(line_num)
            marker_index += 1

        # Only parse questions in the questions section
        if not in_questions_section:
            continue

        # Check for new question (try multiple patterns)
        q_id = None
        q_text = None

        # Try pattern 1: "- 1 ", "- 2a "
        q_match = question_pattern_1.match(line)
        if q_match:
            q_id = q_match.group(1)
            q_text = q_match.group(2).strip()

        # Try pattern 2: "10. 2c Which..." (line num + question ID)
        if not q_id:
            q_match = question_pattern_2.match(line)
            if q_match:
                q_id = q_match.group(1)
                q_text = q_match.group(2).strip()

        # Try pattern 3: "5. The patient..." (just number + text)
        if not q_id:
            q_match = question_pattern_3.match(line)
            if q_match:
                q_id = q_match.group(1)
                q_text = q_match.group(2).strip()

        if q_id and q_text and len(q_text) > 10:  # Ensure it's a real question, not just "5. A"
            # Close previous question
            if current_question:
                current_question.line_end = line_num - 1
                questions.append(current_question)

            # Extract base ID (number without letter)
            base_match = re.match(r'(\d+)', q_id)
            base_id = base_match.group(1) if base_match else q_id

            current_question = Question(
                id=q_id,
                base_id=base_id,
                line_start=line_num,
                line_end=0,
                text=q_text[:200],
                has_choices=False,
                image_markers=[]
            )
            continue

        # Check for sub-question marker (like "2b" on its own line)
        sub_match = subquestion_pattern.match(line.strip())
        if sub_match and current_question:
            # This is a sub-question continuation - keep tracking under same question
            # Or we could create a new question entry - depends on desired granularity
            pass

        # Check for choices (indicates question has standard format)
        if current_question and choice_pattern.match(line):
            current_question.has_choices = True

    # Close final question
    if current_question:
        current_question.line_end = len(lines)
        questions.append(current_question)

    return questions, image_marker_lines


def load_manifest_sorted(manifest_path: str) -> list[dict]:
    """Load and sort manifest by page and y-position."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    manifest.sort(key=lambda x: (x["page"], x["y_position"]))
    return manifest


def create_image_info_list(
    manifest: list[dict],
    image_marker_lines: dict[int, int]
) -> list[ImageInfo]:
    """
    Create ImageInfo objects linking manifest entries to markdown positions.
    """
    # Invert the mapping: marker_index → line_number
    marker_to_line = {v: k for k, v in image_marker_lines.items()}

    images = []
    for i, img in enumerate(manifest):
        line_num = marker_to_line.get(i, -1)
        info = ImageInfo(
            filename=img["filename"],
            page=img["page"],
            y_position=img["y_position"],
            marker_index=i,
            line_number=line_num
        )
        images.append(info)

    return images


def assign_images_to_questions(
    questions: list[Question],
    images: list[ImageInfo]
) -> dict[str, list[ImageInfo]]:
    """
    Assign images to questions based on line number proximity.

    Strategy:
    1. If image is within a question's line range → assign to that question
    2. Group sub-questions (2a, 2b, 2c) together
    3. For images outside question ranges, find nearest question
    """
    question_images: dict[str, list[ImageInfo]] = {}

    # Build a lookup of line ranges
    question_ranges = []
    for q in questions:
        question_ranges.append((q.line_start, q.line_end, q))

    # Sort by line_start
    question_ranges.sort(key=lambda x: x[0])

    for img in images:
        if img.line_number < 0:
            # Image has no markdown marker (filtered out)
            continue

        assigned = False

        # Find which question range this image falls into
        for start, end, q in question_ranges:
            if start <= img.line_number <= end:
                img.question_id = q.id
                img.confidence = "high"
                assigned = True
                break

        if not assigned:
            # Find nearest question by line distance
            min_dist = float('inf')
            nearest_q = None

            for start, end, q in question_ranges:
                dist = min(abs(img.line_number - start), abs(img.line_number - end))
                if dist < min_dist:
                    min_dist = dist
                    nearest_q = q

            if nearest_q and min_dist < 50:  # Within 50 lines
                img.question_id = nearest_q.id
                img.confidence = "medium"
            else:
                img.confidence = "low"

    # Group by question
    for img in images:
        if img.question_id:
            if img.question_id not in question_images:
                question_images[img.question_id] = []
            question_images[img.question_id].append(img)

    return question_images


def verify_with_vision(
    images_dir: str,
    question_images: dict[str, list[ImageInfo]],
    questions: list[Question],
    sample_size: int = 5
) -> dict[str, list[dict]]:
    """
    Use Claude Vision to verify image-question assignments.

    Only verifies a sample of assignments to control costs.
    """
    if not HAS_ANTHROPIC:
        print("Warning: anthropic not installed, skipping vision verification")
        return {}

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set, skipping vision verification")
        return {}

    client = anthropic.Anthropic()
    verification_results = {}

    # Create question text lookup
    q_text_lookup = {q.id: q.text for q in questions}

    # Sample some assignments to verify
    samples = []
    for q_id, imgs in question_images.items():
        if q_id in q_text_lookup:
            for img in imgs[:1]:  # Just first image per question
                samples.append((q_id, img, q_text_lookup[q_id]))
                if len(samples) >= sample_size:
                    break
        if len(samples) >= sample_size:
            break

    print(f"\nVerifying {len(samples)} image assignments with Claude Vision...")

    for q_id, img, q_text in samples:
        img_path = os.path.join(images_dir, img.filename)
        if not os.path.exists(img_path):
            continue

        # Load and encode image
        with open(img_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        ext = Path(img_path).suffix.lower()
        media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")

        # Ask Claude to verify
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
                "text": f"""Does this medical image relate to the following question?

Question {q_id}: {q_text}

Answer with JSON:
{{"matches": true/false, "confidence": "high/medium/low", "reason": "brief explanation"}}"""
            }
        ]

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{"role": "user", "content": message_content}]
            )

            response_text = response.content[0].text

            # Parse JSON
            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            result = json.loads(response_text)
            result["image"] = img.filename
            result["question_id"] = q_id

            verification_results[img.filename] = result

            status = "✓" if result.get("matches") else "✗"
            print(f"  {status} {img.filename} → Q{q_id}: {result.get('reason', 'N/A')[:50]}")

        except Exception as e:
            print(f"  Error verifying {img.filename}: {e}")

    return verification_results


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    markdown_path = sys.argv[1]
    manifest_path = sys.argv[2]
    verify = "--verify" in sys.argv

    print("=" * 60)
    print("Image-to-Question Linker v2")
    print("=" * 60)

    # Step 1: Parse questions from markdown
    print(f"\n1. Parsing questions from: {markdown_path}")
    questions, image_marker_lines = parse_questions(markdown_path)
    print(f"   Found {len(questions)} questions")
    print(f"   Found {len(image_marker_lines)} image markers")

    # Show some question examples
    print("\n   Sample questions:")
    for q in questions[:5]:
        print(f"     Q{q.id} (lines {q.line_start}-{q.line_end}): {q.text[:50]}...")
        if q.image_markers:
            print(f"       Images at lines: {q.image_markers[:3]}{'...' if len(q.image_markers) > 3 else ''}")

    # Step 2: Load manifest
    print(f"\n2. Loading image manifest: {manifest_path}")
    manifest = load_manifest_sorted(manifest_path)
    print(f"   Found {len(manifest)} images")

    # Step 3: Create image info list
    print("\n3. Creating image info...")
    images = create_image_info_list(manifest, image_marker_lines)
    linked_count = len([i for i in images if i.line_number > 0])
    print(f"   {linked_count} images have markdown positions")

    # Step 4: Assign images to questions
    print("\n4. Assigning images to questions...")
    question_images = assign_images_to_questions(questions, images)

    # Count by confidence
    high_conf = sum(1 for imgs in question_images.values() for i in imgs if i.confidence == "high")
    med_conf = sum(1 for imgs in question_images.values() for i in imgs if i.confidence == "medium")

    print(f"   Questions with images: {len(question_images)}")
    print(f"   High confidence assignments: {high_conf}")
    print(f"   Medium confidence assignments: {med_conf}")

    # Step 5: Optional verification
    verification_results = {}
    if verify:
        images_dir = str(Path(manifest_path).parent)
        verification_results = verify_with_vision(
            images_dir, question_images, questions, sample_size=5
        )

    # Step 6: Save outputs
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Question → Images mapping (main output)
    q_img_map = {}
    for q_id, imgs in question_images.items():
        q_img_map[q_id] = [
            {
                "filename": i.filename,
                "page": i.page,
                "confidence": i.confidence
            }
            for i in imgs
        ]

    q_img_file = f"{output_dir}/question_image_map.json"
    with open(q_img_file, "w") as f:
        json.dump(q_img_map, f, indent=2)
    print(f"\n5. Saved question → images map: {q_img_file}")

    # Image → Question mapping (reverse)
    img_q_map = {}
    for q_id, imgs in question_images.items():
        for i in imgs:
            img_q_map[i.filename] = {
                "question_id": q_id,
                "confidence": i.confidence,
                "page": i.page
            }

    img_q_file = f"{output_dir}/image_question_map.json"
    with open(img_q_file, "w") as f:
        json.dump(img_q_map, f, indent=2)
    print(f"   Saved image → question map: {img_q_file}")

    # Save verification results if any
    if verification_results:
        verify_file = f"{output_dir}/verification_results.json"
        with open(verify_file, "w") as f:
            json.dump(verification_results, f, indent=2)
        print(f"   Saved verification results: {verify_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total questions parsed:     {len(questions)}")
    print(f"Total images in manifest:   {len(manifest)}")
    print(f"Images with line positions: {linked_count}")
    print(f"Questions with images:      {len(question_images)}")
    print(f"High confidence matches:    {high_conf}")
    print(f"Medium confidence matches:  {med_conf}")

    # Show distribution
    print("\nImages per question (top 10):")
    sorted_q = sorted(question_images.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for q_id, imgs in sorted_q:
        q_text = next((q.text[:40] for q in questions if q.id == q_id), "")
        print(f"  Q{q_id}: {len(imgs)} images - {q_text}...")


if __name__ == "__main__":
    main()
