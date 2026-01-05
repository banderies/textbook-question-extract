#!/usr/bin/env python3
"""
Chapter-Aware Q&A Parser

This script parses the markdown and creates chapter-aware question IDs.
Since question numbering restarts in each chapter, we prefix IDs with
chapter numbers (e.g., "ch1_2a", "ch8_2a").

Usage:
    python chapter_aware_parser.py <markdown_file> <manifest_json>

Output:
    - output/chapters.json: Chapter metadata
    - output/questions_by_chapter.json: Questions organized by chapter
    - output/chapter_image_map.json: Chapter-aware image-question mapping
"""

import json
import re
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Chapter:
    """A chapter in the book."""
    number: int
    name: str
    line_start: int
    line_end: int
    questions_line_start: int = 0
    questions_line_end: int = 0
    answers_line_start: int = 0
    answers_line_end: int = 0


@dataclass
class Question:
    """A question with chapter-aware ID."""
    chapter_num: int
    local_id: str           # Original ID like "2a"
    full_id: str            # Chapter-prefixed like "ch1_2a"
    text: str
    choices: dict = field(default_factory=dict)
    line_start: int = 0
    line_end: int = 0
    image_lines: list = field(default_factory=list)


def detect_chapters(lines: list[str]) -> list[Chapter]:
    """
    Detect chapter boundaries in the markdown.

    Looks for patterns like:
    - ## 1 Imaging Techniques...
    - ## 2 Normal/Normal Variants
    """
    chapters = []

    # Known chapter patterns from MSK book
    chapter_patterns = [
        (1, r'Imaging\s*Techniques|Physics|Quality\s*and\s*Safety'),
        (2, r'Normal.*Variant'),
        (3, r'Congenital.*Developmental'),
        (4, r'Infection'),
        (5, r'Tumor'),
        (6, r'Trauma'),
        (7, r'Metabolic.*Hematologic'),
        (8, r'Arthropathy'),
        (9, r'Miscellaneous'),
    ]

    # First pass: find chapter headers
    chapter_line_nums = []

    for line_num, line in enumerate(lines, start=1):
        # Look for "## NUMBER ..." pattern
        match = re.match(r'^##\s*(\d+)\s+(.+)', line)
        if match:
            num = int(match.group(1))
            title = match.group(2).strip()
            # Only accept chapters 1-9
            if 1 <= num <= 9:
                chapter_line_nums.append((line_num, num, title))
                continue

        # Also check for chapter content without explicit number
        for ch_num, pattern in chapter_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Check if this looks like a chapter header (short line with ##)
                if line.startswith('##') and len(line) < 150:
                    # Check we haven't already found this chapter
                    if not any(c[1] == ch_num for c in chapter_line_nums):
                        chapter_line_nums.append((line_num, ch_num, line.lstrip('#').strip()))
                    break

    # Sort by line number
    chapter_line_nums.sort(key=lambda x: x[0])

    # Create Chapter objects with boundaries
    for i, (line_num, ch_num, title) in enumerate(chapter_line_nums):
        # End line is start of next chapter or end of file
        if i + 1 < len(chapter_line_nums):
            end_line = chapter_line_nums[i + 1][0] - 1
        else:
            end_line = len(lines)

        chapter = Chapter(
            number=ch_num,
            name=title[:100],
            line_start=line_num,
            line_end=end_line
        )
        chapters.append(chapter)

    # Find QUESTIONS and ANSWERS sections within each chapter
    for chapter in chapters:
        for line_num in range(chapter.line_start, chapter.line_end + 1):
            if line_num <= len(lines):
                line = lines[line_num - 1]
                if 'Q U E S T I O N S' in line and 'A N S W E R S' not in line:
                    chapter.questions_line_start = line_num
                elif 'A N S W E R S' in line:
                    if chapter.questions_line_start > 0 and chapter.questions_line_end == 0:
                        chapter.questions_line_end = line_num - 1
                    chapter.answers_line_start = line_num

        # Set end of answers section
        if chapter.answers_line_start > 0:
            chapter.answers_line_end = chapter.line_end

    return chapters


def parse_questions_in_chapter(
    lines: list[str],
    chapter: Chapter
) -> list[Question]:
    """
    Parse questions within a specific chapter.
    """
    questions = []

    if chapter.questions_line_start == 0:
        return questions

    # Patterns for question detection
    question_patterns = [
        re.compile(r'^-\s+(\d+[a-z]?)\s+(.+)', re.IGNORECASE),
        re.compile(r'^\d+\.\s+(\d+[a-z]?)\s+(.+)', re.IGNORECASE),
        re.compile(r'^(\d+)\.\s+(?![\d])(.{20,})', re.IGNORECASE),
    ]
    choice_pattern = re.compile(r'^-\s+([A-E])\.\s*(.+)')
    image_pattern = re.compile(r'<!--\s*image\s*-->')

    current_question = None

    start = chapter.questions_line_start
    end = chapter.questions_line_end if chapter.questions_line_end > 0 else chapter.answers_line_start

    for line_num in range(start, min(end + 1, len(lines) + 1)):
        line = lines[line_num - 1]

        # Check for image markers
        if image_pattern.search(line):
            if current_question:
                current_question.image_lines.append(line_num)

        # Check for new question
        for pattern in question_patterns:
            match = pattern.match(line)
            if match:
                q_id = match.group(1)
                q_text = match.group(2).strip()

                # Skip if looks like answer
                if 'Answer' in q_text[:20]:
                    continue

                # Skip if too short
                if len(q_text) < 15:
                    continue

                # Save previous question
                if current_question:
                    current_question.line_end = line_num - 1
                    questions.append(current_question)

                # Create new question with chapter-aware ID
                full_id = f"ch{chapter.number}_{q_id}"

                current_question = Question(
                    chapter_num=chapter.number,
                    local_id=q_id,
                    full_id=full_id,
                    text=q_text,
                    choices={},
                    line_start=line_num,
                    image_lines=[]
                )
                break

        # Check for choices
        choice_match = choice_pattern.match(line)
        if choice_match and current_question:
            letter = choice_match.group(1)
            text = choice_match.group(2).strip()
            if not text.startswith('Answer'):
                current_question.choices[letter] = text

    # Save last question
    if current_question:
        current_question.line_end = end
        questions.append(current_question)

    return questions


def link_images_to_questions(
    manifest: list[dict],
    chapters: list[Chapter],
    all_questions: list[Question],
    lines: list[str]
) -> dict[str, dict]:
    """
    Link images to questions with chapter awareness.

    Returns:
        Dict mapping image_filename → {question_full_id, chapter, confidence}
    """
    # Sort manifest by page and y-position
    manifest.sort(key=lambda x: (x["page"], x["y_position"]))

    # Find all image marker positions
    image_pattern = re.compile(r'<!--\s*image\s*-->')
    marker_positions = []  # (line_num, marker_index)

    marker_idx = 0
    for line_num, line in enumerate(lines, start=1):
        if image_pattern.search(line):
            marker_positions.append((line_num, marker_idx))
            marker_idx += 1

    # Create line → marker_index lookup
    line_to_marker = {pos[0]: pos[1] for pos in marker_positions}
    marker_to_line = {pos[1]: pos[0] for pos in marker_positions}

    # Build question lookup by line range
    # For each line, find which question (if any) it belongs to
    line_to_question = {}
    for q in all_questions:
        for line_num in range(q.line_start, q.line_end + 1):
            line_to_question[line_num] = q

    # Now link each image
    image_links = {}

    for i, img in enumerate(manifest):
        filename = img["filename"]

        # Get the line number for this image's marker
        if i in marker_to_line:
            line_num = marker_to_line[i]

            # Find the question this line belongs to
            if line_num in line_to_question:
                q = line_to_question[line_num]
                image_links[filename] = {
                    "question_full_id": q.full_id,
                    "question_local_id": q.local_id,
                    "chapter": q.chapter_num,
                    "confidence": "high",
                    "page": img["page"]
                }
            else:
                # Try to find nearest question
                nearest_q = None
                min_dist = float('inf')

                for q in all_questions:
                    dist = min(abs(line_num - q.line_start), abs(line_num - q.line_end))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_q = q

                if nearest_q and min_dist < 30:
                    image_links[filename] = {
                        "question_full_id": nearest_q.full_id,
                        "question_local_id": nearest_q.local_id,
                        "chapter": nearest_q.chapter_num,
                        "confidence": "medium",
                        "page": img["page"]
                    }
                else:
                    image_links[filename] = {
                        "question_full_id": None,
                        "question_local_id": None,
                        "chapter": None,
                        "confidence": "low",
                        "page": img["page"]
                    }
        else:
            # Image has no marker match
            image_links[filename] = {
                "question_full_id": None,
                "question_local_id": None,
                "chapter": None,
                "confidence": "none",
                "page": img["page"]
            }

    return image_links


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    markdown_path = sys.argv[1]
    manifest_path = sys.argv[2]

    print("=" * 60)
    print("Chapter-Aware Q&A Parser")
    print("=" * 60)

    # Read files
    print(f"\n1. Reading markdown: {markdown_path}")
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"   {len(lines)} lines")

    print(f"\n2. Reading manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    print(f"   {len(manifest)} images")

    # Detect chapters
    print("\n3. Detecting chapters...")
    chapters = detect_chapters(lines)
    print(f"   Found {len(chapters)} chapters:")
    for ch in chapters:
        print(f"      Ch{ch.number}: {ch.name[:50]}... (lines {ch.line_start}-{ch.line_end})")
        if ch.questions_line_start:
            print(f"             Questions: {ch.questions_line_start}-{ch.questions_line_end}")

    # Parse questions
    print("\n4. Parsing questions by chapter...")
    all_questions = []
    questions_by_chapter = {}

    for ch in chapters:
        questions = parse_questions_in_chapter(lines, ch)
        questions_by_chapter[f"ch{ch.number}"] = [
            {
                "full_id": q.full_id,
                "local_id": q.local_id,
                "text": q.text,
                "choices": q.choices,
                "image_count": len(q.image_lines)
            }
            for q in questions
        ]
        all_questions.extend(questions)
        print(f"      Ch{ch.number}: {len(questions)} questions")

    print(f"   Total: {len(all_questions)} questions")

    # Link images
    print("\n5. Linking images to questions...")
    image_links = link_images_to_questions(manifest, chapters, all_questions, lines)

    # Count by confidence
    high = sum(1 for v in image_links.values() if v["confidence"] == "high")
    medium = sum(1 for v in image_links.values() if v["confidence"] == "medium")
    low = sum(1 for v in image_links.values() if v["confidence"] in ["low", "none"])

    print(f"   High confidence: {high}")
    print(f"   Medium confidence: {medium}")
    print(f"   Low/None: {low}")

    # Save outputs
    os.makedirs("output", exist_ok=True)

    # Chapters
    chapters_file = "output/chapters.json"
    with open(chapters_file, "w") as f:
        json.dump([asdict(ch) for ch in chapters], f, indent=2)
    print(f"\n6. Saved chapters to: {chapters_file}")

    # Questions by chapter
    questions_file = "output/questions_by_chapter.json"
    with open(questions_file, "w") as f:
        json.dump(questions_by_chapter, f, indent=2)
    print(f"   Saved questions to: {questions_file}")

    # Image links
    links_file = "output/chapter_image_map.json"
    with open(links_file, "w") as f:
        json.dump(image_links, f, indent=2)
    print(f"   Saved image links to: {links_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for ch_key, questions in questions_by_chapter.items():
        ch_num = ch_key.replace("ch", "")
        ch_name = next((c.name[:40] for c in chapters if c.number == int(ch_num)), "Unknown")
        q_with_images = sum(1 for q in questions if q["image_count"] > 0)
        print(f"  {ch_key}: {len(questions):3d} questions, {q_with_images:3d} with images - {ch_name}...")


if __name__ == "__main__":
    main()
