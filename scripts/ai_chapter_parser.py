#!/usr/bin/env python3
"""
AI-Powered Chapter Parser

Uses Claude to intelligently detect chapter boundaries and parse
the document structure, handling inconsistent formatting.

Usage:
    python ai_chapter_parser.py <markdown_file> <manifest_json>

This script:
1. Samples the markdown to understand structure
2. Uses Claude to identify chapter patterns
3. Parses all chapters with chapter-aware question IDs
4. Links images to questions
"""

import anthropic
import json
import re
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Chapter:
    number: int
    name: str
    line_start: int
    line_end: int
    questions_start: int = 0
    questions_end: int = 0
    answers_start: int = 0
    answers_end: int = 0


@dataclass
class Question:
    chapter_num: int
    local_id: str
    full_id: str
    text: str
    choices: dict = field(default_factory=dict)
    line_start: int = 0
    line_end: int = 0


def get_structure_from_ai(markdown_sample: str, client: anthropic.Anthropic) -> dict:
    """
    Use Claude to analyze the markdown structure and identify chapter patterns.
    """
    prompt = f"""Analyze this markdown sample from a medical textbook and identify:

1. How chapters are delimited (what pattern marks chapter starts)
2. How QUESTIONS sections are marked
3. How ANSWERS sections are marked
4. The pattern for individual questions (e.g., "- 1 ", "5. 2a ")

Here's a sample of the markdown:

{markdown_sample[:8000]}

Return JSON:
{{
    "chapter_pattern": "description or regex of how chapters start",
    "questions_marker": "exact text that marks start of questions section",
    "answers_marker": "exact text that marks start of answers section",
    "question_patterns": ["list of regex patterns that match question starts"],
    "notes": "any other structural observations"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    # Extract JSON
    if "```" in text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            text = match.group(1)

    return json.loads(text)


def find_all_section_markers(lines: list[str]) -> dict:
    """
    Find all QUESTIONS and ANSWERS markers in the document.
    """
    markers = {
        "questions": [],
        "answers": [],
        "chapters": []
    }

    for i, line in enumerate(lines, start=1):
        if 'Q U E S T I O N S' in line and 'A N S W E R S' not in line:
            markers["questions"].append(i)
        if 'A N S W E R S' in line and 'E X P L A N A T I O N' in line:
            markers["answers"].append(i)
        # Look for chapter headers
        if re.match(r'^##\s*\d+\s+\w', line):
            markers["chapters"].append((i, line.strip()))

    return markers


def detect_chapters_from_markers(
    lines: list[str],
    markers: dict
) -> list[Chapter]:
    """
    Create chapter objects by pairing QUESTIONS and ANSWERS markers.

    The book has 9 chapters. Each chapter has one QUESTIONS and one ANSWERS section.
    """
    chapters = []

    questions_lines = markers["questions"]
    answers_lines = markers["answers"]

    # Each chapter should have a QUESTIONS section followed by ANSWERS
    # Pair them up
    pairs = []
    for q_line in questions_lines:
        # Find the next ANSWERS line after this QUESTIONS line
        for a_line in answers_lines:
            if a_line > q_line:
                pairs.append((q_line, a_line))
                break

    # Now create chapters from pairs
    for i, (q_start, a_start) in enumerate(pairs):
        # Chapter number is 1-indexed
        ch_num = i + 1

        # Find chapter end (start of next chapter's questions, or end of file)
        if i + 1 < len(pairs):
            ch_end = pairs[i + 1][0] - 1
        else:
            ch_end = len(lines)

        # Try to find chapter name by looking backwards from QUESTIONS
        ch_name = f"Chapter {ch_num}"
        for j in range(q_start - 1, max(0, q_start - 50), -1):
            line = lines[j - 1] if j > 0 else ""
            # Look for chapter header pattern
            match = re.match(r'^##\s*\d*\s*(.+)', line)
            if match and len(line) < 150:
                ch_name = match.group(1).strip()[:80]
                break

        # Find where questions end (just before answers)
        q_end = a_start - 1

        # Find where answers end (next chapter or EOF)
        a_end = ch_end

        chapter = Chapter(
            number=ch_num,
            name=ch_name,
            line_start=q_start - 50 if q_start > 50 else 1,  # Include some context before
            line_end=ch_end,
            questions_start=q_start,
            questions_end=q_end,
            answers_start=a_start,
            answers_end=a_end
        )
        chapters.append(chapter)

    return chapters


def parse_questions(
    lines: list[str],
    chapter: Chapter
) -> list[Question]:
    """
    Parse questions from a chapter's QUESTIONS section.
    """
    questions = []

    # Patterns
    patterns = [
        re.compile(r'^-\s+(\d+[a-z]?)\s+(.{20,})', re.IGNORECASE),
        re.compile(r'^\d+\.\s+(\d+[a-z]?)\s+(.{20,})', re.IGNORECASE),
        re.compile(r'^(\d+)\.\s+([A-Z].{20,})', re.IGNORECASE),
    ]
    choice_pattern = re.compile(r'^-\s+([A-E])\.\s*(.+)')

    current_q = None

    for line_num in range(chapter.questions_start + 1, chapter.questions_end + 1):
        if line_num > len(lines):
            break

        line = lines[line_num - 1]

        # Skip answer-like lines
        if 'Answer' in line[:30]:
            continue

        # Check for new question
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                q_id = match.group(1)
                q_text = match.group(2).strip()

                # Save previous
                if current_q:
                    current_q.line_end = line_num - 1
                    questions.append(current_q)

                full_id = f"ch{chapter.number}_{q_id}"
                current_q = Question(
                    chapter_num=chapter.number,
                    local_id=q_id,
                    full_id=full_id,
                    text=q_text,
                    choices={},
                    line_start=line_num
                )
                break

        # Check for choices
        choice_match = choice_pattern.match(line)
        if choice_match and current_q:
            letter = choice_match.group(1)
            text = choice_match.group(2).strip()
            if not text.startswith('Answer'):
                current_q.choices[letter] = text

    # Save last
    if current_q:
        current_q.line_end = chapter.questions_end
        questions.append(current_q)

    return questions


def link_images(
    manifest: list[dict],
    chapters: list[Chapter],
    questions: list[Question],
    lines: list[str]
) -> dict:
    """
    Link images to questions.
    """
    manifest.sort(key=lambda x: (x["page"], x["y_position"]))

    # Find image markers
    image_pattern = re.compile(r'<!--\s*image\s*-->')
    marker_to_line = {}
    marker_idx = 0

    for line_num, line in enumerate(lines, start=1):
        if image_pattern.search(line):
            marker_to_line[marker_idx] = line_num
            marker_idx += 1

    # Build line → question lookup
    line_to_q = {}
    for q in questions:
        for ln in range(q.line_start, q.line_end + 1):
            line_to_q[ln] = q

    # Link images
    links = {}
    for i, img in enumerate(manifest):
        fname = img["filename"]

        if i in marker_to_line:
            ln = marker_to_line[i]

            if ln in line_to_q:
                q = line_to_q[ln]
                links[fname] = {
                    "question_full_id": q.full_id,
                    "question_local_id": q.local_id,
                    "chapter": q.chapter_num,
                    "confidence": "high",
                    "page": img["page"]
                }
                continue

            # Find nearest question
            min_dist = float('inf')
            nearest = None
            for q in questions:
                d = min(abs(ln - q.line_start), abs(ln - q.line_end))
                if d < min_dist:
                    min_dist = d
                    nearest = q

            if nearest and min_dist < 50:
                links[fname] = {
                    "question_full_id": nearest.full_id,
                    "question_local_id": nearest.local_id,
                    "chapter": nearest.chapter_num,
                    "confidence": "medium",
                    "page": img["page"]
                }
            else:
                links[fname] = {
                    "question_full_id": None,
                    "question_local_id": None,
                    "chapter": None,
                    "confidence": "low",
                    "page": img["page"]
                }
        else:
            links[fname] = {
                "question_full_id": None,
                "question_local_id": None,
                "chapter": None,
                "confidence": "none",
                "page": img["page"]
            }

    return links


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    markdown_path = sys.argv[1]
    manifest_path = sys.argv[2]

    print("=" * 60)
    print("AI-Powered Chapter Parser")
    print("=" * 60)

    # Read files
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"\n1. Read {len(lines)} lines from markdown")

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"   Read {len(manifest)} images from manifest")

    # Find section markers
    print("\n2. Finding section markers...")
    markers = find_all_section_markers(lines)
    print(f"   Found {len(markers['questions'])} QUESTIONS sections")
    print(f"   Found {len(markers['answers'])} ANSWERS sections")

    # Detect chapters
    print("\n3. Detecting chapters from markers...")
    chapters = detect_chapters_from_markers(lines, markers)
    print(f"   Found {len(chapters)} chapters:")

    for ch in chapters:
        print(f"      Ch{ch.number}: {ch.name[:45]}...")
        print(f"             Q: {ch.questions_start}-{ch.questions_end}, A: {ch.answers_start}-{ch.answers_end}")

    # Parse questions
    print("\n4. Parsing questions...")
    all_questions = []
    questions_by_chapter = {}

    for ch in chapters:
        qs = parse_questions(lines, ch)
        questions_by_chapter[f"ch{ch.number}"] = [
            {"full_id": q.full_id, "local_id": q.local_id, "text": q.text, "choices": q.choices}
            for q in qs
        ]
        all_questions.extend(qs)
        print(f"      Ch{ch.number}: {len(qs)} questions")

    print(f"   Total: {len(all_questions)} questions")

    # Link images
    print("\n5. Linking images...")
    links = link_images(manifest, chapters, all_questions, lines)

    high = sum(1 for v in links.values() if v["confidence"] == "high")
    med = sum(1 for v in links.values() if v["confidence"] == "medium")
    low = sum(1 for v in links.values() if v["confidence"] in ["low", "none"])
    print(f"   High: {high}, Medium: {med}, Low: {low}")

    # Save
    os.makedirs("output", exist_ok=True)

    with open("output/chapters.json", "w") as f:
        json.dump([asdict(ch) for ch in chapters], f, indent=2)

    with open("output/questions_by_chapter.json", "w") as f:
        json.dump(questions_by_chapter, f, indent=2)

    with open("output/chapter_image_map.json", "w") as f:
        json.dump(links, f, indent=2)

    print("\n6. Saved to output/")
    print("   - chapters.json")
    print("   - questions_by_chapter.json")
    print("   - chapter_image_map.json")


if __name__ == "__main__":
    main()
