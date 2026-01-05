#!/usr/bin/env python3
"""
Parse questions from markdown for the review GUI.

This creates a JSON file with question IDs and their full text,
which the review GUI uses to display question context.

Usage:
    python parse_questions_for_gui.py <markdown_file>
"""

import json
import re
import sys
import os


def parse_questions(markdown_path: str) -> dict:
    """
    Parse all questions from the markdown file.

    Handles multiple chapters, each with their own QUESTIONS and ANSWERS sections.
    Only extracts from QUESTIONS sections, not ANSWERS.

    Returns:
        Dict mapping question_id → {text, choices}
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    questions = {}

    # Patterns
    question_patterns = [
        re.compile(r'^-\s+(\d+[a-z]?)\s+(.+)', re.IGNORECASE),
        re.compile(r'^\d+\.\s+(\d+[a-z]?)\s+(.+)', re.IGNORECASE),
        re.compile(r'^(\d+)\.\s+(?![\d])(.+)', re.IGNORECASE),
    ]
    choice_pattern = re.compile(r'^-\s+([A-E])\.\s*(.+)')

    in_questions_section = False
    current_q_id = None
    current_q_text = ""
    current_choices = {}

    for line_num, line in enumerate(lines, start=1):
        # Track sections - handle multiple chapters
        if 'Q U E S T I O N S' in line and 'A N S W E R S' not in line:
            in_questions_section = True
            # Save any pending question when entering new section
            if current_q_id and current_q_text:
                questions[current_q_id] = {
                    "text": current_q_text.strip(),
                    "choices": current_choices.copy(),
                }
            current_q_id = None
            current_q_text = ""
            current_choices = {}
            continue

        if 'A N S W E R S' in line:
            # Save last question before leaving questions section
            if current_q_id and current_q_text:
                questions[current_q_id] = {
                    "text": current_q_text.strip(),
                    "choices": current_choices.copy(),
                }
            in_questions_section = False
            current_q_id = None
            current_q_text = ""
            current_choices = {}
            continue

        if not in_questions_section:
            continue

        # Skip lines that look like answers (contain "Answer")
        if 'Answer' in line and re.match(r'.*Answer\s+[A-E]\.', line):
            continue

        # Check for new question
        matched = False
        for pattern in question_patterns:
            match = pattern.match(line)
            if match:
                q_id = match.group(1)
                q_text = match.group(2).strip()

                # Skip if it looks like an answer
                if q_text.startswith('Answer') or 'Answer' in q_text[:20]:
                    continue

                # Skip if text is too short (probably not a real question)
                if len(q_text) < 15:
                    continue

                # Save previous question
                if current_q_id and current_q_text:
                    questions[current_q_id] = {
                        "text": current_q_text.strip(),
                        "choices": current_choices.copy(),
                    }

                current_q_id = q_id
                current_q_text = q_text
                current_choices = {}
                matched = True
                break

        if matched:
            continue

        # Check for choices
        choice_match = choice_pattern.match(line)
        if choice_match and current_q_id:
            choice_letter = choice_match.group(1)
            choice_text = choice_match.group(2).strip()
            # Skip if choice text looks like an answer explanation
            if not choice_text.startswith('Answer'):
                current_choices[choice_letter] = choice_text

    # Save final question
    if current_q_id and current_q_text:
        questions[current_q_id] = {
            "text": current_q_text.strip(),
            "choices": current_choices.copy(),
        }

    return questions


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_questions_for_gui.py <markdown_file>")
        sys.exit(1)

    markdown_path = sys.argv[1]

    print(f"Parsing questions from: {markdown_path}")
    questions = parse_questions(markdown_path)

    print(f"Found {len(questions)} questions")

    # Save to output
    os.makedirs("output", exist_ok=True)
    output_file = "output/parsed_questions.json"

    with open(output_file, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"Saved to: {output_file}")

    # Show sample
    print("\nSample questions:")
    for q_id in list(questions.keys())[:5]:
        q = questions[q_id]
        print(f"  Q{q_id}: {q['text'][:60]}...")
        if q['choices']:
            print(f"    Choices: {', '.join(q['choices'].keys())}")


if __name__ == "__main__":
    main()
