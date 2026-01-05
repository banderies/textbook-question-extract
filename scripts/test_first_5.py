#!/usr/bin/env python3
"""
Test script: Extract first 5 Q&A pairs from the MSK book.

This demonstrates the agentic call pattern on real data.

Usage:
    python test_first_5.py
"""

import anthropic
import json
import re
import os

# The docling markdown file
MARKDOWN_PATH = "docling/Core Review - Musculoskeletal.md"


def extract_chapter1_content(markdown_path: str) -> tuple[str, str]:
    """
    Extract the first chapter's questions and answers from the markdown.

    Returns:
        Tuple of (questions_text, answers_text)
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the QUESTIONS section (note the spaced format)
    q_pattern = r'## Q U E S T I O N S'
    a_pattern = r'## A N S W E R S.*?A N D.*?E X P L A N A T I O N S'

    # Find first questions section
    q_match = re.search(q_pattern, content)
    if not q_match:
        raise ValueError("Could not find QUESTIONS section")

    # Find first answers section
    a_match = re.search(a_pattern, content)
    if not a_match:
        raise ValueError("Could not find ANSWERS section")

    # Extract questions (between QUESTIONS and ANSWERS headers)
    questions_start = q_match.end()
    questions_end = a_match.start()
    questions_text = content[questions_start:questions_end].strip()

    # For answers, find the next chapter marker or take a chunk
    answers_start = a_match.end()
    # Find next chapter (## 2 ...) or take first 5000 chars
    next_chapter = re.search(r'\n## \d+ ', content[answers_start:])
    if next_chapter:
        answers_end = answers_start + next_chapter.start()
    else:
        answers_end = answers_start + 10000

    answers_text = content[answers_start:answers_end].strip()

    return questions_text, answers_text


def extract_first_n_questions(questions_text: str, n: int = 5) -> str:
    """
    Extract the first N questions from the questions text.

    The format is like:
    - 1 Question text...
    - A. Choice A
    - B. Choice B
    - 2a Next question...
    """
    lines = questions_text.split('\n')

    # Find question boundaries (lines starting with "- " followed by number)
    question_starts = []
    for i, line in enumerate(lines):
        # Match patterns like "- 1 ", "- 2a ", "- 3 "
        if re.match(r'^-\s+\d+[a-z]?\s+\S', line):
            question_starts.append(i)

    if len(question_starts) < n:
        # Return all if we have fewer than n questions
        return questions_text

    # Get the line index where we should stop
    end_idx = question_starts[n] if n < len(question_starts) else len(lines)

    return '\n'.join(lines[:end_idx])


def extract_first_n_answers(answers_text: str, n: int = 5) -> str:
    """
    Extract the first N answers from the answers text.

    The format is like:
    - 1 Answer A. Explanation...
    """
    lines = answers_text.split('\n')

    # Find answer boundaries
    answer_starts = []
    for i, line in enumerate(lines):
        # Match patterns like "- 1 Answer", "- 2a Answer"
        if re.match(r'^-\s+\d+[a-z]?\s+Answer', line):
            answer_starts.append(i)

    if len(answer_starts) < n:
        return answers_text

    # Include everything up to but not including the (n+1)th answer
    end_idx = answer_starts[n] if n < len(answer_starts) else len(lines)

    return '\n'.join(lines[:end_idx])


def call_claude_for_qa_matching(questions: str, answers: str) -> list[dict]:
    """
    Send questions and answers to Claude and get matched pairs.
    """
    client = anthropic.Anthropic()

    system_prompt = """You are an expert at parsing medical education content.
Your task is to match questions to their corresponding answers.

The questions are from a radiology textbook. Each question has:
- A number (like 1, 2a, 2b, 3)
- The question text
- Multiple choice options (A, B, C, D, sometimes E)

The answers section has:
- The same number
- "Answer X" where X is the correct letter
- An explanation

Match them and output JSON:
[
  {
    "question_id": "1",
    "question_text": "Full question here",
    "choices": {
      "A": "Choice A text",
      "B": "Choice B text",
      "C": "Choice C text",
      "D": "Choice D text"
    },
    "correct_answer": "A",
    "explanation": "Full explanation text",
    "has_image": true/false,
    "confidence": "high/medium/low"
  }
]

Note: Questions with <!-- image --> markers have associated images.
Return ONLY valid JSON."""

    user_message = f"""
=== QUESTIONS (first 5) ===
{questions}

=== ANSWERS (first 5) ===
{answers}

Parse and match these Q&A pairs. Return JSON only."""

    print("Calling Claude API...")
    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    response_text = response.content[0].text
    print(f"Response received ({len(response_text)} chars)")

    # Parse JSON (handle markdown code blocks)
    if "```" in response_text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    return json.loads(response_text)


def main():
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Run: export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    print("=" * 60)
    print("Test: Extracting first 5 Q&A pairs from MSK book")
    print("=" * 60)

    # Step 1: Read the markdown
    print("\n1. Reading markdown file...")
    questions_full, answers_full = extract_chapter1_content(MARKDOWN_PATH)
    print(f"   Questions section: {len(questions_full)} chars")
    print(f"   Answers section: {len(answers_full)} chars")

    # Step 2: Extract first 5
    print("\n2. Extracting first 5 questions and answers...")
    questions_5 = extract_first_n_questions(questions_full, 5)
    answers_5 = extract_first_n_answers(answers_full, 5)
    print(f"   First 5 questions: {len(questions_5)} chars")
    print(f"   First 5 answers: {len(answers_5)} chars")

    # Show what we're sending
    print("\n--- QUESTIONS PREVIEW ---")
    print(questions_5[:500] + "..." if len(questions_5) > 500 else questions_5)
    print("\n--- ANSWERS PREVIEW ---")
    print(answers_5[:500] + "..." if len(answers_5) > 500 else answers_5)

    # Step 3: Call Claude
    print("\n3. Calling Claude API for Q&A matching...")
    qa_pairs = call_claude_for_qa_matching(questions_5, answers_5)

    # Step 4: Display results
    print(f"\n4. Results: {len(qa_pairs)} Q&A pairs extracted")
    print("=" * 60)

    for i, qa in enumerate(qa_pairs):
        print(f"\n--- Question {qa.get('question_id', i+1)} ---")
        print(f"Q: {qa.get('question_text', 'N/A')[:100]}...")
        print(f"Correct: {qa.get('correct_answer', '?')}")
        print(f"Has image: {qa.get('has_image', False)}")
        print(f"Confidence: {qa.get('confidence', 'unknown')}")

    # Save to file
    output_file = "output/test_first_5_results.json"
    os.makedirs("output", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"\n5. Saved to {output_file}")


if __name__ == "__main__":
    main()
