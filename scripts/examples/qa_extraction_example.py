#!/usr/bin/env python3
"""
Simple example: Extracting and matching Q&A pairs with Claude.

This demonstrates the core pattern of sending raw text with questions
and answers, and getting back structured JSON with matched pairs.

Usage:
    export ANTHROPIC_API_KEY='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'
    python qa_extraction_example.py
"""

import anthropic
import json
import re


def extract_qa_pairs(questions_text: str, answers_text: str) -> list[dict]:
    """
    Send questions and answers to Claude, get back matched pairs.

    This is the core "agentic" pattern - Claude uses semantic understanding
    to match questions to their answers, handling inconsistent formatting.

    Args:
        questions_text: Raw text containing questions
        answers_text: Raw text containing answers/explanations

    Returns:
        List of matched Q&A pairs as dictionaries
    """
    client = anthropic.Anthropic()

    # The system prompt defines Claude's role and output format
    system_prompt = """You are an expert at parsing educational content.
Your task is to match questions to their corresponding answers.

MATCHING STRATEGY:
- Use question numbers/IDs to match (e.g., "1" matches "Answer 1")
- Use semantic similarity when numbers are ambiguous
- Medical terms in questions should appear in explanations

OUTPUT: Return a JSON array where each element is:
{
    "question_id": "1",
    "question": "Full question text",
    "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "correct_answer": "B",
    "explanation": "Full explanation",
    "confidence": "high/medium/low"
}

Return ONLY valid JSON, no other text."""

    # The user message contains the actual content to process
    user_message = f"""
=== QUESTIONS ===
{questions_text}

=== ANSWERS ===
{answers_text}

Match each question to its answer and return JSON."""

    # Make the API call
    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    # Extract the response text
    response_text = response.content[0].text

    # Parse JSON (handle markdown code blocks)
    if "```" in response_text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    return json.loads(response_text)


# Example with sample data
if __name__ == "__main__":
    # Sample questions (mimicking textbook format)
    sample_questions = """
1. Which of the following MRI parameters would decrease metallic artifact?
   A. Increased field strength
   B. Increased receiver bandwidth
   C. Increased voxel size
   D. Decreased echo time

2. A 45-year-old presents with shoulder pain. Based on the image, what is the most likely diagnosis?
   A. Rotator cuff tear
   B. Labral tear
   C. Biceps tendinopathy
   D. Adhesive capsulitis

3. What is the recommended contrast dose for MR arthrography?
   A. 10-15 mL
   B. 20-25 mL
   C. 30-40 mL
   D. 5-8 mL
"""

    sample_answers = """
1. Answer B. Increasing the receiver bandwidth decreases metallic susceptibility
   artifact by restricting geometric distortion to a smaller region. Higher field
   strength actually increases artifact. Larger voxel size increases artifact.

2. Answer A. The MRI shows a full-thickness tear of the supraspinatus tendon,
   which is the most common location for rotator cuff tears. The tendon shows
   complete discontinuity with retraction.

3. Answer A. For MR arthrography, the recommended intra-articular contrast
   volume is typically 10-15 mL for the shoulder joint. This provides adequate
   joint distension without causing patient discomfort.
"""

    print("Extracting Q&A pairs...")
    print("=" * 50)

    qa_pairs = extract_qa_pairs(sample_questions, sample_answers)

    print(f"\nExtracted {len(qa_pairs)} Q&A pairs:\n")

    for qa in qa_pairs:
        print(f"Q{qa['question_id']}: {qa['question'][:60]}...")
        print(f"  Correct: {qa['correct_answer']}")
        print(f"  Confidence: {qa['confidence']}")
        print()

    # Save to file
    with open("sample_output.json", "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print("Saved to sample_output.json")
