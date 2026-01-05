#!/usr/bin/env python3
"""
Simple example: Using Claude Vision API to match images to questions.

This demonstrates the core pattern of sending an image + questions
and getting back structured matching results.

Usage:
    export ANTHROPIC_API_KEY='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'
    python vision_matching_example.py image.jpg
"""

import anthropic
import base64
import json
import sys
from pathlib import Path


def match_image_to_questions(image_path: str, questions: list[str]) -> dict:
    """
    Send an image and list of questions to Claude, ask which question
    the image belongs to.

    Args:
        image_path: Path to image file (jpg, png)
        questions: List of question texts

    Returns:
        Dict with matched question indices and reasoning
    """
    client = anthropic.Anthropic()

    # Step 1: Encode image as base64
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type from extension
    ext = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "image/jpeg")

    # Step 2: Format questions for the prompt
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    # Step 3: Build the multimodal message
    message_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            }
        },
        {
            "type": "text",
            "text": f"""Look at this medical/educational image and determine which question(s)
it most likely belongs to.

QUESTIONS:
{questions_text}

Return JSON only:
{{
    "matched_indices": [1, 2],  // 1-indexed question numbers
    "confidence": "high/medium/low",
    "image_description": "What the image shows",
    "reasoning": "Why it matches these questions"
}}"""
        }
    ]

    # Step 4: Make the API call
    response = client.messages.create(
        model="claude-opus-4-5-20251101",  # Vision requires Sonnet or Opus
        max_tokens=1024,
        messages=[{"role": "user", "content": message_content}],
        system="You are an expert at analyzing medical images and matching them to questions. Return only valid JSON."
    )

    # Step 5: Parse the response
    response_text = response.content[0].text

    # Handle potential markdown code blocks
    if "```" in response_text:
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    return json.loads(response_text)


# Example usage
if __name__ == "__main__":
    # Sample questions (these would come from your extracted Q&A pairs)
    sample_questions = [
        "What MRI finding is shown in this image of the shoulder?",
        "Which anatomical structure is indicated by the arrow?",
        "What is the diagnosis based on this radiograph?",
        "Which imaging modality was used for this examination?",
    ]

    if len(sys.argv) < 2:
        print("Usage: python vision_matching_example.py <image_path>")
        print("\nThis will match the image to one of these sample questions:")
        for i, q in enumerate(sample_questions, 1):
            print(f"  {i}. {q}")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Analyzing image: {image_path}")
    print("-" * 50)

    result = match_image_to_questions(image_path, sample_questions)

    print(f"Image description: {result.get('image_description')}")
    print(f"Matched to questions: {result.get('matched_indices')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Reasoning: {result.get('reasoning')}")
