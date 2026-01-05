#!/usr/bin/env python3
"""
Agentic Q&A Extractor using Claude API (Opus 4.5)

This script demonstrates how to make programmatic LLM calls for:
1. Extracting questions and answers from any textbook
2. Matching questions to their answers semantically
3. Associating images with questions using vision capabilities

Usage:
    python agentic_qa_extractor.py <markdown_file> [--images-dir <path>]

Environment:
    ANTHROPIC_API_KEY - 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

Author: Book-agnostic Anki generator
"""

import anthropic
import json
import base64
import os
import sys
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Choice:
    label: str  # A, B, C, D, E
    text: str

@dataclass
class Question:
    id: str
    text: str
    choices: list[Choice]
    figure_ref: Optional[str] = None  # e.g., "Figure 37" or "37a"
    page_hint: Optional[int] = None

@dataclass
class Answer:
    question_id: str
    correct_choice: str  # A, B, C, D, E
    explanation: str
    references: Optional[str] = None

@dataclass
class QAPair:
    question: Question
    answer: Answer
    matched_images: list[str]  # filenames
    confidence: str  # high, medium, low


# =============================================================================
# ANTHROPIC API CLIENT
# =============================================================================

class ClaudeAgent:
    """
    Agentic wrapper around Claude API for structured Q&A extraction.

    This demonstrates the key pattern: send context + instructions,
    receive structured JSON output.
    """

    def __init__(self, model: str = "claude-opus-4-5-20251101"):
        """
        Initialize the Claude agent.

        Args:
            model: Model to use. Options:
                - claude-opus-4-5-20251101 (most capable, best for complex matching)
                - claude-sonnet-4-20250514 (faster, good for simpler tasks)
                - claude-3-5-haiku-20241022 (fastest, cheapest)
        """
        self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        self.model = model

    def extract_qa_pairs(
        self,
        questions_text: str,
        answers_text: str,
        book_context: Optional[str] = None
    ) -> list[dict]:
        """
        Extract and match Q&A pairs using semantic understanding.

        This is the core agentic call - Claude reads both sections
        and uses semantic understanding to match them.

        Args:
            questions_text: Raw text containing questions
            answers_text: Raw text containing answers/explanations
            book_context: Optional context about the book (title, subject)

        Returns:
            List of matched Q&A pairs as dictionaries
        """

        system_prompt = """You are an expert at extracting and matching educational content.
Your task is to:
1. Parse the QUESTIONS section to identify each question with its choices
2. Parse the ANSWERS section to identify each answer with its explanation
3. Match questions to answers using semantic understanding

MATCHING STRATEGY:
- Look for explicit identifiers (question numbers like "1", "2a", "37")
- Match figure/image references between questions and answers
- Use content similarity when identifiers are ambiguous
- Medical/technical terms in questions should appear in their explanations

OUTPUT FORMAT:
Return a JSON array where each element has this structure:
{
    "question_id": "1",
    "question_text": "Full question text here",
    "choices": [
        {"label": "A", "text": "First choice"},
        {"label": "B", "text": "Second choice"},
        {"label": "C", "text": "Third choice"},
        {"label": "D", "text": "Fourth choice"},
        {"label": "E", "text": "Fifth choice (if present)"}
    ],
    "correct_answer": "A",
    "explanation": "Full explanation text",
    "references": "Any academic references cited",
    "figure_reference": "37a (if question references a figure)",
    "confidence": "high/medium/low"
}

CONFIDENCE LEVELS:
- high: Clear ID match AND content correlates well
- medium: ID match OR content match, but not both
- low: Best guess based on position or partial matching

Return ONLY valid JSON, no other text."""

        user_message = f"""
{f'BOOK CONTEXT: {book_context}' if book_context else ''}

=== QUESTIONS SECTION ===
{questions_text}

=== ANSWERS SECTION ===
{answers_text}

Extract and match all question-answer pairs. Return JSON only."""

        # Make the API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            messages=[
                {"role": "user", "content": user_message}
            ],
            system=system_prompt
        )

        # Extract JSON from response
        response_text = response.content[0].text

        # Parse JSON (handle potential markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        return json.loads(json_str)

    def match_image_to_question(
        self,
        image_path: str,
        questions: list[dict],
        book_context: Optional[str] = None
    ) -> dict:
        """
        Use Claude's vision capability to match an image to a question.

        This demonstrates multimodal agentic calls - sending images
        along with text for intelligent matching.

        Args:
            image_path: Path to the image file
            questions: List of question dictionaries to match against
            book_context: Optional context about the book

        Returns:
            Dictionary with matching results
        """

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Determine media type
        ext = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_types.get(ext, "image/jpeg")

        # Format questions for the prompt
        questions_summary = "\n".join([
            f"Q{q.get('question_id', i)}: {q.get('question_text', '')[:200]}..."
            for i, q in enumerate(questions)
        ])

        system_prompt = """You are an expert at matching medical/educational images to questions.
Analyze the image and determine which question(s) it most likely belongs to.

Return JSON with this structure:
{
    "matched_question_ids": ["1", "2a"],
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of why this image matches these questions",
    "image_description": "Brief description of what the image shows"
}

Consider:
- Anatomical structures visible in the image
- Imaging modality (X-ray, CT, MRI, ultrasound)
- Any pathology or abnormality shown
- Labels or annotations in the image"""

        user_content = [
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
                "text": f"""
{f'BOOK CONTEXT: {book_context}' if book_context else ''}

QUESTIONS TO MATCH AGAINST:
{questions_summary}

Which question(s) does this image belong to? Return JSON only."""
            }
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_content}
            ],
            system=system_prompt
        )

        response_text = response.content[0].text

        # Parse JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        return json.loads(json_str)

    def detect_book_structure(self, markdown_content: str) -> dict:
        """
        Analyze markdown to detect book structure (chapters, Q&A sections).

        This makes the tool book-agnostic by letting Claude figure out
        the structure rather than hardcoding patterns.

        Args:
            markdown_content: Full or partial markdown from the book

        Returns:
            Dictionary describing detected structure
        """

        # Only send first ~10000 chars for structure detection
        sample = markdown_content[:10000]

        system_prompt = """You are an expert at analyzing document structure.
Analyze this sample and identify:
1. How chapters are delimited (heading patterns)
2. How questions are formatted (numbering scheme)
3. How answers are formatted
4. Any image reference patterns

Return JSON:
{
    "chapter_pattern": "regex or description",
    "question_section_marker": "e.g., 'QUESTIONS' or '## Questions'",
    "answer_section_marker": "e.g., 'ANSWERS AND EXPLANATIONS'",
    "question_numbering": "e.g., '1, 2, 3' or '1a, 1b, 2a'",
    "image_reference_pattern": "e.g., 'Figure X' or '(see image)'",
    "chapters_detected": ["Chapter 1 Name", "Chapter 2 Name"],
    "estimated_questions_per_chapter": 20,
    "notes": "Any other structural observations"
}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": f"Analyze this document structure:\n\n{sample}"}
            ],
            system=system_prompt
        )

        response_text = response.content[0].text
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        return json.loads(json_str)


# =============================================================================
# BOOK-AGNOSTIC PROCESSOR
# =============================================================================

class BookAgnosticProcessor:
    """
    Process any textbook with Q&A format into structured data.

    This class coordinates the agentic workflow:
    1. Detect book structure
    2. Split into chapters
    3. Extract Q&A pairs per chapter
    4. Match images to questions
    """

    def __init__(self, model: str = "claude-opus-4-5-20251101"):
        self.agent = ClaudeAgent(model=model)
        self.structure = None

    def process_book(
        self,
        markdown_path: str,
        images_dir: Optional[str] = None,
        output_dir: str = "output"
    ) -> list[dict]:
        """
        Full pipeline to process a book.

        Args:
            markdown_path: Path to markdown file from docling
            images_dir: Optional directory with extracted images
            output_dir: Where to save JSON outputs

        Returns:
            List of all Q&A pairs with image associations
        """

        Path(output_dir).mkdir(exist_ok=True)

        # Read markdown
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        print("Step 1: Detecting book structure...")
        self.structure = self.agent.detect_book_structure(content)
        print(f"  Found patterns: {self.structure}")

        # Save structure for reference
        with open(f"{output_dir}/structure.json", "w") as f:
            json.dump(self.structure, f, indent=2)

        print("\nStep 2: Splitting into chapters...")
        chapters = self._split_chapters(content)
        print(f"  Found {len(chapters)} chapters")

        all_qa_pairs = []

        for i, chapter in enumerate(chapters):
            print(f"\nStep 3.{i+1}: Processing chapter '{chapter['name']}'...")

            # Extract Q&A pairs for this chapter
            qa_pairs = self.agent.extract_qa_pairs(
                questions_text=chapter["questions"],
                answers_text=chapter["answers"],
                book_context=chapter["name"]
            )

            print(f"  Extracted {len(qa_pairs)} Q&A pairs")

            # Add chapter info
            for qa in qa_pairs:
                qa["chapter"] = chapter["name"]

            all_qa_pairs.extend(qa_pairs)

            # Save chapter output
            chapter_file = f"{output_dir}/chapter_{i+1:02d}_qa.json"
            with open(chapter_file, "w") as f:
                json.dump({"chapter": chapter["name"], "questions": qa_pairs}, f, indent=2)
            print(f"  Saved to {chapter_file}")

        # Step 4: Match images if directory provided
        if images_dir and Path(images_dir).exists():
            print("\nStep 4: Matching images to questions...")
            all_qa_pairs = self._match_images(all_qa_pairs, images_dir)

        # Save combined output
        combined_file = f"{output_dir}/all_qa_pairs.json"
        with open(combined_file, "w") as f:
            json.dump(all_qa_pairs, f, indent=2)
        print(f"\nSaved {len(all_qa_pairs)} total Q&A pairs to {combined_file}")

        return all_qa_pairs

    def _split_chapters(self, content: str) -> list[dict]:
        """
        Split markdown into chapters based on detected structure.

        This is a heuristic approach - you may need to adjust
        based on specific book formats.
        """
        chapters = []

        # Get markers from detected structure
        q_marker = self.structure.get("question_section_marker", "QUESTIONS")
        a_marker = self.structure.get("answer_section_marker", "ANSWERS")

        # Split by common chapter patterns
        # This regex looks for numbered chapter headings
        chapter_pattern = r'(?:^|\n)(#{1,3}\s*\d+\s+[A-Z][^\n]+)'

        parts = re.split(chapter_pattern, content)

        current_chapter = None
        current_content = []

        for part in parts:
            if re.match(r'#{1,3}\s*\d+\s+[A-Z]', part.strip()):
                # This is a chapter heading
                if current_chapter and current_content:
                    chapters.append({
                        "name": current_chapter,
                        "content": "\n".join(current_content)
                    })
                current_chapter = part.strip().lstrip('#').strip()
                current_content = []
            else:
                current_content.append(part)

        # Don't forget last chapter
        if current_chapter and current_content:
            chapters.append({
                "name": current_chapter,
                "content": "\n".join(current_content)
            })

        # Now split each chapter into questions and answers
        for chapter in chapters:
            content = chapter["content"]

            # Find questions and answers sections
            q_pattern = re.compile(rf'{q_marker}', re.IGNORECASE)
            a_pattern = re.compile(rf'{a_marker}', re.IGNORECASE)

            q_match = q_pattern.search(content)
            a_match = a_pattern.search(content)

            if q_match and a_match:
                chapter["questions"] = content[q_match.end():a_match.start()]
                chapter["answers"] = content[a_match.end():]
            else:
                # Fallback: treat first half as questions, second as answers
                mid = len(content) // 2
                chapter["questions"] = content[:mid]
                chapter["answers"] = content[mid:]

        return chapters

    def _match_images(self, qa_pairs: list[dict], images_dir: str) -> list[dict]:
        """
        Match images to questions using vision API.

        For efficiency, this batches questions and processes
        images in groups.
        """
        image_files = list(Path(images_dir).glob("*.jpg")) + \
                      list(Path(images_dir).glob("*.png"))

        print(f"  Found {len(image_files)} images to match")

        # Create question lookup
        questions_by_id = {q.get("question_id"): q for q in qa_pairs}

        # Match each image
        for img_path in image_files[:10]:  # Limit for demo
            try:
                result = self.agent.match_image_to_question(
                    str(img_path),
                    qa_pairs[:20]  # Sample of questions for matching
                )

                matched_ids = result.get("matched_question_ids", [])
                for qid in matched_ids:
                    if qid in questions_by_id:
                        if "matched_images" not in questions_by_id[qid]:
                            questions_by_id[qid]["matched_images"] = []
                        questions_by_id[qid]["matched_images"].append(img_path.name)

                print(f"    {img_path.name} -> Q{matched_ids}")

            except Exception as e:
                print(f"    Warning: Could not match {img_path.name}: {e}")

        return qa_pairs


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point for CLI usage."""

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  export ANTHROPIC_API_KEY='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'")
        print("  python agentic_qa_extractor.py docling/book.md --images-dir images/")
        sys.exit(1)

    markdown_path = sys.argv[1]

    # Parse optional args
    images_dir = None
    if "--images-dir" in sys.argv:
        idx = sys.argv.index("--images-dir")
        if idx + 1 < len(sys.argv):
            images_dir = sys.argv[idx + 1]

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("\nTo get an API key:")
        print("1. Go to https://console.anthropic.com/")
        print("2. Create an account or sign in")
        print("3. Go to API Keys and create a new key")
        print("4. Export it: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # Process the book
    processor = BookAgnosticProcessor(model="claude-opus-4-5-20251101")

    qa_pairs = processor.process_book(
        markdown_path=markdown_path,
        images_dir=images_dir,
        output_dir="output"
    )

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total Q&A pairs: {len(qa_pairs)}")
    print(f"Output saved to: output/")


if __name__ == "__main__":
    main()
