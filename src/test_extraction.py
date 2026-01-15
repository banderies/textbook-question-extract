#!/usr/bin/env python3
"""
Headless test script for the textbook extraction pipeline.
Run this to test extraction without Streamlit UI.

Usage:
    python test_extraction.py [--chapter N] [--model MODEL]

Examples:
    python test_extraction.py --chapter 1 --model claude-3-5-haiku-20241022
    python test_extraction.py --chapter 1
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from anthropic import Anthropic
from llm_extraction import (
    generate_cloze_cards_from_block_llm,
    identify_question_blocks_llm,
    format_raw_block_llm,
    get_prompt,
    load_prompts
)
from pdf_extraction import extract_lines_by_range_mapped


def get_client():
    """Get Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return Anthropic(api_key=api_key)


def load_existing_data(textbook_name: str):
    """Load existing extracted data for a textbook."""
    # Handle both running from scripts/ and from project root
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = script_dir.parent
    output_dir = project_root / "output" / textbook_name

    data = {
        "chapters": [],
        "chapter_text": {},
        "pages": {}
    }

    # Load chapters
    chapters_file = output_dir / "chapters.json"
    if chapters_file.exists():
        with open(chapters_file) as f:
            data["chapters"] = json.load(f)

    # Load chapter text
    chapter_text_file = output_dir / "chapter_text.json"
    if chapter_text_file.exists():
        with open(chapter_text_file) as f:
            data["chapter_text"] = json.load(f)

    # Load pages
    pages_file = output_dir / "pages.json"
    if pages_file.exists():
        with open(pages_file) as f:
            data["pages"] = json.load(f)

    return data


def get_chapter_text_with_lines(data: dict, chapter_num: int) -> str:
    """Get chapter text with line numbers for extraction."""
    chapters = data["chapters"]
    chapter_text_data = data.get("chapter_text", {})

    # Find chapter
    chapter = next((c for c in chapters if c["chapter_number"] == chapter_num), None)
    if not chapter:
        raise ValueError(f"Chapter {chapter_num} not found")

    # Try to get chapter text from chapter_text.json
    ch_key = f"ch{chapter_num}"
    if ch_key in chapter_text_data:
        raw_text = chapter_text_data[ch_key]
    else:
        # Fall back to pages
        pages = data["pages"]
        ch_start = chapter["start_page"]
        ch_idx = next(i for i, c in enumerate(chapters) if c["chapter_number"] == chapter_num)
        ch_end = chapters[ch_idx + 1]["start_page"] if ch_idx + 1 < len(chapters) else 9999

        text_parts = []
        for page_num in range(ch_start, ch_end):
            page_key = str(page_num)
            if page_key in pages:
                text_parts.append(pages[page_key])
        raw_text = '\n'.join(text_parts)

    # Add line numbers
    lines = []
    line_num = 1
    for line in raw_text.split('\n'):
        lines.append(f"[LINE:{line_num:04d}] {line}")
        line_num += 1

    return '\n'.join(lines)


# =============================================================================
# Block Extraction Tests
# =============================================================================

def test_block_identification(client, chapter_num: int, chapter_text: str, model_id: str):
    """Test the block identification (identify_question_blocks_llm)."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Identifying question blocks for Chapter {chapter_num}")
    print(f"{'='*60}")
    print(f"Using model: {model_id}")
    print(f"Chapter text length: {len(chapter_text)} chars")

    # Debug: Check if chapter_text starts correctly
    print(f"  First 200 chars: {chapter_text[:200]}...")

    try:
        # Call without progress callback (more reliable)
        blocks = identify_question_blocks_llm(
            client,
            chapter_num,
            chapter_text,
            model_id,
            on_progress=None
        )

        print(f"\nIdentified {len(blocks)} blocks:")
        for i, b in enumerate(blocks[:10]):  # Show first 10
            block_id = b.get("block_id", "?")
            q_start = b.get("question_start", 0)
            q_end = b.get("question_end", 0)
            a_start = b.get("answer_start", 0)
            a_end = b.get("answer_end", 0)
            print(f"  Block {block_id}: Q lines {q_start}-{q_end}, A lines {a_start}-{a_end}")

        if len(blocks) > 10:
            print(f"  ... and {len(blocks) - 10} more blocks")

        return blocks

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_extract_raw_text(blocks: list, chapter_text: str, chapter_num: int):
    """Test extracting raw text with line numbers preserved."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Extracting raw text with line numbers preserved")
    print(f"{'='*60}")

    lines = chapter_text.split('\n')
    # Create line mapping (1-indexed to array index)
    line_mapping = {i+1: i for i in range(len(lines))}

    raw_blocks = []

    for i, b in enumerate(blocks[:3]):  # Test first 3
        block_id = b.get("block_id", "?")
        q_start = b.get("question_start", 0)
        q_end = b.get("question_end", 0)
        a_start = b.get("answer_start", 0)
        a_end = b.get("answer_end", 0)

        print(f"\n  Block {block_id}:")

        # Extract question text WITH line numbers preserved
        if q_start > 0 and q_end > 0:
            question_text_raw = extract_lines_by_range_mapped(
                lines, q_start, q_end, line_mapping, preserve_line_numbers=True
            )
            print(f"    Q text (lines {q_start}-{q_end}, {len(question_text_raw)} chars)")
            print(f"    First 200 chars: {question_text_raw[:200]}...")
        else:
            question_text_raw = ""
            print(f"    Q text: (no lines)")

        # Extract answer text WITH line numbers preserved
        if a_start > 0 and a_end > 0:
            answer_text_raw = extract_lines_by_range_mapped(
                lines, a_start, a_end, line_mapping, preserve_line_numbers=True
            )
            print(f"    A text (lines {a_start}-{a_end}, {len(answer_text_raw)} chars)")
            print(f"    First 200 chars: {answer_text_raw[:200]}...")
        else:
            answer_text_raw = ""
            print(f"    A text: (no lines)")

        # Verify line numbers are preserved
        if question_text_raw and "[LINE:" not in question_text_raw:
            print(f"    WARNING: Line numbers not preserved in question text!")
        if answer_text_raw and "[LINE:" not in answer_text_raw:
            print(f"    WARNING: Line numbers not preserved in answer text!")

        raw_blocks.append({
            "block_id": f"ch{chapter_num}_{block_id}",
            "block_label": block_id,
            "chapter": chapter_num,
            "question_start": q_start,
            "question_end": q_end,
            "answer_start": a_start,
            "answer_end": a_end,
            "question_text_raw": question_text_raw,
            "answer_text_raw": answer_text_raw,
            "formatted": False
        })

    return raw_blocks


def test_format_block(client, raw_blocks: list, chapter_num: int, model_id: str):
    """Test formatting raw blocks using format_raw_block_llm."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Formatting raw blocks with LLM")
    print(f"{'='*60}")

    formatted_blocks = []

    for raw_block in raw_blocks[:2]:  # Test first 2
        block_id = raw_block.get("block_label", "?")
        print(f"\n  Formatting block {block_id}...")

        try:
            formatted = format_raw_block_llm(
                client,
                block_id,
                raw_block.get("question_text_raw", ""),
                raw_block.get("answer_text_raw", ""),
                model_id,
                chapter_num
            )

            formatted_blocks.append(formatted)

            print(f"    Result:")
            print(f"      block_id: {formatted.get('block_id')}")

            context = formatted.get('context', {})
            context_text = context.get('text', '')[:80] if context.get('text') else "(none)"
            print(f"      context: {context_text}...")

            sub_questions = formatted.get('sub_questions', [])
            print(f"      sub_questions: {len(sub_questions)}")
            for sq in sub_questions[:3]:
                local_id = sq.get('local_id', '?')
                q_text = sq.get('question_text', '')[:50] if sq.get('question_text') else "(none)"
                choices = list(sq.get('choices', {}).keys())
                correct = sq.get('correct_answer', '')
                print(f"        - {local_id}: {q_text}...")
                print(f"          choices: {choices}, correct: {correct}")

            if formatted.get('error'):
                print(f"      ERROR: {formatted['error']}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    return formatted_blocks


def test_card_generation(client, blocks: list, chapter_num: int, model_id: str):
    """Test cloze card generation from blocks."""
    print(f"\n{'='*60}")
    print(f"STEP 4: Generating cloze cards from blocks")
    print(f"{'='*60}")

    all_cards = []

    for block in blocks[:2]:  # Test first 2 blocks
        block_id = block.get("block_id", block.get("block_label", "?"))
        print(f"\n  Generating cards for block {block_id}...")

        try:
            cards = generate_cloze_cards_from_block_llm(
                client, block, chapter_num, model_id
            )

            if cards:
                all_cards.extend(cards)
                print(f"    Generated {len(cards)} cards")
                # Show first card as sample
                if cards:
                    sample = cards[0]
                    cloze = sample.get("cloze_text", "")[:80]
                    print(f"    Sample: {cloze}...")
            else:
                print(f"    No cards generated")

        except Exception as e:
            print(f"    ERROR: {e}")

    return all_cards


def main():
    # Load environment variables from .env file
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = script_dir.parent
    load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(description="Test textbook extraction pipeline")
    parser.add_argument("--chapter", type=int, default=1, help="Chapter number to extract")
    parser.add_argument("--model", type=str, default="claude-3-5-haiku-20241022",
                        help="Model to use (default: claude-3-5-haiku-20241022)")
    parser.add_argument("--textbook", type=str, default="Core_Review_-_Neuro_2e",
                        help="Textbook name (folder in output/)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip cloze card generation step")
    parser.add_argument("--list-prompts", action="store_true",
                        help="List available prompts and exit")

    args = parser.parse_args()

    # List prompts if requested
    if args.list_prompts:
        prompts = load_prompts()
        print("Available prompts:")
        for name, data in prompts.items():
            desc = data.get("description", "No description")
            print(f"  - {name}: {desc}")
        return

    print("="*60)
    print("TEXTBOOK EXTRACTION TEST")
    print("="*60)
    print(f"Textbook: {args.textbook}")
    print(f"Chapter: {args.chapter}")
    print(f"Model: {args.model}")

    # Load existing data
    print(f"\nLoading existing data from output/{args.textbook}/...")
    try:
        data = load_existing_data(args.textbook)
        print(f"  Chapters loaded: {len(data['chapters'])}")
        print(f"  Pages loaded: {len(data['pages'])}")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        print("\nMake sure you have already run Step 1 and Step 2 in the Streamlit UI.")
        return 1

    # Get chapter text
    print(f"\nPreparing chapter {args.chapter} text...")
    try:
        chapter_text = get_chapter_text_with_lines(data, args.chapter)
        print(f"  Chapter text: {len(chapter_text)} chars, {len(chapter_text.splitlines())} lines")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Initialize client
    print("\nInitializing Anthropic client...")
    try:
        client = get_client()
        print("  Client ready")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Run extraction tests
    print("\n" + "="*60)
    print("RUNNING EXTRACTION TESTS")
    print("="*60)

    # Step 1: Identify blocks
    blocks = test_block_identification(client, args.chapter, chapter_text, args.model)

    if not blocks:
        print("\nNo blocks identified. Stopping.")
        return 1

    # Step 2: Extract raw text with line numbers
    raw_blocks = test_extract_raw_text(blocks, chapter_text, args.chapter)

    if not raw_blocks:
        print("\nNo raw text extracted. Stopping.")
        return 1

    # Step 3: Format blocks with LLM
    formatted_blocks = test_format_block(client, raw_blocks, args.chapter, args.model)

    # Step 4 (optional): Generate cloze cards
    if not args.skip_generation and formatted_blocks:
        cards = test_card_generation(client, formatted_blocks, args.chapter, args.model)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

    # Save results for inspection
    output_dir = project_root / "output" / args.textbook

    # Save identified blocks
    with open(output_dir / "test_blocks.json", "w") as f:
        json.dump(blocks, f, indent=2)
    print(f"Blocks saved to: {output_dir}/test_blocks.json")

    # Save raw blocks with line numbers
    with open(output_dir / "test_raw_blocks.json", "w") as f:
        json.dump(raw_blocks, f, indent=2)
    print(f"Raw blocks saved to: {output_dir}/test_raw_blocks.json")

    # Save formatted blocks
    with open(output_dir / "test_formatted_blocks.json", "w") as f:
        json.dump(formatted_blocks, f, indent=2)
    print(f"Formatted blocks saved to: {output_dir}/test_formatted_blocks.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
