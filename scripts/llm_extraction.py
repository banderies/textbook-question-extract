"""
LLM Extraction Module

Contains all LLM-based extraction functions and prompt management.
Prompts are loaded from config/prompts.yaml for easy editing.
"""

import os
import re
import json
from pathlib import Path
from typing import Optional

import yaml

# =============================================================================
# Prompt Loading
# =============================================================================

_cached_prompts = None

def get_prompts_path() -> Path:
    """Get path to prompts.yaml file."""
    return Path(__file__).parent / "config" / "prompts.yaml"


def load_prompts() -> dict:
    """Load prompts from YAML file. Caches after first load."""
    global _cached_prompts

    if _cached_prompts is not None:
        return _cached_prompts

    prompts_path = get_prompts_path()
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with open(prompts_path) as f:
        _cached_prompts = yaml.safe_load(f)

    return _cached_prompts


def get_prompt(name: str, **kwargs) -> str:
    """
    Get a formatted prompt by name.

    Args:
        name: Prompt name (e.g., 'identify_chapters', 'extract_qa_pairs')
        **kwargs: Variables to substitute into the prompt template

    Returns:
        Formatted prompt string
    """
    prompts = load_prompts()

    if name not in prompts:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(prompts.keys())}")

    prompt_template = prompts[name]["prompt"]

    # Use format_map to handle missing keys gracefully
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing variable for prompt '{name}': {e}")


def reload_prompts():
    """Force reload prompts from disk (useful after editing prompts.yaml)."""
    global _cached_prompts
    _cached_prompts = None
    return load_prompts()


# =============================================================================
# Anthropic Client & Model Management
# =============================================================================

# Fallback Claude models (used if API fetch fails)
FALLBACK_MODELS = {
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-3-5-sonnet-20241022": "Claude Sonnet 3.5",
    "claude-3-5-haiku-20241022": "Claude Haiku 3.5",
}
DEFAULT_MODEL_ID = "claude-sonnet-4-20250514"
DEFAULT_MODEL_NAME = "Claude Sonnet 4"

_cached_models = None


def get_anthropic_client():
    """Get Anthropic client, loading API key from .env if needed."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic
    return anthropic.Anthropic()


def fetch_available_models() -> dict:
    """
    Fetch available models from the Anthropic API.
    Returns dict mapping display_name to model_id.
    Falls back to static list if API call fails.
    """
    global _cached_models

    if _cached_models is not None:
        return _cached_models

    try:
        client = get_anthropic_client()
        if not client:
            _cached_models = {v: k for k, v in FALLBACK_MODELS.items()}
            return _cached_models

        response = client.models.list(limit=100)

        models = {}
        for model in response.data:
            display_name = getattr(model, 'display_name', None) or model.id
            models[display_name] = model.id

        if models:
            _cached_models = models
            return _cached_models

    except Exception as e:
        print(f"Failed to fetch models from API: {e}")

    _cached_models = {v: k for k, v in FALLBACK_MODELS.items()}
    return _cached_models


def get_model_options() -> list[str]:
    """Get list of model display names for dropdown."""
    models = fetch_available_models()
    return list(models.keys())


def get_model_id(display_name: str) -> str:
    """Get model ID from display name."""
    models = fetch_available_models()
    return models.get(display_name, DEFAULT_MODEL_ID)


# =============================================================================
# LLM Extraction Functions
# =============================================================================

def identify_chapters_llm(client, pages: list[dict], model_id: str) -> list[dict]:
    """Use Claude to identify chapter boundaries."""
    from pdf_extraction import create_page_index

    page_index = create_page_index(pages)
    prompt = get_prompt("identify_chapters", page_index=page_index)

    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text
    if "```" in response_text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    return json.loads(response_text)


def extract_qa_pairs_llm(client, chapter_num: int, chapter_text: str, model_id: str) -> dict:
    """Use Claude to extract Q&A pairs from a single chapter."""
    prompt = get_prompt("extract_qa_pairs",
                       chapter_num=chapter_num,
                       chapter_text=chapter_text)

    response = client.messages.create(
        model=model_id,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text
    if "```" in response_text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        return {"chapter": chapter_num, "questions": [], "error": str(e)}


def process_chapter_extraction(client, ch_num: int, ch_key: str, ch_text: str, model_id: str) -> tuple[str, list[dict]]:
    """
    Process a single chapter extraction (for parallel execution).

    Returns:
        Tuple of (ch_key, list of formatted questions)
    """
    result = extract_qa_pairs_llm(client, ch_num, ch_text, model_id)

    questions = []
    for q in result.get("questions", []):
        questions.append({
            "full_id": f"ch{ch_num}_{q['id']}",
            "local_id": q["id"],
            "text": q.get("text", ""),
            "choices": q.get("choices", {}),
            "has_image": q.get("has_image", False),
            "image_group": q.get("image_group"),
            "correct_answer": q.get("correct_answer", ""),
            "explanation": q.get("explanation", "")
        })

    return (ch_key, questions)


def match_images_to_questions_llm(client, images: list[dict], chapters: list[dict],
                                   questions: dict, model_id: str) -> dict:
    """
    Use Claude to intelligently match images to questions based on flanking text context.
    Returns dict mapping image filename to question full_id.
    """
    assignments = {}

    for i, ch in enumerate(chapters):
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"
        start_page = ch["start_page"]
        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else 9999

        ch_questions = questions.get(ch_key, [])
        if not ch_questions:
            continue

        ch_images = [img for img in images if start_page <= img["page"] < end_page]
        if not ch_images:
            continue

        # Build question summary for the prompt
        questions_text = []
        for q in ch_questions:
            q_summary = f"- {q['full_id']}: {q['text'][:150]}..."
            if q.get("has_image"):
                q_summary += " [NEEDS IMAGE]"
            questions_text.append(q_summary)

        # Build image context for the prompt
        images_text = []
        for img in ch_images:
            ctx_before = img.get('context_before', '')
            ctx_after = img.get('context_after', '')
            img_info = f"- {img['filename']} (page {img['page']})\n"
            img_info += f"  Text BEFORE image: \"...{ctx_before[-300:]}\"\n"
            img_info += f"  Text AFTER image: \"{ctx_after[:300]}...\""
            images_text.append(img_info)

        prompt = get_prompt("match_images_to_questions",
                           chapter_num=ch_num,
                           questions_text="\n".join(questions_text),
                           images_text="\n".join(images_text))

        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            ch_assignments = json.loads(response_text)

            for img_file, q_id in ch_assignments.items():
                if q_id and q_id != "(none)":
                    assignments[img_file] = q_id

        except Exception as e:
            print(f"LLM matching failed for chapter {ch_num}: {e}")
            continue

    return assignments


def match_images_to_questions_simple(images: list[dict], chapters: list[dict],
                                      questions: dict) -> dict:
    """
    Simple fallback: match images to questions based on page proximity.
    Used when LLM matching is not available.
    """
    assignments = {}

    for i, ch in enumerate(chapters):
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"
        start_page = ch["start_page"]
        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else 9999

        ch_questions = [q for q in questions.get(ch_key, []) if q.get("has_image")]
        ch_images = [img for img in images if start_page <= img["page"] < end_page]

        for img, q in zip(ch_images, ch_questions):
            if q["full_id"] not in assignments.values():
                assignments[img["filename"]] = q["full_id"]

    return assignments


def postprocess_questions_llm(client, questions: dict, model_id: str) -> dict:
    """
    Post-process extracted questions to link context to sub-questions.
    """
    for ch_key, ch_questions in questions.items():
        if not ch_questions:
            continue

        questions_json = json.dumps(ch_questions, indent=2)
        prompt = get_prompt("postprocess_questions", questions_json=questions_json)

        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            updated_questions = json.loads(response_text)
            questions[ch_key] = updated_questions

        except Exception as e:
            print(f"Post-processing failed for {ch_key}: {e}")
            for q in ch_questions:
                if "is_context_only" not in q:
                    has_choices = bool(q.get("choices"))
                    has_letter = any(c.isalpha() for c in q.get("local_id", ""))
                    q["is_context_only"] = not has_choices and not has_letter
                if "context" not in q:
                    q["context"] = ""
                if "context_question_id" not in q:
                    q["context_question_id"] = ""

    return questions


def associate_context_llm(client, questions: dict, image_assignments: dict,
                          model_id: str) -> tuple[dict, dict, dict]:
    """
    Use LLM to identify context relationships and merge context into sub-questions.

    Returns:
        Tuple of (updated_questions, updated_image_assignments, stats)
    """
    if image_assignments is None:
        image_assignments = {}

    updated_assignments = dict(image_assignments)

    stats = {
        "context_questions_found": 0,
        "sub_questions_updated": 0,
        "images_copied": 0
    }

    for ch_key, ch_questions in questions.items():
        if not ch_questions:
            continue

        # Build a summary of questions for the LLM
        questions_summary = []
        for q in ch_questions:
            has_choices = bool(q.get("choices"))
            summary = {
                "full_id": q["full_id"],
                "local_id": q["local_id"],
                "text_preview": q["text"][:200] + "..." if len(q["text"]) > 200 else q["text"],
                "has_choices": has_choices
            }
            questions_summary.append(summary)

        prompt = get_prompt("associate_context",
                           questions_summary=json.dumps(questions_summary, indent=2))

        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            if "```" in response_text:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    response_text = match.group(1)

            result = json.loads(response_text)
            mappings = result.get("context_mappings", [])

            question_by_id = {q["full_id"]: q for q in ch_questions}

            for mapping in mappings:
                context_id = mapping.get("context_id")
                sub_ids = mapping.get("sub_question_ids", [])

                if not context_id or context_id not in question_by_id:
                    continue

                context_q = question_by_id[context_id]
                context_text = context_q.get("text", "").strip()

                context_q["is_context_only"] = True
                stats["context_questions_found"] += 1

                context_images = [
                    img_file for img_file, assigned_to in image_assignments.items()
                    if assigned_to == context_id
                ]

                for sub_id in sub_ids:
                    if sub_id not in question_by_id:
                        continue

                    sub_q = question_by_id[sub_id]

                    if sub_q.get("context_merged"):
                        continue

                    original_text = sub_q.get("text", "").strip()
                    sub_q["text"] = f"{context_text} {original_text}"
                    sub_q["context_merged"] = True
                    sub_q["context_from"] = context_id
                    sub_q["is_context_only"] = False
                    stats["sub_questions_updated"] += 1

                    for img_file in context_images:
                        updated_assignments[img_file] = sub_id
                        stats["images_copied"] += 1

        except Exception as e:
            print(f"LLM context association failed for {ch_key}: {e}")
            continue

    # Ensure all questions have is_context_only set
    for ch_key, ch_questions in questions.items():
        for q in ch_questions:
            if "is_context_only" not in q:
                q["is_context_only"] = False

    return questions, updated_assignments, stats
