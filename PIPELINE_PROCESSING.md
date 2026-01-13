# Pipeline Processing Breakdown: Traditional vs LLM

This document details which parts of each extraction step use traditional (non-LLM) processing versus LLM-based processing. This is important for understanding what may need to change when input data formats change.

## Overview

The pipeline has 7 main steps plus a utility step for editing prompts. Traditional processing is used for PDF parsing, data transformation, and structural operations. LLM processing handles semantic understanding, extraction, and classification tasks.

---

## STEP 1: SELECT SOURCE

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| `extract_text_from_pdf()` | Traditional | `pdf_extraction.py:17` | PyMuPDF extracts raw text per page |
| `extract_images_from_pdf()` | Traditional | `pdf_extraction.py:522` | Extracts images with position coordinates |
| Flanking text extraction | Traditional | `pdf_extraction.py:602-603` | **Hardcoded 500 chars** before/after each image |
| `extract_text_with_lines()` | Traditional | `pdf_extraction.py:82` | Adds `[LINE:NNNN]` markers to text |
| `insert_image_markers()` | Traditional | `pdf_extraction.py:157` | Y-position sorting to interleave `[IMAGE:]` markers |
| Position cache management | Traditional | `pdf_extraction.py:14` | Module-level cache for line positions |

**LLM: None**

---

## STEP 2: EXTRACT CHAPTERS

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| `create_page_index()` | Traditional | `pdf_extraction.py:672` | **Hardcoded 300 chars** preview per page |
| `identify_chapters_llm()` | **LLM** | `llm_extraction.py:691` | Identifies chapter boundaries from page index |
| Chapter sorting | Traditional | `state_management.py:220` | Sort by `chapter_number` |
| `assign_chapters_to_images()` | Traditional | `pdf_extraction.py:637` | Page-range based chapter assignment |

---

## STEP 3: EXTRACT QUESTIONS (Block-Based)

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| `build_chapter_text_with_lines()` | Traditional | `pdf_extraction.py:327` | Builds chapter text preserving global line numbers |
| `identify_question_blocks_llm()` | **LLM** | `llm_extraction.py` | First pass - identifies question BLOCKS with line ranges |
| `extract_lines_by_range()` | Traditional | `pdf_extraction.py:406` | Extracts text between line numbers |
| Leading zero fix | Traditional | `llm_extraction.py:271` | Regex to fix `0941` → `941` in JSON |
| Block structure creation | Traditional | `ui_components.py` | Creates block structure with `block_id`, images, sub-questions |

---

## STEP 4: FORMAT QUESTIONS

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| `format_raw_block_llm()` | **LLM** | `llm_extraction.py` | Formats entire block into structured sub-questions with image distribution |
| `repair_json()` | Traditional | `llm_extraction.py` | Fixes unescaped quotes, control chars, trailing commas |
| Rate limit handling | Traditional | `llm_extraction.py` | Exponential backoff retry logic |
| `full_id` generation | Traditional | `llm_extraction.py` | Concatenates `ch{num}_{local_id}` |
| `build_block_aware_image_assignments()` | Traditional | `ui/helpers.py` | Builds image_assignments dict, sets `context_from` on non-first sub-questions |
| Page number extraction | Traditional | `llm_extraction.py` | Extracts page numbers from [PAGE N] markers |

**Note**: Image distribution is handled by the LLM during `format_raw_block_llm()`:
- Context images → all sub-questions' `image_files`
- Sub-question images → that sub-question's `image_files`
- Answer images → `answer_image_files` arrays

---

## STEP 5: QC QUESTIONS

**All Traditional** - Pure UI rendering and state management, no data processing logic.

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| Question filtering | Traditional | `ui_components.py` | Filter by chapter, review status, context-only |
| Image assignment UI | Traditional | `ui_components.py` | Reassign images between questions |
| QC progress tracking | Traditional | `state_management.py` | Save reviewed/approved status |

---

## STEP 6: GENERATE

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| Block content assembly | Traditional | `ui_components.py` | Combine raw question + answer text for LLM |
| `generate_cloze_cards_from_block_llm()` | **LLM** | `llm_extraction.py` | Generates cloze cards from full block content |
| `generate_cloze_cards_llm()` | **LLM** | `llm_extraction.py` | Legacy: generates from individual explanations |
| Explanation length check | Traditional | `llm_extraction.py` | Skip if < 50 chars (legacy mode) |

---

## STEP 7: EXPORT

**All Traditional** - genanki library builds .apkg file from structured data.

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| Deck structure building | Traditional | `ui_components.py` | Organize by chapter |
| Card HTML generation | Traditional | `ui_components.py` | Build card front/back with images |
| `.apkg` file creation | Traditional | `genanki` library | Package cards and media |

---

## Hardcoded Values at Risk

These hardcoded heuristics may need adjustment when input formats change:

### Character Limits

| Value | Location | Usage |
|-------|----------|-------|
| 500 chars | `pdf_extraction.py:602-603` | Flanking text before/after images |
| 300 chars | `pdf_extraction.py:676` | Page index preview |
| 300 chars | `llm_extraction.py:1073-1074` | Image context for matching |
| 300 chars | `llm_extraction.py:1247` | Question text preview for context association |
| 150 chars | `llm_extraction.py:1062` | Question summary for image matching |
| 25,000 chars | `llm_extraction.py:184` | Max chunk size before splitting |
| 50 chars | `llm_extraction.py:1383` | Minimum explanation length for cloze generation |

### Format Assumptions

| Format | Location | Description |
|--------|----------|-------------|
| `[LINE:NNNN]` | `pdf_extraction.py:399` | Line number marker format |
| `[IMAGE: filename]` | `pdf_extraction.py:210` | Image marker format |
| `ch{num}_{local_id}` | `llm_extraction.py:1015` | Question ID format |
| `[PAGE N]` | `pdf_extraction.py:677` | Page marker in index |

---

## Summary by Step

| Step | Traditional | LLM | Notes |
|------|-------------|-----|-------|
| 1. Source | 100% | 0% | PDF parsing only |
| 2. Chapters | 60% | 40% | LLM identifies boundaries |
| 3. Extract | 40% | 60% | LLM identifies blocks, traditional extracts text |
| 4. Format | 30% | 70% | LLM formats blocks + distributes images, traditional builds assignments |
| 5. QC | 100% | 0% | UI only |
| 6. Generate | 10% | 90% | LLM generates cloze cards from full block content |
| 7. Export | 100% | 0% | File generation only |

---

## Recommendations for Format Changes

To minimize breakage when input formats change:

1. **Move heuristics to prompts** - Instead of hardcoding char limits, let LLM decide relevant context
2. **Have LLM output text directly** - Rather than line ranges that require traditional extraction
3. **Send full context to LLM** - Remove preprocessing that truncates/summarizes
4. **Consolidate post-LLM logic** - Have LLM output final merged structure directly
