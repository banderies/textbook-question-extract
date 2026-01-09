# Pipeline Processing Breakdown: Traditional vs LLM

This document details which parts of each extraction step use traditional (non-LLM) processing versus LLM-based processing. This is important for understanding what may need to change when input data formats change.

## Overview

The pipeline has 8 steps. Traditional processing is used for PDF parsing, data transformation, and structural operations. LLM processing handles semantic understanding, extraction, and classification tasks.

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

## STEP 3: EXTRACT QUESTIONS (Raw)

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| `build_chapter_text_with_lines()` | Traditional | `pdf_extraction.py:327` | Builds chapter text preserving global line numbers |
| `extract_line_ranges_llm()` | **LLM** | `llm_extraction.py:207` | First pass - identifies line ranges for each Q&A |
| `extract_lines_by_range()` | Traditional | `pdf_extraction.py:406` | Extracts text between line numbers |
| Leading zero fix | Traditional | `llm_extraction.py:271` | Regex to fix `0941` → `941` in JSON |
| Image file assignment | Traditional | `llm_extraction.py:494` | Maps images from line range data |

---

## STEP 4: FORMAT QUESTIONS

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| `format_qa_pair_llm()` | **LLM** | `llm_extraction.py:321` | Formats individual Q&A into structured JSON |
| `repair_json()` | Traditional | `llm_extraction.py:287` | Fixes unescaped quotes, control chars, trailing commas |
| Rate limit handling | Traditional | `llm_extraction.py:394` | Exponential backoff retry logic |
| `full_id` generation | Traditional | `llm_extraction.py:1015` | Concatenates `ch{num}_{local_id}` |

---

## STEP 5: ASSOCIATE CONTEXT

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| Build `questions_summary` | Traditional | `llm_extraction.py:1240-1253` | Extracts metadata (has_choices, num_choices, etc.) |
| `associate_context_llm()` | **LLM** | `llm_extraction.py:1212` | Returns context_mappings JSON |
| Self-reference filtering | Traditional | `llm_extraction.py:1291` | Removes mappings where context_id == sub_id |
| Context text merging | Traditional | `llm_extraction.py:1318` | Prepends context text to sub-question text |
| `context_from` assignment | Traditional | `llm_extraction.py:1320` | Links sub-questions to context question |
| `is_context_only` default | Traditional | `llm_extraction.py:1341-1342` | Sets False if not already set |

### Image Matching (also in Step 5)

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| Build `questions_text` | Traditional | `llm_extraction.py:1060-1065` | **Hardcoded 150 chars** + `[NEEDS IMAGE]` flag |
| Build `images_text` | Traditional | `llm_extraction.py:1069-1075` | **Hardcoded 300 chars** context |
| `match_images_to_questions_llm()` | **LLM** | `llm_extraction.py:1032` | Matches images to question IDs |
| `match_images_to_questions_simple()` | Traditional | `llm_extraction.py:1118` | Fallback: page proximity matching |

---

## STEP 6: QC QUESTIONS

**All Traditional** - Pure UI rendering and state management, no data processing logic.

---

## STEP 7: GENERATE

| Processing | Type | Location | Description |
|------------|------|----------|-------------|
| Explanation length check | Traditional | `llm_extraction.py:1383-1385` | Skip if < 50 chars |
| `generate_cloze_cards_llm()` | **LLM** | `llm_extraction.py:1347` | Generates cloze cards from explanation text |

---

## STEP 8: EXPORT

**All Traditional** - genanki library builds .apkg file from structured data.

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
| 3. Extract | 50% | 50% | LLM finds ranges, traditional extracts text |
| 4. Format | 30% | 70% | LLM does main work |
| 5. Context | 40% | 60% | LLM identifies relationships, traditional merges |
| 6. QC | 100% | 0% | UI only |
| 7. Generate | 10% | 90% | LLM generates cards |
| 8. Export | 100% | 0% | File generation only |

---

## Recommendations for Format Changes

To minimize breakage when input formats change:

1. **Move heuristics to prompts** - Instead of hardcoding char limits, let LLM decide relevant context
2. **Have LLM output text directly** - Rather than line ranges that require traditional extraction
3. **Send full context to LLM** - Remove preprocessing that truncates/summarizes
4. **Consolidate post-LLM logic** - Have LLM output final merged structure directly
