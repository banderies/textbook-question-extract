# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Textbook Question Extractor - A Python pipeline for extracting Q&A pairs from PDF textbooks and generating Anki flashcard decks. Uses Claude AI for intelligent semantic extraction and image-question matching.

## Architecture

### Multi-Step Pipeline

```
PDF Input
    ↓
STEP 1: SELECT SOURCE
├── Load PDF from source/ directory
├── PyMuPDF: Extract text pages and images with positions
├── Flanking text: Extract context before/after each image
└── Output: images/*.jpg + output/<textbook>/images.json + pages.json
    ↓
STEP 2: EXTRACT CHAPTERS
├── LLM identifies chapter boundaries from page index
└── Output: chapters.json + chapter_text.json
    ↓
STEP 3: EXTRACT QUESTIONS (Block-Based)
├── First pass: LLM identifies question BLOCKS (grouped by main question number)
├── Output: raw_blocks.json (block boundaries with line ranges)
└── Each block contains: context + sub-questions + shared discussion
    ↓
STEP 4: FORMAT QUESTIONS
├── Second pass: LLM formats each block into structured JSON
├── Block-aware image assignment: shared images → first sub-question
├── Sets context_from on subsequent sub-questions for inheritance
└── Output: questions_by_chapter.json + image_assignments.json + raw_questions.json
    ↓
STEP 5: ASSOCIATE CONTEXT
├── LLM identifies context-only questions (clinical scenarios without choices)
├── Merges context text into sub-questions (e.g., Q1 context → Q1a, Q1b, Q1c)
├── Sub-questions inherit images via context_from field
└── Output: questions_merged.json + image_assignments_merged.json
    ↓
STEP 6: QC QUESTIONS
├── Streamlit GUI: Review/correct assignments
├── Approve/flag questions, reassign images
└── Output: qc_progress.json
    ↓
STEP 7: GENERATE (optional)
├── Generate cloze deletion cards from explanations
└── Output: generated_questions.json
    ↓
STEP 8: EXPORT
└── Output: .apkg file (importable to Anki)
```

### Modular Code Structure

```
scripts/
├── review_gui.py          # Main entry point - initializes app and routes to steps
├── ui_components.py       # All Streamlit UI rendering functions
├── state_management.py    # Session state, file I/O, path management
├── llm_extraction.py      # LLM functions, prompt loading, model management
├── pdf_extraction.py      # PDF text/image extraction, chapter assignment
└── config/
    └── prompts.yaml       # Editable LLM prompts (no code changes needed)
```

| Module | Purpose |
|--------|---------|
| `review_gui.py` | Entry point, session init, step routing |
| `ui_components.py` | Render functions for each step, image callbacks, sidebar |
| `state_management.py` | `st.session_state` management, JSON save/load, paths |
| `llm_extraction.py` | Claude API calls, prompt loading from YAML, model fetching |
| `pdf_extraction.py` | PyMuPDF extraction, flanking text, chapter assignment |
| `config/prompts.yaml` | All LLM prompts - edit to customize extraction behavior |

## Commands

### Environment Setup
```bash
./setup_env.sh                    # Create conda environment
conda activate anki-extractor
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Run the GUI
```bash
streamlit run scripts/review_gui.py
```

## Data Files

All state is persisted to JSON files in `output/<textbook_name>/`:

| File | Contents |
|------|----------|
| `pages.json` | Raw extracted text per page |
| `chapters.json` | Detected chapters with page ranges |
| `chapter_text.json` | Extracted text per chapter |
| `images.json` | Image metadata with flanking text context |
| `raw_blocks.json` | Question blocks with line ranges (Step 3 output) |
| `questions_by_chapter.json` | Formatted Q&A pairs with block_id and context_from |
| `image_assignments.json` | Image filename → question ID mapping (block-aware) |
| `questions_merged.json` | Questions with context merged into sub-questions |
| `image_assignments_merged.json` | Image assignments after context association |
| `qc_progress.json` | QC review status per question |
| `settings.json` | UI state (selected model, current step, QC position) |
| `raw_questions.json` | Line ranges + extracted text from two-pass extraction |
| `generated_questions.json` | Generated cloze deletion cards |
| `extraction.log` | Debug log for LLM extraction operations |

## Key Design Patterns

### Editable Prompts (config/prompts.yaml)
All LLM prompts are stored in YAML for easy editing without code changes:
- `identify_chapters` - Find chapter boundaries
- `identify_question_blocks` - First pass: identify question blocks with line ranges
- `format_raw_block` - Second pass: format blocks into structured Q&A pairs
- `associate_context` - Link context questions to sub-questions
- `generate_cloze_cards_from_block` - Generate cloze cards from block explanations

Legacy prompts (still available):
- `extract_qa_pairs` - Single-pass extraction (deprecated)
- `extract_line_ranges` - Legacy line range extraction
- `format_qa_pair` - Legacy individual Q&A formatting
- `match_images_to_questions` - Assign images using flanking text
- `generate_cloze_cards` - Generate cloze deletion flashcards

### Dynamic Model Selection
- Models fetched from Anthropic API via `client.models.list()`
- Cached after first fetch, falls back to static list if API unavailable
- User can select model per extraction step

### Flanking Text for Image Matching
Images are matched to questions using surrounding text context:
1. Extract 500 chars before and after each image
2. Cross-page boundaries: images at top of page get context from previous page
3. LLM prompt: "Find the LAST question number in text BEFORE image"
4. Multiple images can belong to the same question

### Block-Based Context Inheritance
Questions are grouped into BLOCKS by their main question number. Within each block:

**Image Assignment** (handled by `build_block_aware_image_assignments()`):
- Shared context images → assigned to FIRST sub-question only (e.g., Q1a)
- Question-specific images → assigned to that sub-question directly
- Subsequent sub-questions (Q1b, Q1c) have `context_from: "ch1_1a"` pointing to first

**Image Retrieval** (handled by `get_images_for_question()`):
- Returns both directly assigned images AND inherited images via `context_from`
- Q1b gets its own images PLUS images from Q1a (the context source)

**Example**:
```
Block 1: context + Q1a + Q1b (shared image)
  → image assigned to ch1_1a
  → ch1_1b has context_from: "ch1_1a", inherits image

Block 2: context + Q2a + Q2b (shared image + Q2b-specific image)
  → shared image assigned to ch1_2a
  → Q2b-specific image assigned to ch1_2b
  → ch1_2b has context_from: "ch1_2a", gets BOTH images
```

**Legacy Context-Only Questions**:
For questions with `is_context_only: true` (clinical scenarios without choices):
- Sub-questions have `context_from` pointing to context question
- Images stay assigned to context question; sub-questions inherit

### Chapter-Aware Processing
Question numbering restarts each chapter, so IDs are prefixed: `2a` becomes `ch1_2a` or `ch8_2a`.

### State Persistence
All changes auto-save to JSON. On app restart:
- `init_session_state()` calls `load_saved_data()` and `load_settings()`
- User returns to same step and QC position

## Question Formats Supported

### Standard Format
```
Question text here? 2a
[IMAGE]
A. Choice A  B. Choice B  C. Choice C  D. Choice D
```

### Multi-part Format (with shared context)
```
Context for questions 5a-5c here. 5
[SHARED IMAGE]
Question 5a text? 5a
A. ...
Question 5b text? 5b
A. ...
```

**Block-based processing** (current architecture):
- All questions grouped in one BLOCK with `block_id: "ch1_5"`
- Shared context image assigned to FIRST sub-question (5a)
- Sub-question 5b has `context_from: "ch1_5a"` to inherit image
- Both 5a and 5b include the context text in their `text` field

**Legacy context-only processing**:
- Context question (5) marked as `is_context_only: true`
- Sub-questions (5a, 5b) have `context_from: "ch1_5"`
- Images assigned to context question; sub-questions inherit

## API Usage

```python
# Models are fetched dynamically
response = client.models.list(limit=100)

# Q&A extraction (prompts loaded from YAML)
prompt = get_prompt("extract_qa_pairs", chapter_num=1, chapter_text=text)
response = client.messages.create(
    model=get_model_id(selected_model),
    max_tokens=16000,
    messages=[{"role": "user", "content": prompt}]
)
```

## Dependencies

- Python 3.12+
- `pymupdf` - PDF text and image extraction (import as `fitz`)
- `docling` - PDF to Markdown conversion
- `anthropic` - Claude API client
- `streamlit` - Interactive GUI
- `genanki` - Anki deck generation
- `python-dotenv` - Environment variable loading
- `pyyaml` - Prompt configuration loading (import as `yaml`)
- `pillow` - Image processing for Streamlit

## Debugging

### Extraction Logs
LLM extraction operations are logged to `output/<textbook>/extraction.log`:
- Console shows warnings/errors only
- Log file contains full DEBUG output with timestamps
- Use `get_extraction_logger(output_dir)` to initialize logging for a textbook

### Common Issues
- **Images not matching**: Check flanking text in `images.json` - the text before each image should end with a question number
- **Missing questions**: Review `raw_blocks.json` to verify block boundaries are correct
- **Images not inherited**: Verify `context_from` is set on sub-questions (check `questions_by_chapter.json`)
- **Wrong image assignment**: Check `block_id` field - questions in same block share context images
- **Context not merging**: Questions need matching `block_id` fields; first sub-question gets shared images
