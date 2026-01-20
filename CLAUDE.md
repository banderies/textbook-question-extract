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
├── Output: raw_blocks.json (block boundaries with line ranges + raw text)
└── Each block contains: context + sub-questions + shared discussion
    ↓
STEP 4: FORMAT QUESTIONS
├── Second pass: LLM formats each block into structured JSON
├── Extracts page numbers from [PAGE N] markers in raw text
├── Distributes images: context images to all sub-questions, specific images to their question
├── Separates question images (image_files) from answer images (answer_image_files)
├── Sets context_from on subsequent sub-questions for inheritance
└── Output: questions_by_chapter.json + image_assignments.json
    ↓
STEP 5: QC QUESTIONS
├── Streamlit GUI: Review/correct image assignments
├── Approve/flag questions, reassign images between questions
└── Output: qc_progress.json
    ↓
STEP 6: GENERATE (optional)
├── Generate cloze deletion cards from raw block content
├── Uses full context (question + answer + discussion) for accurate card generation
└── Output: generated_questions.json
    ↓
STEP 7: EXPORT
└── Output: .apkg file (importable to Anki)

UTILITY: EDIT PROMPTS (Step 8)
└── Edit LLM prompts via GUI without code changes
```

### Modular Code Structure

```
src/
├── review_gui.py          # Main entry point - initializes app and routes to steps
├── ui_components.py       # Step rendering functions (Steps 1-7)
├── ui/                    # Modular UI package
│   ├── __init__.py       # Package exports
│   ├── helpers.py        # Utility functions (image handling, sorting, etc.)
│   └── sidebar.py        # Sidebar navigation and status display
├── state_management.py    # Session state, file I/O, path management
├── llm_extraction.py      # LLM functions, prompt loading, model management
├── pdf_extraction.py      # PDF text/image extraction, chapter assignment
└── config/
    └── prompts.yaml       # Editable LLM prompts (no code changes needed)
```

| Module | Purpose |
|--------|---------|
| `review_gui.py` | Entry point, session init, step routing |
| `ui_components.py` | Step rendering functions for each pipeline step |
| `ui/helpers.py` | Utility functions: image handling, sorting, step data management |
| `ui/sidebar.py` | Sidebar navigation, step completion status, progress display |
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
streamlit run src/review_gui.py
```

## Data Files

All state is persisted to JSON files in `output/<textbook_name>/`:

| File | Contents |
|------|----------|
| `pages.json` | Raw extracted text per page |
| `chapters.json` | Detected chapters with page ranges |
| `chapter_text.json` | Extracted text per chapter |
| `images.json` | Image metadata with flanking text context |
| `raw_blocks.json` | Question blocks with line ranges + raw text (Step 3 output) |
| `questions_by_chapter.json` | Formatted Q&A pairs with block_id, context_from, image_files |
| `image_assignments.json` | Image filename → question ID mapping (block-aware) |
| `qc_progress.json` | QC review status per question |
| `settings.json` | UI state (selected model, current step, QC position) |
| `generated_questions.json` | Generated cloze deletion cards |
| `extraction.log` | Debug log for LLM extraction operations |

**Legacy files** (supported for backwards compatibility):
| File | Contents |
|------|----------|
| `raw_questions.json` | Legacy two-pass extraction format |
| `questions_merged.json` | Legacy context association output |
| `image_assignments_merged.json` | Legacy image assignments after context association |

## Design Philosophy

### LLM-First Content Processing
Content extraction, organization, and formatting decisions should be handled by **LLM prompts** (in `config/prompts.yaml`), NOT by hardcoded Python logic. This principle exists because:

1. **Flexibility**: LLMs can adapt to variations in input material that rigid code cannot handle
2. **Maintainability**: Prompt changes don't require code deployment or testing
3. **Intelligence**: LLMs can use context and judgment for edge cases

**Rule**: When adding new content processing logic, ask: "Can the LLM handle this in the prompt?" If yes, modify the prompt. Only use Python code for:
- Infrastructure (file I/O, API calls, UI rendering)
- Data structure operations (building dicts, lists)
- Validation that doesn't depend on content semantics

### Known Hardcoded Logic (Future Refactoring Candidates)
These areas currently use hardcoded logic that could be moved to prompts for better flexibility:

| Area | Location | Current Behavior | Better Approach |
|------|----------|------------------|-----------------|
| Chapter ID format | ui_components.py:930 | Always `ch{N}` | LLM determines ID scheme |
| Question ID format | ui_components.py:948 | Always `ch{N}_{id}` | LLM determines in Step 4 |
| Shared discussion fields | ui_components.py:914-922 | Hardcoded "Imaging Findings", "Discussion" | Prompt specifies expected fields |
| Context inheritance | ui/helpers.py:104-110 | First sub-question owns context | LLM determines in Step 4 |
| Flanking text window | pdf_extraction.py:618 | Fixed 500 chars | Configurable per textbook |
| Cloze categories | prompts.yaml:257 | Medical-specific categories | Per-textbook config |

## Key Design Patterns

### Editable Prompts (config/prompts.yaml)
All LLM prompts are stored in YAML for easy editing without code changes:

- `identify_chapters` - Find chapter boundaries from page index (Step 2)
- `identify_question_blocks` - Identify question blocks with line ranges (Step 3)
- `format_raw_block` - Format blocks into structured Q&A pairs with image distribution (Step 4)
- `generate_cloze_cards_from_block` - Generate cloze cards from raw block content (Step 6)
- `generate_cloze_cards` - Fallback: generate cloze cards from individual question explanations

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

### LLM-Based Image Distribution
The LLM handles image assignment during the `format_raw_block` step (Step 4), following these rules:

**Question vs Answer Images**:
Images are tracked separately based on which section they appear in:
- `image_files` → Images from QUESTION section (shown with the question in Anki)
- `answer_image_files` → Images from ANSWER section (shown only in explanation in Anki)

**Question Image Distribution**:
- **Context images** (in scenario before sub-questions) → included in ALL sub-questions' `image_files`
- **Sub-question-specific images** → only in that sub-question's `image_files`

**Answer Image Distribution**:
- **Sub-question answer images** → in that sub-question's `answer_image_files`
- **Shared discussion images** → in all sub-questions' `answer_image_files`

**Example**:
```
Block 4:
  Question section: context (img1) + Q4a (no image) + Q4b (img2)
  Answer section: shared discussion (img3)

  → Q4a.image_files = ["img1"]
  → Q4a.answer_image_files = ["img3"]
  → Q4b.image_files = ["img1", "img2"]
  → Q4b.answer_image_files = ["img3"]
```

**In Anki Cards**:
- `Image` field: question images (shown on front and back)
- `AnswerImage` field: answer images (shown only on back, after explanation)

**Image Assignment Building** (in `ui/helpers.py`):
- `build_block_aware_image_assignments()` creates the `image_assignments` dict from LLM output (first question with each image wins)
- `get_images_for_question()` → returns question images from `image_files`
- `get_answer_images_for_question()` → returns answer images from `answer_image_files`

**Important**: The LLM determines all image distribution during Step 4. The Python code only reads what the LLM returns - it does not manipulate or redistribute images.

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
[IMAGE 1 - context image]
Question 5a text? 5a
A. ...
[IMAGE 2 - specific to 5b]
Question 5b text? 5b
A. ...
```

**LLM-based processing** (current architecture):
- All questions grouped in one BLOCK with `block_id: "ch1_5"`
- LLM assigns images based on position:
  - IMAGE 1 (context) → included in 5a and 5b's `image_files`
  - IMAGE 2 (after 5b) → only in 5b's `image_files`
- Sub-question 5b has `context_from: "ch1_5a"` for inheritance
- Both 5a and 5b include the context text in their `text` field

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
- **Wrong image assignment**: Check `block_id` field - questions in same block share context images; use QC step to reassign
- **Context not appearing**: Ensure the block contains context text; check `raw_blocks.json` for the raw content
