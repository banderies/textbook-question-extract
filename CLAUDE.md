# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Textbook Question Extractor - A Python pipeline for extracting Q&A pairs from PDF textbooks and generating Anki flashcard decks. Uses Claude AI for intelligent semantic extraction and image-question matching.

## Architecture

### Six-Step Pipeline

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
STEP 3: EXTRACT QUESTIONS
├── Claude API: Extract Q&A pairs per chapter (parallel processing)
├── Image matching: Use flanking text to assign images to questions
└── Output: questions_by_chapter.json + image_assignments.json
    ↓
STEP 4: ASSOCIATE CONTEXT
├── LLM identifies context-only questions (clinical scenarios without choices)
├── Merges context text into sub-questions (e.g., Q1 context → Q1a, Q1b, Q1c)
├── Sub-questions inherit images via context_from field
└── Output: questions_merged.json + image_assignments_merged.json
    ↓
STEP 5: QC QUESTIONS
├── Streamlit GUI: Review/correct assignments
├── Approve/flag questions, reassign images
└── Output: qc_progress.json
    ↓
STEP 6: EXPORT
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
| `questions_by_chapter.json` | Extracted Q&A pairs (pre-merge) |
| `image_assignments.json` | Image filename → question ID mapping (pre-merge) |
| `questions_merged.json` | Questions with context merged into sub-questions |
| `image_assignments_merged.json` | Image assignments after context association |
| `qc_progress.json` | QC review status per question |
| `settings.json` | UI state (selected model, current step, QC position) |

## Key Design Patterns

### Editable Prompts (config/prompts.yaml)
All LLM prompts are stored in YAML for easy editing without code changes:
- `identify_chapters` - Find chapter boundaries
- `extract_qa_pairs` - Extract questions and answers
- `match_images_to_questions` - Assign images using flanking text
- `associate_context` - Link context questions to sub-questions

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

### Context Inheritance
For multi-part questions (Q1 → Q1a, Q1b, Q1c):
- Context question (Q1) has `is_context_only: true`
- Sub-questions have `context_from: "ch1_1"` and `context_merged: true`
- Images stay assigned to context question; sub-questions inherit via `context_from`

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
- Context question (5) marked as `is_context_only`
- Sub-questions (5a, 5b) have `context_from` pointing to parent
- Images inherited through `context_from` lookup

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
- `pymupdf` - PDF text and image extraction
- `anthropic` - Claude API client
- `streamlit` - Interactive GUI
- `genanki` - Anki deck generation
- `python-dotenv` - Environment variable loading
- `pyyaml` - Prompt configuration loading
