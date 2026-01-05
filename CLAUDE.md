# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Textbook Question Extractor - A Python pipeline for extracting Q&A pairs from PDF textbooks and generating Anki flashcard decks. Uses Claude AI for intelligent semantic extraction and image-question matching.

## Architecture

### Three-Phase Pipeline

```
PDF Input
    ↓
PHASE 1: PREPROCESSING
├── PyMuPDF: Extract text pages and images with positions
├── Chapter detection: LLM identifies chapter boundaries
├── Flanking text: Extract context before/after each image
└── Output: images/*.jpg + output/images.json
    ↓
PHASE 2: EXTRACTION & MATCHING
├── Claude API: Extract Q&A pairs per chapter
├── Image matching: Use flanking text to assign images to questions
├── Cross-page context: Images at page boundaries get context from adjacent pages
└── Output: output/questions_by_chapter.json + output/image_assignments.json
    ↓
PHASE 3: QC & EXPORT
├── Streamlit GUI: Review/correct assignments
├── QC progress tracking: Approve/flag questions
└── Output: .apkg file (importable to Anki)
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `review_gui.py` | Main Streamlit GUI - handles all extraction steps |
| `image_pipeline.py` | Extract images from PDF with positional metadata |
| `agentic_qa_extractor.py` | Standalone Q&A extraction using Claude |

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

All state is persisted to JSON files in `output/`:

| File | Contents |
|------|----------|
| `chapters.json` | Detected chapters with page ranges |
| `chapter_text.json` | Extracted text per chapter |
| `questions_by_chapter.json` | Extracted Q&A pairs |
| `images.json` | Image metadata with flanking text context |
| `image_assignments.json` | Image filename → question ID mapping |
| `qc_progress.json` | QC review status per question |
| `settings.json` | UI state (model, step, QC position) |

## Key Design Patterns

### Dynamic Model Selection
- Models fetched from Anthropic API via `GET /v1/models`
- Cached after first fetch, falls back to static list if API unavailable
- User can select model per extraction step

### Flanking Text for Image Matching
Images are matched to questions using surrounding text context:
1. Extract 500 chars before and after each image
2. Cross-page boundaries: images at top of page get context from previous page
3. LLM prompt: "Find the LAST question number in text BEFORE image"

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
- Sub-questions share parent context and image
- Stored with `shared_context` and `image_group` fields

## API Usage

```python
# Models are fetched dynamically
response = client.models.list(limit=100)

# Q&A extraction
response = client.messages.create(
    model=get_selected_model_id(),  # User-selected model
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
