# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Textbook Question Extractor - A Python pipeline for extracting Q&A pairs from PDF textbooks and generating Anki flashcard decks. Uses a hybrid approach combining traditional PDF processing with Claude AI for intelligent semantic matching.

## Architecture

### Three-Phase Pipeline

```
PDF Input
    ↓
PHASE 1: PREPROCESSING
├── docling: PDF → Markdown (structure preservation)
├── PyMuPDF: Extract images with page/Y coordinates
├── Chapter detection: Identify chapters and parse questions
└── Output: docling/*.md + images/ + output/*.json
    ↓
PHASE 2: SEMANTIC MATCHING (AI-Powered)
├── Claude API: Match questions to answers semantically
├── Figure number matching: Use figure refs + content
└── Output: Structured JSON with matched pairs
    ↓
PHASE 3: DECK GENERATION
├── Combine Q&A pairs with images
├── Format for Anki (genanki library)
└── Output: .apkg file (importable to Anki)
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `agentic_qa_extractor.py` | Main Q&A extraction using Claude Opus 4.5 |
| `chapter_aware_parser.py` | Parse chapters with chapter-prefixed IDs (e.g., `ch1_2a`) |
| `image_pipeline.py` | Extract images from PDF with positional metadata |
| `link_images_v2.py` | Block-based image-to-question linking |
| `review_gui_v2.py` | Streamlit GUI for reviewing/correcting assignments |
| `test_first_5.py` | Quick validation on first 5 questions |

## Commands

### Environment Setup
```bash
./setup_env.sh                    # Create conda environment
conda activate anki-extractor
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Full Pipeline
```bash
./run_extraction.sh "path/to/textbook.pdf"
```

### Partial Operations
```bash
./run_extraction.sh --extract-images "pdf_file.pdf"
./run_extraction.sh --extract-qa docling/markdown_file.md
./run_extraction.sh --link-images docling/markdown_file.md
./run_extraction.sh --build-deck
./run_extraction.sh --test
```

### Direct Script Execution
```bash
python scripts/image_pipeline.py extract input.pdf --output images/
python scripts/chapter_aware_parser.py docling/file.md images/manifest.json
python scripts/link_images_v2.py docling/file.md images/manifest.json
streamlit run scripts/review_gui_v2.py
python scripts/test_first_5.py
```

## Key Design Patterns

### Agentic LLM Pattern
Uses Claude for semantic understanding rather than regex. Questions and answers are sent to Claude, which returns structured JSON with matched pairs and confidence levels.

### Chapter-Aware Processing
Question numbering restarts each chapter, so IDs are prefixed: `2a` becomes `ch1_2a` or `ch8_2a`.

### Block-Based Image Linking
Instead of position matching, link_images_v2 identifies question blocks and assigns all images within a block to that question.

## Data Flow

```
Input PDF
    ↓
docling/               ← Markdown output (git-ignored)
images/manifest.json   ← Image metadata (tracked)
images/*.jpg           ← Raw images (git-ignored)
    ↓
output/*.json          ← Parsed chapters, questions, mappings
    ↓
final/*.apkg           ← Anki decks (git-ignored)
```

## Dependencies

- Python 3.12+
- Conda for environment management
- `ANTHROPIC_API_KEY` environment variable required
- Key packages: docling, pymupdf, anthropic, genanki, streamlit
