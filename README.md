# Textbook Question Extractor

A Python pipeline for extracting Q&A pairs from PDF textbooks and generating Anki flashcard decks. Uses a hybrid approach combining traditional PDF processing with Claude AI for intelligent semantic matching.

## Features

- **PDF to Markdown**: Convert textbooks using Docling
- **Image Extraction**: Extract images with positional metadata using PyMuPDF
- **Chapter-Aware Parsing**: Automatically detect chapters and prefix question IDs (e.g., `ch1_2a`, `ch8_2a`)
- **AI-Powered Matching**: Use Claude API for semantic question-answer matching
- **Vision Verification**: Optionally verify image-question assignments using Claude Vision
- **Interactive Review GUI**: Streamlit-based GUI for reviewing and correcting assignments
- **Anki Deck Generation**: Generate `.apkg` files with genanki

## Architecture

```
PDF Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 1: PREPROCESSING                                  │
│  ├── docling: PDF → Markdown                            │
│  ├── PyMuPDF: Extract images with page/Y coordinates    │
│  └── Chapter parser: Detect chapters, parse questions   │
│                                                         │
│  Output: docling/*.md + images/ + output/*.json         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 2: REVIEW & CORRECTION                           │
│  ├── Streamlit GUI for image-question review            │
│  ├── Chapter-aware question assignment                  │
│  └── Manual corrections saved to JSON                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 3: DECK GENERATION                               │
│  ├── Combine Q&A pairs with images                      │
│  └── Generate Anki deck (.apkg)                         │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.12+
- Conda (recommended for environment management)

### Setup

```bash
# Clone the repository
git clone https://github.com/banderies/textbook-question-extract.git
cd textbook-question-extract

# Create conda environment
conda create -n anki-extractor python=3.12 -y
conda activate anki-extractor

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key (for AI features)
export ANTHROPIC_API_KEY='sk-ant-...'
```

Or use the setup script:
```bash
./setup_env.sh
```

## Usage

### 1. Extract from PDF

First, convert your PDF to markdown using docling (in a separate docling environment):
```bash
conda activate docling
python -c "
from docling.document_converter import DocumentConverter
result = DocumentConverter().convert('your_textbook.pdf')
with open('docling/your_textbook.md', 'w') as f:
    f.write(result.document.export_to_markdown())
"
```

### 2. Extract Images

```bash
conda activate anki-extractor
python scripts/image_pipeline.py extract your_textbook.pdf --output images/
```

### 3. Parse Chapters and Link Images

```bash
python scripts/ai_chapter_parser.py docling/your_textbook.md images/manifest.json
```

### 4. Review and Correct (GUI)

```bash
streamlit run scripts/review_gui_v2.py
```

Open http://localhost:8501 in your browser to review image-question assignments.

### 5. Build Anki Deck

```bash
python scripts/build_deck.py output/ final/
```

## Project Structure

```
textbook-question-extract/
├── scripts/
│   ├── image_pipeline.py       # Extract images from PDF
│   ├── ai_chapter_parser.py    # Chapter-aware Q&A parsing
│   ├── link_images_v2.py       # Link images to questions
│   ├── review_gui_v2.py        # Streamlit review GUI
│   ├── agentic_qa_extractor.py # Full agentic extraction pipeline
│   ├── test_first_5.py         # Test script
│   └── examples/
│       ├── qa_extraction_example.py
│       └── vision_matching_example.py
├── docling/                    # Markdown output from docling (gitignored)
├── images/                     # Extracted images (gitignored except manifest)
├── output/                     # Generated JSON files (gitignored)
├── final/                      # Final Anki deck (gitignored)
├── requirements.txt
├── setup_env.sh
├── run_extraction.sh
└── README.md
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `image_pipeline.py` | Extract images from PDF with positional metadata |
| `ai_chapter_parser.py` | Detect chapters and parse questions with chapter-aware IDs |
| `review_gui_v2.py` | Interactive GUI for reviewing image-question assignments |
| `agentic_qa_extractor.py` | Full pipeline using Claude API for extraction |

## API Usage

The project demonstrates agentic LLM patterns using the Anthropic API:

```python
import anthropic

client = anthropic.Anthropic()

# Q&A extraction
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=4096,
    system="Extract Q&A pairs and return JSON...",
    messages=[{"role": "user", "content": questions_text + answers_text}]
)

# Vision-based image matching
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
            {"type": "text", "text": "Which question does this image belong to?"}
        ]
    }]
)
```

## Dependencies

- `docling` - PDF to Markdown conversion
- `pymupdf` - Image extraction from PDFs
- `anthropic` - Claude API client
- `streamlit` - Interactive review GUI
- `genanki` - Anki deck generation
- `pillow` - Image processing

## License

MIT

## Acknowledgments

- Built with [Claude Code](https://claude.ai/claude-code)
- Uses [Docling](https://github.com/DS4SD/docling) for PDF processing
- Anki deck generation via [genanki](https://github.com/kerrickstaley/genanki)
