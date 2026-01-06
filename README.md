# Textbook Question Extractor

Extract Q&A pairs from PDF textbooks and generate Anki flashcard decks using Claude AI.

## Quick Start

```bash
# Setup
git clone https://github.com/banderies/textbook-question-extract.git
cd textbook-question-extract
./setup_env.sh
conda activate anki-extractor
export ANTHROPIC_API_KEY='sk-ant-...'

# Run the GUI
streamlit run scripts/review_gui.py
```

Open http://localhost:8501 and follow the 6-step workflow:

1. **Select Source** - Load your PDF textbook
2. **Extract Chapters** - AI identifies chapter boundaries
3. **Extract Questions** - AI extracts Q&A pairs per chapter
4. **Associate Context** - Link context questions to sub-questions (Q1 → Q1a, Q1b)
5. **QC Questions** - Review and correct image assignments
6. **Export** - Generate Anki deck

## Requirements

- Python 3.12+
- Conda (recommended)
- Anthropic API key

## Installation

```bash
# Create environment
conda create -n anki-extractor python=3.12 -y
conda activate anki-extractor

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY='sk-ant-...'
```

## Features

- **AI-Powered Extraction**: Uses Claude to extract questions, answers, and explanations
- **Image Linking**: Automatically matches images to questions using flanking text context
- **Context Association**: Merges clinical scenario context into sub-questions
- **Chapter-Aware IDs**: Questions prefixed with chapter (e.g., `ch1_2a`, `ch8_2a`)
- **Interactive QC**: Streamlit GUI for reviewing and correcting assignments
- **State Persistence**: All progress saved to JSON files
- **Model Selection**: Choose Claude model per extraction step
- **Editable Prompts**: Customize LLM behavior via `config/prompts.yaml`

## Project Structure

```
scripts/
├── review_gui.py          # Main Streamlit app
├── ui_components.py       # UI rendering functions
├── state_management.py    # Session state and file I/O
├── llm_extraction.py      # Claude API and prompt management
├── pdf_extraction.py      # PDF text/image extraction
└── config/
    └── prompts.yaml       # Editable LLM prompts
```

## License

MIT

See [CLAUDE.md](CLAUDE.md) for technical architecture and development details.
