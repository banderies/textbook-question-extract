# MSK Radiology Anki Deck Generator

## Project Overview

Convert "Musculoskeletal Imaging: A Core Review" PDF into an Anki flashcard deck using a hybrid traditional-processing + LLM-analysis pipeline.

**Key Insight**: Simple regex/positional parsing fails because PDF structure is unpredictable. Instead, we use traditional tools for extraction and LLM intelligence for semantic matching.

## Architecture

```
PDF Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 1: PREPROCESSING (Python scripts)                │
│  ├── docling: PDF → Markdown (preserves structure)      │
│  ├── PyMuPDF: Extract images with page/Y coordinates    │
│  └── Chapter splitter: Separate content by chapter      │
│                                                         │
│  Output: chapters/*.md + images/*.jpg + manifest.json   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 2: LLM PROCESSING (Claude Code, per-chapter)     │
│  ├── Input: Single chapter markdown + image list        │
│  ├── Task: Match questions to answers semantically      │
│  └── Output: Structured JSON with Q&A pairs             │
│                                                         │
│  Key: Fresh context per chapter = no accumulation       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 3: POSTPROCESSING (Python script)                │
│  ├── Combine all chapter JSONs                          │
│  ├── Associate images with questions                    │
│  └── Build Anki deck with genanki                       │
│                                                         │
│  Output: MSK_Radiology_Core_Review.apkg                 │
└─────────────────────────────────────────────────────────┘
```

## Directory Structure

```
msk-anki-project/
├── CLAUDE.md                 # This file - instructions for Claude Code
├── input/
│   └── Core_Review_MSK.pdf   # Source PDF (you provide this)
├── scripts/
│   ├── 01_preprocess.py      # Extract markdown + images
│   ├── 02_split_chapters.py  # Split into chapter files
│   └── 03_build_deck.py      # Combine JSONs → Anki deck
├── chapters/                  # Generated: per-chapter markdown
│   ├── ch01_imaging_techniques_questions.md
│   ├── ch01_imaging_techniques_answers.md
│   ├── ch02_normal_variants_questions.md
│   └── ...
├── images/                    # Generated: extracted images
│   ├── manifest.json          # Image metadata (page, y-position)
│   └── *.jpg
├── output/                    # Generated: LLM outputs
│   ├── ch01_qa.json
│   ├── ch02_qa.json
│   └── ...
└── final/
    └── MSK_Radiology.apkg    # Final Anki deck
```

## Chapter Definitions

The book has 9 chapters. Page numbers are 1-indexed as they appear in PDF:

| Chapter | Name | Questions Pages | Answers Pages |
|---------|------|-----------------|---------------|
| 1 | Imaging Techniques/Physics/Quality and Safety | 14-29 | 30-36 |
| 2 | Normal/Normal Variants | 37-42 | 43-44 |
| 3 | Congenital and Developmental | 45-54 | 55-58 |
| 4 | Infection | 60-67 | 68-71 |
| 5 | Tumors and Tumor-Like Conditions | 72-106 | 107-126 |
| 6 | Trauma | 127-166 | 167-187 |
| 7 | Metabolic and Hematologic Disorders | 187-193 | 194-198 |
| 8 | Arthropathy | 199-215 | 216-226 |
| 9 | Miscellaneous | 227-229 | 230-233 |

---

## PHASE 1: Preprocessing

### Step 1.1: Install Dependencies

```bash
pip install docling pymupdf genanki
```

### Step 1.2: Create Preprocessing Script

Create `scripts/01_preprocess.py`:

```python
#!/usr/bin/env python3
"""
Phase 1: Extract markdown and images from PDF.

Usage:
    python scripts/01_preprocess.py input/Core_Review_MSK.pdf
"""

import fitz  # PyMuPDF
import json
import os
import sys
from pathlib import Path
from docling.document_converter import DocumentConverter

def extract_images(pdf_path: str, output_dir: str) -> list:
    """Extract all images with their page numbers and Y positions."""
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    manifest = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        images = page.get_images()
        
        for img_idx, img in enumerate(images):
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                
                base_image = doc.extract_image(xref)
                ext = base_image["ext"]
                filename = f"p{page_idx + 1:03d}_y{int(rects[0].y0):04d}_x{xref}.{ext}"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, "wb") as f:
                    f.write(base_image["image"])
                
                manifest.append({
                    "filename": filename,
                    "page": page_idx + 1,
                    "y_position": rects[0].y0,
                    "xref": xref
                })
            except Exception as e:
                print(f"Warning: Could not extract image {xref} from page {page_idx + 1}: {e}")
    
    doc.close()
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Extracted {len(manifest)} images to {output_dir}")
    return manifest

def convert_to_markdown(pdf_path: str, output_path: str) -> str:
    """Convert PDF to markdown using docling."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    markdown = result.document.export_to_markdown()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"Converted PDF to markdown: {output_path}")
    return markdown

def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_preprocess.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Create output directories
    Path("images").mkdir(exist_ok=True)
    Path("chapters").mkdir(exist_ok=True)
    
    # Extract images
    print("\n=== Extracting images ===")
    extract_images(pdf_path, "images")
    
    # Convert to markdown
    print("\n=== Converting to markdown ===")
    convert_to_markdown(pdf_path, "full_document.md")
    
    print("\n=== Phase 1 complete ===")
    print("Next: Run 02_split_chapters.py to split by chapter")

if __name__ == "__main__":
    main()
```

### Step 1.3: Create Chapter Splitter Script

Create `scripts/02_split_chapters.py`:

```python
#!/usr/bin/env python3
"""
Phase 1b: Split the full markdown into chapter-specific files.

This script attempts to automatically identify chapter boundaries.
You may need to adjust the splitting logic based on docling's output format.

Usage:
    python scripts/02_split_chapters.py full_document.md
"""

import re
import os
import sys
from pathlib import Path

# Chapter definitions with expected title patterns
CHAPTERS = [
    {
        "id": "ch01",
        "name": "Imaging Techniques/Physics/Quality and Safety",
        "patterns": ["imaging technique", "physics", "quality and safety"],
    },
    {
        "id": "ch02", 
        "name": "Normal/Normal Variants",
        "patterns": ["normal variant", "normal/normal"],
    },
    {
        "id": "ch03",
        "name": "Congenital and Developmental",
        "patterns": ["congenital", "developmental"],
    },
    {
        "id": "ch04",
        "name": "Infection",
        "patterns": ["infection"],
    },
    {
        "id": "ch05",
        "name": "Tumors and Tumor-Like Conditions",
        "patterns": ["tumor", "tumour"],
    },
    {
        "id": "ch06",
        "name": "Trauma",
        "patterns": ["trauma"],
    },
    {
        "id": "ch07",
        "name": "Metabolic and Hematologic Disorders",
        "patterns": ["metabolic", "hematologic"],
    },
    {
        "id": "ch08",
        "name": "Arthropathy",
        "patterns": ["arthropathy"],
    },
    {
        "id": "ch09",
        "name": "Miscellaneous",
        "patterns": ["miscellaneous"],
    },
]

def split_markdown_by_chapters(markdown_path: str, output_dir: str):
    """
    Split markdown into chapter files.
    
    NOTE: This is a template. You may need to adjust based on 
    how docling formats the output. Look at full_document.md first
    to understand the structure.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    with open(markdown_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Strategy: Look for chapter headings (usually ## or # followed by chapter name)
    # This regex looks for markdown headings that might contain chapter titles
    
    # First, let's identify where "QUESTIONS" and "ANSWERS AND EXPLANATIONS" sections are
    # These are key structural markers in the book
    
    questions_pattern = r'(?:^|\n)(#{1,3}\s*)?QUESTIONS?\s*\n'
    answers_pattern = r'(?:^|\n)(#{1,3}\s*)?ANSWERS?\s+AND\s+EXPLANATIONS?\s*\n'
    
    # Find all matches
    q_matches = list(re.finditer(questions_pattern, content, re.IGNORECASE))
    a_matches = list(re.finditer(answers_pattern, content, re.IGNORECASE))
    
    print(f"Found {len(q_matches)} QUESTIONS sections")
    print(f"Found {len(a_matches)} ANSWERS sections")
    
    # For each chapter, try to extract its questions and answers sections
    # This is a simplified approach - you may need to refine based on actual output
    
    if len(q_matches) == 0:
        print("\nWARNING: Could not find QUESTIONS markers in markdown.")
        print("The markdown structure may be different than expected.")
        print("Please examine full_document.md manually and adjust this script.")
        
        # Fallback: save entire document for manual processing
        output_path = os.path.join(output_dir, "full_content.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved full content to {output_path}")
        return
    
    print("\nChapter splitting complete. Check the chapters/ directory.")
    print("You may need to manually verify and adjust the splits.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 02_split_chapters.py <markdown_path>")
        sys.exit(1)
    
    split_markdown_by_chapters(sys.argv[1], "chapters")

if __name__ == "__main__":
    main()
```

---

## PHASE 2: LLM Processing (The Key Step)

This is where Claude Code processes each chapter. The key insight is that each chapter is processed in a **fresh conversation context**, preventing context overflow.

### Step 2.1: Create the Processing Instructions

Create `CHAPTER_PROCESSING_PROMPT.md` - this is the prompt template for processing each chapter:

```markdown
# Chapter Processing Instructions

You are processing a chapter from "Musculoskeletal Imaging: A Core Review" to create Anki flashcards.

## Your Task

1. Read the QUESTIONS section and extract each question with its choices
2. Read the ANSWERS AND EXPLANATIONS section
3. Match each question to its correct answer using SEMANTIC UNDERSTANDING
4. Output structured JSON

## Critical Matching Rules

- Questions and answers are linked by FIGURE NUMBERS (e.g., "37", "38a", "38b")
- Figure numbers appear near questions and at the start of answer explanations
- Some questions share figure numbers (e.g., 38a, 38b refer to same image)
- Use CONTENT MATCHING when figure numbers are ambiguous:
  - Medical terms in question should appear in its explanation
  - The correct choice text often appears in the explanation
  - Disease names, anatomical terms, imaging findings should match

## Output Format

Output a JSON array with this structure:

```json
{
  "chapter": "Chapter Name",
  "questions": [
    {
      "figure_num": "37",
      "question": "Full question text...",
      "choices": {
        "A": "Choice A text",
        "B": "Choice B text", 
        "C": "Choice C text",
        "D": "Choice D text",
        "E": "Choice E text (if exists)"
      },
      "correct_answer": "A",
      "explanation": "Full explanation text...",
      "image_pages": [26],
      "confidence": "high"
    }
  ]
}
```

## Confidence Levels

- "high": Clear figure number match AND content matches
- "medium": Either figure number OR content matches well
- "low": Best guess based on position or partial matching

## Image Association

Questions that reference images ("image below", "following radiograph", "images provided") 
should have their page numbers noted. Use the image manifest to find relevant images.

## Now Process This Chapter

[CHAPTER CONTENT WILL BE INSERTED HERE]

[IMAGE MANIFEST FOR THIS CHAPTER WILL BE INSERTED HERE]
```

### Step 2.2: Claude Code Processing Script

Create `scripts/process_chapter.sh` - a wrapper to call Claude Code for each chapter:

```bash
#!/bin/bash
# Process a single chapter with Claude Code
# Usage: ./process_chapter.sh chapters/ch01_questions.md chapters/ch01_answers.md output/ch01_qa.json

QUESTIONS_FILE=$1
ANSWERS_FILE=$2
OUTPUT_FILE=$3
CHAPTER_NAME=$4

# Combine inputs for Claude
PROMPT="Process this radiology chapter and output JSON matching questions to answers.

## Questions Section:
$(cat "$QUESTIONS_FILE")

## Answers Section:
$(cat "$ANSWERS_FILE")

## Image Manifest (for this chapter's pages):
$(cat images/manifest.json | jq '[.[] | select(.page >= 14 and .page <= 36)]')

Output only valid JSON following the schema in CHAPTER_PROCESSING_PROMPT.md"

# Call Claude Code
# Note: This assumes 'claude' CLI is installed and configured
claude --print "$PROMPT" > "$OUTPUT_FILE"

echo "Processed $CHAPTER_NAME -> $OUTPUT_FILE"
```

### Step 2.3: Master Processing Script

Create `scripts/run_all_chapters.sh`:

```bash
#!/bin/bash
# Process all chapters through Claude Code

CHAPTERS=(
    "ch01:Imaging Techniques:14:36"
    "ch02:Normal Variants:37:44"
    "ch03:Congenital:45:58"
    "ch04:Infection:60:71"
    "ch05:Tumors:72:126"
    "ch06:Trauma:127:187"
    "ch07:Metabolic:187:198"
    "ch08:Arthropathy:199:226"
    "ch09:Miscellaneous:227:233"
)

mkdir -p output

for chapter_info in "${CHAPTERS[@]}"; do
    IFS=':' read -r id name start_page end_page <<< "$chapter_info"
    
    echo "Processing $name..."
    
    # Note: Adjust file paths based on how chapters were actually split
    ./scripts/process_chapter.sh \
        "chapters/${id}_questions.md" \
        "chapters/${id}_answers.md" \
        "output/${id}_qa.json" \
        "$name"
    
    echo "Completed $name"
    echo "---"
done

echo "All chapters processed!"
```

---

## PHASE 3: Postprocessing

### Step 3.1: Build Anki Deck Script

Create `scripts/03_build_deck.py`:

```python
#!/usr/bin/env python3
"""
Phase 3: Combine all chapter JSONs and build Anki deck.

Usage:
    python scripts/03_build_deck.py output/ final/MSK_Radiology.apkg
"""

import genanki
import json
import os
import sys
import hashlib
import html
from pathlib import Path

def generate_id(seed: str) -> int:
    return int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)

def escape_html(text: str) -> str:
    if not text:
        return ""
    return html.escape(str(text)).replace('\n', '<br>')

# Card styling
CSS = '''
.card { 
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
    font-size: 16px; 
    line-height: 1.6; 
    color: #1a1a1a; 
    background: #fff; 
    max-width: 800px; 
    margin: auto; 
    padding: 20px; 
}
.chapter { 
    font-size: 12px; 
    color: #666; 
    margin-bottom: 15px; 
    text-transform: uppercase; 
    border-bottom: 1px solid #e0e0e0; 
    padding-bottom: 8px; 
}
.question { 
    font-size: 17px; 
    font-weight: 600; 
    color: #1a1a1a; 
    margin-bottom: 20px; 
}
.images { 
    margin: 15px 0; 
    text-align: center; 
}
.images img { 
    max-width: 100%; 
    max-height: 400px; 
    margin: 5px; 
    border: 1px solid #ddd; 
    border-radius: 4px; 
}
.choice { 
    margin: 12px 0; 
    padding: 10px 15px; 
    background: #f5f5f5; 
    border-radius: 6px; 
    color: #1a1a1a; 
    border-left: 3px solid #ddd; 
}
hr#answer { 
    border: none; 
    border-top: 2px solid #4CAF50; 
    margin: 25px 0; 
}
.correct-answer { 
    color: #2e7d32; 
    font-size: 18px; 
    font-weight: 600; 
    margin: 15px 0; 
    padding: 10px; 
    background: #e8f5e9; 
    border-radius: 6px; 
}
.explanation { 
    margin-top: 20px; 
    padding: 20px; 
    background: #f8f9fa; 
    border-left: 4px solid #2196F3; 
    border-radius: 0 8px 8px 0; 
    color: #333; 
}
.night_mode .card { background: #1e1e1e; color: #e0e0e0; }
.night_mode .question, .night_mode .choice { color: #e0e0e0; }
.night_mode .choice { background: #2d2d2d; }
.night_mode .correct-answer { background: #1b3d1f; color: #81c784; }
.night_mode .explanation { background: #2d2d2d; color: #e0e0e0; }
'''

def create_model():
    return genanki.Model(
        generate_id('msk_radiology_mc'),
        'MSK Radiology - Multiple Choice',
        fields=[
            {'name': 'Question'},
            {'name': 'Images'},
            {'name': 'ChoiceA'},
            {'name': 'ChoiceB'},
            {'name': 'ChoiceC'},
            {'name': 'ChoiceD'},
            {'name': 'ChoiceE'},
            {'name': 'CorrectAnswer'},
            {'name': 'Explanation'},
            {'name': 'Chapter'},
        ],
        templates=[{
            'name': 'Card 1',
            'qfmt': '''
<div class="chapter">{{Chapter}}</div>
<div class="question">{{Question}}</div>
{{#Images}}<div class="images">{{Images}}</div>{{/Images}}
<div class="choices">
{{#ChoiceA}}<div class="choice"><b>A.</b> {{ChoiceA}}</div>{{/ChoiceA}}
{{#ChoiceB}}<div class="choice"><b>B.</b> {{ChoiceB}}</div>{{/ChoiceB}}
{{#ChoiceC}}<div class="choice"><b>C.</b> {{ChoiceC}}</div>{{/ChoiceC}}
{{#ChoiceD}}<div class="choice"><b>D.</b> {{ChoiceD}}</div>{{/ChoiceD}}
{{#ChoiceE}}<div class="choice"><b>E.</b> {{ChoiceE}}</div>{{/ChoiceE}}
</div>
''',
            'afmt': '''
{{FrontSide}}
<hr id="answer">
<div class="correct-answer">✓ Answer: {{CorrectAnswer}}</div>
<div class="explanation">{{Explanation}}</div>
''',
        }],
        css=CSS
    )

def load_chapter_jsons(input_dir: str) -> list:
    """Load all chapter JSON files and combine."""
    all_questions = []
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('_qa.json'):
            filepath = os.path.join(input_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chapter_name = data.get('chapter', filename.replace('_qa.json', ''))
                questions = data.get('questions', [])
                
                for q in questions:
                    q['chapter'] = chapter_name
                    all_questions.append(q)
                
                print(f"Loaded {len(questions)} questions from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return all_questions

def find_images_for_question(question: dict, image_manifest: list) -> list:
    """Find image files for a question based on page numbers."""
    image_pages = question.get('image_pages', [])
    matching_images = []
    
    for img in image_manifest:
        if img['page'] in image_pages:
            matching_images.append(img['filename'])
    
    return matching_images

def build_deck(questions: list, image_manifest: list, image_dir: str, output_path: str):
    """Build the Anki deck from questions."""
    model = create_model()
    deck = genanki.Deck(generate_id('msk_radiology'), 'MSK Radiology Core Review')
    media_files = []
    
    for q in questions:
        # Build image HTML
        images_html = ""
        question_images = find_images_for_question(q, image_manifest)
        for img_filename in question_images:
            img_path = os.path.join(image_dir, img_filename)
            if os.path.exists(img_path):
                media_files.append(img_path)
                images_html += f'<img src="{img_filename}">'
        
        choices = q.get('choices', {})
        
        note = genanki.Note(
            model=model,
            fields=[
                escape_html(q.get('question', '')),
                images_html,
                escape_html(choices.get('A', '')),
                escape_html(choices.get('B', '')),
                escape_html(choices.get('C', '')),
                escape_html(choices.get('D', '')),
                escape_html(choices.get('E', '')),
                q.get('correct_answer', '?'),
                escape_html(q.get('explanation', 'No explanation available')),
                q.get('chapter', 'Unknown'),
            ],
            tags=['MSK', q.get('chapter', '').replace(' ', '_').replace('/', '_')]
        )
        deck.add_note(note)
    
    # Create package
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    package = genanki.Package(deck)
    package.media_files = list(set(media_files))
    package.write_to_file(output_path)
    
    print(f"\nCreated Anki deck: {output_path}")
    print(f"  Total cards: {len(questions)}")
    print(f"  Images included: {len(set(media_files))}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python 03_build_deck.py <input_dir> <output_apkg>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load image manifest
    manifest_path = "images/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            image_manifest = json.load(f)
    else:
        print("Warning: No image manifest found")
        image_manifest = []
    
    # Load and combine all chapter JSONs
    questions = load_chapter_jsons(input_dir)
    print(f"\nTotal questions loaded: {len(questions)}")
    
    # Build deck
    build_deck(questions, image_manifest, "images", output_path)

if __name__ == "__main__":
    main()
```

---

## Usage Instructions for Claude Code

### Initial Setup

```bash
# Create project directory
mkdir msk-anki-project
cd msk-anki-project

# Copy PDF to input directory
mkdir -p input
cp /path/to/Core_Review_MSK.pdf input/

# Install dependencies
pip install docling pymupdf genanki

# Create directory structure
mkdir -p scripts chapters images output final
```

### Run Phase 1: Preprocessing

```bash
python scripts/01_preprocess.py input/Core_Review_MSK.pdf
python scripts/02_split_chapters.py full_document.md
```

### Run Phase 2: LLM Processing

For each chapter, ask Claude Code:

```
Read chapters/ch01_questions.md and chapters/ch01_answers.md.
Match each question to its answer using semantic understanding.
Output JSON following the schema in CHAPTER_PROCESSING_PROMPT.md.
Save to output/ch01_qa.json.
```

Repeat for each chapter (ch01 through ch09).

### Run Phase 3: Build Deck

```bash
python scripts/03_build_deck.py output/ final/MSK_Radiology.apkg
```

---

## Troubleshooting

### docling output issues
- Check `full_document.md` to understand the actual structure
- Adjust `02_split_chapters.py` regex patterns accordingly
- You may need to manually split chapters if structure is unusual

### Image matching issues
- Check `images/manifest.json` for page numbers
- Verify page numbers in chapter definitions
- Images are matched by page number proximity

### JSON parsing errors
- Ensure Claude Code outputs valid JSON
- Check for unclosed quotes or brackets
- Validate JSON before running Phase 3

---

## Alternative: Manual Processing

If automated splitting doesn't work well, you can:

1. Open `full_document.md` in a text editor
2. Manually copy each chapter's questions and answers to separate files
3. Process each file individually with Claude Code

This is more work but guarantees correct chapter boundaries.

---

## Key Learning Points

This project demonstrates a powerful pattern:

1. **Traditional tools for extraction** - PyMuPDF, docling handle the mechanical work
2. **LLM for semantic understanding** - Claude matches content that requires comprehension
3. **Fresh context per chunk** - Processing chapters separately prevents context overflow
4. **Structured output** - JSON schema ensures consistent, parseable results

This pattern applies to many scenarios:
- Processing legal documents
- Analyzing research papers
- Converting documentation formats
- Data extraction from PDFs
