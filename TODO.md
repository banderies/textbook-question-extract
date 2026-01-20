# TODO: Refactoring Hardcoded Logic to LLM Prompts

This document tracks areas where hardcoded Python logic should be moved to LLM prompts for better flexibility across different textbook types.

## Design Principle

Content extraction, organization, and formatting decisions should be handled by **LLM prompts**, not hardcoded Python logic. This allows the system to adapt to variations in input material without code changes.

---

## High Priority

### 1. Chapter ID Format
**Location:** `ui_components.py:930, 948`

**Current behavior:**
```python
"full_id": f"ch{ch_num}_{block_id}",
```
Always generates IDs like `ch1`, `ch2`, `ch1_2a`.

**Problem:** Different textbooks may use Roman numerals, section letters, or other numbering schemes.

**Suggested fix:**
- Add `id_prefix` field to `chapters.json` that LLM determines in Step 2
- Prompt change: Ask LLM to specify the appropriate ID prefix for each chapter based on how the textbook labels them
- Python reads the prefix from chapter data instead of hardcoding `ch{N}`

---

### 2. Question ID Format
**Location:** `ui_components.py:948, 977`

**Current behavior:**
```python
full_id = f"ch{ch_num}_{local_id}"
```

**Problem:** Assumes all question IDs follow `chapter_question` pattern.

**Suggested fix:**
- Let LLM determine full question ID during Step 4 formatting
- Add `full_id` to the `format_raw_block` prompt output schema
- Python uses whatever ID the LLM returns

---

### 3. Shared Discussion Field Names
**Location:** `ui_components.py:914-922`

**Current behavior:**
```python
if shared.get("imaging_findings"):
    parts.append(f"**Imaging Findings:** {shared['imaging_findings']}")
if shared.get("discussion"):
    parts.append(f"**Discussion:** {shared['discussion']}")
if shared.get("differential_diagnosis"):
    parts.append(f"**Differential Diagnosis:** {shared['differential_diagnosis']}")
```

**Problem:** Hardcodes medical textbook section names. Other textbooks might have "Clinical Pearls", "Key Points", "Evidence Base", etc.

**Suggested fix:**
- Change prompt to return `shared_discussion.sections` as an array of `{header, content}` objects
- Python iterates over whatever sections the LLM returns
- Example output: `[{"header": "Key Points", "content": "..."}, {"header": "References", "content": "..."}]`

---

### 4. Context Inheritance Logic
**Location:** `ui/helpers.py:104-110`

**Current behavior:**
```python
for block_id, questions in block_questions.items():
    if len(questions) > 1:
        first_q_id = questions[0]["full_id"]
        for q in questions[1:]:
            if not q.get("context_from"):
                q["context_from"] = first_q_id
```

**Problem:** Assumes first sub-question always owns the context.

**Suggested fix:**
- Add `context_owner` field to `format_raw_block` prompt output
- LLM determines which question (if any) owns the shared context
- Python reads `context_from` directly from LLM output, doesn't set it

---

### 5. Context Image Distribution
**Location:** `ui_components.py:958-969`

**Current behavior:**
```python
all_images = list(sq.get("image_files", []))
for img in context_images:
    if img not in all_images:
        all_images.append(img)
```

**Problem:** Code distributes context images to all sub-questions. Should be LLM's decision.

**Suggested fix:**
- Update `format_raw_block` prompt to explicitly include ALL images for each sub-question
- Prompt already instructs this; remove the Python code that re-distributes
- Trust LLM output completely for `image_files` arrays

---

### 6. Cloze Card Categories
**Location:** `config/prompts.yaml:257`

**Current behavior:**
```
Categories: anatomy, pathology, imaging, clinical, differential, statistics, mechanism
```

**Problem:** Medical-specific categories hardcoded in prompt.

**Suggested fix:**
- Create per-textbook config in `config/textbook_types.yaml`
- Each textbook type defines its own categories
- Prompt loads categories dynamically: `{cloze_categories}`

---

## Medium Priority

### 7. Text Concatenation Separator
**Location:** `ui_components.py:950-953`

**Current behavior:**
```python
if context_text and local_id != block_id:
    q_text = context_text + "\n\n" + q_text
```

**Problem:** Hardcoded `\n\n` separator between context and question.

**Suggested fix:**
- LLM should return fully-formed `question_text` including context
- Remove Python concatenation; trust LLM output

---

### 8. Flanking Text Window Size
**Location:** `pdf_extraction.py:618-620`

**Current behavior:**
```python
context_before = " ".join(text_before)[-500:] if text_before else ""
context_after = " ".join(text_after)[:500] if text_after else ""
```

**Problem:** Fixed 500 character window may be too small or large for different PDFs.

**Suggested fix:**
- Add `flanking_text_chars` to `config/settings.yaml` or per-textbook config
- Default to 500, allow override per textbook

---

### 9. Quote/Dash Normalization
**Location:** `pdf_extraction.py:807-816`

**Current behavior:**
```python
replacements = {
    '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
    '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
    '\u201C': '"',  # LEFT DOUBLE QUOTATION MARK
    '\u201D': '"',  # RIGHT DOUBLE QUOTATION MARK
    '\u2013': '-',  # EN DASH
    '\u2014': '-',  # EM DASH
}
```

**Problem:** Hardcoded character mappings.

**Suggested fix:**
- Move to `config/unicode_normalization.yaml`
- Allow per-textbook overrides if needed

---

### 10. Page Range Boundary Default
**Location:** `pdf_extraction.py:664-672`

**Current behavior:**
```python
end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else 9999
```

**Problem:** Hardcoded `9999` as "last page" sentinel.

**Suggested fix:**
- Use actual PDF page count instead of magic number
- Allow chapters to specify explicit `end_page` in LLM output

---

### 11. Markdown Formatting for Headers
**Location:** `ui_components.py:915-921`

**Current behavior:**
```python
parts.append(f"**Imaging Findings:** {shared['imaging_findings']}")
```

**Problem:** Assumes Markdown bold (`**...**`) is correct output format.

**Suggested fix:**
- Add output format config: `markdown`, `html`, or `plain`
- Apply formatting based on config, not hardcoded

---

### 12. Step Order for Cascade Clearing
**Location:** `ui/helpers.py:128`

**Current behavior:**
```python
step_order = ["source", "chapters", "questions", "format", "qc", "generate", "export"]
```

**Problem:** Hardcoded step sequence.

**Suggested fix:**
- Define step order and dependencies in `config/pipeline.yaml`
- Include which steps depend on which, for smarter cache invalidation

---

## Low Priority

### 13. Line Number Marker Format
**Location:** `ui_components.py:887-888`

**Current behavior:**
```python
question_text = re.sub(r'\[LINE:\d+\]\s*', '', question_text)
```

**Problem:** Assumes `[LINE:NNNN]` format.

**Suggested fix:**
- Move regex pattern to config
- Unlikely to change, but good for consistency

---

### 14. Image Filename Format
**Location:** `pdf_extraction.py:623`

**Current behavior:**
```python
filename = f"p{page_num + 1:03d}_y{int(y_position):04d}_x{int(x_position):04d}_{img_idx}.{image_ext}"
```

**Problem:** Fixed digit counts (3 for page, 4 for position).

**Suggested fix:**
- Add image naming template to config
- Allow simpler names like `img_001.jpg` if preferred

---

### 15. JSON Repair Regex Patterns
**Location:** `llm_extraction.py:228-250`

**Current behavior:** Hardcoded regex patterns to fix malformed JSON from LLM.

**Problem:** May not catch all malformation types.

**Suggested fix:**
- Add logging when repair is triggered
- Consider using a JSON5 parser as fallback
- Keep as code (infrastructure), but improve error handling

---

## Implementation Notes

When refactoring these items:

1. **Backward compatibility**: Ensure existing `questions_by_chapter.json` files still load correctly
2. **Default values**: When moving to config, set sensible defaults matching current behavior
3. **Prompt testing**: Test prompt changes against multiple textbook types before deploying
4. **Gradual migration**: Refactor one area at a time, verify each change works

## Config File Structure (Proposed)

```
config/
├── prompts.yaml           # LLM prompts (existing)
├── pipeline.yaml          # Step order, dependencies
├── textbook_types/
│   ├── medical.yaml       # Medical textbook settings
│   ├── default.yaml       # Default settings
│   └── ...
└── unicode_normalization.yaml
```

Each textbook type config would include:
- Cloze categories
- Expected section headers
- Flanking text window size
- ID format preferences

---

## Prompt Language: Rigid Instructions to Make Flexible

The following prompt sections use overly rigid language that limits the LLM's ability to adapt to different textbook formats. The LLM should **understand content semantically** and make decisions based on that understanding, not follow hard formatting rules.

### High Priority

#### P1. Section Header Names (identify_question_blocks, format_raw_block)
**Location:** `prompts.yaml:288-292, 420-423, 452-458`

**Current (rigid):**
```
- "Imaging Findings:" sections
- "Discussion:" sections
- "Differential Diagnosis:" sections
- "References:" sections
```

**Problem:** Lists specific section names; other textbooks may use "Clinical Pearls", "Key Points", "Teaching Points", "Evidence Base", etc.

**Suggested fix:** Change to "educational sections such as discussions, findings, differentials, key points, or any other explanatory content - identify these based on the content's purpose, not specific header text"

---

#### P2. Answer Format Patterns (identify_question_blocks, format_raw_block)
**Location:** `prompts.yaml:286, 310-311, 380-381`

**Current (rigid):**
```
- Lines like "1a Answer B." or "Answer 1a. B." with brief explanations
- answer_start: First line of answers for this block (e.g., "1a  Answer B." or "Answer 1a.")
```

**Problem:** Assumes specific answer line formats that may not apply to all textbooks.

**Suggested fix:** Change to "identify where the answer/explanation section begins based on content - this typically follows the question choices and contains the correct answer with explanation"

---

#### P3. Cloze Categories (generate_cloze_cards)
**Location:** `prompts.yaml:257, 591`

**Current (rigid):**
```
Categories: anatomy, pathology, imaging, clinical, differential, statistics, mechanism
```

**Problem:** Medical-specific categories won't work for other textbook types (history, programming, etc.).

**Suggested fix:** Either make categories configurable per textbook type, or change to "assign a category that describes the type of knowledge being tested (e.g., concept, fact, relationship, process, terminology)"

---

### Medium Priority

#### P4. Chapter Structure Assumptions (identify_chapters)
**Location:** `prompts.yaml:10-13`

**Current (rigid):**
```
This is a medical textbook where each chapter has:
- A chapter header (e.g., "Chapter 1: Title" or "1 Title")
- A QUESTIONS section with numbered questions
- An ANSWERS section with explanations
```

**Problem:** Assumes all textbooks follow this exact structure.

**Suggested fix:** Change to "Analyze the textbook structure to identify chapters that contain question-and-answer content. Look for patterns in how this textbook organizes its Q&A sections."

---

#### P5. Sub-question ID Format Examples (identify_question_blocks, format_raw_block)
**Location:** `prompts.yaml:272-275, 350-353`

**Current (rigid):**
```
- Block "1" = all content for question 1 (context + 1a + 1b + 1c, etc.)
- Look for numbers like "1", "2", "3" OR sub-question IDs like "1a", "1b", "2a", "2b"
```

**Problem:** Only shows letter suffixes; some textbooks use roman numerals (1i, 1ii), numbers (1.1, 1.2), or other schemes.

**Suggested fix:** Change to "sub-questions may use various suffixes (letters like 1a/1b, roman numerals like 1i/1ii, decimals like 1.1/1.2, or other patterns) - identify the scheme used in this textbook"

---

#### P6. Opposite/Complement Pair Lists (generate_cloze_cards)
**Location:** `prompts.yaml:96-97, 536`

**Current (rigid):**
```
Apply to: younger/older, pediatric/adult, increased/decreased, higher/lower, proximal/distal,
anterior/posterior, medial/lateral, superior/inferior, early/late, acute/chronic, larger/smaller, etc.
```

**Problem:** Exhaustive list implies these are the only pairs; misses domain-specific pairs.

**Suggested fix:** Change to "Apply this principle to any complementary or opposite pairs where knowing one reveals the other - use your judgment to identify such relationships in the content"

---

#### P7. Question Pattern Detection (identify_chapters)
**Location:** `prompts.yaml:21`

**Current (rigid):**
```
Look for patterns like "QUESTIONS" or numbered questions (1., 2., etc.)
```

**Problem:** May miss other heading styles like "Practice Problems", "Review Questions", "Self-Assessment", "Exercises".

**Suggested fix:** Change to "Look for sections containing practice questions, review problems, self-assessment items, or similar Q&A content - these may be labeled in various ways"

---

### Implementation Notes for Prompt Changes

When updating prompts:

1. **Preserve examples**: Keep concrete examples but frame them as "such as" or "for example" rather than exhaustive lists
2. **Emphasize understanding**: Use phrases like "based on your understanding of the content" rather than "look for pattern X"
3. **Avoid rigid patterns**: Replace "lines like X or Y" with "content that serves the purpose of..."
4. **Trust the LLM**: The model can identify semantic patterns; don't constrain it with formatting rules
5. **Test thoroughly**: Changes to prompts affect all future extractions; test against multiple textbook styles
