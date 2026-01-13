# Prompt Analysis: Specificity vs Flexibility

This document analyzes each LLM prompt for format-specific assumptions that may break with different input sources. The goal is to identify what needs to change to make the pipeline robust to varying textbook formats.

---

## Summary of Format Assumptions

### Active Prompts (Current Pipeline)

| Prompt | Step | Critical Assumptions | Flexibility Rating |
|--------|------|---------------------|-------------------|
| `identify_chapters` | 2 | Medical textbook, QUESTIONS/ANSWERS sections, chapter headers | Low |
| `identify_question_blocks` | 3 | Block structure with context + sub-questions, line markers | Medium |
| `format_raw_block` | 4 | Block contains context, sub-questions, shared discussion, image distribution | Medium |
| `generate_cloze_cards_from_block` | 6 | Medical content, raw block text with explanations | Medium |

### Legacy Prompts (Available for Backwards Compatibility)

| Prompt | Critical Assumptions | Flexibility Rating |
|--------|---------------------|-------------------|
| `extract_qa_pairs` | QUESTIONS/ANSWERS sections, A-E choices, numbered questions | Low |
| `match_images_to_questions` | Question number at END of text before image, specific layout | Very Low |
| `associate_context` | Data-driven (has_choices, etc.) - no longer a separate step | High |
| `postprocess_questions` | image_group field, numeric vs letter ID patterns | Medium |
| `extract_line_ranges` | QUESTIONS/ANSWERS sections, "Answer X." format, specific layout | Very Low |
| `format_qa_pair` | A-E choices format, verbatim extraction | Medium |
| `generate_cloze_cards` | Medical content focus, individual question explanations | Medium |

---

---

# ACTIVE PROMPTS (Current Pipeline)

---

## Prompt: `identify_chapters` (Step 2)

### Current Assumptions

```
This is a medical textbook where each chapter has:
- A chapter header (e.g., "Chapter 1: Title" or "1 Title")
- A QUESTIONS section with numbered questions
- An ANSWERS section with explanations
```

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| Domain | "medical textbook" | Could be any subject |
| Chapter format | "Chapter 1: Title" or "1 Title" | "Unit 1", "Module 1", "Part I", "Section 1.1" |
| Section names | "QUESTIONS" and "ANSWERS" | "Exercises", "Problems", "Review Questions", "Solutions", "Self-Assessment" |
| Structure | Separate Q and A sections | Interleaved Q&A, answers at back of book, no answers provided |

### What Works Well
- Asks for JSON output with clear schema
- Uses `[PAGE X]` markers from preprocessing (decoupled from raw PDF)
- Flexible on `has_questions: true` filtering

### Recommended Changes

```yaml
# MORE FLEXIBLE VERSION
identify_chapters:
  prompt: |
    Analyze this document's page index and identify all sections that contain
    question-and-answer content, practice problems, exercises, or assessments.

    ADAPT TO THE DOCUMENT'S STRUCTURE:
    - Sections may be called: chapters, units, modules, parts, sections
    - Questions may be labeled: questions, exercises, problems, review, self-assessment, practice
    - Answers may be labeled: answers, solutions, explanations, answer key
    - Some documents have Q&A together, others separate them

    For each section with Q&A content, provide:
    - section_number: The section identifier (number or roman numeral)
    - title: The section title as it appears
    - start_page: Page number from [PAGE X] markers
    - qa_structure: "separate" if Q and A in different locations, "interleaved" if together

    Return ONLY a JSON array...
```

---

## Prompt: `identify_question_blocks` (Step 3)

### Purpose
Identifies question BLOCKS - groups of related questions that share context and possibly images.

### Current Assumptions

```
- Each block contains: main question number, context text, sub-questions (a, b, c, etc.)
- Blocks are separated by changes in main question number
- Uses [LINE:NNNN] markers from preprocessing
- Uses [IMAGE: filename] markers for image positions
```

### What Works Well
- Block-based approach preserves context relationships
- Line markers enable precise text extraction
- Handles multi-part questions naturally
- Output includes both question and answer line ranges

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| Block structure | Context + sub-questions pattern | May have standalone questions, no sub-questions |
| ID pattern | "1", "1a", "1b" numbering | "1.1", "1.2" or "1-a", "1-b" patterns |
| Image markers | `[IMAGE: filename]` format | Depends on preprocessing |

---

## Prompt: `format_raw_block` (Step 4)

### Purpose
Formats a raw text block into structured JSON with context, sub-questions, and shared discussion.

### Current Assumptions

```
- Block contains: context area, one or more sub-questions, shared answer/discussion
- Images in context area are shared by all sub-questions
- Images near specific sub-questions belong to that sub-question
- Each sub-question has: local_id, question_text, choices, correct_answer, explanation
```

### What Works Well
- Separates context from question-specific content
- Handles image placement (context vs question-specific)
- Preserves shared discussion text
- Outputs normalized choice format (A, B, C, D)

### Post-Processing
The `build_block_aware_image_assignments()` function in `ui_components.py` handles:
- Building the `image_assignments` dict (first question with each image wins)
- Setting `context_from` on subsequent sub-questions in the block

### Image Distribution (LLM-handled)
The LLM now handles image distribution directly:
- Context images → all sub-questions' `image_files`
- Sub-question images → that sub-question's `image_files`
- Answer images → `answer_image_files` arrays

---

## Prompt: `generate_cloze_cards_from_block` (Step 6)

### Purpose
Generates cloze deletion flashcards from the raw block content (question + answer + discussion).

### Current Assumptions

```
- Block contains raw text extracted from PDF
- Text includes: clinical scenario, question stem, choices, correct answer, explanation, discussion
- May contain [IMAGE: filename] markers (ignored for card generation)
- May contain formatting artifacts from PDF extraction
```

### What Works Well
- Uses full raw block content (no data loss from intermediate formatting)
- Medical content focus with appropriate categories
- Multi-cloze support (c1, c2, c3 on same card for related facts)
- Handles raw/unformatted text gracefully

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| Domain | Medical content | Could be any subject |
| Categories | anatomy, pathology, imaging, etc. | Need subject-specific categories |

---

# LEGACY PROMPTS (Available for Backwards Compatibility)

The following prompts are available but no longer used in the main pipeline.

---

## Prompt (Legacy): `extract_qa_pairs`

### Current Assumptions

```
1. Find all questions in the QUESTIONS section
2. Find all answers in the ANSWERS section
3. Match each question to its answer
4. Identify the correct answer choice (A, B, C, D, or E)
```

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| Section structure | Separate QUESTIONS and ANSWERS | Interleaved, inline answers, no answers |
| Choice format | "A, B, C, D, or E" | "1, 2, 3, 4", "I, II, III, IV", "(a), (b), (c)", True/False |
| Question numbering | "1", "1a", "2a" | "1.1", "Q1", "Problem 1", roman numerals |
| Answer format | Letter choice | Could be numeric, fill-in-blank, short answer |
| Multi-part | "1", "1a", "1b" pattern | "1(a)", "1.a", "1-a", "1i", "1ii" |

### What Works Well
- Verbatim extraction instruction is format-agnostic
- `has_image` detection based on text references
- `image_group` concept is flexible

### Recommended Changes

```yaml
# MORE FLEXIBLE VERSION
extract_qa_pairs:
  prompt: |
    Extract all questions and their corresponding answers from this chapter.

    ADAPT TO THE DOCUMENT'S FORMAT:
    - Questions may be numbered: 1, 2, 3 OR 1., 2., 3. OR Q1, Q2 OR 1.1, 1.2
    - Sub-questions may use: a, b, c OR (a), (b) OR i, ii, iii OR 1a, 1b
    - Choices may use: A, B, C, D OR 1, 2, 3, 4 OR (a), (b), (c) OR I, II, III
    - Some questions have no choices (open-ended, fill-in-blank)

    DETECT THE STRUCTURE:
    - If answers are in a separate section, match by question number
    - If answers follow each question, extract inline
    - If no answers provided, leave explanation empty

    For choice-based questions, normalize choices to A, B, C, D, E format in output
    regardless of how they appear in the source.

    Return questions with:
    - id: Question identifier as it appears in source
    - id_normalized: Standardized ID (e.g., "1a" regardless of source format)
    ...
```

---

## Prompt (Legacy): `match_images_to_questions`

### Current Assumptions

```
TEXTBOOK STRUCTURE (critical for correct matching):
In this textbook, each question appears in this order:
1. Question text ending with the question number (e.g., "...needle placement? 2a")
2. IMAGE(S) - one OR MORE images may appear here
3. Answer choices (A, B, C, D)
```

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| Question number position | At END of question text | At START ("2a. What is..."), or separate line |
| Image position | Between question and choices | Before question, after choices, in margins, grouped at end |
| Layout | Linear top-to-bottom | Multi-column, floating figures, referenced by "See Figure 3" |
| Reference style | Implicit (image follows question) | Explicit ("refer to Figure 2.1"), numbered figures |

### What Works Well
- Uses flanking text context (flexible)
- Allows multiple images per question
- Has "(none)" fallback for unmatched images

### Critical Problem
This prompt is **highly specific to one textbook's layout**. The assumption that "question number at END of text before image" is very brittle.

### Recommended Changes

```yaml
# MORE FLEXIBLE VERSION
match_images_to_questions:
  prompt: |
    Match images to questions using the surrounding text context.

    DETECTION STRATEGIES (use whichever applies):

    1. PROXIMITY: Image appears immediately after a question
       - Look for question numbers/IDs in text BEFORE the image
       - The closest preceding question number likely owns the image

    2. EXPLICIT REFERENCE: Question text references a figure
       - "See Figure 3", "refer to the image below", "the radiograph shown"
       - Match the figure number/reference to the image

    3. CAPTION: Image has a caption with question reference
       - "Figure for Question 5", "Q3 Image"

    4. SHARED CONTEXT: Multiple questions reference same image
       - Questions 5a, 5b, 5c all say "based on the image above"
       - All should link to the same image

    For each image, analyze:
    - Text immediately before (last 300 chars)
    - Text immediately after (first 300 chars)
    - Any figure numbers or captions
    - Question references to images/figures

    Return your best match, or "(none)" if no clear association.
```

---

## Prompt (Legacy): `postprocess_questions`

### Current Assumptions

```
- A question is "context-only" if it has no choices (choices is empty {})
  AND its ID is just a number (e.g., "5") not a letter suffix (e.g., "5a")
- Sub-questions share the same image_group as their context question
```

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| ID pattern | Number = context, letter suffix = sub | Could be "5.1", "5.2" or "5-a", "5-b" |
| Grouping | Uses `image_group` field | May need to infer from text references |

### What Works Well
- Relies on extracted data structure rather than raw text
- Clear rules for identifying context

### Recommended Changes
- Make ID pattern detection more flexible
- Don't assume `image_group` exists - infer from ID patterns

---

## Prompt (Legacy): `associate_context`

**Note**: This prompt is no longer used as a separate step. Context handling is now done during the `format_raw_block` step.

### Current Assumptions

This prompt is **already well-designed for flexibility**:

```
AVAILABLE DATA FOR EACH QUESTION:
- full_id, local_id, text_preview
- has_choices, num_choices
- has_correct_answer, has_explanation
```

### What Works Well
- **Data-driven**: Uses metadata (has_choices, num_choices) not format patterns
- **Clear rules**: "has_choices: true can NEVER be a context_id"
- **Examples of valid/invalid patterns**
- **Conservative default**: "If unsure, return empty mappings"

### Minor Issues
- `local_id` patterns still assume "5" vs "5a" format
- Could add more diverse ID pattern examples

### This is the model for other prompts
The `associate_context` prompt demonstrates good practice:
1. Provide structured metadata, not raw text
2. Define rules based on data properties
3. Give concrete valid/invalid examples
4. Have a safe fallback

---

## Prompt (Legacy): `extract_line_ranges`

### Current Assumptions

```
TEXTBOOK STRUCTURE:
- QUESTIONS section: Contains numbered questions with answer choices (A/B/C/D/E)
- ANSWERS section: Contains explanations starting with "Answer X." where X is the correct letter
- Questions and answers are in the same order (Question 1 matches Answer 1, etc.)
```

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| Section names | "QUESTIONS" and "ANSWERS" | Many variations |
| Answer format | "Answer X." prefix | "1. A", "Q1: B", just the explanation |
| Ordering | Sequential matching | Answers may be reordered, grouped differently |
| Line markers | `[LINE:NNNN]` format | Depends on preprocessing |
| Image markers | `[IMAGE: filename]` | Depends on preprocessing |

### What Works Well
- Uses preprocessed markers (decoupled from raw PDF)
- Outputs minimal structured data (line ranges)
- Handles multi-part questions

### Critical Problem
The "Answer X." format assumption is very specific. Many textbooks use:
- "1. A - Explanation..."
- Just the explanation with no prefix
- Letter followed by explanation
- Numbered list matching question numbers

### Recommended Changes

```yaml
# MORE FLEXIBLE VERSION
extract_line_ranges:
  prompt: |
    Identify question-answer pairs in this chapter text.

    Line numbers are marked as [LINE:NNNN]. Images are marked as [IMAGE: filename].

    FIND THE STRUCTURE:
    First, identify how this chapter is organized:
    - Are questions and answers in separate sections, or interleaved?
    - How are questions numbered? (1, 2, 3 or 1., 2., 3. or Q1, Q2, etc.)
    - How are answers identified? (Answer 1, 1. A, just explanations, etc.)
    - Are answers in the same order as questions?

    Then extract each Q&A pair:
    - question_id: The identifier as it appears
    - question_start/end: Line range for question + choices
    - answer_start/end: Line range for explanation (0 if not found)
    - correct_letter: The correct choice if identifiable
    - image_files: Images within the question's line range

    HANDLE VARIATIONS:
    - If answers are interleaved, answer may immediately follow choices
    - If answers are in a separate section, match by question number
    - If answer format is unclear, extract the explanation text anyway
```

---

## Prompt (Legacy): `format_qa_pair`

### Current Assumptions

```
Extract each answer choice (A, B, C, D, and E if present)
```

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| Choice labels | A, B, C, D, E | 1, 2, 3, 4 or (a), (b), (c) or I, II, III |
| Choice count | 4-5 choices | 2 (T/F), 3, 6+ |
| Question type | Multiple choice | Fill-in-blank, matching, short answer |

### What Works Well
- Verbatim extraction
- Handles context-only questions
- Simple, focused task

### Recommended Changes

```yaml
# MORE FLEXIBLE VERSION
format_qa_pair:
  prompt: |
    Format this question-answer pair into structured JSON.

    QUESTION ID: {question_id}
    QUESTION TEXT: {question_text}
    ANSWER TEXT: {answer_text}

    DETECT QUESTION TYPE:
    - Multiple choice: Has labeled options (A/B/C/D, 1/2/3/4, etc.)
    - True/False: Two options, typically True and False
    - Fill-in-blank: Has blank spaces to complete
    - Open-ended: No predefined choices

    NORMALIZE OUTPUT:
    - Convert any choice labels to A, B, C, D, E format
    - If True/False, use A=True, B=False
    - If no choices, leave choices empty

    Return JSON with:
    - question_type: "multiple_choice", "true_false", "fill_blank", "open_ended"
    - choices: Normalized to A/B/C/D/E keys (empty for non-choice questions)
    ...
```

---

## Prompt (Legacy): `generate_cloze_cards`

### Current Assumptions

```
Focus on: anatomical structures, pathways, clinical findings,
diagnostic criteria, percentages/statistics, differential diagnosis features
```

### Format-Specific Issues

| Issue | Current Assumption | Alternative Formats |
|-------|-------------------|---------------------|
| Domain | Medical content | Could be any subject |
| Categories | anatomy, pathology, imaging, etc. | Need subject-specific categories |

### What Works Well
- Cloze syntax is universal (Anki standard)
- Quality guidelines are generalizable
- Good/bad examples help calibrate

### Recommended Changes

```yaml
# MORE FLEXIBLE VERSION
generate_cloze_cards:
  prompt: |
    Create Anki cloze deletion flashcards from this explanation text.

    SOURCE: {question_id}
    SUBJECT DOMAIN: {domain}  # NEW: Pass domain from chapter detection

    EXPLANATION:
    {explanation}

    UNIVERSAL GUIDELINES:
    - Each card tests ONE key fact or relationship
    - Include enough context for the card to stand alone
    - Focus on: definitions, relationships, quantities, sequences, comparisons

    DOMAIN-SPECIFIC FOCUS:
    - Medical: anatomy, pathology, clinical findings, diagnostics
    - Science: formulas, processes, classifications, properties
    - History: dates, events, causes, effects, figures
    - Language: vocabulary, grammar rules, usage patterns

    Detect the domain from the content and categorize appropriately.
```

---

## Flexibility Improvement Priorities

### Active Prompts - Priority for Improvement

1. **`identify_chapters`** (High) - "QUESTIONS/ANSWERS" section names, medical textbook assumption
2. **`identify_question_blocks`** (Medium) - Block structure assumptions, line marker format
3. **`format_raw_block`** (Medium) - Context/sub-question structure
4. **`generate_cloze_cards_from_block`** (Low) - Domain is only constraint, handles raw text well

### Legacy Prompts (Lower Priority - Not in Active Pipeline)

- **`extract_line_ranges`** - "Answer X." format assumption
- **`match_images_to_questions`** - Layout-specific assumptions
- **`extract_qa_pairs`** - Choice format (A/B/C/D)
- **`format_qa_pair`** - Choice format normalization
- **`associate_context`** - Already data-driven and flexible
- **`generate_cloze_cards`** - Domain is only constraint

---

## Design Principles for Robust Prompts

Based on analyzing `associate_context` (the most robust prompt):

### 1. Use Structured Metadata, Not Raw Patterns
```
BAD:  "Look for 'Answer X.' at the start of the line"
GOOD: "has_correct_answer: true/false indicates if answer was found"
```

### 2. Detect Structure Before Extracting
```
BAD:  "The QUESTIONS section contains..."
GOOD: "First identify how Q&A content is organized, then extract accordingly"
```

### 3. Provide Valid AND Invalid Examples
```
GOOD: "VALID: Question 5 has no choices → IS context
       INVALID: Questions 2a, 2b both have choices → NEITHER is context"
```

### 4. Define Safe Fallbacks
```
GOOD: "If structure is unclear, return empty array rather than guessing"
```

### 5. Normalize Output Regardless of Input
```
GOOD: "Convert any choice format (1/2/3, a/b/c, I/II/III) to A/B/C/D in output"
```

### 6. Pass Detection Results Forward
```
GOOD: Have identify_chapters output qa_structure: "separate" | "interleaved"
      Then extract_qa_pairs can adapt its strategy based on this
```
