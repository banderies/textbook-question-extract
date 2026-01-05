#!/bin/bash
# =============================================================================
# Anki Q&A Extractor - Main Runner Script
# =============================================================================
#
# This script activates the conda environment and runs the extraction pipeline.
#
# Usage:
#   ./run_extraction.sh <pdf_file>                    # Full pipeline
#   ./run_extraction.sh --extract-images <pdf_file>   # Extract images only
#   ./run_extraction.sh --extract-qa <markdown_file>  # Extract Q&A only
#   ./run_extraction.sh --build-deck                  # Build Anki deck from JSON
#
# Environment Variables:
#   ANTHROPIC_API_KEY - Required for Q&A extraction (Claude API)
#
# =============================================================================

set -e

# Configuration
ENV_NAME="anki-extractor"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Find and initialize conda
find_conda() {
    local conda_paths=(
        "$HOME/opt/miniconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/opt/anaconda3/etc/profile.d/conda.sh"
    )

    for path in "${conda_paths[@]}"; do
        if [ -f "$path" ]; then
            source "$path"
            return 0
        fi
    done

    if command -v conda &> /dev/null; then
        local conda_base=$(conda info --base 2>/dev/null)
        if [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
            source "$conda_base/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    echo -e "${RED}Error: Could not find conda installation${NC}"
    exit 1
}

# Activate environment
activate_env() {
    find_conda

    if ! conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${RED}Error: Environment '$ENV_NAME' not found${NC}"
        echo "Run ./setup_env.sh first to create it"
        exit 1
    fi

    conda activate "$ENV_NAME"
}

# Check API key
check_api_key() {
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo -e "${YELLOW}Warning: ANTHROPIC_API_KEY not set${NC}"
        echo "Q&A extraction requires the Claude API."
        echo ""
        echo "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Show usage
show_usage() {
    echo "Anki Q&A Extractor"
    echo ""
    echo "Usage:"
    echo "  $0 <pdf_file>                      Full pipeline (images + Q&A + deck)"
    echo "  $0 --extract-images <pdf_file>    Extract images only"
    echo "  $0 --extract-qa <markdown_file>   Extract Q&A pairs only"
    echo "  $0 --link-images <markdown_file>  Link images to markdown"
    echo "  $0 --build-deck                   Build Anki deck from JSON"
    echo "  $0 --test                         Run test on first 5 questions"
    echo ""
    echo "Environment Variables:"
    echo "  ANTHROPIC_API_KEY   Claude API key (required for Q&A extraction)"
    echo ""
    echo "Examples:"
    echo "  $0 'Core Review - Musculoskeletal.pdf'"
    echo "  $0 --extract-qa docling/book.md"
}

# Extract images from PDF
extract_images() {
    local pdf_file="$1"
    echo -e "${CYAN}=== Extracting Images ===${NC}"
    python "$SCRIPT_DIR/scripts/image_pipeline.py" extract "$pdf_file" --output images/
}

# Link images to markdown
link_images() {
    local markdown_file="$1"
    echo -e "${CYAN}=== Linking Images to Markdown ===${NC}"
    python "$SCRIPT_DIR/scripts/link_images.py" "$markdown_file" images/manifest.json
}

# Extract Q&A pairs
extract_qa() {
    local markdown_file="$1"
    check_api_key
    echo -e "${CYAN}=== Extracting Q&A Pairs ===${NC}"
    python "$SCRIPT_DIR/scripts/agentic_qa_extractor.py" "$markdown_file" --images-dir images/
}

# Build Anki deck
build_deck() {
    echo -e "${CYAN}=== Building Anki Deck ===${NC}"
    python "$SCRIPT_DIR/scripts/build_deck.py" output/ final/
}

# Run test
run_test() {
    check_api_key
    echo -e "${CYAN}=== Running Test (First 5 Questions) ===${NC}"
    python "$SCRIPT_DIR/scripts/test_first_5.py"
}

# Full pipeline
full_pipeline() {
    local pdf_file="$1"
    local basename=$(basename "$pdf_file" .pdf)
    local markdown_file="docling/${basename}.md"

    echo -e "${GREEN}=== Full Extraction Pipeline ===${NC}"
    echo "PDF: $pdf_file"
    echo ""

    # Step 1: Check if docling output exists
    if [ ! -f "$markdown_file" ]; then
        echo -e "${YELLOW}Docling markdown not found at: $markdown_file${NC}"
        echo "Please run docling first to convert PDF to markdown."
        echo ""
        echo "If you have a docling environment:"
        echo "  conda activate docling"
        echo "  python -c \"from docling.document_converter import DocumentConverter; print(DocumentConverter().convert('$pdf_file').document.export_to_markdown())\" > \"$markdown_file\""
        exit 1
    fi

    echo -e "${GREEN}✓ Found docling output: $markdown_file${NC}"

    # Step 2: Extract images
    if [ ! -f "images/manifest.json" ]; then
        extract_images "$pdf_file"
    else
        echo -e "${GREEN}✓ Images already extracted (images/manifest.json exists)${NC}"
    fi

    # Step 3: Link images to markdown
    link_images "$markdown_file"

    # Step 4: Extract Q&A
    extract_qa "$markdown_file"

    # Step 5: Build deck
    build_deck

    echo ""
    echo -e "${GREEN}=== Pipeline Complete ===${NC}"
    echo "Anki deck saved to: final/"
}

# Main
activate_env

case "${1:-}" in
    --help|-h)
        show_usage
        ;;
    --extract-images)
        shift
        extract_images "$1"
        ;;
    --extract-qa)
        shift
        extract_qa "$1"
        ;;
    --link-images)
        shift
        link_images "$1"
        ;;
    --build-deck)
        build_deck
        ;;
    --test)
        run_test
        ;;
    "")
        show_usage
        ;;
    *)
        full_pipeline "$1"
        ;;
esac
