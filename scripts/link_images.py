#!/usr/bin/env python3
"""
Image-to-Markdown Linker

This script creates a mapping between `<!-- image -->` markers in the docling
markdown output and the actual image files extracted by PyMuPDF.

The key insight: Both docling and PyMuPDF process images in the same order
(page by page, top to bottom). So the Nth `<!-- image -->` marker corresponds
to the Nth extracted image when sorted by (page, y_position).

Usage:
    python link_images.py <markdown_file> <manifest_json>

Output:
    - image_links.json: Mapping of image filenames to their markdown context
    - markdown_with_images.md: Markdown with image placeholders replaced by filenames

Example:
    python link_images.py docling/book.md images/manifest.json
"""

import json
import re
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ImageMarker:
    """Represents an <!-- image --> marker in the markdown."""
    marker_index: int      # 0-indexed position of this marker
    line_number: int       # Line number in the markdown file
    context_before: str    # Text before the marker (for matching)
    context_after: str     # Text after the marker
    nearby_question_id: Optional[str] = None  # If near a question


@dataclass
class ImageLink:
    """Links an image file to its markdown context."""
    image_filename: str
    page: int
    y_position: float
    marker_index: int
    line_number: int
    context_before: str
    context_after: str
    nearby_question_id: Optional[str]


def parse_markdown_for_images(markdown_path: str) -> list[ImageMarker]:
    """
    Find all <!-- image --> markers in the markdown and extract their context.

    Returns:
        List of ImageMarker objects with position and context information
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.split('\n')

    markers = []
    marker_pattern = re.compile(r'<!--\s*image\s*-->')

    # Also track question numbers for context
    question_pattern = re.compile(r'^-\s+(\d+[a-z]?)\s+')

    current_question_id = None
    marker_index = 0

    for line_num, line in enumerate(lines, start=1):
        # Check if this line contains a question ID
        q_match = question_pattern.match(line)
        if q_match:
            current_question_id = q_match.group(1)

        # Check for image markers
        for match in marker_pattern.finditer(line):
            # Get context: surrounding lines
            context_start = max(0, line_num - 4)
            context_end = min(len(lines), line_num + 3)

            context_before = '\n'.join(lines[context_start:line_num-1])
            context_after = '\n'.join(lines[line_num:context_end])

            # Clean up context (remove other image markers, truncate)
            context_before = marker_pattern.sub('[IMAGE]', context_before)[-200:]
            context_after = marker_pattern.sub('[IMAGE]', context_after)[:200]

            marker = ImageMarker(
                marker_index=marker_index,
                line_number=line_num,
                context_before=context_before.strip(),
                context_after=context_after.strip(),
                nearby_question_id=current_question_id
            )
            markers.append(marker)
            marker_index += 1

    return markers


def load_and_sort_manifest(manifest_path: str) -> list[dict]:
    """
    Load the image manifest and sort by page, then y_position.

    This ensures images are in the same order as they appear in the document.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Sort by page number first, then by Y position (top to bottom)
    manifest.sort(key=lambda x: (x["page"], x["y_position"]))

    return manifest


def create_image_links(
    markers: list[ImageMarker],
    manifest: list[dict]
) -> list[ImageLink]:
    """
    Link image markers to actual image files.

    The Nth marker corresponds to the Nth image in the sorted manifest.
    """
    links = []

    for i, marker in enumerate(markers):
        if i < len(manifest):
            img = manifest[i]
            link = ImageLink(
                image_filename=img["filename"],
                page=img["page"],
                y_position=img["y_position"],
                marker_index=marker.marker_index,
                line_number=marker.line_number,
                context_before=marker.context_before,
                context_after=marker.context_after,
                nearby_question_id=marker.nearby_question_id
            )
            links.append(link)
        else:
            # More markers than images - create placeholder
            print(f"Warning: Marker {i} at line {marker.line_number} has no corresponding image")
            link = ImageLink(
                image_filename="[NO_IMAGE]",
                page=-1,
                y_position=-1,
                marker_index=marker.marker_index,
                line_number=marker.line_number,
                context_before=marker.context_before,
                context_after=marker.context_after,
                nearby_question_id=marker.nearby_question_id
            )
            links.append(link)

    # Check for extra images
    if len(manifest) > len(markers):
        print(f"Note: {len(manifest) - len(markers)} images have no corresponding marker")

    return links


def create_markdown_with_images(
    markdown_path: str,
    links: list[ImageLink],
    output_path: str
) -> None:
    """
    Create a new markdown file with <!-- image --> replaced by actual image references.

    Uses markdown image syntax: ![description](filename)
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Create a mapping from marker index to filename
    link_map = {link.marker_index: link for link in links}

    # Replace markers in order (need to do this carefully to preserve indices)
    marker_pattern = re.compile(r'<!--\s*image\s*-->')

    marker_index = 0

    def replace_marker(match):
        nonlocal marker_index
        if marker_index in link_map:
            link = link_map[marker_index]
            filename = link.image_filename
            if filename != "[NO_IMAGE]":
                # Create markdown image reference
                replacement = f'![Image {marker_index + 1}](images/{filename})'
            else:
                replacement = '<!-- image: NOT FOUND -->'
        else:
            replacement = '<!-- image: NO LINK -->'
        marker_index += 1
        return replacement

    new_content = marker_pattern.sub(replace_marker, content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Created markdown with image references: {output_path}")


def create_question_image_map(links: list[ImageLink]) -> dict[str, list[str]]:
    """
    Create a mapping from question IDs to their associated images.

    This is useful for the Q&A extraction step.
    """
    question_images: dict[str, list[str]] = {}

    for link in links:
        if link.nearby_question_id and link.image_filename != "[NO_IMAGE]":
            qid = link.nearby_question_id
            if qid not in question_images:
                question_images[qid] = []
            question_images[qid].append(link.image_filename)

    return question_images


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    markdown_path = sys.argv[1]
    manifest_path = sys.argv[2]

    # Validate inputs
    if not os.path.exists(markdown_path):
        print(f"Error: Markdown file not found: {markdown_path}")
        sys.exit(1)

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found: {manifest_path}")
        sys.exit(1)

    print("=" * 60)
    print("Image-to-Markdown Linker")
    print("=" * 60)

    # Step 1: Parse markdown for image markers
    print(f"\n1. Parsing markdown: {markdown_path}")
    markers = parse_markdown_for_images(markdown_path)
    print(f"   Found {len(markers)} image markers")

    # Step 2: Load and sort manifest
    print(f"\n2. Loading manifest: {manifest_path}")
    manifest = load_and_sort_manifest(manifest_path)
    print(f"   Found {len(manifest)} images")

    # Step 3: Create links
    print("\n3. Linking images to markers...")
    links = create_image_links(markers, manifest)
    print(f"   Created {len(links)} links")

    # Step 4: Save outputs
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save image links JSON
    links_file = f"{output_dir}/image_links.json"
    with open(links_file, "w") as f:
        json.dump([asdict(link) for link in links], f, indent=2)
    print(f"\n4. Saved image links to: {links_file}")

    # Save question-image mapping
    question_images = create_question_image_map(links)
    question_images_file = f"{output_dir}/question_images.json"
    with open(question_images_file, "w") as f:
        json.dump(question_images, f, indent=2)
    print(f"   Saved question-image mapping to: {question_images_file}")
    print(f"   Questions with images: {len(question_images)}")

    # Create markdown with image references
    md_basename = Path(markdown_path).stem
    output_md = f"{output_dir}/{md_basename}_with_images.md"
    create_markdown_with_images(markdown_path, links, output_md)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total image markers:    {len(markers)}")
    print(f"Total images in PDF:    {len(manifest)}")
    print(f"Successfully linked:    {len([l for l in links if l.image_filename != '[NO_IMAGE]'])}")
    print(f"Questions with images:  {len(question_images)}")

    # Show first few links as example
    print("\nFirst 5 image links:")
    for link in links[:5]:
        print(f"  [{link.marker_index}] {link.image_filename}")
        print(f"      Page {link.page}, Y={link.y_position:.0f}")
        if link.nearby_question_id:
            print(f"      Near question: {link.nearby_question_id}")
        print()


if __name__ == "__main__":
    main()
