#!/usr/bin/env python3
"""
Image-Question Assignment Review GUI

A Streamlit-based GUI for efficiently reviewing and correcting
image-to-question assignments.

Usage:
    streamlit run scripts/review_gui.py

Features:
    - Visual side-by-side view of images and questions
    - Easy navigation with keyboard shortcuts
    - Reassign images to different questions
    - Mark images as "no question" or "skip"
    - Auto-save progress
    - Export corrected mappings
"""

import streamlit as st
import json
import os
from pathlib import Path
from PIL import Image
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Image-Question Review",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths (relative to where streamlit is run from)
IMAGES_DIR = "images"
OUTPUT_DIR = "output"
MANIFEST_FILE = f"{IMAGES_DIR}/manifest.json"
IMAGE_QUESTION_MAP_FILE = f"{OUTPUT_DIR}/image_question_map.json"
QUESTION_IMAGE_MAP_FILE = f"{OUTPUT_DIR}/question_image_map.json"
CORRECTIONS_FILE = f"{OUTPUT_DIR}/corrections.json"
QUESTIONS_FILE = f"{OUTPUT_DIR}/parsed_questions.json"


@st.cache_data
def load_manifest():
    """Load image manifest."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)
        # Sort by page and y_position
        manifest.sort(key=lambda x: (x["page"], x["y_position"]))
        return manifest
    return []


@st.cache_data
def load_image_question_map():
    """Load image → question mapping."""
    if os.path.exists(IMAGE_QUESTION_MAP_FILE):
        with open(IMAGE_QUESTION_MAP_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_question_image_map():
    """Load question → images mapping."""
    if os.path.exists(QUESTION_IMAGE_MAP_FILE):
        with open(QUESTION_IMAGE_MAP_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_parsed_questions():
    """Load parsed questions with text and choices."""
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE) as f:
            return json.load(f)
    return {}


def load_corrections():
    """Load saved corrections (not cached - needs to be fresh)."""
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE) as f:
            return json.load(f)
    return {"corrections": {}, "reviewed": [], "metadata": {}}


def save_corrections(corrections_data):
    """Save corrections to file."""
    corrections_data["metadata"]["last_saved"] = datetime.now().isoformat()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(corrections_data, f, indent=2)


def get_all_question_ids(question_map):
    """Get sorted list of all question IDs."""
    ids = list(question_map.keys())

    # Sort by numeric value, handling letters (1, 2, 2a, 2b, 3, 10, 11, etc.)
    def sort_key(x):
        import re
        match = re.match(r'(\d+)([a-z]?)', str(x))
        if match:
            num = int(match.group(1))
            letter = match.group(2) or ''
            return (num, letter)
        return (999999, str(x))

    return sorted(ids, key=sort_key)


def display_image(filename, max_height=500):
    """Display an image with error handling."""
    img_path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            st.image(img, use_container_width=True)
            return True
        except Exception as e:
            st.error(f"Error loading image: {e}")
    else:
        st.warning(f"Image not found: {filename}")
    return False


def main():
    st.title("🔍 Image-Question Assignment Review")

    # Load data
    manifest = load_manifest()
    img_q_map = load_image_question_map()
    q_img_map = load_question_image_map()
    parsed_questions = load_parsed_questions()

    if not manifest:
        st.error("No images found. Run the image extraction first.")
        st.code("python scripts/image_pipeline.py extract <pdf_file> --output images/")
        return

    if not img_q_map:
        st.error("No image-question mapping found. Run the linker first.")
        st.code("python scripts/link_images_v2.py <markdown_file> images/manifest.json")
        return

    # Load corrections (fresh each time)
    if "corrections_data" not in st.session_state:
        st.session_state.corrections_data = load_corrections()

    corrections_data = st.session_state.corrections_data

    # Get all question IDs for dropdown
    all_question_ids = ["(none)", "(skip)"] + get_all_question_ids(q_img_map)

    # Initialize session state for navigation
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    # Sidebar - Navigation and Stats
    with st.sidebar:
        st.header("📊 Progress")

        total_images = len(manifest)
        reviewed_count = len(corrections_data.get("reviewed", []))
        corrected_count = len(corrections_data.get("corrections", {}))

        # Progress bar
        progress = reviewed_count / total_images if total_images > 0 else 0
        st.progress(progress)
        st.write(f"**{reviewed_count}** / {total_images} reviewed ({progress*100:.1f}%)")
        st.write(f"**{corrected_count}** corrections made")

        st.divider()

        # Navigation
        st.header("🧭 Navigation")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
                st.rerun()
        with col2:
            if st.button("➡️ Next", use_container_width=True):
                st.session_state.current_idx = min(total_images - 1, st.session_state.current_idx + 1)
                st.rerun()

        # Jump to specific image
        new_idx = st.number_input(
            "Go to image #",
            min_value=1,
            max_value=total_images,
            value=st.session_state.current_idx + 1,
            step=1
        )
        if new_idx - 1 != st.session_state.current_idx:
            st.session_state.current_idx = new_idx - 1
            st.rerun()

        st.divider()

        # Filter options
        st.header("🔎 Filter")
        filter_option = st.radio(
            "Show images:",
            ["All", "Unreviewed only", "Corrected only", "Low confidence"],
            index=0
        )

        st.divider()

        # Save button
        st.header("💾 Save")
        if st.button("Save Progress", type="primary", use_container_width=True):
            save_corrections(corrections_data)
            st.success("Saved!")

        # Auto-save indicator
        if corrections_data.get("metadata", {}).get("last_saved"):
            st.caption(f"Last saved: {corrections_data['metadata']['last_saved'][:19]}")

        st.divider()

        # Export
        st.header("📤 Export")
        if st.button("Export Final Mapping", use_container_width=True):
            # Create final mapping with corrections applied
            final_mapping = {}
            for img in manifest:
                filename = img["filename"]
                if filename in corrections_data.get("corrections", {}):
                    q_id = corrections_data["corrections"][filename]
                elif filename in img_q_map:
                    q_id = img_q_map[filename].get("question_id")
                else:
                    q_id = None

                if q_id and q_id not in ["(none)", "(skip)"]:
                    final_mapping[filename] = q_id

            # Save final mapping
            final_file = f"{OUTPUT_DIR}/final_image_question_map.json"
            with open(final_file, "w") as f:
                json.dump(final_mapping, f, indent=2)
            st.success(f"Exported to {final_file}")

    # Apply filter
    filtered_indices = list(range(len(manifest)))
    if filter_option == "Unreviewed only":
        filtered_indices = [
            i for i in filtered_indices
            if manifest[i]["filename"] not in corrections_data.get("reviewed", [])
        ]
    elif filter_option == "Corrected only":
        filtered_indices = [
            i for i in filtered_indices
            if manifest[i]["filename"] in corrections_data.get("corrections", {})
        ]
    elif filter_option == "Low confidence":
        filtered_indices = [
            i for i in filtered_indices
            if img_q_map.get(manifest[i]["filename"], {}).get("confidence") in ["low", "medium"]
        ]

    if not filtered_indices:
        st.info("No images match the current filter.")
        return

    # Ensure current index is within filtered range
    if st.session_state.current_idx >= len(manifest):
        st.session_state.current_idx = 0

    # Get current image
    current_img = manifest[st.session_state.current_idx]
    filename = current_img["filename"]

    # Main content area - two columns
    col_img, col_info = st.columns([3, 2])

    with col_img:
        st.subheader(f"Image {st.session_state.current_idx + 1} of {total_images}")
        st.caption(f"📁 {filename}")
        st.caption(f"📄 Page {current_img['page']} | Y: {current_img['y_position']:.0f}")

        # Display image
        display_image(filename)

    with col_info:
        st.subheader("📝 Assignment")

        # Get current assignment
        current_mapping = img_q_map.get(filename, {})
        original_question_id = current_mapping.get("question_id", "(none)")
        confidence = current_mapping.get("confidence", "unknown")

        # Check for correction
        corrected_question_id = corrections_data.get("corrections", {}).get(filename)

        # Status indicator
        if corrected_question_id:
            st.success(f"✏️ Corrected: {original_question_id} → **{corrected_question_id}**")
            current_assignment = corrected_question_id
        else:
            current_assignment = original_question_id
            if confidence == "high":
                st.info(f"🎯 Auto-assigned: **Q{original_question_id}** (high confidence)")
            elif confidence == "medium":
                st.warning(f"⚠️ Auto-assigned: **Q{original_question_id}** (medium confidence)")
            else:
                st.error(f"❓ Auto-assigned: **Q{original_question_id}** (low confidence)")

        st.divider()

        # Question details (if assigned)
        if current_assignment and current_assignment not in ["(none)", "(skip)"]:
            st.write("**Current Question:**")

            # Get question text and choices
            q_data = parsed_questions.get(current_assignment, {})
            if q_data:
                st.markdown(f"**Q{current_assignment}:** {q_data.get('text', 'No text found')}")

                # Show choices
                choices = q_data.get('choices', {})
                if choices:
                    for letter in sorted(choices.keys()):
                        st.markdown(f"- **{letter}.** {choices[letter]}")
            else:
                st.write(f"Question **{current_assignment}** (text not found)")

            # Show how many images are assigned
            q_images = q_img_map.get(current_assignment, [])
            if q_images:
                st.caption(f"📷 {len(q_images)} image(s) assigned to this question")

        st.divider()

        # Reassignment dropdown
        st.write("**Reassign to:**")

        # Find current selection index
        try:
            default_idx = all_question_ids.index(current_assignment) if current_assignment in all_question_ids else 0
        except ValueError:
            default_idx = 0

        new_assignment = st.selectbox(
            "Select question",
            all_question_ids,
            index=default_idx,
            key=f"reassign_{filename}",
            label_visibility="collapsed"
        )

        # Quick action buttons
        st.write("**Quick actions:**")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("✓ Confirm", use_container_width=True, help="Mark as reviewed (no change)"):
                if filename not in corrections_data["reviewed"]:
                    corrections_data["reviewed"].append(filename)
                st.session_state.current_idx = min(len(manifest) - 1, st.session_state.current_idx + 1)
                st.rerun()

        with col_b:
            if st.button("🚫 No Q", use_container_width=True, help="This image has no question"):
                corrections_data["corrections"][filename] = "(none)"
                if filename not in corrections_data["reviewed"]:
                    corrections_data["reviewed"].append(filename)
                st.session_state.current_idx = min(len(manifest) - 1, st.session_state.current_idx + 1)
                st.rerun()

        with col_c:
            if st.button("⏭️ Skip", use_container_width=True, help="Skip for now"):
                st.session_state.current_idx = min(len(manifest) - 1, st.session_state.current_idx + 1)
                st.rerun()

        # Apply reassignment if changed
        if new_assignment != current_assignment:
            if new_assignment == original_question_id:
                # Reverting to original - remove correction
                if filename in corrections_data.get("corrections", {}):
                    del corrections_data["corrections"][filename]
            else:
                corrections_data["corrections"][filename] = new_assignment

            if filename not in corrections_data["reviewed"]:
                corrections_data["reviewed"].append(filename)

            # Auto-save on change
            save_corrections(corrections_data)
            st.rerun()

        st.divider()

        # Keyboard shortcuts info
        with st.expander("⌨️ Keyboard shortcuts"):
            st.markdown("""
            - **←/→** or **A/D**: Previous/Next image
            - **Enter**: Confirm current assignment
            - **N**: Mark as no question
            - **S**: Skip
            """)

    # Bottom navigation bar
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("⏮️ First", use_container_width=True):
            st.session_state.current_idx = 0
            st.rerun()

    with col2:
        if st.button("⬅️ -10", use_container_width=True):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 10)
            st.rerun()

    with col3:
        st.write(f"**{st.session_state.current_idx + 1}** / {total_images}")

    with col4:
        if st.button("➡️ +10", use_container_width=True):
            st.session_state.current_idx = min(len(manifest) - 1, st.session_state.current_idx + 10)
            st.rerun()

    with col5:
        if st.button("⏭️ Last", use_container_width=True):
            st.session_state.current_idx = len(manifest) - 1
            st.rerun()


if __name__ == "__main__":
    main()
