#!/usr/bin/env python3
"""
Image-Question Assignment Review GUI v2 - Chapter Aware

A Streamlit-based GUI for reviewing and correcting image-to-question
assignments with full chapter awareness.

Usage:
    streamlit run scripts/review_gui_v2.py

Features:
    - Chapter-aware question IDs (ch1_2a, ch8_2a)
    - Filter by chapter
    - Shows question text and choices
    - Easy navigation and corrections
    - Auto-save progress
"""

import streamlit as st
import json
import os
from pathlib import Path
from PIL import Image
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Image-Question Review v2",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
IMAGES_DIR = "images"
OUTPUT_DIR = "output"
MANIFEST_FILE = f"{IMAGES_DIR}/manifest.json"
CHAPTER_IMAGE_MAP_FILE = f"{OUTPUT_DIR}/chapter_image_map.json"
QUESTIONS_BY_CHAPTER_FILE = f"{OUTPUT_DIR}/questions_by_chapter.json"
CHAPTERS_FILE = f"{OUTPUT_DIR}/chapters.json"
CORRECTIONS_FILE = f"{OUTPUT_DIR}/corrections_v2.json"


@st.cache_data
def load_manifest():
    """Load image manifest."""
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)
        manifest.sort(key=lambda x: (x["page"], x["y_position"]))
        return manifest
    return []


@st.cache_data
def load_chapter_image_map():
    """Load chapter-aware image → question mapping."""
    if os.path.exists(CHAPTER_IMAGE_MAP_FILE):
        with open(CHAPTER_IMAGE_MAP_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_questions_by_chapter():
    """Load questions organized by chapter."""
    if os.path.exists(QUESTIONS_BY_CHAPTER_FILE):
        with open(QUESTIONS_BY_CHAPTER_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_chapters():
    """Load chapter metadata."""
    if os.path.exists(CHAPTERS_FILE):
        with open(CHAPTERS_FILE) as f:
            return json.load(f)
    return []


def load_corrections():
    """Load saved corrections."""
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


def get_all_question_ids(questions_by_chapter: dict) -> list[str]:
    """Get all full question IDs sorted by chapter then number."""
    all_ids = []
    for ch_key in sorted(questions_by_chapter.keys(), key=lambda x: int(x.replace("ch", ""))):
        for q in questions_by_chapter[ch_key]:
            all_ids.append(q["full_id"])
    return all_ids


def get_question_data(full_id: str, questions_by_chapter: dict) -> dict:
    """Get question data by full ID."""
    for ch_key, questions in questions_by_chapter.items():
        for q in questions:
            if q["full_id"] == full_id:
                return q
    return {}


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
    st.title("📚 Image-Question Review (Chapter Aware)")

    # Load data
    manifest = load_manifest()
    chapter_image_map = load_chapter_image_map()
    questions_by_chapter = load_questions_by_chapter()
    chapters = load_chapters()

    if not manifest:
        st.error("No images found. Run the extraction first.")
        st.code("python scripts/image_pipeline.py extract <pdf> --output images/")
        return

    if not chapter_image_map:
        st.error("No chapter-aware mapping found. Run the parser first.")
        st.code("python scripts/ai_chapter_parser.py <markdown> images/manifest.json")
        return

    # Load corrections
    if "corrections_data" not in st.session_state:
        st.session_state.corrections_data = load_corrections()

    corrections_data = st.session_state.corrections_data

    # Get all question IDs for dropdown
    all_question_ids = ["(none)", "(skip)"] + get_all_question_ids(questions_by_chapter)

    # Initialize navigation
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    # Sidebar
    with st.sidebar:
        st.header("📊 Progress")

        total_images = len(manifest)
        reviewed = len(corrections_data.get("reviewed", []))
        corrected = len(corrections_data.get("corrections", {}))

        st.progress(reviewed / total_images if total_images else 0)
        st.write(f"**{reviewed}** / {total_images} reviewed ({100*reviewed/total_images:.1f}%)")
        st.write(f"**{corrected}** corrections made")

        st.divider()

        # Chapter filter
        st.header("📖 Filter by Chapter")
        chapter_options = ["All chapters"] + [f"Ch{ch['number']}: {ch['name'][:30]}..." for ch in chapters]
        selected_chapter = st.selectbox("Chapter", chapter_options, index=0)

        st.divider()

        # Confidence filter
        st.header("🎯 Filter by Confidence")
        conf_filter = st.radio(
            "Show:",
            ["All", "High confidence", "Medium/Low", "Unreviewed"],
            index=0
        )

        st.divider()

        # Navigation
        st.header("🧭 Navigation")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Prev", use_container_width=True):
                st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
                st.rerun()
        with col2:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.current_idx = min(total_images - 1, st.session_state.current_idx + 1)
                st.rerun()

        jump_to = st.number_input("Go to #", 1, total_images, st.session_state.current_idx + 1)
        if jump_to - 1 != st.session_state.current_idx:
            st.session_state.current_idx = jump_to - 1
            st.rerun()

        st.divider()

        # Save
        if st.button("💾 Save Progress", type="primary", use_container_width=True):
            save_corrections(corrections_data)
            st.success("Saved!")

        if corrections_data.get("metadata", {}).get("last_saved"):
            st.caption(f"Last: {corrections_data['metadata']['last_saved'][:19]}")

        st.divider()

        # Export
        if st.button("📤 Export Final", use_container_width=True):
            final = {}
            for img in manifest:
                fn = img["filename"]
                if fn in corrections_data.get("corrections", {}):
                    q_id = corrections_data["corrections"][fn]
                elif fn in chapter_image_map:
                    q_id = chapter_image_map[fn].get("question_full_id")
                else:
                    q_id = None

                if q_id and q_id not in ["(none)", "(skip)"]:
                    final[fn] = q_id

            with open(f"{OUTPUT_DIR}/final_chapter_image_map.json", "w") as f:
                json.dump(final, f, indent=2)
            st.success("Exported!")

    # Apply filters
    filtered_indices = list(range(len(manifest)))

    # Chapter filter
    if selected_chapter != "All chapters":
        ch_num = int(selected_chapter.split(":")[0].replace("Ch", ""))
        filtered_indices = [
            i for i in filtered_indices
            if chapter_image_map.get(manifest[i]["filename"], {}).get("chapter") == ch_num
        ]

    # Confidence filter
    if conf_filter == "High confidence":
        filtered_indices = [
            i for i in filtered_indices
            if chapter_image_map.get(manifest[i]["filename"], {}).get("confidence") == "high"
        ]
    elif conf_filter == "Medium/Low":
        filtered_indices = [
            i for i in filtered_indices
            if chapter_image_map.get(manifest[i]["filename"], {}).get("confidence") in ["medium", "low", "none"]
        ]
    elif conf_filter == "Unreviewed":
        filtered_indices = [
            i for i in filtered_indices
            if manifest[i]["filename"] not in corrections_data.get("reviewed", [])
        ]

    if not filtered_indices:
        st.info("No images match the current filters.")
        return

    # Clamp index
    if st.session_state.current_idx >= len(manifest):
        st.session_state.current_idx = 0

    # Get current image
    current_img = manifest[st.session_state.current_idx]
    filename = current_img["filename"]
    mapping = chapter_image_map.get(filename, {})

    # Main content
    col_img, col_info = st.columns([3, 2])

    with col_img:
        # Header with chapter info
        ch_num = mapping.get("chapter")
        ch_name = next((c["name"][:40] for c in chapters if c["number"] == ch_num), "Unknown") if ch_num else "Unknown"

        st.subheader(f"Image {st.session_state.current_idx + 1} / {total_images}")
        if ch_num:
            st.info(f"📖 **Chapter {ch_num}**: {ch_name}")

        st.caption(f"📁 {filename} | Page {current_img['page']}")

        display_image(filename)

    with col_info:
        st.subheader("📝 Assignment")

        original_full_id = mapping.get("question_full_id")
        confidence = mapping.get("confidence", "unknown")

        # Check for correction
        corrected_id = corrections_data.get("corrections", {}).get(filename)

        current_id = corrected_id if corrected_id else original_full_id

        # Status
        if corrected_id:
            st.success(f"✏️ Corrected: {original_full_id} → **{corrected_id}**")
        elif confidence == "high":
            st.info(f"🎯 **{current_id}** (high confidence)")
        elif confidence == "medium":
            st.warning(f"⚠️ **{current_id}** (medium confidence)")
        else:
            st.error(f"❓ **{current_id or 'None'}** (low confidence)")

        st.divider()

        # Show question details
        if current_id and current_id not in ["(none)", "(skip)"]:
            q_data = get_question_data(current_id, questions_by_chapter)
            if q_data:
                st.markdown(f"**{current_id}:** {q_data.get('text', 'N/A')}")

                choices = q_data.get("choices", {})
                if choices:
                    st.write("**Choices:**")
                    for letter in sorted(choices.keys()):
                        st.markdown(f"- **{letter}.** {choices[letter]}")

        st.divider()

        # Reassignment
        st.write("**Reassign to:**")

        try:
            default_idx = all_question_ids.index(current_id) if current_id in all_question_ids else 0
        except ValueError:
            default_idx = 0

        new_id = st.selectbox(
            "Select question",
            all_question_ids,
            index=default_idx,
            key=f"reassign_{filename}",
            label_visibility="collapsed"
        )

        # Quick actions
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("✓ OK", use_container_width=True, help="Confirm"):
                if filename not in corrections_data["reviewed"]:
                    corrections_data["reviewed"].append(filename)
                save_corrections(corrections_data)
                st.session_state.current_idx = min(total_images - 1, st.session_state.current_idx + 1)
                st.rerun()

        with col_b:
            if st.button("🚫 None", use_container_width=True, help="No question"):
                corrections_data["corrections"][filename] = "(none)"
                if filename not in corrections_data["reviewed"]:
                    corrections_data["reviewed"].append(filename)
                save_corrections(corrections_data)
                st.session_state.current_idx = min(total_images - 1, st.session_state.current_idx + 1)
                st.rerun()

        with col_c:
            if st.button("⏭️ Skip", use_container_width=True):
                st.session_state.current_idx = min(total_images - 1, st.session_state.current_idx + 1)
                st.rerun()

        # Handle reassignment
        if new_id != current_id:
            if new_id == original_full_id:
                if filename in corrections_data.get("corrections", {}):
                    del corrections_data["corrections"][filename]
            else:
                corrections_data["corrections"][filename] = new_id

            if filename not in corrections_data["reviewed"]:
                corrections_data["reviewed"].append(filename)

            save_corrections(corrections_data)
            st.rerun()

    # Bottom navigation
    st.divider()
    cols = st.columns(5)
    with cols[0]:
        if st.button("⏮️ First", use_container_width=True):
            st.session_state.current_idx = 0
            st.rerun()
    with cols[1]:
        if st.button("⬅️ -10", use_container_width=True):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 10)
            st.rerun()
    with cols[2]:
        st.markdown(f"**{st.session_state.current_idx + 1}** / {total_images}")
    with cols[3]:
        if st.button("+10 ➡️", use_container_width=True):
            st.session_state.current_idx = min(total_images - 1, st.session_state.current_idx + 10)
            st.rerun()
    with cols[4]:
        if st.button("Last ⏭️", use_container_width=True):
            st.session_state.current_idx = total_images - 1
            st.rerun()


if __name__ == "__main__":
    main()
