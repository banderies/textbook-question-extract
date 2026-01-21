"""
UI Sidebar Module

Contains sidebar navigation and status display.
"""

import streamlit as st

from state_management import get_pdf_slug, save_settings
from cost_tracking import get_session_summary, format_cost, format_tokens


def get_step_completion_status() -> dict[str, bool]:
    """Check which steps are completed based on session state data."""
    # QC is complete only when ALL questions have been reviewed
    total_questions = sum(len(qs) for qs in st.session_state.questions.values())
    reviewed_count = len(st.session_state.qc_progress.get("reviewed", {}))
    qc_complete = total_questions > 0 and reviewed_count >= total_questions

    # Generate step is complete when we have generated cards
    generated_cards = st.session_state.generated_questions.get("generated_cards", {})
    generate_complete = bool(generated_cards) and sum(len(c) for c in generated_cards.values()) > 0

    return {
        "source": st.session_state.pages is not None and len(st.session_state.pages) > 0,
        "chapters": st.session_state.chapters is not None and len(st.session_state.chapters) > 0,
        "questions": bool(st.session_state.get("raw_blocks")) and sum(len(bs) for bs in st.session_state.raw_blocks.values()) > 0,
        "format": bool(st.session_state.questions) and sum(len(qs) for qs in st.session_state.questions.values()) > 0,
        "qc": qc_complete,
        "generate": generate_complete,
        "export": False,  # Export is an action, not a state
        "prompts": False,  # Prompts step is always accessible, not completable
        "stats": False,  # Stats step is always accessible, not completable
    }


def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.title("Textbook Q&A Extractor")

    if st.session_state.current_pdf:
        textbook_name = get_pdf_slug(st.session_state.current_pdf).replace('_', ' ')
        st.sidebar.caption(f"Working on: **{textbook_name}**")

    st.sidebar.markdown("---")

    # Get step completion status
    completion_status = get_step_completion_status()

    # CSS for green completed step buttons
    st.sidebar.markdown("""
        <style>
        /* Green styling for completed step buttons (primary type) in sidebar */
        section[data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #28a745 !important;
            color: white !important;
            border-color: #28a745 !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"]:hover,
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background-color: #218838 !important;
            border-color: #1e7e34 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    steps = [
        ("source", "1. Select Source"),
        ("chapters", "2. Extract Chapters"),
        ("questions", "3. Extract Questions"),
        ("format", "4. Format Questions"),
        ("qc", "5. QC Questions"),
        ("generate", "6. Generate Questions"),
        ("export", "7. Export"),
        ("prompts", "8. Edit Prompts"),
        ("stats", "9. Stats")
    ]

    for step_id, step_name in steps:
        is_complete = completion_status.get(step_id, False)
        btn_type = "primary" if is_complete else "secondary"

        if st.sidebar.button(step_name, key=f"nav_{step_id}", type=btn_type):
            st.session_state.current_step = step_id
            save_settings()

    st.sidebar.markdown("---")

    # Status summary - Order: Chapters, Images, Raw Questions, Formatted, Context, QC
    st.sidebar.subheader("Status")
    if st.session_state.chapters:
        st.sidebar.success(f"Chapters: {len(st.session_state.chapters)}")
    else:
        st.sidebar.info("Chapters: Not extracted")

    img_count = len(st.session_state.images)
    if img_count > 0:
        assigned = len(st.session_state.image_assignments)
        st.sidebar.success(f"Images: {img_count} ({assigned} assigned)")
    else:
        st.sidebar.info("Images: Not extracted")

    raw_block_count = sum(len(bs) for bs in st.session_state.get("raw_blocks", {}).values())
    if raw_block_count > 0:
        st.sidebar.success(f"Raw Blocks: {raw_block_count}")
    else:
        st.sidebar.info("Raw Blocks: Not extracted")

    q_count = sum(len(qs) for qs in st.session_state.questions.values())
    if q_count > 0:
        st.sidebar.success(f"Formatted: {q_count}")
    else:
        st.sidebar.info("Formatted: Not done")

    reviewed = len(st.session_state.qc_progress.get("reviewed", {}))
    if reviewed > 0:
        st.sidebar.success(f"QC'd: {reviewed}/{q_count}")

    # API Usage summary
    summary = get_session_summary()
    if summary["total_cost"] > 0:
        st.sidebar.markdown("---")
        st.sidebar.subheader("API Usage")
        st.sidebar.metric("Session Cost", format_cost(summary["total_cost"]))
        st.sidebar.caption(f"{format_tokens(summary['total_input_tokens'])} in / {format_tokens(summary['total_output_tokens'])} out")
