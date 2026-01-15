"""
UI Package

Modular UI components for the Textbook Q&A Extractor.

This package provides helper functions and sidebar rendering.
Step rendering functions remain in ui_components.py to avoid circular imports.

Usage:
    from ui import render_sidebar
    from ui.helpers import play_completion_sound, clear_step_data, ...
"""

# Import helper functions
from ui.helpers import (
    play_completion_sound,
    build_block_aware_image_assignments,
    get_selected_model_id,
    clear_step_data,
    sort_chapter_keys,
    prepare_chapter_for_two_pass,
    question_sort_key,
    get_images_for_question,
    get_answer_images_for_question,
    get_all_question_options,
)

# Import sidebar
from ui.sidebar import (
    get_step_completion_status,
    render_sidebar,
)

__all__ = [
    # Helpers
    'play_completion_sound',
    'build_block_aware_image_assignments',
    'get_selected_model_id',
    'clear_step_data',
    'sort_chapter_keys',
    'prepare_chapter_for_two_pass',
    'question_sort_key',
    'get_images_for_question',
    'get_answer_images_for_question',
    'get_all_question_options',
    # Sidebar
    'get_step_completion_status',
    'render_sidebar',
]
