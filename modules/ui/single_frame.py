import gradio as gr
import random
import numpy as np
from PIL import Image
import logging

from diffusers_helper.bucket_tools import find_nearest_bucket

PRESET_FILE = ".framepack/generation_presets.json"


def create_single_frame_ui(lora_names, default_prompt, DUMMY_LORA_NAME, settings):
    """
    Create a lightweight UI for Single Frame generation.
    Returns a dict of components similar to create_generate_ui but minimal.
    """
    with gr.Column():

        prompt = gr.Textbox(label="Prompt", value=default_prompt, scale=10)
        n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True)

        with gr.Row():
            steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
            seed = gr.Number(label="Seed", value=2500, precision=0)
            randomize_seed = gr.Checkbox(label="Randomize", value=True)

        with gr.Row("Resolution"):
            resolutionW = gr.Slider(
                label="Width", minimum=128, maximum=768, value=640, step=32
            )
            resolutionH = gr.Slider(
                label="Height", minimum=128, maximum=768, value=640, step=32
            )

        # LoRA selector (keep for parity)
        lora_selector = gr.Dropdown(
            choices=lora_names,
            label="Select LoRAs to Load",
            multiselect=True,
            value=[],
            info="Select one or more LoRAs to use for this job",
            scale=10,
        )
        lora_names_states = gr.State(lora_names)
        # placeholders for lora sliders (hidden until selected)
        lora_sliders = {
            lora: gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.01,
                label=f"{lora} Weight",
                visible=False,
                interactive=True,
            )
            for lora in lora_names
        }

        # Outputs / controls
        result_image = gr.Image(
            label="Finished Frame (PNG)",
            interactive=False,
            visible=True,
            show_label=True,
            elem_classes="contain-image",
            image_mode="RGB",
            type="filepath",
        )
        preview_image = gr.Image(
            label="Preview",
            height=150,
            visible=True,
            type="numpy",
            interactive=False,
            elem_classes="contain-image",
            image_mode="RGB",
        )
        progress_desc = gr.Markdown("", elem_classes="no-generating-animation")
        progress_bar = gr.HTML("", elem_classes="no-generating-animation")

        current_job_id = gr.Textbox(label="Current Job ID", value="", visible=True, interactive=True)
        start_button = gr.Button(value="🚀 Add Single Frame Job", variant="primary")
        end_button = gr.Button(value="❌ Cancel Current Job", interactive=True, visible=False)

        components = {
            "prompt": prompt,
            "n_prompt": n_prompt,
            "steps": steps,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "resolutionW": resolutionW,
            "resolutionH": resolutionH,
            "lora_selector": lora_selector,
            "lora_names_states": lora_names_states,
            "lora_sliders": lora_sliders,
            "result_image": result_image,
            "preview_image": preview_image,
            "progress_desc": progress_desc,
            "progress_bar": progress_bar,
            "current_job_id": current_job_id,
            "start_button": start_button,
            "end_button": end_button,
            "ui_components": {
                "prompt": prompt,
                "n_prompt": n_prompt,
                "steps": steps,
                "seed": seed,
                "randomize_seed": randomize_seed,
                "resolutionW": resolutionW,
                "resolutionH": resolutionH,
            },
        }

    return components


def connect_single_frame_events(sf, s, q, f):
    """
    Connect events for the single frame UI. Uses the same process_fn as the main UI.
    The process_fn signature is large; we build a minimal set of arguments and pass sensible defaults.
    """

    def process_single_frame_with_queue_update(*args):
        # args order matches the components we will wire when clicking start_button
        (
            prompt_text_arg,
            n_prompt_arg,
            steps_arg,
            seed_arg,
            randomize_seed_arg,
            resolutionW_arg,
            resolutionH_arg,
            selected_loras_arg,
            *lora_slider_values_tuple,
        ) = args

        backend_model_type = "Single Frame"
        is_ui_video_model = False
        input_data = None  # Single Frame does not require an input image
        actual_end_frame_image_for_backend = None
        actual_end_frame_strength_for_backend = 1.0

        # sensible defaults for parameters not exposed in this minimal UI
        total_second_length_arg = 4.0 / 30.0
        latent_window_size_arg = 1
        # steps_arg and seed_arg are provided by the UI; coerce to ints safely
        try:
            steps_arg = int(steps_arg) if steps_arg is not None else 25
        except Exception:
            steps_arg = 25
        try:
            seed_arg = int(seed_arg) if seed_arg is not None else 2500
        except Exception:
            seed_arg = 2500

        # Heuristic safeguard: if steps seems unreasonably large (e.g., >1000),
        # but seed is small, they may have been swapped by the frontend. Swap them.
        if steps_arg > 1000 and seed_arg <= 1000:
            logging.warning(
                f"Detected unusually large steps ({steps_arg}) and small seed ({seed_arg}). Swapping values."
            )
            steps_arg, seed_arg = seed_arg, steps_arg

        cfg_arg = 1.0
        gs_arg = 10.0
        rs_arg = 0.0
        use_teacache_arg = False
        teacache_num_steps_arg = 25
        teacache_rel_l1_thresh_arg = 0.15
        use_magcache_arg = False
        magcache_threshold_arg = 0.1
        magcache_max_consecutive_skips_arg = 2
        magcache_retention_ratio_arg = 0.25
        blend_sections_arg = 0
        latent_type_arg = "Noise"
        clean_up_videos_arg = s.get("clean_up_videos", True) if isinstance(s, dict) else True
        resolutionW_arg = resolutionW_arg or 640
        resolutionH_arg = resolutionH_arg or 640
        input_image_path = None
        combine_with_source_arg = False
        num_cleaned_frames_arg = 5
        lora_names_states_arg = sf.get("lora_names_states") if sf.get("lora_names_states") else []

        # Randomize seed if requested
        if randomize_seed_arg:
            seed_arg = random.randint(0, 21474)

        # Build positional args in the exact order expected by the main process() function
        call_args = [
            backend_model_type,
            input_data,
            actual_end_frame_image_for_backend,
            actual_end_frame_strength_for_backend,
            prompt_text_arg,
            n_prompt_arg,
            seed_arg,
            total_second_length_arg,
            latent_window_size_arg,
            steps_arg,
            cfg_arg,
            gs_arg,
            rs_arg,
            use_teacache_arg,
            teacache_num_steps_arg,
            teacache_rel_l1_thresh_arg,
            use_magcache_arg,
            magcache_threshold_arg,
            magcache_max_consecutive_skips_arg,
            magcache_retention_ratio_arg,
            blend_sections_arg,
            latent_type_arg,
            clean_up_videos_arg,
            selected_loras_arg,
            resolutionW_arg,
            resolutionH_arg,
            input_image_path,
            combine_with_source_arg,
            num_cleaned_frames_arg,
        ]

        # Append any LoRA slider values as positional lora_args (matches studio.process *lora_args)
        call_args.extend(list(lora_slider_values_tuple))

        # Log the call arguments for debugging
        logging.debug(f"Single Frame call args (first 20): {call_args[:20]}")

        # Call the shared process function with positional args to avoid accidental ordering issues
        result = f["process_fn"](*call_args)

        return result

    # Hook up the start button
    ips = [
        sf["prompt"],
        sf["n_prompt"],
        sf["steps"],
        sf["seed"],
        sf["randomize_seed"],
        sf["resolutionW"],
        sf["resolutionH"],
        sf["lora_selector"],
    ]

    def update_button_before_processing(*_args):
        queue_status_data, queue_stats_text = f["update_stats"]()
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value="⏳ Adding...", interactive=False),
            gr.update(),
        )

    sf["start_button"].click(
        fn=update_button_before_processing,
        inputs=ips,
        outputs=[
            sf["result_image"],
            sf["current_job_id"],
            sf["preview_image"],
            sf["progress_desc"],
            sf["progress_bar"],
            sf["start_button"],
            sf["end_button"],
        ],
    ).then(
        fn=process_single_frame_with_queue_update,
        inputs=ips,
        outputs=[
            sf["result_image"],
            sf["current_job_id"],
            sf["preview_image"],
            sf["progress_desc"],
            sf["progress_bar"],
            sf["start_button"],
            sf["end_button"],
        ],
    ).then(
        fn=lambda: f["update_start_button_state"]("Single Frame", None),
        inputs=None,
        outputs=[sf["start_button"], sf["end_button"]],
    )

    # Wire end button to cancel
    sf["end_button"].click(
        fn=lambda: (f["end_process_with_update"]()),
        outputs=[q["queue_status"], q["queue_stats_display"], sf["end_button"], sf["current_job_id"]],
    )

    return
