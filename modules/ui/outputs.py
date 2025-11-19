from typing import List, Any
from pathlib import Path
import gradio as gr
import os
import json
import logging

from modules.ui.audio import _ensure_mmaudio_on_path, _ensure_melqc_on_path

_ensure_mmaudio_on_path()
_ensure_melqc_on_path()

from mmaudio.eval_utils import all_model_cfg  # noqa: E402

logger = logging.getLogger(__name__)


def create_outputs_ui(settings):
    outputDirectory_video = settings.get(
        "output_dir", settings.default_settings["output_dir"]
    )
    outputDirectory_metadata = settings.get(
        "metadata_dir", settings.default_settings["metadata_dir"]
    )

    os.makedirs(outputDirectory_video, exist_ok=True)
    os.makedirs(outputDirectory_metadata, exist_ok=True)

    gallery_items_state = gr.State([])
    selected_original_video_path_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            thumbs = gr.Gallery(
                columns=[4], allow_preview=False, object_fit="cover", height="auto"
            )
            refresh_gallery_button = gr.Button("🔄 Update Gallery")
            delete_btn = gr.Button("🗑️ Delete Selected", visible=False)

            gen_audio_acc = gr.Accordion(label="Audio", open=False, visible=False)
            with gen_audio_acc:
                with gr.Accordion(label="MMAudio"):
                    mmaudio_model_dropdown = gr.Dropdown(
                        label="Select mmaudio model",
                        choices=all_model_cfg.keys(),
                        multiselect=False,
                        value="large_44k_v2",
                        info="Select one mmaudio model to use, to generate audio",
                        # scale=1,
                    )
                    mmaudio_overwrite_chkbox = gr.Checkbox(
                        label="Overwrite audio", visible=False
                    )

                    with gr.Blocks():
                        mmaudio_prompt_chkbox = gr.Checkbox(label="Append")
                        mmaudio_prompt_txt = gr.Textbox(label="Positive prompt")

                    with gr.Blocks():
                        mmaudio_prompt_neg_chkbox = gr.Checkbox(label="Append")
                        mmaudio_prompt_neg_txt = gr.Textbox(label="Negative prompt")

                    mmaudio_gen_btn = gr.Button("Generate audio")
                with gr.Accordion(label="MelQC"):
                    melqc_overwrite_chkbox = gr.Checkbox(
                        label="Overwrite audio", visible=False
                    )

                    with gr.Blocks():
                        melqc_prompt_chkbox = gr.Checkbox(label="Append")
                        melqc_prompt_txt = gr.Textbox(label="Positive prompt")

                    with gr.Blocks():
                        melqc_prompt_neg_chkbox = gr.Checkbox(label="Append")
                        melqc_prompt_neg_txt = gr.Textbox(label="Negative prompt")

                    melqc_gen_btn = gr.Button("Generate audio")
                audio_delete_btn = gr.Button("🗑️ Delete")
        with gr.Column(scale=5):
            video_out = gr.Video(sources=[], autoplay=True, loop=True, visible=False)
        with gr.Column(scale=1):
            info_out = gr.Textbox(label="Generation info", visible=False)
            send_to_toolbox_btn = gr.Button("➡️ Send to Post-processing", visible=False)

    return {
        "gallery_items_state": gallery_items_state,
        "selected_original_video_path_state": selected_original_video_path_state,
        "thumbs": thumbs,
        "refresh_gallery_button": refresh_gallery_button,
        "video_out": video_out,
        "info_out": info_out,
        "send_to_toolbox_btn": send_to_toolbox_btn,
        "outputDirectory_video": outputDirectory_video,
        "outputDirectory_metadata": outputDirectory_metadata,
        "delete_btn": delete_btn,
        "selected_prefix_state": gr.State(None),
        "gen_audio_acc": gen_audio_acc,
        "audio_delete_btn": audio_delete_btn,
        "mmaudio_gen_btn": mmaudio_gen_btn,
        "mmaudio_overwrite_chkbox": mmaudio_overwrite_chkbox,
        "mmaudio_model_dropdown": mmaudio_model_dropdown,
        "mmaudio_prompt_chkbox": mmaudio_prompt_chkbox,
        "mmaudio_prompt_neg_chkbox": mmaudio_prompt_neg_chkbox,
        "mmaudio_prompt_txt": mmaudio_prompt_txt,
        "mmaudio_prompt_neg_txt": mmaudio_prompt_neg_txt,
        "melqc_gen_btn": melqc_gen_btn,
        "melqc_overwrite_chkbox": melqc_overwrite_chkbox,
        "melqc_prompt_chkbox": melqc_prompt_chkbox,
        "melqc_prompt_neg_chkbox": melqc_prompt_neg_chkbox,
        "melqc_prompt_txt": melqc_prompt_txt,
        "melqc_prompt_neg_txt": melqc_prompt_neg_txt,
    }


def connect_outputs_events(
    o, tb_target_video_input: gr.Tab, main_tabs_component: gr.Tabs
):
    def get_gallery_items() -> List[tuple[str, str]]:
        if not os.path.exists(o["outputDirectory_metadata"]):
            logging.error(
                f"Error: Metadata directory not found at {o['outputDirectory_metadata']}"
            )
            return []

        files_with_mtime = []

        all_video_files = os.listdir(o["outputDirectory_video"])

        for f in os.listdir(o["outputDirectory_metadata"]):
            if f.endswith(".png"):
                prefix = os.path.splitext(f)[0]

                matching_videos = []
                for video_file in all_video_files:
                    if video_file.startswith(prefix) and video_file.endswith(".mp4"):
                        matching_videos.append(
                            os.path.join(o["outputDirectory_video"], video_file)
                        )

                if matching_videos:
                    latest_video = max(matching_videos, key=os.path.getmtime)
                    files_with_mtime.append(
                        (
                            os.path.join(o["outputDirectory_metadata"], f),
                            prefix,
                            os.path.getmtime(latest_video),
                        )
                    )

        files_with_mtime.sort(key=lambda x: x[2], reverse=True)

        return [(thumb, prefix) for thumb, prefix, _ in files_with_mtime]

    def refresh_gallery() -> tuple[gr.State, gr.Gallery]:
        new_items = get_gallery_items()
        return (
            new_items,  # gallery_items_state
            gr.update(value=[item[0] for item in new_items]),  # thumbs
        )

    def get_latest_video_version(prefix) -> str:
        max_number = -1
        selected_file = None
        for f in os.listdir(o["outputDirectory_video"]):
            if f.startswith(prefix + "_") and f.endswith(".mp4"):
                if "combined" in f:
                    continue
                try:
                    num_str = f.replace(prefix + "_", "").replace(".mp4", "")
                    if num_str.isdigit():
                        num = int(num_str)
                        if num > max_number:
                            max_number = num
                            selected_file = f
                except (ValueError, TypeError):
                    continue
        return selected_file

    def load_video_and_info_from_prefix(
        prefix,
    ) -> tuple[None, str, gr.Button, None] | tuple[str, str, gr.Button, str]:
        video_file = get_latest_video_version(prefix)
        if not video_file:
            video_file = f"{prefix}.mp4"

        video_path = os.path.join(o["outputDirectory_video"], video_file)
        json_path = os.path.join(o["outputDirectory_metadata"], f"{prefix}.json")

        if not os.path.exists(video_path) or not os.path.exists(json_path):
            return None, "Video or JSON not found.", gr.update(visible=False), None

        with open(json_path, "r", encoding="utf-8") as f:
            info_content = json.load(f)

        return (
            video_path,
            json.dumps(info_content, indent=2, ensure_ascii=False),
            gr.update(visible=True),
            video_path,
        )

    def delete_selected_item(
        selected_prefix: str,
    ) -> tuple[gr.Video, gr.State, List, gr.Gallery, gr.Button, gr.Accordion]:
        if not selected_prefix:
            return (
                gr.update(visible=False),  # video out
                None,  # selected_original_video_path_state
                [],  # gallery_items_state
                gr.update(value=[]),  # thumbs
                gr.update(visible=False),  # delete button
                gr.update(visible=False, open=False),  # generate audio accordion
            )

        deleted_files = []

        # Delete all MP4 files with the pattern prefix_*.mp4
        for filename in os.listdir(o["outputDirectory_video"]):
            if filename.startswith(selected_prefix + "_") and filename.endswith(".mp4"):
                file_path = os.path.join(o["outputDirectory_video"], filename)
                try:
                    os.remove(file_path)
                    deleted_files.append(filename)
                except Exception as e:
                    logger.error(f"Error deleting {filename}: {e}")

        # Also delete the base prefix.mp4 if it exists
        base_file = f"{selected_prefix}.mp4"
        base_path = os.path.join(o["outputDirectory_video"], base_file)
        if os.path.exists(base_path):
            try:
                os.remove(base_path)
                deleted_files.append(base_file)
            except Exception as e:
                logger.error(f"Error deleting {base_file}: {e}")

        # Delete metadata files (json and png)
        for ext in [".json", ".png"]:
            meta_file = f"{selected_prefix}{ext}"
            meta_path = os.path.join(o["outputDirectory_metadata"], meta_file)
            if os.path.exists(meta_path):
                try:
                    os.remove(meta_path)
                    deleted_files.append(meta_file)
                except Exception as e:
                    logger.error(f"Error deleting {meta_file}: {e}")

        print(f"Deleted files: {deleted_files}")

        # Refresh the gallery
        new_items = get_gallery_items()
        return (
            gr.update(visible=False),  # video out
            None,  # selected_original_video_path_state
            new_items,  # gallery_items_state
            gr.update(value=[item[0] for item in new_items]),  # thumbs
            gr.update(visible=False),  # delete button
            gr.update(
                visible=False, open=False
            ),  # Make generate audio accordion invisible
        )

    def on_select(
        gallery_items, evt: gr.SelectData
    ) -> tuple[
        gr.Video,
        gr.Textbox,
        gr.Button,
        gr.State,
        gr.Button,
        gr.State,
        gr.Accordion,
        gr.Checkbox,
        gr.Checkbox,
    ]:
        if evt.index is None or not gallery_items or evt.index >= len(gallery_items):
            return (
                gr.update(visible=False),  # video_out
                gr.update(visible=False),  # info_out
                gr.update(visible=False),  # send_to_toolbox_btn
                None,  # selected_original_video_path_state
                gr.update(visible=False),  # delete_btn
                None,  # selected_prefix_state
                gr.update(visible=False),  # gen_audio_acc
                gr.update(visible=False),  # overwrite_mmaudio_chkbox
                gr.update(visible=False),  # melqc_overwrite_chkbox
            )

        prefix = gallery_items[evt.index][1]
        original_video_path, info_string, button_visibility, new_selected_path = (
            load_video_and_info_from_prefix(prefix)
        )

        overwrite_mmaudio_checkbox_visibility = check_audio(prefix)[0]

        return (
            gr.update(
                value=original_video_path, visible=bool(original_video_path)
            ),  # video_out
            gr.update(value=info_string, visible=bool(original_video_path)),  # info_out
            gr.update(visible=bool(original_video_path)),  # send_to_toolbox_btn
            new_selected_path,  # selected_original_video_path_state
            gr.update(visible=bool(original_video_path)),  # delete_btn
            prefix,  # selected_prefix_state
            gr.update(visible=bool(original_video_path)),  # gen_audio_acc
            overwrite_mmaudio_checkbox_visibility,  # overwrite_audio_chkbox
            overwrite_mmaudio_checkbox_visibility,  # melqc_overwrite_chkbox
        )

    def send_to_toolbox(selected_video_path) -> tuple[gr.Tab, gr.Tabs]:
        return gr.update(value=selected_video_path), gr.update(selected="toolbox_tab")

    def get_video_file_and_metadata(prefix) -> tuple[Path, str]:
        original_video_path, metadata, button_visibility, new_selected_path = (
            load_video_and_info_from_prefix(prefix)
        )
        if original_video_path is None:
            logging.error(f"Video file and metadata for prefix {prefix}, not found")
        return Path(original_video_path), metadata

    def get_audio(selected_prefix: str) -> tuple[Path, Any, int]:
        video_file_path = get_video_file_and_metadata(selected_prefix)[0]

        from moviepy import VideoFileClip

        video_clip = VideoFileClip(video_file_path)

        # get the audio from the video
        video_audio = video_clip.audio

        # get the duration of the video
        video_duration = video_clip.duration

        return (
            video_file_path,
            video_audio,
            video_duration,
        )

    def generate_mmaudio(
        selected_prefix: str,
        mmaudio_model_dropdown: gr.Dropdown,
        mmaudio_overwrite_chkbox: gr.Checkbox,
        mmaudio_prompt_txt: gr.Textbox,
        mmaudio_prompt_neg_txt: gr.Textbox,
        mmaudio_prompt_chkbox: gr.Checkbox,
        mmaudio_prompt_neg_chkbox: gr.Checkbox,
    ) -> tuple[gr.State]:
        if not selected_prefix:
            return (
                None,  # selected_original_video_path_state
            )

        video_file, metadata = get_video_file_and_metadata(selected_prefix)
        # load the metadata as a json
        metadata = json.loads(metadata)

        video_file_path, video_audio, duration = get_audio(selected_prefix)

        logger.info("GENAudio: Loading video")

        if video_audio is not None and mmaudio_overwrite_chkbox is False:
            logging.info(
                "GENAudio: Audio exists but overwrite audio is unchecked -> skipping"
            )
            return (
                None,  # selected_original_video_path_state
            )

        def get_prompt(p: str, b: str) -> str:
            if p == "":
                return b
            if b != "":
                return b
            return p

        prompt = metadata.get("prompt", "")

        if mmaudio_prompt_chkbox:
            prompt += mmaudio_prompt_txt
        else:
            prompt = get_prompt(metadata.get("prompt", ""), mmaudio_prompt_txt)

        negative_prompt = metadata.get("negative_prompt", "")

        if mmaudio_prompt_neg_chkbox:
            negative_prompt += mmaudio_prompt_neg_txt
        else:
            negative_prompt = get_prompt(
                metadata.get("negative_prompt", ""), mmaudio_prompt_neg_txt
            )

        # Append missing words from the DEFAULT_AUDIO_NEGATIVE_PROMPT list to the negative prompt
        from modules.MMAudio.app import DEFAULT_AUDIO_NEGATIVE_PROMPT

        missing_words = [
            word
            for word in DEFAULT_AUDIO_NEGATIVE_PROMPT
            if word not in negative_prompt
        ]
        if missing_words:
            if not negative_prompt.endswith(","):
                negative_prompt += ", "
            negative_prompt += ", ".join(missing_words)

        steps = metadata.get("steps", 25)
        cfg_strength = metadata.get("cfg", 1)

        model_config = None

        if mmaudio_model_dropdown:
            model_config = all_model_cfg.get(mmaudio_model_dropdown)

        logger.info(
            f"GENAudio: Generating audio with\n"
            f" prompt:{str(prompt)}\n"
            f" negative prompt:{str(negative_prompt)}\n"
            f" steps:{str(steps)}\n"
            f" cfg:{str(cfg_strength)}"
        )

        from modules.MMAudio.app import get_mmaudio_model, add_audio_to_video

        audio_net_model, features_utils, sequence_config = get_mmaudio_model(
            model_config
        )
        video_with_audio_path = add_audio_to_video(
            video_path=video_file,
            prompt=prompt,
            audio_negative_prompt=negative_prompt,
            audio_steps=steps,
            audio_cfg_strength=cfg_strength,
            duration=duration,
            audio_net=audio_net_model,
            audio_feature_utils=features_utils,
            audio_seq_cfg=sequence_config,
            overwrite_orig_file=True,
        )

        if video_with_audio_path is not None:
            logger.info(
                "Generating audio finished for video: " + video_with_audio_path.name
            )
        else:
            logger.info("Error generating audio for video: " + video_file.name)

        return (
            None,  # selected_original_video_path_state
        )

    def generate_melqc(
        selected_prefix: str,
        melqc_overwrite_chkbox: gr.Checkbox,
        melqc_prompt_txt: gr.Textbox,
        melqc_prompt_neg_txt: gr.Textbox,
        melqc_prompt_chkbox: gr.Checkbox,
        melqc_prompt_neg_chkbox: gr.Checkbox,
    ) -> tuple[gr.State]:
        if not selected_prefix:
            return (
                None,  # selected_original_video_path_state
            )

        video_file, metadata = get_video_file_and_metadata(selected_prefix)
        # load the metadata as a json
        metadata = json.loads(metadata)

        video_file_path, video_audio, duration = get_audio(selected_prefix)

        logger.info("GENAudio: Loading video")

        if video_audio is not None and melqc_overwrite_chkbox is False:
            logging.info(
                "GENAudio: Audio exists but overwrite audio is unchecked -> skipping"
            )
            return (
                None,  # selected_original_video_path_state
            )

        def get_prompt(p: str, b: str) -> str:
            if p == "":
                return b
            if b != "":
                return b
            return p

        prompt = metadata.get("prompt", "")

        if melqc_prompt_chkbox:
            prompt += melqc_prompt_txt
        else:
            prompt = get_prompt(metadata.get("prompt", ""), melqc_prompt_txt)

        negative_prompt = metadata.get("negative_prompt", "")

        if melqc_prompt_neg_chkbox:
            negative_prompt += melqc_prompt_neg_txt
        else:
            negative_prompt = get_prompt(
                metadata.get("negative_prompt", ""), melqc_prompt_neg_txt
            )

        # Append missing words from the DEFAULT_AUDIO_NEGATIVE_PROMPT list to the negative prompt
        from modules.MMAudio.app import DEFAULT_AUDIO_NEGATIVE_PROMPT

        missing_words = [
            word
            for word in DEFAULT_AUDIO_NEGATIVE_PROMPT
            if word not in negative_prompt
        ]
        if missing_words:
            if not negative_prompt.endswith(","):
                negative_prompt += ", "
            negative_prompt += ", ".join(missing_words)

        steps = metadata.get("steps", 25)
        cfg_strength = metadata.get("cfg", 1)

        logger.info(
            f"GENAudio: Generating audio with\n"
            f" prompt:{str(prompt)}\n"
            f" negative prompt:{str(negative_prompt)}\n"
            f" steps:{str(steps)}\n"
            f" cfg:{str(cfg_strength)}"
        )

        try:
            from modules.MelQC.app import add_audio_to_video

            config_path = Path("./modules/MelQC/models/cldm_v15.yaml")

            video_with_audio_path = add_audio_to_video(
                config_path=config_path,
                video_path=video_file,
                prompt=prompt,
                negative_prompt=negative_prompt,
                sample_step=steps,
                cfg_scale=cfg_strength,
                duration=duration,
                logger=logger,
                overwrite_orig_file=True,
            )
            if video_with_audio_path is not None:
                logger.info(
                    "Generating audio finished for video: " + video_with_audio_path.name
                )
            else:
                logger.info("Error generating audio for video: " + video_file.name)
        except ModuleNotFoundError as e:
            msg = str(e)
            logger.error(f"Error generating audio -> {msg}", exc_info=True)

        return (
            None,  # selected_original_video_path_state
        )

    def delete_audio(selected_prefix: str):
        if not selected_prefix:
            return

        video_file_path, video_audio, duration = get_audio(selected_prefix)

        if video_audio is not None:
            from mmaudio.eval_utils import load_video

            video_info = load_video(video_file_path, duration)

            from mmaudio.data.av_utils import reencode_without_audio

            reencode_without_audio(video_info, video_file_path)

            logger.info(f"Deleted audio for video {str(video_file_path)}")
        else:
            logger.info(f"No audio found for video {str(video_file_path)}")

    def check_audio(selected_prefix: str) -> tuple[gr.Checkbox, gr.Checkbox]:
        if not selected_prefix:
            return (
                gr.update(visible=False),  # overwrite_audio_chkbox
                gr.update(visible=False),  # melqc_overwrite_chkbox
            )

        try:
            video_file_path, video_audio, duration = get_audio(selected_prefix)
            if video_audio is not None:
                return (
                    gr.update(visible=True, value=False),  # overwrite_audio_chkbox
                    gr.update(visible=True, value=False),  # melqc_overwrite_chkbox
                )
            else:
                return (
                    gr.update(visible=False),  # overwrite_audio_chkbox
                    gr.update(visible=False),  # melqc_overwrite_chkbox
                )
        except Exception as e:
            logger.error(f"Error checking audio: {e}")
            return (gr.update(visible=False),)

    o["refresh_gallery_button"].click(
        fn=refresh_gallery, inputs=[], outputs=[o["gallery_items_state"], o["thumbs"]]
    )
    o["thumbs"].select(
        fn=on_select,
        inputs=[o["gallery_items_state"]],
        outputs=[
            o["video_out"],
            o["info_out"],
            o["send_to_toolbox_btn"],
            o["selected_original_video_path_state"],
            o["delete_btn"],
            o["selected_prefix_state"],
            o["gen_audio_acc"],
            o["mmaudio_overwrite_chkbox"],
            o["melqc_overwrite_chkbox"],
        ],
    )
    o["send_to_toolbox_btn"].click(
        fn=send_to_toolbox,
        inputs=[o["selected_original_video_path_state"]],
        outputs=[tb_target_video_input, main_tabs_component],
    )
    o["delete_btn"].click(
        fn=delete_selected_item,
        inputs=[o["selected_prefix_state"]],
        outputs=[
            o["video_out"],
            o["selected_original_video_path_state"],
            o["gallery_items_state"],
            o["thumbs"],
            o["delete_btn"],
            o["gen_audio_acc"],
        ],
    )
    o["mmaudio_gen_btn"].click(
        fn=generate_mmaudio,
        inputs=[
            o["selected_prefix_state"],
            o["mmaudio_model_dropdown"],
            o["mmaudio_overwrite_chkbox"],
            o["mmaudio_prompt_txt"],
            o["mmaudio_prompt_neg_txt"],
            o["mmaudio_prompt_chkbox"],
            o["mmaudio_prompt_neg_chkbox"],
        ],
        outputs=[
            o["selected_original_video_path_state"],
        ],
    )
    o["melqc_gen_btn"].click(
        fn=generate_melqc,
        inputs=[
            o["selected_prefix_state"],
            o["melqc_overwrite_chkbox"],
            o["melqc_prompt_txt"],
            o["melqc_prompt_neg_txt"],
            o["melqc_prompt_chkbox"],
            o["melqc_prompt_neg_chkbox"],
        ],
        outputs=[
            o["selected_original_video_path_state"],
        ],
    )
    o["audio_delete_btn"].click(
        fn=delete_audio,
        inputs=[
            o["selected_prefix_state"],
        ],
        outputs=[],
    )
    o["gen_audio_acc"].expand(
        fn=check_audio,
        inputs=[
            o["selected_prefix_state"],
        ],
        outputs=[
            o["mmaudio_overwrite_chkbox"],
            o["melqc_overwrite_chkbox"],
        ],
    )
    return get_gallery_items
