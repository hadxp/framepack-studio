import logging
import yaml
import cv2, os
import einops
import numpy as np
import torch
import psutil
import soundfile as sf
from einops import rearrange, repeat
from torch.nn import functional as F
from ldm.models.diffusion.plms import PLMSSampler
from pytorch_lightning import seed_everything
from cldm.model import create_model_args, load_state_dict
from moviepy import VideoFileClip, AudioFileClip
from foleycrafter.pipelines.auffusion_pipeline import Generator

from SyncFormer.utils import instantiate_from_config
from CodePredictor.generate_code import VideoCodeGenerator

from transformers import CLIPImageProcessor, CLIPModel


def denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200,
    min_value: float = 1e-5,
    power: float = 1,
):
    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))

    max_value = np.log(max_value)
    min_value = np.log(min_value)
    data = torch.flip(data, [1])
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]
    data = torch.pow(data, 1 / power)
    spectrogram = data * (max_value - min_value) + min_value

    return spectrogram

from pathlib import Path
def process_video(mp4_path):
    video_clip = VideoFileClip(mp4_path)
    frames = np.array([frame for frame in video_clip.iter_frames()])
    fps = video_clip.fps
    video_duration = video_clip.duration
    if fps != 25:
        resampled_indices = np.linspace(
            0, len(frames) - 1, int(video_duration * 25), dtype=int
        )
        frames = frames[resampled_indices]

    frames_pad = []
    for i in range(len(frames)):
        frame = frames[i]
        h, w, _ = frame.shape
        padl = max(h, w)
        frame_pad = np.pad(
            frame,
            (
                ((padl - h) // 2, (padl - h) - (padl - h) // 2),
                ((padl - w) // 2, (padl - w) - (padl - w) // 2),
                (0, 0),
            ),
            "constant",
        )
        assert frame_pad.shape[0] == padl
        assert frame_pad.shape[1] == padl
        frames_pad.append(frame_pad)

    return frames_pad


def get_video_features(frames, model_sync):
    num_frames = len(frames)
    frames = torch.cat(
        [
            frames[0].unsqueeze(0).repeat(8, 1, 1, 1),
            frames,
            frames[-1].unsqueeze(0).repeat(7, 1, 1, 1),
        ],
        axis=0,
    )
    segment_indices = (
        torch.arange(16).unsqueeze(0) + torch.arange(num_frames).unsqueeze(1)
    ).cuda()
    segments = frames[segment_indices]
    segments = segments.permute(0, 2, 1, 3, 4).unsqueeze(0)
    rst, _ = model_sync(segments, for_loop=True)
    return rst[0]


class MelQCD:
    def __init__(
        self,
        logger: logging.Logger,
        checkpoint_path: Path = Path("./pretrain/MelQCD.ckpt"),
        config_path: Path = Path("./models/cldm_v15.yaml"),
        num_vstar: int = 32,
        num_win: int = 8,  # number of frequency windows
        num_quan: int = 3,  # number of qmel values
    ):
        self.logger = logger
        self.basedir = os.getcwd()
        self.model_dir = os.path.join(self.basedir, checkpoint_path)
        self.loaded = False
        self.load_model(checkpoint_path, config_path, num_vstar, num_win, num_quan)

    def load_model(
        self,
        base_local_download_dir: Path = Path("./melqc"),
        config_path: Path = Path("./models/cldm_v15.yaml"),
        num_vstar: int = 32,
        num_win: int = 8,  # number of frequency windows
        num_quan: int = 3,  # number of qmel values
    ):
        self.logger.info("Start Load Models...")

        checkpoint_path = base_local_download_dir / "MelQCD.ckpt"
        syncformer_checkpoint_path = base_local_download_dir / "SyncFormer.pt"
        codepredictor_checkpoint_path = base_local_download_dir / "CodePredictor.ckpt"

        if not checkpoint_path.exists() or not syncformer_checkpoint_path.exists() or not codepredictor_checkpoint_path.exists():
            return "Failed"

        self.base_local_download_dir = base_local_download_dir

        # load clip and syncformer model
        self.model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor_clip = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model_clip.cuda().eval()

        self.processor_sync = CLIPImageProcessor(image_mean=0.5, image_std=0.5)
        with open("./modules/MelQC/SyncFormer/vfeat.yaml") as stream:
            cfg = yaml.safe_load(stream)
        self.model_sync = instantiate_from_config(cfg)
        ckpt = torch.load(
            str(syncformer_checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        new_dict = {
            k.replace("module.v_encoder.", ""): v
            for k, v in ckpt["state_dict"].items()
            if "v_encoder" in k
        }
        self.model_sync.load_state_dict(new_dict, strict=True)
        self.model_sync.cuda().eval()

        # load code predictor model
        self.vcgen = VideoCodeGenerator(
            str(codepredictor_checkpoint_path), num_win, num_quan
        )

        self.melqcd_model = create_model_args(config_path, num_vstar, 768)
        self.melqcd_model.load_state_dict(
            load_state_dict(checkpoint_path, location="cuda"), strict=True
        )
        self.melqcd_model = self.melqcd_model.cuda()

        self.vocoder = Generator.from_pretrained(str(base_local_download_dir)).cuda()

        self.loaded = True
        self.logger.info("Load Finish!")

        return "Load"

    def foley(
        self,
        video_path: Path,
        prompt: str,
        negative_prompt: str,
        sample_step: int,
        cfg_scale: int,
        seed: str,
        duration: int,
        overwrite_orig_file: bool,
        sampling_rate: int = 44100,
    ) -> Path:
        # prepare features
        frames_pad = process_video(video_path)

        # clip
        frames_p = self.processor_clip(np.array(frames_pad), return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            rst_clip = self.model_clip.get_image_features(**frames_p).cpu().numpy()
        print("extract clip feature down")

        # syncformer
        for i in range(256 - len(frames_pad)):
            frames_pad.append(np.zeros_like(frames_pad[-1]))

        frames_p = self.processor_sync(np.array(frames_pad), return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            rst_sync = (
                get_video_features(frames_p["pixel_values"], self.model_sync)
                .cpu()
                .numpy()
            )

        # code predictor
        effect_length = min(rst_sync.shape[0], 250)
        effect_length = torch.tensor(effect_length)[None].cuda()

        rst_sync = torch.from_numpy(rst_sync)
        rst_sync = F.pad(
            rst_sync[:effect_length], (0, 0, 0, 250 - effect_length), value=0
        )
        rst_sync = rst_sync[None].cuda()
        prediction, pred_mean, pred_std, tseq = self.vcgen.generate(
            rst_sync, effect_length
        )

        emb_pred = tseq[0]
        mean_pred = pred_mean[0]
        std_pred = pred_std[0]

        # melqcd
        ## create controlmap
        H, W, C = 256, 1024, 3
        upper_index = torch.linspace(0, 999, 1024).long()
        Mean = mean_pred[upper_index].unsqueeze(0).unsqueeze(-1).repeat(8, 1, 3)
        Std = std_pred[upper_index].unsqueeze(0).unsqueeze(-1).repeat(8, 1, 3)
        Spec = F.pad(
            emb_pred, (0, 0, 0, 1024 - emb_pred.size(0)), "constant", 0
        )  # 1024, 8
        Spec = Spec.unsqueeze(-1).repeat(1, 1, 3).permute(1, 0, 2)  # 8, 1024, 3
        f_control = Mean + Std * Spec
        whole_map = torch.zeros(H, W, C)
        for env in range(8):
            whole_map[env * H // 8 : (env + 1) * H // 8] = (
                f_control[env].unsqueeze(0).repeat(H // 8, 1, 1)
            )
        Control_map = whole_map
        Control_map = torch.stack([Control_map for _ in range(1)], dim=0).float()
        Control_map = einops.rearrange(Control_map, "b h w c -> b c h w").clone().cuda()
        Control_map = Control_map * 2 - 1

        ## create Textual inversion feature
        rst_clip = torch.from_numpy(rst_clip)
        global_video_feat_mean = torch.mean(rst_clip, dim=0)[None]
        global_video_feat_mean = global_video_feat_mean.unsqueeze(0).cuda()

        ## start melqcd inference
        ddim_sampler = PLMSSampler(self.melqcd_model)
        with torch.no_grad():
            seed_everything(seed)
            cond = {
                "c_scale_mean": [global_video_feat_mean],
                "c_concat": [Control_map],
                "c_crossattn": [[prompt]],
            }
            un_cond = {
                "c_scale_mean": [global_video_feat_mean],
                "c_concat": [Control_map],
                "c_crossattn": [
                    self.melqcd_model.get_learned_conditioning(
                        [negative_prompt] * 1
                    )
                ],
            }
            shape = (4, 256 // 8, 1024 // 8)

            strength = 1.0
            self.melqcd_model.control_scales = (
                [strength] * 13
            )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(
                sample_step,
                1,
                shape,
                cond,
                verbose=False,
                eta=0,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=un_cond,
            )

            x_samples = self.melqcd_model.decode_first_stage(samples)[0]

        results = ((x_samples + 1) / 2).clip(0, 1)
        res_mel = denormalize_spectrogram(results)
        audio = self.vocoder.inference(res_mel, lengths=sampling_rate)[0]
        audio = audio[: int(10 * sampling_rate)]

        video_filename = video_path.name

        if overwrite_orig_file:
            # Create video with audio, in the same location as the original file
            video_with_audio_filename = video_filename
        else:
            # Create video with audio, in the same folder with the original video
            video_with_audio_filename = video_filename + "_audio.mp4"

        video_dir = video_path.parent
        video_with_audio_path = Path(video_dir) / video_with_audio_filename

        self.logger.info("Audio generated, adding to video")

        video_clip = VideoFileClip(str(video_path))
        temp_wavfile_path = self.base_local_download_dir / "temp" / 'temp.wav'
        sf.write(temp_wavfile_path, audio, sampling_rate)
        audio_clip = AudioFileClip(str(temp_wavfile_path))
        audio_clip = audio_clip.subclip(0, 10)
        video_with_new_audio = video_clip.set_audio(audio_clip)
        video_with_new_audio.write_videofile(str(video_with_audio_path), codec='libx264', audio_codec='aac')

        return video_with_audio_path

def add_audio_to_video(
        video_path: Path,
        prompt: str,
        negative_prompt: str,
        sample_step: int,
        cfg_scale: int,
        duration: int,
        overwrite_orig_file: bool,
        logger: logging.Logger,
        seed: str = 42,
        sampling_rate: int = 16000,
        base_local_download_dir: Path = Path("./melqc"),
        config_path: Path = Path("./models/cldm_v15.yaml"),
        num_vstar: int = 32,
        num_win: int = 8,  # number of frequency windows
        num_quan: int = 3,  # number of qmel values
) -> Path:
    # download models
    from modelscope.hub.snapshot_download import snapshot_download
    # Set ModelScope domain
    os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
    # download the models
    snapshot_download(
        model_id='iic/MelQCD',
        local_dir=base_local_download_dir,
    )
    logger.info("Models downloaded")
    # Instantiate and load model
    melqcd = MelQCD(logger, base_local_download_dir, config_path, num_vstar, num_win, num_quan)
    logger.info("Generating audio")
    # generate and add audio to video
    return melqcd.foley(video_path, prompt, negative_prompt, sample_step, cfg_scale, seed, duration, overwrite_orig_file, sampling_rate)
