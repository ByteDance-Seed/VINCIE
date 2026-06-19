# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os
import random
from collections import defaultdict
from functools import partial
from itertools import chain
from typing import List, Optional, Tuple, Union
import copy

import mediapy
import torch
import torch.nn as nn
import yaml
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch import Tensor
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.nn.modules.utils import _triple
from torchvision.transforms import Compose, Normalize, ToTensor

from common.config import create_object
from common.decorators import log_on_entry
from common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
    create_training_timesteps_from_config,
)
from common.distributed import (
    barrier_if_distributed,
    get_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    init_torch,
)
from common.distributed.advanced import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_sequence_parallel,
)
from common.distributed.meta_init_utils import (
    meta_non_persistent_buffer_init_fn,
    meta_param_init_fn,
)
from common.entrypoint import Entrypoint
from common.fs import is_hdfs_path, mkdir, move
from common.mfu import Flops
from common.partition import partition_by_groups_balance_by_repeat, partition_by_size
from common.seed import shift_seed
from data.image.transforms.aspect_ratio_crop import AspectRatioCrop
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from models.dit import na
from models.text.encoder import TextEncoder


class SesEdit7BGenerator(Entrypoint):

    def entrypoint(self):
        init_torch(cudnn_benchmark=False)
        self.configure_sequence_parallel()
        self.configure_models()
        self.configure_diffusion()
        self.inference_loop()

    @log_on_entry
    def configure_sequence_parallel(self):
        sp_size = self.config.generation.get("sequence_parallel", 1)
        if sp_size > 1:
            assert self.config.generation.seed is not None, "Sequence parallel requires a fixed seed."
            init_sequence_parallel(sp_size)

    @log_on_entry
    def configure_models(self):
        self.configure_dit_model()
        self.configure_vae_model()
        self.configure_text_model()
        self.configure_dit_fsdp_model()
        self.configure_text_fsdp_model()

    @log_on_entry
    def configure_dit_model(self, device="cpu"):
        checkpoint = self.config.dit.get("checkpoint", None)

        with torch.device("cpu"):
            self.dit = create_object(self.config.dit.model)
            if self.config.prompt_id_embedding.fusion_strategy == "none":
                self.dit.prompt_id_embedding = nn.Identity()
            else:
                self.dit.prompt_id_embedding = create_object(self.config.prompt_id_embedding.model)

        self.dit.set_gradient_checkpointing(self.config.dit.gradient_checkpoint)

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", mmap=True)
            state = Flops.unwrap_state_dict(state)
            loading_info = self.dit.load_state_dict(state, strict=False, assign=True)
            self.logger.info(f"Loading pretrained ckpt from {checkpoint}")
            self.logger.info(f"Loading info: {loading_info}")
            self.dit = meta_non_persistent_buffer_init_fn(self.dit)

        num_params = sum(p.numel() for p in self.dit.parameters() if p.requires_grad)
        self.logger.info(f"DiT trainable parameters: {num_params:,}")

    @log_on_entry
    def configure_vae_model(self):
        dtype = getattr(torch, self.config.vae.dtype)
        self.vae = create_object(self.config.vae.model)
        self.vae.requires_grad_(False).eval()
        self.vae.to(device=get_device(), dtype=dtype)

        if self.config.vae.checkpoint:
            state = torch.load(self.config.vae.checkpoint, map_location=get_device(), mmap=True)
            self.vae.load_state_dict(state)

        if hasattr(self.vae, "set_causal_slicing") and hasattr(self.config.vae, "slicing"):
            self.vae.set_causal_slicing(**self.config.vae.slicing)

        if self.config.vae.compile:
            self.vae.encode = torch.compile(self.vae.encode, dynamic=True)
            self.vae.decode = torch.compile(self.vae.decode, dynamic=True)

    @log_on_entry
    def configure_text_model(self, device="cpu"):
        dtype = getattr(torch, self.config.text.dtype)
        self.text_encoder = TextEncoder(self.config.text)
        self.text_encoder.requires_grad_(False).eval()
        self.text_encoder.to(dtype=dtype)

        if self.config.text.compile:
            self.text_encoder = torch.compile(self.text_encoder)

        if isinstance(self.config.text.dropout, ListConfig):
            assert len(self.config.text.dropout) == len(self.config.text.models)

    @log_on_entry
    def configure_dit_fsdp_model(self):
        from models.dit.blocks import dit_blocks
        from models.dit.nablocks import nadit_blocks

        all_dit_blocks = tuple(
            chain(
                dit_blocks.values(),
                nadit_blocks.values(),
            )
        )

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return recurse or isinstance(module, all_dit_blocks)

        fsdp_strategy = ShardingStrategy[self.config.dit.fsdp.sharding_strategy]

        self.dit = FullyShardedDataParallel(
            self.dit,
            auto_wrap_policy=custom_auto_wrap_policy,
            sharding_strategy=fsdp_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=get_local_rank(),
            use_orig_params=False,
            sync_module_states=True,
            forward_prefetch=True,
            limit_all_gathers=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            param_init_fn=meta_param_init_fn,
        )

    @log_on_entry
    def configure_text_fsdp_model(self):
        if not self.config.text.fsdp.enabled:
            self.text_encoder.to(get_device())
            return

        from transformers.models.t5.modeling_t5 import T5Block, T5Model

        text_blocks = (T5Model, T5Block)

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return recurse or isinstance(module, text_blocks)

        text_encoder_dtype = getattr(torch, self.config.text.dtype)
        fsdp_strategy = ShardingStrategy[self.config.text.fsdp.sharding_strategy]

        self.text_encoder = FullyShardedDataParallel(
            module=self.text_encoder,
            auto_wrap_policy=custom_auto_wrap_policy,
            sharding_strategy=fsdp_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=get_local_rank(),
            use_orig_params=False,
            sync_module_states=False,
            forward_prefetch=True,
            limit_all_gathers=True,
            mixed_precision=MixedPrecision(
                param_dtype=text_encoder_dtype,
                reduce_dtype=text_encoder_dtype,
                buffer_dtype=text_encoder_dtype,
            ),
        )
        self.text_encoder.to(get_device()).requires_grad_(False)

    def configure_diffusion(self):
        self.schedule = create_schedule_from_config(
            config=self.config.diffusion.schedule,
            device=get_device(),
        )
        self.training_timesteps = create_training_timesteps_from_config(
            config=self.config.diffusion.timesteps.training,
            schedule=self.schedule,
            device=get_device(),
        )
        self.sampling_timesteps = create_sampling_timesteps_from_config(
            config=self.config.diffusion.timesteps.sampling,
            schedule=self.schedule,
            device=get_device(),
        )
        self.sampler = create_sampler_from_config(
            config=self.config.diffusion.sampler,
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
        )

    def gather_dicts(self, my_dict):
        world_size = get_world_size() // get_sequence_parallel_world_size()
        rank = get_global_rank() // get_sequence_parallel_world_size()
        device = torch.device(f"cuda:{rank}")

        my_dict_str = json.dumps(my_dict)
        my_dict_bytes = my_dict_str.encode("utf-8")
        my_dict_tensor = torch.tensor(list(my_dict_bytes), dtype=torch.uint8, device=device)

        tensor_size = torch.tensor([my_dict_tensor.size(0)], dtype=torch.int64, device=device)
        tensor_sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_sizes, tensor_size)

        max_size = max(s.item() for s in tensor_sizes)
        gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
        tensor_padded = torch.cat([
            my_dict_tensor,
            torch.zeros(max_size - my_dict_tensor.size(0), dtype=torch.uint8, device=device),
        ])
        torch.distributed.all_gather(gathered_tensors, tensor_padded)

        all_dicts = [
            json.loads(tensor[:size.item()].cpu().numpy().tobytes().decode("utf-8"))
            for tensor, size in zip(gathered_tensors, tensor_sizes)
        ]

        combined_dict = {}
        for d in all_dicts:
            for key, value in d.items():
                if key not in combined_dict:
                    combined_dict[key] = value
                else:
                    combined_dict[key].update(value)

        my_dict.update(combined_dict)
        return my_dict

    @log_on_entry
    def inference_loop(self):
        gen_config = self.config.generation
        world_size = get_world_size() // get_sequence_parallel_world_size()
        rank = get_global_rank() // get_sequence_parallel_world_size()

        # Load prompts: local directory with meta.yaml  OR  inline single-image config.
        if gen_config.positive_prompt.get("path", None):
            prompt_dir = gen_config.positive_prompt.path
            with open(os.path.join(prompt_dir, "meta.yaml"), "r") as f:
                pos_prompts = yaml.safe_load(f)
            for pp in pos_prompts:
                pp["img_paths"] = [
                    os.path.join(prompt_dir, os.path.basename(p))
                    for p in pp["img_paths"]
                ]
        else:
            pos_prompts = [
                dict(
                    index=0,
                    img_paths=list(gen_config.positive_prompt["image_path"]),
                    context=list(gen_config.positive_prompt["prompts"]),
                )
            ]

        mkdir(gen_config.output.dir)

        all_num_turns = [len(pp["context"]) - len(pp["img_paths"]) + 1 for pp in pos_prompts]
        max_num_turns = max(all_num_turns)

        idx2pos_prompts = {pp["index"]: pp for pp in pos_prompts}

        # Initialise turn 0: context = initial images only.
        pos_prompts_turn_i_lst = []
        for pp in pos_prompts:
            new_pp = copy.deepcopy(pp)
            context_end_idx = len(new_pp["img_paths"])
            new_pp["context"] = new_pp["context"][:context_end_idx]
            new_pp["img_paths"] = new_pp["img_paths"][:context_end_idx]
            pos_prompts_turn_i_lst.append(new_pp)

        self.idx2turn2imgpath = defaultdict(dict)

        for i_turn in range(max_num_turns):
            if i_turn > 0:
                self.idx2turn2imgpath = self.gather_dicts(self.idx2turn2imgpath)
                new_lst = []
                for pp in pos_prompts_turn_i_lst:
                    all_context = idx2pos_prompts[pp["index"]]["context"]
                    if i_turn < len(all_context):
                        pp["context"] = all_context[: i_turn + 1]
                        pp["img_paths"].append(self.idx2turn2imgpath[str(pp["index"])][str(i_turn - 1)])
                        new_lst.append(pp)
                pos_prompts_turn_i_lst = new_lst

            barrier_if_distributed()

            pos_prompts_turn_i = ListConfig(pos_prompts_turn_i_lst)
            pos_prompts_turn_i_rank = partition_by_groups_balance_by_repeat(pos_prompts_turn_i, world_size)[rank]
            pos_prompts_turn_i_rank = partition_by_size(pos_prompts_turn_i_rank, gen_config.batch_size)

            if len(pos_prompts_turn_i_rank) == 0:
                pos_prompts_turn_i_rank = [[pos_prompts_turn_i[0]]]

            for batch_pos_prompts in pos_prompts_turn_i_rank:
                for repeat_index in range(gen_config.repeat):
                    prepare_input = partial(self.prepare_input, repeat_idx=repeat_index)
                    texts_pos, conditions, noises, indices, seeds = zip(
                        *map(prepare_input, batch_pos_prompts)
                    )
                    texts_neg = [gen_config.negative_prompt] * len(batch_pos_prompts)

                    if isinstance(texts_pos[0], ListConfig):
                        texts_pos = [list(tp) for tp in texts_pos]

                    gc.collect()
                    torch.cuda.empty_cache()

                    samples = self.inference(
                        noises=noises,
                        conditions=conditions,
                        texts_pos=texts_pos,
                        texts_neg=texts_neg,
                    )

                    for sample, text_pos, index, seed in zip(samples, texts_pos, indices, seeds):
                        if get_sequence_parallel_rank() == 0:
                            self.save_sample(
                                sample=sample,
                                text=text_pos,
                                index=index,
                                repeat_index=repeat_index,
                                seed=seed,
                                turn_index=i_turn,
                            )

    def prepare_input(
        self,
        prompt: Union[str, DictConfig],
        repeat_idx: int,
        device: torch.device = get_device(),
    ):
        if isinstance(prompt, DictConfig):
            if prompt.get("img_paths"):
                task = "v2i"
                text = list(prompt.context)

                prompt_prefix = os.environ.get("CAMERA_PREFIX", "")
                text = [prompt_prefix + t_ for t_ in text]

                if self.config.generation.get("pad_img_placehoder", False):
                    if "context_imgPlaceholder" in prompt:
                        text = list(prompt.context_imgPlaceholder)
                    else:
                        text = [
                            f"Based on <IMG{i_c}>, {str_x.strip().strip('.')}. Output <IMG{i_c+1}>: "
                            for i_c, str_x in enumerate(text)
                        ]

                default_aspect_ratio = self.config.generation.get("aspect_ratio", "keep_ratio")
                aspect_ratio = prompt.get("aspect_ratio", default_aspect_ratio)
                cond = self.load_image_latent_v2i(list(prompt.img_paths), aspect_ratio)
                cond = torch.cat([cond, torch.zeros((1, *cond.shape[1:]), device=device)], dim=0)
            else:
                raise ValueError("prompt.img_paths is required")
        else:
            raise ValueError("prompt must be a DictConfig")

        index = prompt.index
        seed = shift_seed(self.config.generation.seed, index + repeat_idx)
        seed = seed if seed is not None else random.randint(0, 100000)
        generator = torch.Generator(device).manual_seed(seed)
        noise = torch.randn(size=cond.shape, device=device, generator=generator)

        cond = self.get_condition(cond, task)
        return text, cond, noise, index, seed

    def load_image_latent_v2i(self, path: List[str], aspect_ratio: str = "keep_ratio"):
        resolution = self.config.generation.resolution
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        _, ph, pw = _triple(self.config.dit.model.patch_size)
        ratio_cands = self.config.generation.get("adaptive_aspect_ratio_cands", ["9:16", "1:1", "16:9"])

        transform = Compose([
            ToTensor(),
            AspectRatioCrop(aspect_ratio=aspect_ratio, adaptive_aspect_ratio_cands=ratio_cands),
            NaResize(resolution=resolution, mode="area", downsample_only=False),
            DivisibleCrop((vs * ph, vs * pw)),
            Normalize(0.5, 0.5),
        ])

        img_list = []
        ori_size = None
        for p in path:
            with Image.open(p) as img:
                img = img.convert("RGB")
                if ori_size is None:
                    ori_size = img.size
                else:
                    img = img.resize(ori_size)
                img = transform(img)
            img_list.append(img)

        img_tensor = torch.stack(img_list, dim=0)
        img_tensor = rearrange(img_tensor, "t c h w -> c t h w")
        return self.vae_encode([img_tensor])[0]

    def get_condition(self, latent: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if t == 1:
            return cond
        cond[:-1, ..., :-1] = latent[:-1]
        cond[:-1, ..., -1:] = 1.0
        return cond

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor]) -> List[Tensor]:
        latents = []
        if not samples:
            return latents

        device = get_device()
        dtype = getattr(torch, self.config.vae.dtype)
        scale = self.config.vae.scaling_factor
        shift = self.config.vae.get("shifting_factor", 0.0)

        if isinstance(scale, ListConfig):
            scale = torch.tensor(scale, device=device, dtype=dtype)
        if isinstance(shift, ListConfig):
            shift = torch.tensor(shift, device=device, dtype=dtype)

        def pix2latent(sample):
            sample = sample.to(device, dtype)
            if hasattr(self.vae, "preprocess"):
                sample = self.vae.preprocess(sample)
            latent = self.vae.encode(sample).latent
            latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
            latent = rearrange(latent, "b c ... -> b ... c")
            latent = (latent - shift) * scale
            return latent

        for sample in samples:
            sample = sample.unsqueeze(0)
            if sample.ndim == 5:
                frames = [pix2latent(sample[:, :, i]) for i in range(sample.shape[2])]
                latent = torch.cat(frames, dim=1)
            else:
                latent = pix2latent(sample)
            latents.append(latent.squeeze(0))

        return latents

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor]) -> List[Tensor]:
        samples = []
        if not latents:
            return samples

        device = get_device()
        dtype = getattr(torch, self.config.vae.dtype)
        scale = self.config.vae.scaling_factor
        shift = self.config.vae.get("shifting_factor", 0.0)

        if isinstance(scale, ListConfig):
            scale = torch.tensor(scale, device=device, dtype=dtype)
        if isinstance(shift, ListConfig):
            shift = torch.tensor(shift, device=device, dtype=dtype)

        for latent in latents:
            latent = latent.unsqueeze(0).to(device, dtype)
            latent = latent / scale + shift
            latent = rearrange(latent, "b ... c -> b c ...")
            latent = latent.squeeze(2)

            if latent.ndim == 5:
                frames = []
                for i in range(latent.shape[2]):
                    s = self.vae.decode(latent[:, :, i]).sample
                    if s.ndim == 5:
                        s = s.squeeze(2)
                    frames.append(s)
                sample = torch.stack(frames, dim=2)
            else:
                sample = self.vae.decode(latent).sample

            if hasattr(self.vae, "postprocess"):
                sample = self.vae.postprocess(sample)
            samples.append(sample.squeeze(0))

        return samples

    @torch.no_grad()
    def text_encode(self, texts: List[str]):
        text_outputs = self.text_encoder(texts)
        if isinstance(text_outputs.embeddings, list):
            text_embeds = [e[m] for e, m in zip(text_outputs.embeddings, text_outputs.masks)]
            text_shapes = [m.sum(-1).unsqueeze(-1) for m in text_outputs.masks]
        else:
            text_embeds = [text_outputs.embeddings[text_outputs.masks]]
            text_shapes = [text_outputs.masks.sum(-1).unsqueeze(-1)]
        return text_embeds, text_shapes

    def fuse_prompt_id_emb(self, strategy, sentence_tuple, text_id_emb, text_shapes):
        if strategy == "seq_concat":
            sentence_list_with_id_emb = [
                torch.cat([tid_emb.unsqueeze(0), stc_seq], dim=0)
                for tid_emb, stc_seq in zip(text_id_emb[0], sentence_tuple)
            ]
            text_shapes_with_id_emb = text_shapes + 1
        elif strategy == "dim_add":
            sentence_list_with_id_emb = [
                tid_emb.unsqueeze(0) + stc_seq
                for tid_emb, stc_seq in zip(text_id_emb[0], sentence_tuple)
            ]
            text_shapes_with_id_emb = text_shapes
        elif strategy == "none":
            sentence_list_with_id_emb = list(sentence_tuple)
            text_shapes_with_id_emb = text_shapes
        else:
            raise NotImplementedError(f"Unsupported fusion strategy: {strategy}")
        return sentence_list_with_id_emb, text_shapes_with_id_emb

    def get_session_text_embedding(self, texts: Union[List[List[str]], List[str]]):
        is_session = isinstance(texts[0], list)

        if is_session:
            text_list = [t for sample in texts for t in sample]
            text_id_flattend = [i_t for sample in texts for i_t in range(len(sample))]
        else:
            text_list = texts
            text_id_flattend = [0] * len(text_list)

        all_text_embeds, all_text_shapes = self.text_encode(text_list)
        all_out_embeds, all_out_shapes = [], []

        for text_embeds, text_shapes in zip(all_text_embeds, all_text_shapes):
            sentence_tuple = torch.split(text_embeds, text_shapes.flatten().tolist())
            text_id_emb = self.dit(
                is_forward_pie=True,
                prompt_id=torch.LongTensor(text_id_flattend).to(get_device()).unsqueeze(0),
            )

            assert len(sentence_tuple) == text_id_emb.shape[1]

            sentence_list_with_id_emb, text_shapes_with_id_emb = self.fuse_prompt_id_emb(
                self.config.prompt_id_embedding.get("fusion_strategy", "seq_concat"),
                sentence_tuple,
                text_id_emb,
                text_shapes,
            )

            if is_session:
                start_idx = 0
                text_embeds_vid, text_shapes_vid = [], []
                for sample in texts:
                    n = len(sample)
                    text_embeds_vid.append(
                        torch.cat(sentence_list_with_id_emb[start_idx : start_idx + n], dim=0)
                    )
                    text_shapes_vid.append(
                        text_shapes_with_id_emb[start_idx : start_idx + n].sum(dim=0)
                    )
                    start_idx += n
                text_embeds = torch.cat(text_embeds_vid, dim=0)
                text_shapes = torch.stack(text_shapes_vid, dim=0)
            else:
                text_embeds = torch.cat(sentence_list_with_id_emb, dim=0)
                text_shapes = text_shapes_with_id_emb

            all_out_embeds.append(text_embeds)
            all_out_shapes.append(text_shapes)

        return all_out_embeds, all_out_shapes

    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        texts_pos: Union[List[str], List[List[str]]],
        texts_neg: Union[List[str], List[List[str]]],
        cfg_scale: Optional[float] = None,
    ) -> List[Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)
        if batch_size == 0:
            return []

        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale

        text_pos_embeds, text_pos_shapes = self.get_session_text_embedding(texts_pos)
        text_neg_embeds, text_neg_shapes = self.get_session_text_embedding(texts_neg)

        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)

        was_training = self.dit.training
        self.dit.eval()

        latents = self.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(
                    cfg_scale
                    if (args.i + 1) / len(self.sampler.timesteps)
                    <= self.config.diffusion.cfg.get("partial", 1)
                    else 1.0
                ),
                rescale=self.config.diffusion.cfg.rescale,
            ),
        )

        self.dit.train(was_training)
        latents = na.unflatten(latents, latents_shapes)
        return self.vae_decode(latents)

    def save_sample(self, *, sample, index, repeat_index, text, seed, turn_index):
        gen_config = self.config.generation
        sample = sample[:, -1, :, :]  # session editing: only the last frame is the output

        extension = ".mp4" if sample.ndim == 4 else ".png"
        if isinstance(text, list):
            prompt = text[-1].replace("/", "_").replace(":", "_")
        else:
            prompt = str(text).replace("/", "_").replace(":", "_")

        filename = gen_config.output.filename.format(
            index=index,
            repeat_index=repeat_index,
            seed=seed,
            turn_index=turn_index,
        ) + extension

        pathname = os.path.join(gen_config.output.dir, filename)
        self.idx2turn2imgpath[str(index)][str(turn_index)] = pathname
        tempname = os.path.join("/tmp", filename) if is_hdfs_path(gen_config.output.dir) else pathname

        sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).to("cpu", torch.uint8)

        if sample.ndim == 4:
            mediapy.write_video(
                path=tempname,
                images=rearrange(sample, "c t h w -> t h w c").numpy(),
                fps=gen_config.fps,
            )
        elif sample.ndim == 3:
            mediapy.write_image(
                path=tempname,
                image=rearrange(sample, "c h w -> h w c").numpy(),
            )
        else:
            raise ValueError(f"Unexpected sample ndim: {sample.ndim}")

        move(tempname, pathname)
