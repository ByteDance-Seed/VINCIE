# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0

import os
import random
from functools import partial
from typing import Union
import mediapy
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from PIL import Image
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import copy
import torch.distributed as dist
import json
import traceback
from itertools import chain
import gc

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torchvision.transforms import Compose, Normalize, ToTensor
from torch import Tensor

from common.decorators import log_on_entry
from common.distributed import get_device, get_global_rank, get_world_size, init_torch, barrier_if_distributed
from common.distributed.advanced import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_sequence_parallel,
)
from common.fs import is_hdfs_path, mkdir, move
from common.partition import partition_by_size, partition_by_groups_balance_by_repeat
from common.seed import shift_seed
from common.decorators import log_on_entry, log_runtime
from common.config import create_object
from common.mfu import Flops
from common.distributed.meta_init_utils import (
    meta_non_persistent_buffer_init_fn,
    meta_param_init_fn,
)
from common.distributed.advanced import (
    get_sequence_parallel_world_size,
    init_model_shard_group,
    init_sequence_parallel,
)
from common.distributed import (
    get_device,
    get_global_rank,
    get_local_rank,
    get_world_size,
    init_torch,
)
from common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
    create_training_timesteps_from_config,
)
from common.entrypoint import Entrypoint
from torch.distributed.device_mesh import init_device_mesh

from data.image.transforms.aspect_ratio_crop import AspectRatioCrop
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize

from models.text.encoder import TextEncoder
from models.dit import na


class VINCIEGenerator(Entrypoint):
    def entrypoint(self):
        init_torch(cudnn_benchmark=False)
        self.configure_sequence_parallel()
        self.configure_persistence()
        self.configure_models()
        self.configure_diffusion()
        self.inference_loop()

    @log_on_entry
    def configure_sequence_parallel(self):
        gen_config = self.config.generation
        sp_size = gen_config.sequence_parallel
        if sp_size > 1:
            assert gen_config.seed is not None, "Sequence parallel must use seed."
            init_sequence_parallel(sp_size)

    @log_on_entry
    def configure_persistence(self):
        # No need for persistence for generation.
        self.resume = None

    @log_on_entry
    def configure_models(self):
        # Initialize models.
        # FSDP modules must be first loaded onto cpu.
        self.configure_dit_model()
        self.configure_vae_model()
        self.configure_text_model()

        # Initialize fsdp.
        self.configure_dit_fsdp_model()
        self.configure_text_fsdp_model()

    @log_on_entry
    @log_runtime
    def configure_dit_model(self, device="cpu"):
        # Load dit checkpoint.
        checkpoint = self.config.dit.get("checkpoint", None)
        if self.resume:
            checkpoint = self.resume.models["dit"].states.path

        # For fast init & resume,
        #   when training from scratch, rank0 init DiT on cpu, then sync to other ranks with FSDP.
        #   otherwise, all ranks init DiT on meta device, then load_state_dict with assign=True.
        if self.config.dit.get("init_with_meta_device", False):
            init_device = "cpu" if get_global_rank() == 0 and checkpoint is None else "meta"
        else:
            init_device = "cpu"

        # Create dit model.
        with torch.device(init_device):
            self.dit = create_object(self.config.dit.model)
            # prompt id embedding for session editing
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

        if device in [get_device(), "cuda"]:
            self.dit.to(get_device())

        # Print model size.
        num_params = sum(p.numel() for p in self.dit.parameters() if p.requires_grad)
        self.logger.info(f"DiT trainable parameters: {num_params:,}")

    @log_on_entry
    @log_runtime
    def configure_vae_model(self):
        # Create vae model.
        dtype = getattr(torch, self.config.vae.dtype)
        self.vae = create_object(self.config.vae.model)
        self.vae.requires_grad_(False).eval()
        self.vae.to(device=get_device(), dtype=dtype)

        # Load vae checkpoint.
        state = torch.load(
            self.config.vae.checkpoint, map_location=get_device(), mmap=True
        )
        self.vae.load_state_dict(state)

        # Set causal slicing.
        if hasattr(self.vae, "set_causal_slicing") and hasattr(self.config.vae, "slicing"):
            self.vae.set_causal_slicing(**self.config.vae.slicing)

        # Compile vae if needed.
        if self.config.vae.compile:
            self.vae.encode = torch.compile(self.vae.encode, dynamic=True)
            self.vae.decode = torch.compile(self.vae.decode, dynamic=True)

    @log_on_entry
    def configure_text_model(self, device="cpu"):
        # Create text model.
        dtype = getattr(torch, self.config.text.dtype)
        self.text_encoder = TextEncoder(self.config.text)
        self.text_encoder.requires_grad_(False).eval()
        self.text_encoder.to(dtype=dtype)

        # Compile text model if needed.
        if self.config.text.compile:
            self.text_encoder = torch.compile(self.text_encoder)

        if device in [get_device(), "cuda"]:
            self.text_encoder.to(get_device())

        if isinstance(self.config.text.dropout, ListConfig):
            assert len(self.config.text.dropout) == len(self.config.text.models)

    @log_on_entry
    @log_runtime
    def configure_dit_fsdp_model(self):
        from models.dit.nablocks import nadit_blocks as nadit_blocks_v2

        dit_blocks = tuple(
            chain(
                nadit_blocks_v2.values(),
            )
        )

        # Assert that dit has wrappable blocks.
        assert any(isinstance(m, dit_blocks) for m in self.dit.modules())

        # Define wrap policy on all dit blocks.
        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return recurse or isinstance(module, dit_blocks)

        # Configure FSDP settings.
        device_mesh, fsdp_strategy = self.get_fsdp_sharding_config(
            self.config.dit.fsdp.sharding_strategy,
            self.config.dit.fsdp.get("device_mesh", None),
        )

        # Init model_shard_cpu_group for saving checkpoint with sharded state_dict.
        init_model_shard_group(sharding_strategy=fsdp_strategy, device_mesh=device_mesh)

        settings = dict(
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
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            device_mesh=device_mesh,
            param_init_fn=meta_param_init_fn,
        )

        zero2_mapper = {
            ShardingStrategy.HYBRID_SHARD: ShardingStrategy._HYBRID_SHARD_ZERO2,
            ShardingStrategy.FULL_SHARD: ShardingStrategy.SHARD_GRAD_OP,
        }
        if fsdp_strategy in zero2_mapper and self.config.dit.fsdp.get(
            "skip_reshard_last_block", False
        ):
            # Skip reshard for the last block.
            self.dit.blocks[-1]._wrap_overrides = dict(
                sharding_strategy=zero2_mapper[fsdp_strategy],
            )

        # Apply FSDP.
        self.dit = FullyShardedDataParallel(self.dit, **settings)

        if hasattr(self, "ema"):
            self.ema = FullyShardedDataParallel(self.ema, **settings)
            self.ema.requires_grad_(False)

    @log_on_entry
    def configure_text_fsdp_model(self):
        # If FSDP is not enabled, put text_encoder to GPU and return.
        if not self.config.text.fsdp.enabled:
            self.text_encoder.to(get_device())
            return

        from transformers.models.clip.modeling_clip import CLIPEncoderLayer
        from transformers.models.t5.modeling_t5 import T5Block, T5Model

        text_blocks = (T5Model, T5Block, CLIPEncoderLayer)
        text_blocks_names = (
            "QWenBlock",
            "QWenModel",
            "Qwen2DecoderLayer",
            "Qwen2Model",
        )  # QWen cannot be imported. Use str.

        # Assert that text_encoder has wrappable blocks.
        assert any(
            isinstance(m, text_blocks) or m.__class__.__name__ in text_blocks_names
            for m in self.text_encoder.modules()
        )

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return (
                recurse
                or isinstance(module, text_blocks)
                or module.__class__.__name__ in text_blocks_names
            )

        # Apply FSDP.
        text_encoder_dtype = getattr(torch, self.config.text.dtype)
        device_mesh, fsdp_strategy = self.get_fsdp_sharding_config(
            self.config.text.fsdp.sharding_strategy,
            self.config.text.fsdp.get("device_mesh", None),
        )
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
            device_mesh=device_mesh,
        )
        self.text_encoder.to(get_device()).requires_grad_(False)

    def get_fsdp_sharding_config(self, sharding_strategy, device_mesh_config):
        device_mesh = None
        fsdp_strategy = ShardingStrategy[sharding_strategy]
        if (
            fsdp_strategy in [ShardingStrategy._HYBRID_SHARD_ZERO2, ShardingStrategy.HYBRID_SHARD]
            and device_mesh_config is not None
        ):
            device_mesh = init_device_mesh("cuda", tuple(device_mesh_config))
        return device_mesh, fsdp_strategy

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
        # Compute rank and world size.
        world_size = get_world_size() // get_sequence_parallel_world_size()
        rank = get_global_rank() // get_sequence_parallel_world_size()

        # Move to the appropriate device
        device = torch.device(f'cuda:{rank}')

        # Serialize the dict to a JSON string and then to a tensor
        my_dict_str = json.dumps(my_dict)
        my_dict_bytes = my_dict_str.encode('utf-8')
        my_dict_tensor = torch.tensor(list(my_dict_bytes), dtype=torch.uint8, device=device)

        # Gather sizes, ensuring the use of int64 for uniformity
        tensor_size = torch.tensor([my_dict_tensor.size(0)], dtype=torch.int64, device=device)
        tensor_sizes = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(tensor_sizes, tensor_size)

        max_size = max(tensor_size.item() for tensor_size in tensor_sizes)

        # Gather all tensors, ensuring they are on the same device
        gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
        tensor_padded = torch.cat([my_dict_tensor, torch.zeros(max_size - my_dict_tensor.size(0), dtype=torch.uint8, device=device)])
        dist.all_gather(gathered_tensors, tensor_padded)

        # Deserialize the gathered tensors
        all_dicts = [
            json.loads(tensor[:size.item()].cpu().numpy().tobytes().decode('utf-8')) 
            for tensor, size in zip(gathered_tensors, tensor_sizes)
        ]

        combined_dict = {}
        for d in all_dicts:
            for key, value in d.items():
                if key not in combined_dict:
                    combined_dict[key] = value
                else:
                    combined_dict[key].update(value)

        # Distribute the combined dict to all ranks if needed
        # For example, you can assign it back to my_dict
        my_dict.update(combined_dict)
        return my_dict

    @log_on_entry
    def inference_loop(self):
        gen_config = self.config.generation
        # Compute rank and world size.
        world_size = get_world_size() // get_sequence_parallel_world_size()
        rank = get_global_rank() // get_sequence_parallel_world_size()

        with open(os.path.join(gen_config.positive_prompt.path, 'meta.yaml'), 'r') as file:
            pos_prompts = yaml.safe_load(file)

        for pp in pos_prompts:
            pp['img_paths'] = [os.path.join(gen_config.positive_prompt.path, os.path.basename(p)) for p in pp['img_paths']]

        # Create output dir.
        mkdir(gen_config.output.dir)

        all_num_turns = [len(pp['context']) for pp in pos_prompts]
        max_num_turns = max(all_num_turns)


        context_end_idx = 1
        idx2pos_prompts = {pp['index']: pp for pp in pos_prompts}

        # init
        # i_turn == 0
        pos_prompts_turn_i_lst = []
        for i_pp, pp in enumerate(pos_prompts):
            new_pp = copy.deepcopy(pp)
            new_pp['context'] = new_pp['context'][:context_end_idx]
            new_pp['img_paths'] = new_pp['img_paths'][:context_end_idx]
            pos_prompts_turn_i_lst.append(new_pp)
        self.idx2turn2imgpath = defaultdict(dict)

        for i_turn in range(max_num_turns):
            # combine new generated images into context
            if i_turn > 0:
                self.idx2turn2imgpath = self.gather_dicts(self.idx2turn2imgpath)
                new_pos_prompts_turn_i_lst = []
                for pp in pos_prompts_turn_i_lst:
                    all_context = idx2pos_prompts[pp['index']]['context']
                    if i_turn < len(all_context):
                        pp['context'] = all_context[:i_turn + 1]
                        data_sample_ = self.idx2turn2imgpath[str(pp['index'])]
                        pp['img_paths'].append(data_sample_[str(i_turn - 1)])
                        new_pos_prompts_turn_i_lst.append(pp)
                pos_prompts_turn_i_lst = new_pos_prompts_turn_i_lst

            barrier_if_distributed()

            # Partition positive prompts by rank and batch size.
            pos_prompts_turn_i = ListConfig(pos_prompts_turn_i_lst)
            assert isinstance(pos_prompts_turn_i, ListConfig)
            pos_prompts_turn_i_rank = partition_by_groups_balance_by_repeat(pos_prompts_turn_i, world_size)[rank]
            pos_prompts_turn_i_rank = partition_by_size(pos_prompts_turn_i_rank, gen_config.batch_size)

            if len(pos_prompts_turn_i_rank) == 0: # make sure the batch size is not 0 for each rank, otherwise it will be blocked
                pos_prompts_turn_i_rank = [[pos_prompts_turn_i[0]]]

            # Start generation.
            for batch_pos_prompts in pos_prompts_turn_i_rank:
                # Repeat
                for repeat_index in range(gen_config.repeat):
                    # Prepare inputs.
                    prepare_input = partial(self.prepare_input, repeat_idx=repeat_index)
                    texts_pos, conditions, noises, indices, seeds = zip(
                        *map(prepare_input, batch_pos_prompts)
                    )
                    texts_neg = [gen_config.negative_prompt] * len(batch_pos_prompts)

                    if isinstance(texts_pos[0], ListConfig):
                        texts_pos = [list(tp) for tp in texts_pos]

                    gc.collect()            
                    torch.cuda.empty_cache()  
                    torch.cuda.ipc_collect() 
                    # Generate samples as a list of [C T H W] or [C H W].
                    samples = self.inference(
                        noises=noises,
                        conditions=conditions,
                        texts_pos=texts_pos,
                        texts_neg=texts_neg,
                    )

                    # Save samples.
                    for sample, text_pos, index, seed in zip(samples, texts_pos, indices, seeds):
                        if get_sequence_parallel_rank() == 0:
                            self.save_sample(
                                sample=sample,
                                text=text_pos,
                                index=index,
                                repeat_index=repeat_index,
                                seed=seed,
                                turn_index=i_turn
                            )
            

    def prepare_input(
        self,
        prompt: Union[str, DictConfig],
        repeat_idx: int,
        device: torch.device = get_device(),
    ):
        # Compute latent size.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        ch = self.config.vae.model.latent_channels

        # Parse prompt.
        if isinstance(prompt, DictConfig):
            if prompt.get("img_paths"):
                task = "v2i"
                text = prompt.context
                prompt_prefix = os.environ.get("CAMERA_PREFIX", "")
                text = [prompt_prefix + t_ for t_ in text]

                if self.config.generation.get("pad_img_placehoder", False):
                    if 'context_imgPlaceholder' in prompt: 
                        text = prompt.context_imgPlaceholder
                    elif 'mse_bench' in self.config.generation.positive_prompt.path:
                        text = [f'Based on <IMG{i_c}>, {str_x.strip().strip(".")}. Output <IMG{i_c+1}>: '
                                for i_c, str_x in enumerate(text)]
                    else:
                        raise NotImplementedError(self.config.generation.positive_prompt.path)

                default_aspect_ratio = self.config.generation.get("aspect_ratio", "keep_ratio")
                aspect_ratio = prompt.get("aspect_ratio", default_aspect_ratio)
                cond = self.load_image_latent_v2i(prompt.img_paths, aspect_ratio) # (T, h, w, c),
                cond = torch.cat([cond, torch.zeros((1, *cond.shape[1:]), device=device)], dim=0) # (T+1, h, w, c)
            else:
                raise ValueError('prompt.img_paths is empty. ')
        else:
            raise ValueError('prompt is not a DictConfig. ')

        # Generate noise.
        index = prompt.index
        seed = shift_seed(self.config.generation.seed, index + repeat_idx)
        seed = seed if seed is not None else random.randint(0, 100000)
        generator = torch.Generator(device).manual_seed(seed)
        noise = torch.randn(size=cond.shape, device=device, generator=generator)

        # Post process cond.
        cond = self.get_condition(cond, task)
        return text, cond, noise, index, seed

    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        texts_pos: Union[List[str], List[Tensor], List[Tuple[Tensor]], List[List[str]]],
        texts_neg: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        cfg_scale: Optional[float] = None,
    ) -> List[Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []

        # Set cfg scale
        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale

        # Text embeddings.
        if isinstance(texts_pos[0], list):  # session texts
            assert isinstance(texts_pos[0][0], str) and isinstance(texts_neg[0], str)
            text_pos_embeds, text_pos_shapes = self.get_session_text_embedding(texts_pos)
            text_neg_embeds, text_neg_shapes = self.get_session_text_embedding(texts_neg)
        elif isinstance(texts_pos[0], str):  # image
            assert type(texts_pos[0]) is type(texts_neg[0])
            text_pos_embeds, text_pos_shapes = self.get_session_text_embedding(texts_pos)
            text_neg_embeds, text_neg_shapes = self.get_session_text_embedding(texts_neg)
        else:
            raise NotImplementedError

        # Flatten.
        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)

        # Enter eval mode.
        was_training = self.dit.training
        self.dit.eval()

        # Sampling.
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

        # Exit eval mode.
        self.dit.train(was_training)

        # Unflatten.
        latents = na.unflatten(latents, latents_shapes)

        # Vae decode.
        samples = self.vae_decode(latents)
        return samples

    def load_image_latent_v2i(self, path: List[str], aspect_ratio: str = "keep_ratio"):
        # Load size.
        resolution = self.config.generation.resolution
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        _, ph, pw = _triple(self.config.dit.model.patch_size)
        ratio_cands = self.config.generation.get(
            "adaptive_aspect_ratio_cands", ["9:16", "1:1", "16:9"]
        )

        # Define transform.
        transform = Compose(
            [
                ToTensor(),
                AspectRatioCrop(
                    aspect_ratio=aspect_ratio, adaptive_aspect_ratio_cands=ratio_cands
                ),
                NaResize(
                    resolution=resolution,
                    mode="area",
                    downsample_only=False,  # Upsample image, model only trained for high res.
                ),
                DivisibleCrop((vs * ph, vs * pw)),
                Normalize(0.5, 0.5),
            ]
        )
        # Load all images.
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

        # Vae encode.
        return self.vae_encode([img_tensor])[0]


    def get_condition(self, latent: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if t == 1:  # t2i
            return cond
        else:  # session editing, v2i
            cond[:-1, ..., :-1] = latent[:-1]
            cond[:-1, ..., -1:] = 1.0
            return cond

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor]) -> List[Tensor]:
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            is_session_img = self.config.data.video.get("type") == "session_image"

            # Vae process by each group.
            def pix2latent(sample): 
                sample = sample.to(device, dtype)
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)
                if use_sample:
                    latent = self.vae.encode(sample).latent
                else:
                    # Deterministic vae encode, only used for i2v inference (optionally)
                    latent = self.vae.encode(sample).posterior.mode().squeeze(2)
                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = rearrange(latent, "b c ... -> b ... c")
                latent = (latent - shift) * scale
                return latent

            for sample in batches:
                if (
                    is_session_img and sample.ndim == 5
                ):  # # sample should in [1, C, T, H, W] for video or session image, or [1, C, H, W] for single image
                    latent = []
                    for i_frame in range(sample.shape[2]):
                        latent_frame_i = pix2latent(sample[:, :, i_frame, :, :])
                        latent.append(latent_frame_i)
                    latent = torch.cat(latent, dim=1)  # (1, T, h, w, c)
                else:
                    latent = pix2latent(sample)
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.config.vae.grouping:
                latents = na.unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]

        return latents

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor]) -> List[Tensor]:
        samples = []
        if len(latents) > 0:
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group latents of the same shape to batches if enabled.
            if self.config.vae.grouping:
                latents, indices = na.pack(latents)
            else:
                latents = [latent.unsqueeze(0) for latent in latents]

            is_session_img = self.config.data.video.get("type") == "session_image"
            # Vae process by each group.
            for latent in latents:
                latent = latent.to(device, dtype)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                latent = latent.squeeze(2)

                if is_session_img and latent.ndim == 5:
                    assert (
                        not self.config.vae.grouping
                    ), "vae.grouping has not been implemented for session image"
                    sample = []
                    for i_frame in range(latent.shape[2]):
                        s = self.vae.decode(latent[:, :, i_frame, :, :]).sample
                        sample.append(s)
                    sample = torch.stack(sample, dim=2)
                    assert sample.ndim == 5
                else:
                    sample = self.vae.decode(latent).sample

                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)
                samples.append(sample)

            # Ungroup back to individual sample with the original order.
            if self.config.vae.grouping:
                samples = na.unpack(samples, indices)
            else:
                samples = [sample.squeeze(0) for sample in samples]

        return samples

    def get_session_text_embedding(self, texts: Union[List[List[str]], List[str]]):
        is_session = True if isinstance(texts[0], list) else False
        if is_session:
            text_list = [text for sample in texts for text in sample]  # flatten
            text_id_flattend = [i_t for sample in texts for i_t in range(len(sample))]
        else:
            text_list = texts
            text_id_flattend = [0] * len(text_list)

        text_embeds, text_shapes = self.text_encode(text_list)
        sentence_tuple = torch.split(text_embeds, text_shapes.flatten().tolist())
        text_id_emb = self.dit(
            is_forward_pie=True,
            prompt_id=torch.LongTensor(text_id_flattend).to(get_device()).unsqueeze(0),
        )  # (1, n_sentence, dim)
        assert (
            len(sentence_tuple) == text_id_emb.shape[1]
        ), f"{len(sentence_tuple)}, {text_id_emb.shape}"

        sentence_list_with_id_emb, text_shapes_with_id_emb = self.fuse_prompt_id_emb(
            self.config.prompt_id_embedding.get("fusion_strategy", "seq_concat"),
            sentence_tuple,
            text_id_emb,
            text_shapes,
        )

        if is_session:
            # concate all prompts in one session along sequence dim
            start_idx = 0
            text_embeds_vid, text_shapes_vid = [], []
            for sample in texts:
                n_stc_per_session = len(sample)
                text_embeds_vid.append(
                    torch.cat(
                        sentence_list_with_id_emb[start_idx : start_idx + n_stc_per_session], dim=0
                    )
                )  # (n_token, dim)
                text_shapes_vid.append(
                    text_shapes_with_id_emb[start_idx : start_idx + n_stc_per_session].sum(dim=0)
                )
                start_idx += n_stc_per_session
            assert start_idx == len(text_list)
            text_embeds = torch.cat(text_embeds_vid, dim=0)
            text_shapes = torch.stack(text_shapes_vid, dim=0)
        else:
            text_embeds = torch.cat(sentence_list_with_id_emb, dim=0)
            text_shapes = text_shapes_with_id_emb

        return text_embeds, text_shapes

    @torch.no_grad()
    def text_encode(
        self, texts: List[str]
    ) -> Tuple[Union[Tensor, List[Tensor]], Union[Tensor, List[Tensor]]]:
        # Text encoder forward.
        text_outputs = self.text_encoder(texts)
        # Convert to nadit input format.
        if isinstance(text_outputs.embeddings, list):
            raise NotImplementedError("List of embeddings not supported yet.")
        else:
            text_embeds = text_outputs.embeddings[text_outputs.masks]
            text_shapes = text_outputs.masks.sum(-1).unsqueeze(-1)
        # Return flattened embeddings and shapes.
        return text_embeds, text_shapes

    def fuse_prompt_id_emb(
        self,
        strategy: str,
        sentence_tuple: Tuple[torch.Tensor],
        text_id_emb: torch.Tensor,
        text_shapes: torch.Tensor,
    ):
        if strategy == "seq_concat": 
            # fusion strategy 1: sequence-level concat, id_emb as the first token
            sentence_list_with_id_emb = [
                torch.cat([tid_emb.unsqueeze(0), stc_seq], dim=0)
                for tid_emb, stc_seq in zip(text_id_emb[0], sentence_tuple)
            ]
            text_shapes_with_id_emb = text_shapes + 1  # (n_stc, 1)
        elif strategy == "dim_add":  
            # fusion strategy 2: dimension-level add
            sentence_list_with_id_emb = [
                tid_emb.unsqueeze(0) + stc_seq
                for tid_emb, stc_seq in zip(text_id_emb[0], sentence_tuple)
            ]
            text_shapes_with_id_emb = text_shapes
        elif strategy == "none":  
            # fusion strategy 3: no fusion
            sentence_list_with_id_emb = sentence_tuple
            text_shapes_with_id_emb = text_shapes
        else:
            raise NotImplementedError(f"Unsupported fusion strategy: {strategy}")
        return sentence_list_with_id_emb, text_shapes_with_id_emb

    def save_sample(
        self,
        *,
        sample: torch.Tensor,
        index: int,
        repeat_index: int,
        text: str,
        seed: int, 
        turn_index: int
    ):
        gen_config = self.config.generation
        sample = sample[:, -1, :, :] # session editing, so only the last frame is valid.

        # Prepare file path.
        extension = ".mp4" if sample.ndim == 4 else ".png"
        if isinstance(text, str):
            prompt = text.replace("/", "_").replace(":", "_")
        elif isinstance(text, list):
            prompt = text[-1].replace("/", "_").replace(":", "_")
        else:
            raise NotImplementedError
            
        filename = gen_config.output.filename.format(
            index=index,
            repeat_index=repeat_index,
            seed=seed,
            turn_index=turn_index
        )

        filename += extension
        pathname = os.path.join(gen_config.output.dir, filename)
        self.idx2turn2imgpath[str(index)][str(turn_index)] = pathname

        tempname = (
            os.path.join("/tmp", filename) if is_hdfs_path(gen_config.output.dir) else pathname
        )

        # Convert sample.
        sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).to("cpu", torch.uint8)

        # Save file.
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
            raise ValueError

        # Move to final location.
        move(tempname, pathname)
