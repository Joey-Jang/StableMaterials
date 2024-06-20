import contextlib
import inspect
from typing import Any, Dict, List, Optional, Union, get_args

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.models.transformers import Transformer2DModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    BaseOutput,
    deprecate,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import (
    Image,
    Jpeg2KImagePlugin,
    JpegImagePlugin,
    PngImagePlugin,
    TiffImagePlugin,
)
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModel,
)

from diffusers import AutoencoderKL, DiffusionPipeline, UNet2DConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
from dataclasses import dataclass

ImageInput = Union[
    PipelineImageInput,
    JpegImagePlugin.JpegImageFile,
    Jpeg2KImagePlugin.Jpeg2KImageFile,
    PngImagePlugin.PngImageFile,
    TiffImagePlugin.TiffImageFile,
]

import math


def postprocess(
    image: torch.FloatTensor,
    output_type: str = "pil",
):
    """
    Postprocess the image output from tensor to `output_type`.

    Args:
        image (`torch.FloatTensor`):
            The image input, should be a pytorch tensor with shape `B x C x H x W`.
        output_type (`str`, *optional*, defaults to `pil`):
            The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.

    Returns:
        `PIL.Image.Image`, `np.ndarray` or `torch.FloatTensor`:
            The postprocessed image.
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
        )
    if output_type not in ["latent", "pt", "np", "pil"]:
        deprecation_message = (
            f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
            "`pil`, `np`, `pt`, `latent`"
        )
        deprecate(
            "Unsupported output_type", "1.0.0", deprecation_message, standard_warn=False
        )
        output_type = "np"

    image = image.detach().cpu()
    image = image.to(torch.float32)

    if output_type == "latent":
        return image

    # denormalize the image
    image = image * 0.5 + 0.5  # .clamp(0, 1)

    materials = []
    for i in range(image.shape[0]):

        material = StableMaterialsMaterial()
        material.init_from_tensor(image[i], mode=output_type)

        materials.append(material)

    return materials


@dataclass
class StableMaterialsMaterial:
    basecolor: torch.FloatTensor
    normal: torch.FloatTensor
    height: torch.FloatTensor
    roughness: torch.FloatTensor
    metallic: torch.FloatTensor
    _mode: str = "tensor"  # Default mode is tensor

    def __init__(
        self,
        basecolor: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        normal: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        height: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        roughness: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        metallic: Optional[Union[Image.Image, np.ndarray, torch.FloatTensor]] = None,
        mode: str = "tensor",
    ):
        self._basecolor = self._to_pt(basecolor)
        self._normal = self._to_pt(normal)
        self._height = self._to_pt(height)
        self._roughness = self._to_pt(roughness)
        self._metallic = self._to_pt(metallic)
        self._mode = mode

    def init_from_tensor(self, image: torch.FloatTensor, mode: str = "tensor"):
        assert image.shape[0] >= 8, "Input tensor should have at least 8 channels"
        self._basecolor = image[:3].clamp(0, 1)
        self._normal = self.compute_normal_map_z_component(image[3:5])
        self._height = image[5:6].clamp(0, 1)
        self._roughness = image[6:7].clamp(0, 1)
        self._metallic = image[7:8].clamp(0, 1)
        self._mode = mode

    def resize(self, size, antialias=True):
        self._basecolor = TF.resize(self._basecolor, size, antialias=antialias)
        self._normal = TF.resize(self._normal, size, antialias=antialias)
        self._height = TF.resize(self._height, size, antialias=antialias)
        self._roughness = TF.resize(self._roughness, size, antialias=antialias)
        self._metallic = TF.resize(self._metallic, size, antialias=antialias)
        return self

    def tile(self, num_tiles):
        self._basecolor = self._basecolor.repeat(1, num_tiles, num_tiles)
        self._normal = self._normal.repeat(1, num_tiles, num_tiles)
        self._height = self._height.repeat(1, num_tiles, num_tiles)
        self._roughness = self._roughness.repeat(1, num_tiles, num_tiles)
        self._metallic = self._metallic.repeat(1, num_tiles, num_tiles)
        return self

    def _to_numpy(self, image: torch.FloatTensor):
        if image is None:
            return None
        return image.numpy()

    def _to_pil(self, image: torch.FloatTensor, mode: str = "RGB"):
        if image is None:
            return None
        return TF.to_pil_image(image).convert(mode)

    def _to_pt(self, image):
        if image is None:
            return None
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        elif isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        return image.cpu()

    def compute_normal_map_z_component(self, normal: torch.FloatTensor):
        normal = normal * 2 - 1
        sum_sq = (normal**2).sum(dim=0, keepdim=True)[0]
        z = torch.zeros_like(sum_sq)
        mask = sum_sq <= 1
        z[mask] = torch.sqrt(1 - sum_sq[mask])
        mask_outlier = sum_sq > 1
        scale_factor = torch.sqrt(sum_sq[mask_outlier])
        normal[:, mask_outlier] = normal[:, mask_outlier] / scale_factor
        normal = torch.cat([normal, z.unsqueeze(0)], dim=0)
        normal = normal * 0.5 + 0.5
        return normal.clamp(0, 1)

    def _convert(self, image, mode="RGB"):
        if self._mode == "numpy":
            return self._to_numpy(image)
        elif self._mode == "pil":
            return self._to_pil(image, mode)
        return image

    @property
    def size(self):
        return list(self._basecolor.shape[-2:])

    @property
    def basecolor(self):
        return self._convert(self._basecolor, mode="RGB")

    @property
    def normal(self):
        return self._convert(self._normal, mode="RGB")

    @property
    def height(self):
        return self._convert(self._height, mode="L")

    @property
    def roughness(self):
        return self._convert(self._roughness, mode="L")

    @property
    def metallic(self):
        return self._convert(self._metallic, mode="L")

    def as_dict(self):
        return {
            "basecolor": self.basecolor,
            "normal": self.normal,
            "height": self.height,
            "roughness": self.roughness,
            "metallic": self.metallic,
        }

    def as_list(self):
        return [
            self.basecolor,
            self.normal,
            self.height,
            self.roughness,
            self.metallic,
        ]

    def as_tensor(self):
        return torch.cat(
            [
                self._basecolor,
                self._normal[:2],
                self._height,
                self._roughness,
                self._metallic,
            ],
            dim=0,
        )


@dataclass
class StableMaterialsPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: List[StableMaterialsMaterial]


def patch(x, patch_factor=2):
    if isinstance(x, (list, tuple)):
        pass

    b, c, h, w = x.shape
    patch_size = h // patch_factor

    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, c, patch_size, patch_size)

    n_patches = x.shape[0] // b

    return x, (b, h), n_patches, patch_size


def unpatch(x, b, h, n_patches, patch_size=32):
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]
        else:
            pass

    factor = patch_size / x.shape[-1]
    h, w = int(h / factor), int(h / factor)

    c, patch_size = x.shape[1], x.shape[2]
    n_patches = x.shape[0] // b

    x = x.reshape(b, n_patches, c, patch_size, patch_size)
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(b, c * patch_size * patch_size, -1)

    x = F.fold(
        x,
        output_size=(h, w),
        kernel_size=patch_size,
        stride=patch_size,
    )

    return x


def roll(x):
    roll_h = torch.randint(0, 256, (1,)).item() // 2 * 2
    roll_w = torch.randint(0, 256, (1,)).item() // 2 * 2

    x = torch.roll(x, shifts=(roll_h, roll_w), dims=(2, 3))

    return x, (roll_h, roll_w)


def unroll(x, roll_h, roll_w, factor=1.0):
    roll_h = int(roll_h * factor)
    roll_w = int(roll_w * factor)
    x = torch.roll(x, shifts=(-roll_h, -roll_w), dims=(2, 3))
    return x


@contextlib.contextmanager
def rolled_conv(enabled=True):
    conv = torch.nn.Conv2d

    if enabled:
        # Save the original conv's constructor
        orig_forward = conv.forward

        def forward(self, x, *args, **kwargs):
            x, (roll_h, roll_w) = roll(x)

            pad = 4
            x = F.pad(x, (pad, pad, pad, pad), mode="circular")
            h = x.shape[-2]

            x = orig_forward(self, x, *args, **kwargs)
            h1 = x.shape[-2]
            factor = h1 / h

            pad = int(pad * factor)
            x = x[..., pad:-pad, pad:-pad]
            x = unroll(x, roll_h, roll_w, factor)

            return x

        # Patch conv's constructor
        conv.forward = forward
        # conv.__init__ = __init__
        yield conv

        # Restore the original conv's constructor
        conv.forward = orig_forward
    else:
        # Use the original conv
        yield conv


@contextlib.contextmanager
def tiled_attn(enabled=True, scale_multiplier=4):
    conv = Transformer2DModel

    if enabled:
        # Save the original conv's constructor
        orig_forward = conv.forward
        # mult = scale_multiplier

        def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
            hidden_states, (roll_h, roll_w) = roll(hidden_states)
            hidden_states, (b, h), n_patches, patch_size = patch(
                hidden_states, self.scale_multiplier
            )
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                n_patches, dim=0
            )
            chunks = math.ceil(len(hidden_states) / 8)
            hidden_states = hidden_states.chunk(chunks, dim=0)
            encoder_hidden_states = encoder_hidden_states.chunk(chunks, dim=0)
            result = []
            for i in range(chunks):
                result.append(
                    orig_forward(
                        self,
                        hidden_states[i],
                        encoder_hidden_states[i],
                        *args,
                        **kwargs,
                    )[0]
                )
            hidden_states = torch.cat(result, dim=0)
            hidden_states = unpatch(hidden_states, b, h, n_patches, patch_size)
            hidden_states = unroll(hidden_states, roll_h, roll_w)
            return (hidden_states,)

        # Patch conv's constructor
        conv.scale_multiplier = scale_multiplier
        conv.forward = forward
        yield conv

        # Restore the original conv's constructor
        conv.forward = orig_forward
    else:
        # Use the original conv
        yield conv


class StableMaterialsPipeline(DiffusionPipeline, FromSingleFileMixin):

    model_cpu_offload_seq = "prompt_encoder->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        # prompt_encoder: nn.Module,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        vision_encoder: CLIPVisionModel,
        processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            # prompt_encoder=prompt_encoder,
            scheduler=scheduler,
            # Conditioning modules
            tokenizer=tokenizer,
            processor=processor,
            text_encoder=text_encoder,
            vision_encoder=vision_encoder,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def __encode_text(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        outputs = self.text_encoder(**inputs)
        return outputs.text_embeds.unsqueeze(1)

    def __encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        outputs = self.vision_encoder(**inputs)
        return outputs.image_embeds.unsqueeze(1)

    def __encode_prompt(
        self,
        prompt,
    ):
        if type(prompt) != list:
            prompt = [prompt]

        embs = []
        for prompt in prompt:
            if isinstance(prompt, str):
                embs.append(self.__encode_text(prompt))
            elif type(prompt) in get_args(ImageInput):
                embs.append(self.__encode_image(prompt))
            else:
                raise NotImplementedError

        return torch.cat(embs, dim=0)

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if (
            prompt is not None
            and isinstance(prompt, str)
            or isinstance(prompt, Image.Image)
        ):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self.__encode_prompt(prompt)

        if self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                # uncond_tokens = [""] * batch_size
                uncond_tokens = [Image.new("RGB", (512, 512), (0, 0, 0))] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif len(negative_prompt) != batch_size:
                raise ValueError(
                    "The `negative_prompt` must be a string, a list of strings of length `batch_size`, or `None`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds = self.__encode_prompt(uncond_tokens)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

        return prompt_embeds, negative_prompt_embeds

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, (str, list, Image.Image))):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[
            str, List[str], PipelineImageInput, List[PipelineImageInput]
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        tileable: bool = False,
        patched: bool = False,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        **kwargs,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and (
            isinstance(prompt, str) or isinstance(prompt, Image.Image)
        ):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                scale_multiplier = (
                    latent_model_input.shape[-1]
                ) // self.unet.config.sample_size

                past_mid = i >= len(timesteps) // 4
                # predict the noise residual
                with rolled_conv(enabled=(tileable & past_mid)):
                    with tiled_attn(enabled=patched, scale_multiplier=scale_multiplier):
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            return_dict=False,
                        )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if not output_type == "latent":
            if tileable:
                # decode padded latent to preserve tileability
                l_height = height // self.vae_scale_factor
                l_width = width // self.vae_scale_factor
                pad = l_height // 4
                latents = TF.center_crop(
                    latents.repeat(1, 1, 3, 3), (l_height + pad, l_width + pad)
                )

            # decode the latents
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]

            # crop to original size
            image = TF.center_crop(image, (height, width))
        else:
            image = latents

        image = postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return StableMaterialsPipelineOutput(images=image)
