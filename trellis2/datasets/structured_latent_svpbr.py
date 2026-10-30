import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import json
import logging
from typing import *
import numpy as np
import torch
import cv2
from .. import models
from .components import StandardDatasetBase, ImageConditionedMixin
from ..modules.sparse import SparseTensor, sparse_cat
from ..representations import MeshWithVoxel
from ..renderers import PbrMeshRenderer, EnvMap
from ..utils.data_utils import load_balanced_group_indices
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


logger = logging.getLogger(__name__)


def procedural_studio_envmap(height: int = 256, width: int = 512) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError(f'Environment map size must be positive, got {height}x{width}')

    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    azimuth = np.linspace(-np.pi, np.pi, width, dtype=np.float32)[None, :]
    altitude = np.float32(np.pi) * (np.float32(0.5) - vertical)

    horizon_blend = (1.0 / (1.0 + np.exp((vertical - 0.5) * 24.0)))[..., None]
    sky = np.stack([
        0.55 - 0.20 * vertical,
        0.62 - 0.22 * vertical,
        0.78 - 0.28 * vertical,
    ], axis=-1)
    ground = np.stack([
        0.10 + 0.10 * vertical,
        0.09 + 0.09 * vertical,
        0.08 + 0.08 * vertical,
    ], axis=-1)
    gradient = ground * (1.0 - horizon_blend) + sky * horizon_blend
    image = np.array(np.broadcast_to(gradient, (height, width, 3)), dtype=np.float32)

    key_direction = (0.9, 1.25)
    fill_direction = (-2.2, 0.55)
    rim_direction = (2.6, 0.35)
    lights = (
        (key_direction, 5.5, np.array([1.00, 0.95, 0.88], dtype=np.float32)),
        (fill_direction, 2.2, np.array([0.80, 0.88, 1.00], dtype=np.float32)),
        (rim_direction, 1.4, np.array([0.95, 0.92, 0.98], dtype=np.float32)),
    )
    for (light_azimuth, light_altitude), intensity, color in lights:
        angular_distance = np.arccos(np.clip(
            np.sin(altitude) * np.sin(light_altitude)
            + np.cos(altitude) * np.cos(light_altitude) * np.cos(azimuth - light_azimuth),
            -1.0, 1.0,
        ))
        falloff = np.exp(-0.5 * (angular_distance / 0.22) ** 2) * intensity
        image = image + falloff[..., None] * color[None, None, :]

    image = np.clip(image, 0.0, 16.0).astype(np.float32)
    return torch.tensor(image, dtype=torch.float32, device='cuda')


def load_visualization_envmap() -> EnvMap:
    envmap_path = os.environ.get('TRELLIS_ENVMAP')
    if envmap_path:
        if os.path.isfile(envmap_path):
            image = cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f'Unable to decode the environment map {envmap_path!r}')
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            else:
                image = image.astype(np.float32)
            if image.ndim == 2:
                image = image[:, :, None]
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(
                    f'Unsupported channel count {image.shape[2]} in the environment map {envmap_path!r}'
                )
            return EnvMap(torch.tensor(
                np.ascontiguousarray(image),
                dtype=torch.float32, device='cuda'
            ))
        logger.warning(
            'Environment map %r not found, falling back to the procedural studio environment',
            envmap_path,
        )
    return EnvMap(procedural_studio_envmap())


class SLatPbrVisMixin:
    def __init__(
        self,
        *args,
        pretrained_pbr_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pbr_slat_dec = None
        self.pretrained_pbr_slat_dec = pretrained_pbr_slat_dec
        self.pbr_slat_dec_path = pbr_slat_dec_path
        self.pbr_slat_dec_ckpt = pbr_slat_dec_ckpt
        self.shape_slat_dec = None
        self.pretrained_shape_slat_dec = pretrained_shape_slat_dec
        self.shape_slat_dec_path = shape_slat_dec_path
        self.shape_slat_dec_ckpt = shape_slat_dec_ckpt

    def _loading_slat_dec(self):
        if self.pbr_slat_dec is not None and self.shape_slat_dec is not None:
            return
        if self.pbr_slat_dec_path is not None:
            with open(os.path.join(self.pbr_slat_dec_path, 'config.json'), 'r') as f:
                cfg = json.load(f)
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.pbr_slat_dec_path, 'ckpts', f'decoder_{self.pbr_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_pbr_slat_dec)
        self.pbr_slat_dec = decoder.cuda().eval()

        if self.shape_slat_dec_path is not None:
            with open(os.path.join(self.shape_slat_dec_path, 'config.json'), 'r') as f:
                cfg = json.load(f)
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.shape_slat_dec_path, 'ckpts', f'decoder_{self.shape_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_shape_slat_dec)
        decoder.set_resolution(self.resolution)
        self.shape_slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        del self.pbr_slat_dec
        self.pbr_slat_dec = None
        del self.shape_slat_dec
        self.shape_slat_dec = None

    @torch.no_grad()
    def decode_latent(self, z, shape_z, batch_size=4):
        self._loading_slat_dec()
        reps = []
        if self.shape_slat_normalization is not None:
            shape_z = shape_z * self.shape_slat_std.to(z.device) + self.shape_slat_mean.to(z.device)
        if self.pbr_slat_normalization is not None:
            z = z * self.pbr_slat_std.to(z.device) + self.pbr_slat_mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            mesh, subs = self.shape_slat_dec(shape_z[i:i+batch_size], return_subs=True)
            vox = self.pbr_slat_dec(z[i:i+batch_size], guide_subs=subs) * 0.5 + 0.5
            reps.extend([
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / self.resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout = self.layout,
                )
                for m, v in zip(mesh, [vox[j] for j in range(len(mesh))])
            ])
        self._delete_slat_dec()
        return reps

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        shape_z = sample['concat_cond'].cuda()
        z = sample['x_0'].cuda()
        reps = self.decode_latent(z, shape_z)

        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        tile_resolution = 512
        tile = [2, 2]

        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = tile_resolution
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8
        envmap = load_visualization_envmap()

        images = {}
        for representation in reps:
            mosaics = {}
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, envmap=envmap)
                row = j // tile[1]
                col = j % tile[1]
                for k, v in res.items():
                    if k not in mosaics:
                        mosaics[k] = torch.zeros(
                            3, tile_resolution * tile[0], tile_resolution * tile[1], device='cuda'
                        )
                    mosaics[k][
                        :,
                        tile_resolution * row:tile_resolution * (row + 1),
                        tile_resolution * col:tile_resolution * (col + 1),
                    ] = v
            for k, mosaic in mosaics.items():
                if k not in images:
                    images[k] = []
                images[k].append(mosaic)
        for k in images.keys():
            images[k] = torch.stack(images[k], dim=0)
        return images


class SLatPbr(SLatPbrVisMixin, StandardDatasetBase):
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        full_pbr: bool = False,
        pbr_slat_normalization: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        attrs: Optional[List[str]] = None,
        pretrained_pbr_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        if attrs is None:
            attrs = ['base_color', 'metallic', 'roughness', 'emissive', 'alpha']
        self.resolution = resolution
        self.pbr_slat_normalization = pbr_slat_normalization
        self.shape_slat_normalization = shape_slat_normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.full_pbr = full_pbr
        self.value_range = (0, 1)

        super().__init__(
            roots,
            pretrained_pbr_slat_dec=pretrained_pbr_slat_dec,
            pbr_slat_dec_path=pbr_slat_dec_path,
            pbr_slat_dec_ckpt=pbr_slat_dec_ckpt,
            pretrained_shape_slat_dec=pretrained_shape_slat_dec,
            shape_slat_dec_path=shape_slat_dec_path,
            shape_slat_dec_ckpt=shape_slat_dec_ckpt,
            **kwargs
        )

        self.loads = [self.metadata.loc[sha256, 'pbr_latent_tokens'] for _, sha256 in self.instances]

        if self.pbr_slat_normalization is not None:
            self.pbr_slat_mean = torch.tensor(self.pbr_slat_normalization['mean']).reshape(1, -1)
            self.pbr_slat_std = torch.tensor(self.pbr_slat_normalization['std']).reshape(1, -1)

        if self.shape_slat_normalization is not None:
            self.shape_slat_mean = torch.tensor(self.shape_slat_normalization['mean']).reshape(1, -1)
            self.shape_slat_std = torch.tensor(self.shape_slat_normalization['std']).reshape(1, -1)

        self.attrs = list(attrs)
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in self.attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]

    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata['pbr_latent_encoded'] == True]
        stats['With PBR latent'] = len(metadata)
        metadata = metadata[metadata['shape_latent_encoded'] == True]
        stats['With shape latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['pbr_latent_tokens'] <= self.max_tokens]
        stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        if self.full_pbr:
            metadata = metadata[metadata['num_basecolor_tex'] > 0]
            metadata = metadata[metadata['num_metallic_tex'] > 0]
            metadata = metadata[metadata['num_roughness_tex'] > 0]
            stats['Full PBR'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        with np.load(os.path.join(root['pbr_latent'], f'{instance}.npz')) as data:
            coords = torch.tensor(data['coords']).int()
            feats = torch.tensor(data['feats']).float()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        if self.pbr_slat_normalization is not None:
            feats = (feats - self.pbr_slat_mean) / self.pbr_slat_std
        pbr_z = SparseTensor(feats, coords)

        with np.load(os.path.join(root['shape_latent'], f'{instance}.npz')) as data:
            coords = torch.tensor(data['coords']).int()
            feats = torch.tensor(data['feats']).float()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        if self.shape_slat_normalization is not None:
            feats = (feats - self.shape_slat_mean) / self.shape_slat_std
        shape_z = SparseTensor(feats, coords)

        assert torch.equal(shape_z.coords, pbr_z.coords), \
            f"Shape latent and PBR latent have different coordinates: {shape_z.coords.shape} vs {pbr_z.coords.shape}"

        return {
            'x_0': pbr_z,
            'concat_cond': shape_z,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['x_0'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], SparseTensor):
                    pack[k] = sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


class ImageConditionedSLatPbr(ImageConditionedMixin, SLatPbr):
    pass
