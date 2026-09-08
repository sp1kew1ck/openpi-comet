import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.configs.robots.base_config import RobotConfig
from openpi.models import model as _model


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data, robot_config: RobotConfig) -> np.ndarray:
    """Extract state from proprioception data based on robot configuration.

    We assume perfect correlation for the two gripper fingers.

    Args:
        proprio_data: Raw proprioception data
        robot_config: RobotConfig instance containing robot configuration

    Returns:
        Extracted state array
    """
    state = []
    for proprio in robot_config.proprio:
        if proprio.is_eef:
            # Sum the gripper finger positions to get a single width value
            state.append(proprio_data[..., proprio.indices].sum(axis=-1, keepdims=True))
        else:
            state.append(proprio_data[..., proprio.indices])
    return np.concatenate(state, axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_seg_image(image, max_seg=8) -> np.ndarray:
    image = np.asarray(image)
    image = image / max_seg * 255
    image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    return image.astype(np.uint8)


def depth_to_pcd(depth_image: np.ndarray, camera_intrinsics: np.ndarray, downsample: int = 6) -> np.ndarray:
    """
    Convert depth image to point cloud.
    """
    depth_image = np.asarray(depth_image)
    h, w = depth_image.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image
    pcd_xyz = np.stack([x, y, z], axis=-1)  # (h, w, 3)

    return pcd_xyz[::downsample, ::downsample].reshape(16, -1, 3)


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    meta_image_keys: list[str] = dataclasses.field(default_factory=list)

    depth_as_pcd: bool = False

    pcd_downsample: int = 6

    # Robot configuration object
    robot_config: RobotConfig = dataclasses.field(default=None)

    # Statistics computation does not need to decode camera frames.
    include_images: bool = True

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data, self.robot_config)
        if "actions" in data:
            action = data["actions"]

        inputs = {"state": state}

        if self.include_images:
            # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
            # stores as float32 (C,H,W), gets skipped for policy inference
            base_image = _parse_image(data["observation/egocentric_camera"])
            wrist_image_left = _parse_image(data["observation/wrist_image_left"])
            wrist_image_right = _parse_image(data["observation/wrist_image_right"])

            meta_images, meta_image_names = [], []
            if "observation/egocentric_seg" in self.meta_image_keys:
                seg_image = _parse_seg_image(data["observation/egocentric_seg"])
                meta_images.append(seg_image)
                meta_image_names.append("base_0_seg")

            match self.model_type:
                case _model.ModelType.PI0 | _model.ModelType.PI05:
                    names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                    images = (base_image, wrist_image_left, wrist_image_right)
                    image_masks = (np.True_, np.True_, np.True_)
                case _model.ModelType.PI0_FAST:
                    names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                    # We don't mask out padding images for FAST models.
                    images = (base_image, wrist_image_left, wrist_image_right)
                    image_masks = (np.True_, np.True_, np.True_)
                case _:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

            names += tuple(meta_image_names)
            images += tuple(meta_images)
            image_masks += tuple(np.True_ for _ in meta_image_names)
            inputs["image"] = dict(zip(names, images, strict=True))
            inputs["image_mask"] = dict(zip(names, image_masks, strict=True))

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if self.depth_as_pcd:
            depth_image = data["observation/egocentric_depth"]
            inputs["pcd_xyz"] = depth_to_pcd(depth_image, self.robot_config.camera_intrinsics["head"], self.pcd_downsample)

        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
