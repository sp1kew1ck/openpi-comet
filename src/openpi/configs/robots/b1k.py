import numpy as np

from .base_config import ObservationConfig
from .base_config import RobotConfig
from .base_config import StateActionConfig
from .base_config import register_robot

# R1Pro Robot Configuration
# Dual-arm mobile manipulator with base, torso, and multiple camera views
R1Pro = RobotConfig(
    name="robot",
    robot_type="R1Pro",
    observations={
        "observation/egocentric_camera": ObservationConfig(
            name="head",
            obs_key="robot::robot:zed_link:Camera:0::rgb",
            dataset_key="observation.rgb.zed_link_camera_0",
            resolution=[240, 240]
        ),
        "observation/wrist_image_left": ObservationConfig(
            name="left_wrist",
            obs_key="robot::robot:left_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.left_realsense_link_camera_0",
            resolution=[240, 240]
        ),
        "observation/wrist_image_right": ObservationConfig(
            name="right_wrist",
            obs_key="robot::robot:right_realsense_link:Camera:0::rgb",
            dataset_key="observation.rgb.right_realsense_link_camera_0",
            resolution=[240, 240]
        ),
    },
    action_key="action",
    action_dim=23,
    action=[
        StateActionConfig(name="base", indices=list(range(3))),
        StateActionConfig(name="torso", indices=list(range(3, 7)), needs_delta_comp=True),
        StateActionConfig(name="left_arm", indices=list(range(7, 14)), needs_delta_comp=True),
        StateActionConfig(name="left_gripper", indices=[14], is_eef=True),
        StateActionConfig(name="right_arm", indices=list(range(15, 22)), needs_delta_comp=True),
        StateActionConfig(name="right_gripper", indices=[22], is_eef=True),
    ],
    proprio=[
        StateActionConfig(name="base_qvel", indices=list(range(3))),
        StateActionConfig(name="trunk_qpos", indices=list(range(53, 57))),
        StateActionConfig(name="left_arm_qpos", indices=list(range(3, 10))),
        StateActionConfig(name="left_gripper_qpos", indices=list(range(24, 26)), is_eef=True),
        StateActionConfig(name="right_arm_qpos", indices=list(range(28, 35))),
        StateActionConfig(name="right_gripper_qpos", indices=list(range(49, 51)), is_eef=True),
    ],
    camera_intrinsics={
        "head": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),  # 720x720
        "left_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
        "right_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
    },
)

# Register robots in the global registry
register_robot("b1k/R1Pro", R1Pro)
