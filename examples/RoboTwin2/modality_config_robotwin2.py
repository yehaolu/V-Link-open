"""Fixed public RoboTwin2 modality config.

The state/action key schema matches the dataset meta/modality.json of
the public RoboTwin2 LeRobot dataset (left/right gripper + left/right arm EEF).
Video and training targets use the same three-view order: ``image``,
``left_wrist_image``, and ``right_wrist_image``. The action horizon is 15.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


robotwin2_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["image", "left_wrist_image", "right_wrist_image"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_gripper",
            "right_gripper",
            "left_arm_eef",
            "right_arm_eef",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(15)),
        modality_keys=[
            "left_gripper",
            "right_gripper",
            "left_arm_eef",
            "right_arm_eef",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_gripper",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_gripper",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="left_arm_eef",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_arm_eef",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
    # Training-only target transport: the selected LeRobot loader places the
    # depth_m and mesh_segmentation arrays from camera_meta/*.npz in
    # VLAStepData.camera_calibration for processor target extraction. The
    # public inference policy does not accept or expose camera matrices.
    "camera_calibration": ModalityConfig(
        delta_indices=[0],
        modality_keys=["camera_calibration"],
    ),
}


register_modality_config(
    robotwin2_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
