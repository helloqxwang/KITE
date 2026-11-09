"""Require the decoder to supply motion commands for the released method."""
import numpy as np
from omegaconf import OmegaConf

PAPER_DISCRETE_TASKS = {'object_suite_pick_place', 'keyboard_press', 'pumpbottle_use'}


def paper_expected_frame_count(cfg, section):
    """Require complete source trajectories for the released discrete tasks."""
    if OmegaConf.select(cfg, 'task.name', default='') in PAPER_DISCRETE_TASKS:
        return 120
    return OmegaConf.select(cfg, f'{section}.expected_frame_count', default=None)


def validate_paper_control_dt(cfg, dt):
    """Check the compiled simulator timestep as well as the YAML settings."""
    if OmegaConf.select(cfg, 'task.name', default='') in PAPER_DISCRETE_TASKS:
        if not np.isclose(float(dt), .05, rtol=0., atol=1e-9):
            raise ValueError('Released discrete tasks require 120 frames at 20 Hz (50 ms per prediction)')


def validate_paper_execution(cfg, section):
    if OmegaConf.select(cfg, f'{section}.proprio_source', default='executed') != 'executed':
        raise ValueError('The decoder must receive the measured current configuration, not the previous command')
    for name in ('phase_controller_steps', 'phase_finger_gains', 'phase_finger_effort'):
        if OmegaConf.select(cfg, f'{section}.{name}.enabled', default=False):
            raise ValueError(f'{section}.{name} uses source phase labels; released evaluation requires a fixed controller')
    if OmegaConf.select(cfg, 'task.name', default='') in PAPER_DISCRETE_TASKS:
        if OmegaConf.select(cfg, f'{section}.controller_steps_per_prediction', default=1) != 1:
            raise ValueError('Released discrete tasks require one controller step per prediction at 20 Hz')
        for field in ('expected_frame_count', 'max_frames', 'max_steps'):
            value = OmegaConf.select(cfg, f'{section}.{field}', default=None)
            if value is not None and value != 120:
                raise ValueError(f'{section}.{field} must be 120 for the released evaluation protocol')
        validate_paper_control_dt(cfg, float(OmegaConf.select(cfg, 'env.timestep', default=.002))
                                  * int(OmegaConf.select(cfg, 'env.frame_skip', default=25)))
    if OmegaConf.select(cfg, 'task.name', default='') == 'keyboard_press':
        fraction = float(OmegaConf.select(cfg, 'keyboard.key_success_fraction', default=.8))
        if not np.isclose(fraction, .5, rtol=0., atol=1e-9):
            raise ValueError('The main-paper keyboard criterion is 50% of key travel')
        initial_press = float(OmegaConf.select(cfg, 'keyboard.target_key_initial_push_down_offset_fraction', default=0.))
        if initial_press != 0.:
            raise ValueError('Paper keyboard evaluation must not pre-depress target keys')
    for name in ('grasp_hold', 'robot_model_execution_root_offset',
                 'effector_execution_offset', 'robotiq_functional_contact_mapping',
                 'robotiq_gripper_min_command', 'robotiq_stick_execution_offset'):
        if OmegaConf.select(cfg, f'{section}.{name}.enabled', default=False):
            raise ValueError(f'{section}.{name} overrides decoder commands and is outside the paper method')
    offset = OmegaConf.select(cfg, f'{section}.gripper_execution_action_offset_xyz', default=[0., 0., 0.])
    if np.any(np.asarray(offset, dtype=float) != 0):
        raise ValueError('Gripper execution offsets override decoder commands and are outside the paper method')
