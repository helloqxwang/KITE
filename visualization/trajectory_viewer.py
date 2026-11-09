"""Interactive source-to-target lifting trajectory replay."""
from pathlib import Path
import time

import hydra
import numpy as np
import torch

from visualization.decoder_runtime import contact_errors, label, nearest_surface
from visualization.decoder_scene import Viewer, INITIAL, CONTACT
from visualization.trajectory_library import build_source_trajectory, load_library, validate_record
from visualization.trajectory_rollout import roll_out


class TrajectoryViewer(Viewer):
    def __init__(self, cfg):
        super().__init__(cfg, "KITE · Trajectory following", "Curated source lift and finger motion → interaction intents → autoregressive target decoding.")
        self.library = load_library()
        self.source = self.embodiment_selector("Source embodiment")
        self.target = self.embodiment_selector("Target embodiment")
        target = str(self.cfg.trajectory.target_embodiment)
        if target in self.runtime.names:
            self.target.value = label(target)
        self.info = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Trajectory settings", expand_by_default=False):
            self.steps = self.server.gui.add_slider("Source frames", min=20, max=120, step=10, initial_value=int(self.cfg.trajectory.steps))
            self.hold = self.server.gui.add_slider("Extra initial hold frames", min=0, max=15, step=1, initial_value=int(self.cfg.trajectory.hold_steps))
            self.height = self.server.gui.add_slider("Lift height (cm)", min=4., max=20., step=1., initial_value=100.*float(self.cfg.trajectory.lift_height_m))
            self.fingers = self.server.gui.add_slider("Finger motion (%)", min=0., max=100., step=5., initial_value=100.*float(self.cfg.trajectory.finger_motion))
            self.calls = self.server.gui.add_slider("Decoder calls per frame", min=1, max=5, step=1, initial_value=int(self.cfg.trajectory.calls_per_step))
            self.clamp = self.server.gui.add_checkbox("Clamp joint limits", initial_value=bool(self.cfg.trajectory.clamp_joint_limits))
        self.frame_index = self.server.gui.add_slider("Playback frame (0 = initial)", min=0, max=int(self.steps.value), step=1, initial_value=0)
        self.play = self.server.gui.add_checkbox("Play trajectory", initial_value=False)
        self.fps = self.server.gui.add_slider("Playback frames / second", min=1, max=30, step=1, initial_value=15)
        self.server.gui.add_button("Restart playback").on_click(self._restart)
        self.server.gui.add_button("Frame view").on_click(lambda _: self.frame())
        with self.server.gui.add_folder("Display", expand_by_default=False):
            self.normals = self.server.gui.add_checkbox("Surface normals", initial_value=True)
            self.error_lines = self.server.gui.add_checkbox("Position error lines", initial_value=True)
            self.normal_length = self.server.gui.add_slider("Normal length (m)", min=.005, max=.05, step=.005, initial_value=.02)
            self.overlay = self.server.gui.add_checkbox("Overlay source and target", initial_value=False)
            self.trails = self.server.gui.add_checkbox("Contact trajectories", initial_value=True)
        with self.server.gui.add_folder("Interpretation", expand_by_default=False):
            self.notes = self.server.gui.add_markdown("")
        self._key = None
        for control in [self.source, self.target, self.steps, self.hold, self.height, self.fingers, self.calls, self.clamp]:
            control.on_update(self._selection_changed)
        for control in [self.frame_index, self.normals, self.error_lines, self.normal_length, self.trails]:
            control.on_update(self._display_changed)
        self.overlay.on_update(lambda _: self.update(reframe=True))

    def _selection_changed(self, _):
        self.play.value = False
        self.update(reframe=True)

    def _display_changed(self, _):
        # Viser invokes callbacks synchronously even for a same-value Python
        # assignment. Resetting the playback slider during prepare must not
        # recursively recompute the rollout.
        if not self.busy:
            self.update()

    def _restart(self, _):
        self.frame_index.value = 0
        self.play.value = True

    def prepare(self):
        source = self.name_by_label[self.source.value]
        target = self.name_by_label[self.target.value]
        key = (source, target, int(self.steps.value), int(self.hold.value), float(self.height.value),
               float(self.fingers.value), int(self.calls.value), self.clamp.value)
        if key == self._key:
            return
        for name in [source,target]:
            if name not in self.library["embodiments"]:
                raise ValueError(f"No curated lifting motion for {label(name)} in this library")
        source_examples, target_examples = self.runtime.get(source), self.runtime.get(target)
        record = self.library["embodiments"][source]
        validate_record(target_examples.model, self.library["embodiments"][target])
        trajectory = build_source_trajectory(source_examples.model,source,record,steps=key[2],hold_steps=key[3],
                                             lift_height=key[4]/100.,finger_motion=key[5]/100.)
        alignment = self.library["initializations"][source][target]
        result = roll_out(self.runtime,target,trajectory,torch.tensor(alignment["q_init"]),
                          calls_per_step=key[6],clamp_limits=key[7],
                          progress=lambda step,total: setattr(self.status,"content",f"Decoding {label(target)}: {step}/{total} frames…"))
        self.source_examples, self.target_examples = source_examples,target_examples
        self.trajectory,self.result,self.alignment = trajectory,result,alignment
        self._key = key
        self.frame_index.max = key[2]
        self.frame_index.value = 0

    def render(self):
        frame = min(int(self.frame_index.value),len(self.trajectory.intents))
        index = max(0,frame-1)
        source,target = self.source_examples,self.target_examples
        intent = self.trajectory.intents[index]
        q = self.result.initial if frame == 0 else self.result.states[index,-1]
        points,normals,_ = nearest_surface(target.model,q,intent)
        rms,angle = contact_errors(intent,points,normals)
        offset = np.zeros(3) if self.overlay.value else np.array([.55,0.,0.])
        center = self.trajectory.intents[0,:,:3].mean(0).numpy()
        root = self.scene.panel("/trajectory_source")
        self.scene.mesh(root,"source",source.model,self.trajectory.configurations[index],CONTACT,opacity=.35 if self.overlay.value else .85)
        self.contacts(root,intent)
        self.scene.caption(root,f"Source · {label(source.name)}",center)
        target_root = self.scene.panel("/trajectory_target",offset)
        self.scene.mesh(target_root,"initial",target.model,self.result.initial,INITIAL,opacity=.16,offset=offset)
        self.scene.mesh(target_root,"prediction",target.model,q,offset=offset)
        self.contacts(target_root,intent,points,normals)
        if not self.overlay.value:
            self.scene.caption(target_root,f"Target · {label(target.name)}",center)
        if self.trails.value:
            positions = self.trajectory.intents[:,:,:3].numpy()
            lines = np.stack([positions[:-1],positions[1:]],axis=2).reshape(-1,2,3)
            for panel in [root,target_root]:
                self.server.scene.add_line_segments(panel+"/contact_trails",lines,(200,80,80),line_width=1.)
        # Frame the entire lift, including the initial pose, so playback keeps
        # a stable camera rather than panning at every frame.
        path = self.trajectory.intents[:,:,:3].reshape(-1,3).numpy()
        self.scene.bounds.extend([path.min(0)-.15,path.max(0)+.15,path.min(0)+offset-.15,path.max(0)+offset+.15])
        self.info.content = (
            f"**{label(source.name)} → {label(target.name)}** · Frame **{frame}/{len(self.trajectory.intents)}**\n\n"
            f"Position RMS: **{rms:.2f} mm** · Normal error: **{angle:.1f}°**  \n"
            f"Trajectory mean / final RMS: **{self.result.position_rms_mm.mean():.2f} / {self.result.position_rms_mm[-1]:.2f} mm**  \n"
            f"Initial selected-contact alignment: **{self.alignment['selected_contact_rms_mm']:.2f} mm**  \n"
            f"Clamped coordinates over rollout: **{self.result.clipped_coordinates}**"
        )
        self.notes.content = (
            "Curated source poses lift while fingers move. The same three surface attachments supply positions and outward normals throughout. "
            "The target starts from a curated native grasp, aligned using only the first source contact set (positions and normals). "
            "The independent gripper and stick can be aligned separately. "
            "This initial alignment does not guarantee reachability; no target reference trajectory is used. "
            "Each frame is encoded once. Every decoder call receives the preceding target prediction, including between frames. "
            "Repeated calls and joint-limit clamping follow the controls above. Playback only replays the cached rollout. "
            "Errors use nearest surface-template samples. This is kinematic replay, not a physics rollout or a task-success evaluation.\n\n"
            "Source contacts: "+", ".join(f"{name} (sample {index})" for name,index in zip(self.trajectory.contact_labels,self.trajectory.contact_indices))
        )

    def run(self):
        self.update(reframe=True)
        print(f"Viewer URL: http://{self.cfg.visualization.host}:{self.server.get_port()}",flush=True)
        try:
            while True:
                if self.play.value and not self.busy:
                    if self.frame_index.value >= self.frame_index.max:
                        self.play.value = False
                    else:
                        self.frame_index.value += 1
                time.sleep(1./float(self.fps.value))
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


@hydra.main(version_base="1.2",config_path=str(Path(__file__).resolve().parents[1]/"conf"),config_name="visualize_contact_trajectory")
def main(cfg):
    torch.set_float32_matmul_precision("high")
    TrajectoryViewer(cfg).run()
