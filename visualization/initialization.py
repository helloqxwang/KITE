"""Hold the latent intent fixed and inspect dependence on the initial configuration."""
from pathlib import Path

import hydra
import numpy as np
import torch

from visualization.decoder_runtime import aligned_initial, contact_errors, interpolate_configuration, label, nearest_surface, perturb_initial
from visualization.decoder_scene import Viewer, INITIAL


class InitializationViewer(Viewer):
    def __init__(self, cfg):
        super().__init__(cfg, "KITE · Initialization sensitivity", "Keep an interaction intent fixed while changing the decoder's initial configuration.")
        self.embodiment = self.embodiment_selector()
        self.sample = self.example_slider("Intent example")
        self.use_original = self.server.gui.add_checkbox("Use intent example's initial configuration", initial_value=True)
        self.initial_sample = self.example_slider("Initial configuration example")
        self.align = self.server.gui.add_checkbox("Align initial contact-set center", initial_value=True)
        self.info = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Initial configuration perturbation"):
            self.translation = [self.server.gui.add_slider(f"Translate {axis} (cm)", min=-30., max=30., step=1., initial_value=0.) for axis in "XYZ"]
            self.rotation = [self.server.gui.add_slider(f"Rotate {axis} (°)", min=-180., max=180., step=5., initial_value=0.) for axis in "XYZ"]
            self.joints = self.server.gui.add_slider("Joint noise (% of range)", min=0., max=30., step=1., initial_value=0.)
            self.seed = self.server.gui.add_slider("Perturbation seed", min=0, max=1000, step=1, initial_value=0)
            reset = self.server.gui.add_button("Reset initial configuration")
        self.display_controls()
        self._intent_key = self._prediction_key = None
        self._resetting = False
        self._sync_controls()
        for control in (self.embodiment, self.sample, self.use_original, self.initial_sample, self.align, *self.translation, *self.rotation, self.joints, self.seed):
            control.on_update(self._changed)
        reset.on_click(self._reset)

    def _sync_controls(self):
        self.initial_sample.disabled = self.use_original.value or self.runtime.num_samples == 1
        self.align.disabled = self.use_original.value

    def _changed(self, _):
        if not self._resetting:
            self._sync_controls()
            self.update(reframe=True)

    def _reset(self, _):
        with self.lock:
            self._resetting = True
            with self.server.atomic():
                self.use_original.value = True
                self.initial_sample.value = self.sample.value
                self.align.value = True
                for control in (*self.translation, *self.rotation, self.joints, self.seed):
                    control.value = 0
                self.play.value = False
                self.fraction.value = 1.
                self._sync_controls()
            self._resetting = False
        self.update(reframe=True)

    def prepare(self):
        name = self.name_by_label[self.embodiment.value]
        ex = self.runtime.get(name)
        i = min(int(self.sample.value), ex.count-1)
        intent_key = (name, i)
        if intent_key != self._intent_key:
            self.intent = ex.intent(i)
            self.latent = self.runtime.encode(name, self.intent)
            self.baseline_initial = ex.configuration("q1_padded", i)
            self.baseline_prediction = self.runtime.decode(name, self.latent, self.baseline_initial)
            self._intent_key = intent_key
        initial_index = i if self.use_original.value else min(int(self.initial_sample.value), ex.count-1)
        translation = tuple(float(c.value)/100. for c in self.translation)
        rotation = tuple(float(c.value) for c in self.rotation)
        key = (intent_key, initial_index, self.use_original.value, self.align.value, translation, rotation, self.joints.value, self.seed.value)
        if key != self._prediction_key:
            initial = (aligned_initial(ex, initial_index, self.intent)
                       if self.align.value and not self.use_original.value else ex.configuration("q1_padded", initial_index))
            initial = perturb_initial(ex.model, initial, translation=translation, rotation_deg=rotation,
                                      joint_fraction=float(self.joints.value)/100., seed=int(self.seed.value))
            self.changed_prediction = (self.baseline_prediction.clone() if torch.equal(initial, self.baseline_initial)
                                       else self.runtime.decode(name, self.latent, initial))
            self.changed_initial = initial
            self._prediction_key = key
        self.examples = ex

    def render(self):
        ex = self.examples
        center = self.intent[:, :3].mean(0).numpy()
        rows = ["| Initial configuration | RMS (mm) | Normal (°) |", "|---|---:|---:|"]
        for n, (title, q_init, q_pred) in enumerate([
            ("Original", self.baseline_initial, self.baseline_prediction),
            ("Changed", self.changed_initial, self.changed_prediction),
        ]):
            offset = np.array([.65*n, 0., 0.])
            root = self.scene.panel(f"/initial_{n}", offset)
            q = interpolate_configuration(ex.model, q_init, q_pred, self.fraction.value)
            self.scene.mesh(root, "initial", ex.model, q_init, INITIAL, opacity=.22, offset=offset)
            self.scene.mesh(root, "prediction", ex.model, q, offset=offset)
            points, normals, indices = nearest_surface(ex.model, q, self.intent)
            rms, angle = contact_errors(self.intent, points, normals)
            self.contacts(root, self.intent, points, normals)
            self.scene.caption(root, f"{label(ex.name)} · {title}", center)
            rows.append(f"| {title} | {rms:.2f} | {angle:.1f} |")
            if not hasattr(ex.model, "_mano_weights"):
                link_ids = ex.model.surface_template_link_indices[indices.to(ex.model.device)].cpu().unique().tolist()
                links = ", ".join(ex.model.mesh_link_names[j] for j in link_ids)
                if n == 1:
                    changed_links = links
        self.info.content = f"**Fixed intent: example {int(self.sample.value)}**\n\n" + "\n".join(rows)
        self.notes.content = (
            "The interaction intent and its encoded latent stay fixed when the initial configuration changes. "
            "Errors use **nearest surface-template samples** and their normals; they do not prescribe finger correspondences. "
            "Translations and rotations move all free components together. Joint noise is deterministic and clamped to joint limits. "
            "Rotation controls use intrinsic XYZ in world coordinates, matching the kinematic models. "
            "Errors describe the displayed preview pose; preview = 1 shows the decoder prediction. "
            "Predictions use a single decoder call, without output clipping or iterative refinement."
        )
        if not hasattr(ex.model, "_mano_weights"):
            self.notes.content += f"\n\nNearest links (changed): {changed_links}"


@hydra.main(version_base="1.2", config_path=str(Path(__file__).resolve().parents[1] / "conf"), config_name="visualize_decoder_initialization")
def main(cfg):
    torch.set_float32_matmul_precision("high")
    InitializationViewer(cfg).run()
