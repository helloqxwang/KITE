"""Decode one shared latent intent on independently selected target embodiments."""
from pathlib import Path

import hydra
import numpy as np
import torch

from visualization.decoder_runtime import aligned_initial, closest_initial, contact_errors, interpolate_configuration, label, nearest_surface
from visualization.decoder_scene import Viewer, INITIAL, CONTACT


class CrossEmbodimentViewer(Viewer):
    def __init__(self, cfg):
        super().__init__(cfg, "KITE · Cross-embodiment decoding", "One interaction intent → one latent intent → selected kinematic decoders.")
        self.source = self.embodiment_selector("Source embodiment")
        self.sample = self.example_slider("Source intent example")
        self.info = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Target embodiments"):
            self.targets = {name: self.server.gui.add_checkbox(label(name), initial_value=name != "mano_right")
                            for name in self.runtime.names}
        self.initialization = self.server.gui.add_dropdown("Target initialization",
            ["Closest centered example", "Choose example"], initial_value="Closest centered example")
        self.initial_sample = self.example_slider("Target initial example")
        self.display_controls()
        self._key = None
        for control in (self.source, self.sample, self.initialization, self.initial_sample, *self.targets.values()):
            control.on_update(lambda _: self.update(reframe=True))

    def prepare(self):
        source_name = self.name_by_label[self.source.value]
        source = self.runtime.get(source_name)
        index = min(int(self.sample.value), source.count-1)
        selected = tuple(name for name, control in self.targets.items() if control.value)
        key = (source_name, index, selected, self.initialization.value, int(self.initial_sample.value))
        if key == self._key:
            return
        intent = source.intent(index)
        latent = self.runtime.encode(source_name, intent)
        decoded, failures = [], []
        for name in selected:
            try:
                ex = self.runtime.get(name)
                if self.initialization.value == "Closest centered example":
                    initial_index, q_init = closest_initial(ex, intent)
                else:
                    initial_index = min(int(self.initial_sample.value), ex.count-1)
                    q_init = aligned_initial(ex, initial_index, intent)
                q_pred = self.runtime.decode(name, latent, q_init)
                decoded.append((ex, initial_index, q_init, q_pred))
            except Exception as exc:
                failures.append(f"{label(name)}: {exc}")
        self.source_examples, self.source_index = source, index
        self.intent, self.latent, self.decoded, self.failures = intent, latent, decoded, failures
        self._key = key

    def render(self):
        ex, i = self.source_examples, self.source_index
        center = self.intent[:, :3].mean(0).numpy()
        root = self.scene.panel("/source")
        self.scene.mesh(root, "contact_configuration", ex.model, ex.configuration("q2_padded", i), CONTACT)
        self.contacts(root, self.intent)
        self.scene.caption(root, f"Source intent · {label(ex.name)}", center)
        lines = ["| Target embodiment | Initial example | RMS (mm) | Normal (°) |",
                 "|---|---:|---:|---:|"]
        for n, (target, initial_index, q_init, q_pred) in enumerate(self.decoded, 1):
            offset = np.array([(n % 3) * .65, -(n // 3) * .65, 0.])
            root = self.scene.panel(f"/target_{target.name}", offset)
            q = interpolate_configuration(target.model, q_init, q_pred, self.fraction.value)
            self.scene.mesh(root, "initial", target.model, q_init, INITIAL, opacity=.18, offset=offset)
            self.scene.mesh(root, "prediction", target.model, q, offset=offset)
            points, normals, _ = nearest_surface(target.model, q, self.intent)
            rms, angle = contact_errors(self.intent, points, normals)
            self.contacts(root, self.intent, points, normals)
            self.scene.caption(root, label(target.name), center)
            lines.append(f"| {label(target.name)} | {initial_index} | {rms:.2f} | {angle:.1f} |")
        selected_count = sum(control.value for control in self.targets.values())
        summary = f"**{len(self.decoded)}/{selected_count} target embodiments displayed.**"
        if not selected_count:
            summary += " Select targets below."
        self.info.content = summary + "\n\n" + "\n".join(lines) + "\n\n" + "\n\n".join(f"**Unavailable:** {f}" for f in self.failures)
        self.notes.content = (
            "Errors use **nearest surface-template samples**, not source-to-target link correspondences. "
            "Normals are measured at those same samples. Panel spacing affects display only; every decoder receives the same latent intent. "
            "Initial configurations are translated to align contact-set centroids; the default selects a nearby centered pattern. "
            "This position-only heuristic does not fit normals, orientation, or joint posture, and cannot guarantee reachability. "
            "Each prediction uses one decoder call. Motion preview interpolates configurations; this is not a physics rollout."
        )


@hydra.main(version_base="1.2", config_path=str(Path(__file__).resolve().parents[1] / "conf"), config_name="visualize_cross_embodiment")
def main(cfg):
    torch.set_float32_matmul_precision("high")
    CrossEmbodimentViewer(cfg).run()
