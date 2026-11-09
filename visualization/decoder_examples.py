"""Single-embodiment synthetic examples with in-browser embodiment selection."""
import hydra
import torch
from pathlib import Path

from visualization.decoder_runtime import attached_contacts, contact_errors, interpolate_configuration, label, pose_groups
from visualization.decoder_scene import Viewer, INITIAL, CONTACT, GOAL


VIEWS = ["All configurations", "Goal and prediction", "Contact configuration and prediction",
         "Initial, goal and prediction", "Contact links only", "Initial to prediction"]


class ExampleViewer(Viewer):
    def __init__(self, cfg):
        super().__init__(cfg, "KITE · Kinematic decoder", "Explore synthetic interaction intents and their decoded goal configurations.")
        self.embodiment = self.embodiment_selector()
        self.sample = self.example_slider()
        initial_mode = min(5, max(0, int(self.cfg.visualization.initial_vis_mode)))
        self.mode = self.server.gui.add_dropdown("View", VIEWS, initial_value=VIEWS[initial_mode])
        self.center_origin = self.server.gui.add_checkbox("Center at origin", initial_value=False)
        self.display_controls()
        self.embodiment.on_update(lambda _: self.update(reframe=True))
        self.sample.on_update(lambda _: self.update(reframe=not self.center_origin.value))
        self.center_origin.on_update(lambda _: self.update(reframe=True))
        self.mode.on_update(lambda _: self.update())

    def prepare(self):
        self.examples = self.runtime.get(self.name_by_label[self.embodiment.value])
        self.index = min(int(self.sample.value), self.examples.count - 1)

    def render(self):
        ex, i = self.examples, self.index
        q_init = ex.configuration("q1_padded", i)
        q_contact = ex.configuration("q2_padded", i)
        q_goal = ex.configuration("action", i)
        q_pred = ex.prediction[i, :ex.model.dof]
        q_preview = interpolate_configuration(ex.model, q_init, q_pred, self.fraction.value)
        intent = ex.intent(i)
        points, normals = attached_contacts(ex, i, q_preview)
        rms, angle = contact_errors(intent, points, normals)
        # One display-only translation keeps configurations, intent, and errors
        # aligned. Composite embodiments retain their relative component poses.
        offset = (0., 0., 0.)
        if self.center_origin.value:
            bases = torch.stack([q_init[group["translation_indices"]] for group in pose_groups(ex.model)])
            offset = (-bases.mean(0)).numpy()
        root = self.scene.panel("/example", offset=offset)
        mode = VIEWS.index(self.mode.value)
        links = None
        if mode == 4 and not hasattr(ex.model, "_mano_weights"):
            valid = ex.batch["contact_valid_mask"][i].bool()
            ids = ex.batch["contact_link_indices"][i, valid].tolist()
            links = {ex.model.mesh_link_names[j] for j in ids}
        if mode in {0, 3, 5}:
            self.scene.mesh(root, "initial", ex.model, q_init, INITIAL, opacity=.18 if mode == 5 else .45, offset=offset)
        if mode in {0, 2}:
            self.scene.mesh(root, "contact_configuration", ex.model, q_contact, CONTACT, opacity=.45, offset=offset)
        if mode in {0, 1, 3, 4}:
            self.scene.mesh(root, "goal", ex.model, q_goal, GOAL, opacity=.4, links=links, offset=offset)
        self.scene.mesh(root, "prediction", ex.model, q_preview, links=links, offset=offset)
        self.contacts(root, intent, points, normals)
        self.info.content = (
            f"**{label(ex.name)}** · Example **{i+1}/{ex.count}**  \n"
            f"Contact position RMS: **{rms:.2f} mm** · Mean normal error: **{angle:.1f}°**"
        )
        self.notes.content = (
            "Errors use the selected template points at the displayed configuration. "
            "Preview = 1 shows the decoder output.  \n"
            "Orange: contact configuration · Yellow: goal configuration. "
            "The goal retains the initial values of joints unrelated to the selected contacts."
            + ("  \nCentered display: the initial base position is at the origin "
               "(the midpoint of the bases for composite embodiments)." if self.center_origin.value else "")
        )


@hydra.main(version_base="1.2", config_path=str(Path(__file__).resolve().parents[1] / "conf"), config_name="visualize_decoder")
def main(cfg):
    torch.set_float32_matmul_precision("high")
    ExampleViewer(cfg).run()
