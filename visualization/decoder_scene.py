"""Viser scene primitives and UI utilities shared by the decoder explorers."""
from __future__ import annotations

import time
import traceback

import numpy as np
import viser
from omegaconf import OmegaConf

from visualization.decoder_runtime import DecoderRuntime, label


INITIAL = (90, 140, 255)
CONTACT = (255, 150, 60)
GOAL = (245, 220, 65)
PREDICTED = (80, 220, 140)


def mesh_arrays(model, q, links=None):
    payload = model.get_trimesh_q(q.to(model.device), mode="original")
    if links is None:
        mesh = payload["visual"]
        return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)
    vertices, faces, count = [], [], 0
    for name, part in payload["parts"].items():
        if name in links:
            vertices.append(np.asarray(part.vertices, np.float32))
            faces.append(np.asarray(part.faces, np.int32) + count)
            count += len(part.vertices)
    if not vertices:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.concatenate(vertices), np.concatenate(faces)


def normal_arrows(points, normals, length):
    points, normals = np.asarray(points), np.asarray(normals)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    tips = points + length * normals
    axis = np.tile([0., 0., 1.], (len(points), 1))
    axis[np.abs(normals[:, 2]) > .9] = [1., 0., 0.]
    side = np.cross(normals, axis)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-8)
    wings = tips - length * .25 * normals
    return np.concatenate([np.stack([points, tips], 1), np.stack([tips, wings + length * .12 * side], 1),
                           np.stack([tips, wings - length * .12 * side], 1)]).astype(np.float32)


class Scene:
    def __init__(self, server, *, point_size=.004, opacity=.85):
        self.server = server
        self.point_size, self.opacity = point_size, opacity
        self.roots = []
        self.bounds = []

    def clear(self):
        for handle in self.roots:
            handle.remove()
        self.roots.clear()
        self.bounds.clear()

    def panel(self, name, offset=(0., 0., 0.)):
        offset = np.asarray(offset, dtype=float)
        root = self.server.scene.add_frame(name, position=offset, show_axes=False)
        # Recreating a named frame in an atomic update can retain its previous
        # client-side pose. Viser omits zero/default positions on creation, so
        # force an explicit final transform (the queued intermediate is replaced).
        root.position = np.asarray(offset) + [1., 0., 0.]
        root.position = offset
        self.roots.append(root)
        return name

    def mesh(self, root, name, model, q, color=PREDICTED, opacity=None, offset=(0., 0., 0.), links=None):
        vertices, faces = mesh_arrays(model, q, links)
        if len(vertices):
            self.server.scene.add_mesh_simple(f"{root}/{name}", vertices, faces, color=color,
                opacity=self.opacity if opacity is None else opacity)
            self.bounds.extend([vertices.min(0) + offset, vertices.max(0) + offset])

    def contacts(self, root, intent, points=None, normals=None, *, show_normals=True, show_errors=True, normal_length=.02):
        desired = intent.detach().cpu().numpy()
        self.server.scene.add_point_cloud(f"{root}/intent", desired[:, :3], (255, 55, 55), point_size=self.point_size)
        if show_normals:
            self.server.scene.add_line_segments(f"{root}/intent_normals", normal_arrows(desired[:, :3], desired[:, 3:6], normal_length),
                                                (255, 145, 145), line_width=2.)
        if points is not None:
            points = points.detach().cpu().numpy()
            self.server.scene.add_point_cloud(f"{root}/realized", points, (35, 240, 90), point_size=self.point_size)
            if show_errors:
                self.server.scene.add_line_segments(f"{root}/position_errors", np.stack([desired[:, :3], points], 1),
                                                    (250, 210, 70), line_width=1.5)
            if show_normals:
                self.server.scene.add_line_segments(f"{root}/realized_normals", normal_arrows(points, normals.cpu().numpy(), normal_length),
                                                    (150, 255, 170), line_width=2.)

    def caption(self, root, text, center):
        self.server.scene.add_label(f"{root}/caption", text, position=np.asarray(center) + [0, 0, .23],
                                   anchor="bottom-center", font_screen_scale=.8)

    def frame(self, camera):
        if not self.bounds:
            return
        bounds = np.asarray(self.bounds)
        center = (bounds.min(0) + bounds.max(0)) / 2
        radius = max(.18, .68 * float(np.linalg.norm(bounds.max(0) - bounds.min(0))))
        # Viser translates look_at along with position. Set the target last or
        # a reframe pans away from the scene.
        camera.position = center + radius * np.asarray([.3, -.9, .55])
        camera.look_at = center
        if hasattr(camera, "up_direction"):
            camera.up_direction = (0., 0., 1.)
        else:
            camera.up = (0., 0., 1.)


class Viewer:
    def __init__(self, cfg, title, explanation):
        self.runtime = DecoderRuntime(cfg)
        self.cfg = self.runtime.cfg
        self.lock = self.runtime.lock
        self.server = viser.ViserServer(host=str(self.cfg.visualization.host), port=int(self.cfg.visualization.port))
        self.server.scene.set_up_direction("+z")
        self.server.gui.configure_theme(control_layout="floating", control_width="large", dark_mode=True)
        self.server.gui.add_markdown(f"## {title}\n{explanation}")
        self.server.gui.add_markdown("Blue: initial · Green: predicted configuration  \n"
            "Red: interaction intent · Green points: realization  \n"
            "Kinematic motion preview")
        self.status = self.server.gui.add_markdown("")
        self.scene = Scene(self.server, point_size=float(self.cfg.visualization.point_size), opacity=float(self.cfg.visualization.hand_opacity))
        self.frame_needed = True
        self.busy = False
        self.name_by_label = {label(n): n for n in self.runtime.names}
        self.server.on_client_connect(self._client_connected)

    def _client_connected(self, client):
        # Initial camera messages can arrive while the browser is connecting or
        # while examples are still being generated. Frame once the client has
        # reported its camera; reconnects must also see the current scene.
        with self.lock, client.atomic():
            self.scene.frame(client.camera)

    def embodiment_selector(self, title="Embodiment"):
        initial = str(OmegaConf.select(self.cfg, "visualization.robot_name", default=self.runtime.names[0]))
        if initial not in self.runtime.names:
            initial = self.runtime.names[0]
        return self.server.gui.add_dropdown(title, list(self.name_by_label), initial_value=label(initial))

    def example_slider(self, title="Example"):
        return self.server.gui.add_slider(title, min=0, max=max(1, self.runtime.num_samples-1), step=1,
                                          initial_value=0, disabled=self.runtime.num_samples == 1)

    def display_controls(self):
        with self.server.gui.add_folder("Display", expand_by_default=False):
            self.normals = self.server.gui.add_checkbox("Surface normals", initial_value=True)
            self.error_lines = self.server.gui.add_checkbox("Position error lines", initial_value=True)
            self.normal_length = self.server.gui.add_slider("Normal length (m)", min=.005, max=.05, step=.005, initial_value=.02)
        self.fraction = self.server.gui.add_slider("Motion preview", min=0., max=1., step=.02, initial_value=1.)
        self.play = self.server.gui.add_checkbox("Animate", initial_value=False)
        self.server.gui.add_button("Frame view").on_click(lambda _: self.frame())
        if not hasattr(self, "info"):
            self.info = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Interpretation", expand_by_default=False):
            self.notes = self.server.gui.add_markdown("")
        for control in (self.normals, self.error_lines, self.normal_length, self.fraction):
            control.on_update(lambda _: self.update())

    def contacts(self, root, intent, points=None, normals=None):
        self.scene.contacts(root, intent, points, normals, show_normals=self.normals.value,
                            show_errors=self.error_lines.value, normal_length=self.normal_length.value)

    def frame(self):
        with self.lock:
            self.scene.frame(self.server.initial_camera)
            for client in self.server.get_clients().values():
                with client.atomic():
                    self.scene.frame(client.camera)

    def update(self, *, reframe=False):
        with self.lock:
            self.busy = True
            try:
                self.status.content = "Loading / updating…"
                self.prepare()
                with self.server.atomic():
                    self.scene.clear()
                    self.render()
                self.status.content = ""
                if reframe or self.frame_needed:
                    self.frame()
                    self.frame_needed = False
            except Exception as exc:
                self.play.value = False
                with self.server.atomic():
                    self.scene.clear()
                    self.info.content = ""
                self.status.content = f"**Unable to display this selection:** {exc}  \nChoose another embodiment or example."
                traceback.print_exc()
            finally:
                self.busy = False

    def run(self):
        self.update(reframe=True)
        print(f"Viewer URL: http://{self.cfg.visualization.host}:{self.server.get_port()}", flush=True)
        try:
            while True:
                if self.play.value and not self.busy:
                    self.fraction.value = 0. if self.fraction.value >= 1. else min(1., round(self.fraction.value + .04, 2))
                time.sleep(.08)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
