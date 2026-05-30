"""Build a compact terrain with stable semantics and improved visuals (less flicker)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import random
import tempfile

import numpy as np

import pybullet as p
import pybullet_data


@dataclass
class TerrainSceneConfig:
    seed: int = 42
    # Playable / semantic region (trees, rocks, roads, water, drone clamp).
    terrain_half_extent_m: float = 20.0
    # Huge flat apron so chase cam does not reveal a square edge.
    visual_ground_half_extent_m: float = 320.0
    safe_clearing_radius_m: float = 9.0
    n_trees: int = 18
    n_rocks: int = 28
    water_patch: bool = True
    water_half_size_m: float = 4.5
    add_roads: bool = True
    road_half_width_m: float = 1.7
    rock_scale_min: float = 0.35
    rock_scale_max: float = 0.85
    tree_trunk_radius_m: float = 0.16
    tree_trunk_height_m: float = 1.35
    ground_height_noise_m: float = 0.04
    prefer_opengl_render: bool = True
    grass_texture_size: int = 2048
    # Tiled visuals can z-fight and shimmer in OpenGL; default is one high-res slab.
    ground_use_tiled_visuals: bool = False
    ground_tiles_per_axis: int = 14
    # Urban clutter — unsafe in scene-semantic fusion (see safety_mapping: building, car).
    n_buildings: int = 5
    n_parked_cars: int = 10


@dataclass
class PyBulletLandingScene:
    client: int
    drone_id: int
    cfg: TerrainSceneConfig
    terrain_z: float = 0.0
    obstacle_body_ids: list[int] = field(default_factory=list)
    tree_crowns: list[tuple[float, float, float]] = field(default_factory=list)
    rock_patches: list[tuple[float, float, float]] = field(default_factory=list)
    water_patches: list[tuple[float, float, float]] = field(default_factory=list)
    road_patches: list[tuple[float, float, float, float]] = field(default_factory=list)
    building_footprints: list[tuple[float, float, float]] = field(default_factory=list)
    car_patches: list[tuple[float, float, float]] = field(default_factory=list)
    _renderer: int = field(default=p.ER_TINY_RENDERER, repr=False)
    render_far_plane: float = 900.0

    @staticmethod
    def _pick_renderer(*, prefer_opengl: bool, far_plane: float) -> int:
        if not prefer_opengl:
            return p.ER_TINY_RENDERER
        view = p.computeViewMatrix([0.0, -8.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        proj = p.computeProjectionMatrixFOV(
            fov=55.0, aspect=1.0, nearVal=0.05, farVal=float(max(120.0, far_plane))
        )
        for renderer in (p.ER_BULLET_HARDWARE_OPENGL, p.ER_TINY_RENDERER):
            _, _, rgb, _, _ = p.getCameraImage(
                width=48,
                height=48,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=renderer,
            )
            arr = np.asarray(rgb, dtype=np.uint8).reshape((48, 48, 4))
            if int(arr[..., :3].max()) > 8:
                return renderer
        return p.ER_TINY_RENDERER

    @staticmethod
    def _write_grass_texture_png(path: str, size: int, seed: int) -> None:
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python is required for grass textures") from None
        rng = np.random.default_rng(seed)
        # Original procedural turf: saturated green noise + smooth blobs + vignette (no photo texture).
        base = np.array([52.0, 118.0, 62.0], dtype=np.float32)
        n = rng.normal(0.0, 14.0, (size, size, 3))
        g = np.clip(base + n, 0.0, 255.0).astype(np.uint8)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        blob = 18.0 * np.sin(xx * 0.04 + yy * 0.03) + 12.0 * np.sin(xx * 0.11)
        g = np.clip(g.astype(np.float32) + blob[..., None] * np.array([0.6, 0.9, 0.5]), 0, 255).astype(np.uint8)
        cx = cy = size * 0.5
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size * 0.71)
        vig = np.clip(1.0 - 0.35 * d, 0.65, 1.0)
        g = (g.astype(np.float32) * vig[..., None]).astype(np.uint8)
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=0.85, sigmaY=0.85)
        cv2.imwrite(path, cv2.cvtColor(g, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _write_facade_texture_png(path: str, size: int, seed: int) -> None:
        """Procedural modern curtain-wall / concrete panel facade (BGR on disk)."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python is required for facade textures") from None
        rng = np.random.default_rng(seed)
        s = int(size)
        base = np.full((s, s, 3), (165, 175, 188), dtype=np.uint8)
        # Vertical mullions (cool gray metal)
        ncol = 7
        col_w = s // ncol
        for c in range(ncol + 2):
            x0 = c * col_w
            cv2.rectangle(base, (x0, 0), (min(x0 + 3, s - 1), s - 1), (120, 128, 138), -1)
        # Per-panel slight tone + horizontal spandrels
        for c in range(ncol):
            x0 = c * col_w + 4
            x1 = min((c + 1) * col_w - 2, s - 1)
            shade = int(rng.integers(-12, 13))
            tint = (
                int(np.clip(160 + shade, 140, 200)),
                int(np.clip(168 + shade, 148, 205)),
                int(np.clip(178 + shade, 155, 210)),
            )
            cv2.rectangle(base, (x0, 0), (x1, s - 1), tint, -1)
            for y in range(0, s, max(14, s // 18)):
                cv2.rectangle(base, (x0, y), (x1, min(y + 4, s - 1)), (95, 102, 112), -1)
        # Glass bands (bluish reflective)
        nb = int(rng.integers(4, 8))
        for _ in range(nb):
            x0 = int(rng.integers(6, s - 50))
            y0 = int(rng.integers(6, s - 40))
            bw = int(rng.integers(28, 55))
            bh = int(rng.integers(18, 36))
            cv2.rectangle(base, (x0, y0), (x0 + bw, y0 + bh), (95, 118, 145), -1)
            cv2.rectangle(base, (x0 + 2, y0 + 2), (x0 + bw - 2, y0 + bh - 2), (130, 168, 205), 1)
            cv2.line(base, (x0 + 4, y0 + 4), (x0 + bw - 4, y0 + bh - 4), (180, 210, 230), 1)
        img = cv2.GaussianBlur(base, (0, 0), sigmaX=0.65, sigmaY=0.65)
        cv2.imwrite(path, img)

    @classmethod
    def build(cls, cfg: TerrainSceneConfig, *, use_gui: bool = False) -> PyBulletLandingScene:
        rng = random.Random(cfg.seed)
        mode = p.GUI if use_gui else p.DIRECT
        client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # No plane.urdf — its checkerboard moirés badly under TinyRenderer / video compression.
        p.resetDebugVisualizerCamera(cameraDistance=35, cameraYaw=40, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])

        playable = float(cfg.terrain_half_extent_m)
        vhalf = float(max(cfg.visual_ground_half_extent_m, playable))
        render_far = max(520.0, vhalf * 2.6)
        renderer = cls._pick_renderer(prefer_opengl=cfg.prefer_opengl_render, far_plane=render_far)
        rname = "OpenGL" if renderer == p.ER_BULLET_HARDWARE_OPENGL else "TinyRenderer"
        print(f"[scene] PyBullet camera renderer: {rname}", flush=True)

        ground_half = [vhalf, vhalf, 0.08]
        ground_z_center = -ground_half[2]  # top surface at z=0
        ground_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=ground_half)

        grass_tex_uid: int | None = None
        tex_path: str | None = None
        try:
            fd, tex_path = tempfile.mkstemp(prefix="yoloslam_grass_", suffix=".png")
            os.close(fd)
            cls._write_grass_texture_png(tex_path, int(cfg.grass_texture_size), int(cfg.seed))
            grass_tex_uid = p.loadTexture(tex_path)
        except Exception:
            grass_tex_uid = None
        finally:
            if tex_path is not None:
                try:
                    os.remove(tex_path)
                except OSError:
                    pass

        if cfg.ground_use_tiled_visuals:
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=ground_col,
                baseVisualShapeIndex=-1,
                basePosition=[0.0, 0.0, ground_z_center],
            )
            nt = max(4, int(cfg.ground_tiles_per_axis))
            t_half = float(vhalf) / float(nt)
            tile_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[t_half, t_half, 0.08],
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
            )
            for i in range(nt):
                for j in range(nt):
                    cx = -vhalf + (2 * i + 1) * t_half
                    cy = -vhalf + (2 * j + 1) * t_half
                    tid = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=-1,
                        baseVisualShapeIndex=tile_vis,
                        basePosition=[cx, cy, ground_z_center],
                    )
                    if grass_tex_uid is not None:
                        try:
                            p.changeVisualShape(tid, -1, textureUniqueId=grass_tex_uid)
                        except Exception:
                            pass
        else:
            ground_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=ground_half,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
            )
            ground_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=ground_col,
                baseVisualShapeIndex=ground_vis,
                basePosition=[0.0, 0.0, ground_z_center],
            )
            if grass_tex_uid is not None:
                try:
                    p.changeVisualShape(ground_id, -1, textureUniqueId=grass_tex_uid)
                except Exception:
                    pass

        obstacle_ids: list[int] = []
        tree_crowns: list[tuple[float, float, float]] = []
        rock_patches: list[tuple[float, float, float]] = []
        water_patches: list[tuple[float, float, float]] = []
        road_patches: list[tuple[float, float, float, float]] = []
        building_footprints: list[tuple[float, float, float]] = []
        car_patches: list[tuple[float, float, float]] = []

        if cfg.water_patch:
            wh = cfg.water_half_size_m
            wx = playable - wh - 1.2
            wy = -playable + wh + 1.0
            wz = 0.03
            water_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh, wh, 0.02])
            water_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[wh, wh, 0.02],
                rgbaColor=[0.10, 0.32, 0.78, 0.96],
            )
            wid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=water_col,
                baseVisualShapeIndex=water_vis,
                basePosition=[wx, wy, wz],
            )
            obstacle_ids.append(wid)
            water_patches.append((wx, wy, wh))

        if cfg.add_roads:
            rw = cfg.road_half_width_m
            road_z = 0.015
            hx, hy = playable * 0.92, rw
            road_col_h = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, 0.012])
            road_vis_h = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, 0.012],
                rgbaColor=[0.28, 0.28, 0.29, 1.0],
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=road_col_h,
                baseVisualShapeIndex=road_vis_h,
                basePosition=[0.0, -playable * 0.25, road_z],
            )
            road_patches.append((0.0, -playable * 0.25, hx, hy))
            vx, vy = rw, playable * 0.78
            road_col_v = p.createCollisionShape(p.GEOM_BOX, halfExtents=[vx, vy, 0.012])
            road_vis_v = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[vx, vy, 0.012],
                rgbaColor=[0.30, 0.30, 0.31, 1.0],
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=road_col_v,
                baseVisualShapeIndex=road_vis_v,
                basePosition=[playable * 0.12, 0.0, road_z + 0.002],
            )
            road_patches.append((playable * 0.12, 0.0, vx, vy))

        for _bi in range(int(cfg.n_buildings)):
            ang = 2.0 * math.pi * (_bi / max(1, int(cfg.n_buildings))) + rng.uniform(-0.4, 0.4)
            rad = rng.uniform(max(cfg.safe_clearing_radius_m + 5.5, playable * 0.52), playable - 2.5)
            bx = math.cos(ang) * rad
            by = math.sin(ang) * rad
            b_ids, b_fp = _spawn_building(bx, by, rng)
            obstacle_ids.extend(b_ids)
            building_footprints.append(b_fp)
            for bid in b_ids:
                _try_apply_facade_texture(bid, rng.randint(0, 1_000_000))

        car_colors: list[list[float]] = [
            [0.75, 0.12, 0.10],
            [0.12, 0.24, 0.72],
            [0.86, 0.87, 0.90],
            [0.07, 0.07, 0.08],
            [0.92, 0.78, 0.18],
        ]
        n_cars = int(cfg.n_parked_cars)
        cars_placed = 0
        car_xy_placed: list[tuple[float, float]] = []
        min_car_sep = 2.95
        min_car_from_building = 2.15
        y_h = -playable * 0.25
        x_v = playable * 0.12

        def _car_spot_clear(cx: float, cy: float) -> bool:
            if math.hypot(cx, cy) < cfg.safe_clearing_radius_m + 1.3:
                return False
            for ox, oy in car_xy_placed:
                if math.hypot(cx - ox, cy - oy) < min_car_sep:
                    return False
            for bx, by, br in building_footprints:
                if math.hypot(cx - bx, cy - by) < float(br) + min_car_from_building:
                    return False
            return True

        def _place_car(cx: float, cy: float, yaw_c: float) -> None:
            nonlocal cars_placed
            if cars_placed >= n_cars:
                return
            if not _car_spot_clear(cx, cy):
                return
            rgba = rng.choice(car_colors)
            cid, cpat = _spawn_parked_car(cx, cy, yaw_c, rgba)
            obstacle_ids.append(cid)
            car_patches.append(cpat)
            car_xy_placed.append((cx, cy))
            cars_placed += 1

        for _ in range(max(n_cars, 1) * 3):
            if cars_placed >= n_cars:
                break
            sx = rng.uniform(-playable * 0.74, playable * 0.74)
            _place_car(sx, y_h + rng.uniform(-0.65, 0.65), 0.0 if rng.random() > 0.5 else math.pi)
        for _ in range(max(n_cars, 1) * 3):
            if cars_placed >= n_cars:
                break
            sy = rng.uniform(-playable * 0.72, playable * 0.72)
            _place_car(x_v + rng.uniform(-0.55, 0.55), sy, math.pi / 2.0 if rng.random() > 0.5 else -math.pi / 2.0)
        while cars_placed < n_cars:
            ang = rng.uniform(0.0, 2.0 * math.pi)
            rad = rng.uniform(cfg.safe_clearing_radius_m + 2.0, playable - 2.0)
            _place_car(math.cos(ang) * rad, math.sin(ang) * rad, rng.uniform(-math.pi, math.pi))

        for _ in range(cfg.n_rocks):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            rad = rng.uniform(cfg.safe_clearing_radius_m + 0.6, playable - 1.2)
            px = math.cos(ang) * rad
            py = math.sin(ang) * rad
            sx = rng.uniform(cfg.rock_scale_min, cfg.rock_scale_max)
            sy = rng.uniform(cfg.rock_scale_min, cfg.rock_scale_max)
            sz = rng.uniform(cfg.rock_scale_min * 0.55, cfg.rock_scale_max * 0.65)
            rz = sz * 0.5 + rng.uniform(0.0, cfg.ground_height_noise_m)
            rock_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx * 0.5, sy * 0.5, sz * 0.5])
            rock_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[sx * 0.5, sy * 0.5, sz * 0.5],
                rgbaColor=[0.40, 0.30, 0.22, 1.0],
            )
            yaw = rng.uniform(0.0, math.pi)
            orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
            rid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=rock_col,
                baseVisualShapeIndex=rock_vis,
                basePosition=[px, py, rz],
                baseOrientation=orn,
            )
            obstacle_ids.append(rid)
            rock_patches.append((px, py, 0.5 * max(sx, sy)))

        for _ in range(cfg.n_trees):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            rad = rng.uniform(cfg.safe_clearing_radius_m + 0.9, playable - 1.5)
            px = math.cos(ang) * rad
            py = math.sin(ang) * rad
            crown_r, t_ids = _spawn_realistic_tree(
                px,
                py,
                rng,
                trunk_r=cfg.tree_trunk_radius_m,
                trunk_h=cfg.tree_trunk_height_m,
                scale=rng.uniform(0.85, 1.15),
            )
            obstacle_ids.extend(t_ids)
            tree_crowns.append((px, py, crown_r))

        drone_id = _spawn_x500_style_quad()

        inst = cls(
            client=client,
            drone_id=drone_id,
            cfg=cfg,
            obstacle_body_ids=obstacle_ids,
            tree_crowns=tree_crowns,
            rock_patches=rock_patches,
            water_patches=water_patches,
            road_patches=road_patches,
            building_footprints=building_footprints,
            car_patches=car_patches,
            _renderer=renderer,
            render_far_plane=render_far,
        )
        return inst

    def set_drone_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(self.drone_id, [x, y, z], quat)

    def render_nadir_rgb(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        width: int,
        height: int,
        *,
        fov_deg: float = 65.0,
        near: float = 0.08,
        far: float | None = None,
        renderer: int | None = None,
    ) -> np.ndarray:
        """Return BGR uint8 image from a downward-facing camera."""
        bgr, _ = self.render_nadir_rgbd(
            x,
            y,
            z,
            yaw,
            width,
            height,
            fov_deg=fov_deg,
            near=near,
            far=far,
            renderer=renderer,
        )
        return bgr

    def render_nadir_rgbd(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        width: int,
        height: int,
        *,
        fov_deg: float = 65.0,
        near: float = 0.08,
        far: float | None = None,
        renderer: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (BGR uint8, depth float32 meters) from a downward-facing camera."""
        far_val = float(self.render_far_plane if far is None else far)
        # Keep the virtual camera above the ground slab when landed (avoids clipping / sky leak).
        z_eye = max(float(z), 0.52)
        eye = np.array([x, y, z_eye], dtype=float)
        target = np.array([x, y, 0.0], dtype=float)
        up = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        aspect = float(width) / float(height)
        near_eff = max(float(near), 0.12) if z_eye < 0.85 else float(near)
        proj = p.computeProjectionMatrixFOV(fov=fov_deg, aspect=aspect, nearVal=near_eff, farVal=far_val)
        r = int(self._renderer if renderer is None else renderer)
        _, _, rgb, depth_buf, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=r,
        )
        rgb = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        bgr = rgb[:, :, ::-1].copy()
        depth_arr = np.array(depth_buf, dtype=np.float32).reshape((height, width))
        depth_m = (far_val * near_eff) / (far_val - (far_val - near_eff) * np.clip(depth_arr, 0.0, 1.0))
        depth_m[~np.isfinite(depth_m)] = np.nan
        depth_m[depth_m <= near_eff] = np.nan
        depth_m[depth_m >= far_val * 0.995] = np.nan
        return bgr, depth_m

    def render_chase_rgb(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        width: int,
        height: int,
        *,
        fov_deg: float = 58.0,
        distance_m: float = 14.0,
        height_above_drone_m: float = 5.0,
        target_height_frac: float = 0.88,
        near: float = 0.12,
        far: float | None = None,
        renderer: int | None = None,
    ) -> np.ndarray:
        """Third-person camera behind the drone, looking toward the UAV and terrain."""
        far_val = float(self.render_far_plane if far is None else far)
        bx = -float(distance_m) * math.cos(yaw)
        by = -float(distance_m) * math.sin(yaw)
        eye = np.array([x + bx, y + by, z + float(height_above_drone_m)], dtype=float)
        tz = max(0.12, float(z) * float(target_height_frac))
        target = np.array([x, y, tz], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        aspect = float(width) / float(height)
        proj = p.computeProjectionMatrixFOV(fov=fov_deg, aspect=aspect, nearVal=near, farVal=far_val)
        r = int(self._renderer if renderer is None else renderer)
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=r,
        )
        rgb = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        return rgb[:, :, ::-1].copy()

    def project_world_to_chase_pixel(
        self,
        wx: float,
        wy: float,
        wz: float,
        drone_x: float,
        drone_y: float,
        drone_z: float,
        yaw: float,
        width: int,
        height: int,
        *,
        fov_deg: float = 58.0,
        distance_m: float = 14.0,
        height_above_drone_m: float = 5.0,
        target_height_frac: float = 0.88,
        near: float = 0.12,
        far: float | None = None,
    ) -> tuple[int | None, int | None]:
        """Project a world point onto chase-camera image pixels (matches render_chase_rgb)."""
        far_val = float(self.render_far_plane if far is None else far)
        bx = -float(distance_m) * math.cos(yaw)
        by = -float(distance_m) * math.sin(yaw)
        eye = [drone_x + bx, drone_y + by, drone_z + float(height_above_drone_m)]
        tz = max(0.12, float(drone_z) * float(target_height_frac))
        target = [drone_x, drone_y, tz]
        up = [0.0, 0.0, 1.0]
        view = p.computeViewMatrix(eye, target, up)
        aspect = float(width) / float(height)
        proj = p.computeProjectionMatrixFOV(
            fov=fov_deg, aspect=aspect, nearVal=near, farVal=far_val
        )
        vm = np.asarray(view, dtype=np.float64).reshape((4, 4), order="F")
        pm = np.asarray(proj, dtype=np.float64).reshape((4, 4), order="F")
        world_h = np.array([wx, wy, wz, 1.0], dtype=np.float64)
        clip = pm @ (vm @ world_h)
        w_clip = float(clip[3])
        if abs(w_clip) < 1e-9:
            return None, None
        ndc = clip[:3] / w_clip
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None, None
        u = int(round((float(ndc[0]) * 0.5 + 0.5) * float(width)))
        v = int(round((1.0 - (float(ndc[1]) * 0.5 + 0.5)) * float(height)))
        return u, v

    def disconnect(self) -> None:
        try:
            p.disconnect(self.client)
        except Exception:
            pass


def _try_apply_facade_texture(body_id: int, tex_seed: int) -> None:
    fd, fpath = tempfile.mkstemp(prefix="yoloslam_facade_", suffix=".png")
    os.close(fd)
    try:
        PyBulletLandingScene._write_facade_texture_png(fpath, 384, int(tex_seed))
        tex_uid = p.loadTexture(fpath)
        p.changeVisualShape(body_id, -1, textureUniqueId=tex_uid)
    except Exception:
        pass
    finally:
        try:
            os.remove(fpath)
        except OSError:
            pass


def _spawn_building(
    px: float,
    py: float,
    rng: random.Random,
) -> tuple[list[int], tuple[float, float, float]]:
    """Simple block building (unsafe semantic footprint). Returns (body ids, (x,y,xy radius))."""
    hx = rng.uniform(1.8, 3.2)
    hy = rng.uniform(1.8, 3.2)
    hz = rng.uniform(2.8, 5.5)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    wall = [
        0.58 + rng.uniform(0.0, 0.14),
        0.56 + rng.uniform(0.0, 0.12),
        0.54 + rng.uniform(0.0, 0.10),
    ]
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
        rgbaColor=[wall[0], wall[1], wall[2], 1.0],
    )
    yaw = rng.uniform(0.0, math.pi)
    orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    bid = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[px, py, hz + 0.02],
        baseOrientation=orn,
    )
    r = math.hypot(hx, hy) * 1.12
    return [bid], (px, py, r)


def _spawn_parked_car(
    px: float,
    py: float,
    yaw: float,
    rgba: list[float],
) -> tuple[int, tuple[float, float, float]]:
    """Low-poly sedan: one box body + four wheel cylinders (visual)."""
    hx, hy, hz = 1.05, 0.42, 0.34
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    body_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx * 0.98, hy * 0.98, hz * 0.55],
        rgbaColor=[rgba[0], rgba[1], rgba[2], 1.0],
    )
    roof_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx * 0.55, hy * 0.75, hz * 0.35],
        rgbaColor=[rgba[0] * 0.92, rgba[1] * 0.92, rgba[2] * 0.92, 1.0],
    )
    wheel_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.11,
        length=0.08,
        rgbaColor=[0.12, 0.12, 0.13, 1.0],
    )
    z_body = hz + 0.02
    link_masses = [0.01, 0.01, 0.01, 0.01, 0.01]
    link_collision = [-1, -1, -1, -1, -1]
    link_visual = [roof_vis, wheel_vis, wheel_vis, wheel_vis, wheel_vis]
    wx = hx * 0.72
    wy = hy * 0.72
    zw = -hz + 0.13
    link_positions = [
        [0.0, 0.0, hz * 0.48],
        [wx, wy, zw],
        [wx, -wy, zw],
        [-wx, wy, zw],
        [-wx, -wy, zw],
    ]
    link_ori = [p.getQuaternionFromEuler([0.0, 0.0, 0.0])] * 5
    link_ori[1] = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    link_ori[2] = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    link_ori[3] = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    link_ori[4] = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    cid = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=body_vis,
        basePosition=[px, py, z_body],
        baseOrientation=orn,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision,
        linkVisualShapeIndices=link_visual,
        linkPositions=link_positions,
        linkOrientations=link_ori,
        linkInertialFramePositions=[[0.0, 0.0, 0.0]] * 5,
        linkInertialFrameOrientations=[p.getQuaternionFromEuler([0.0, 0.0, 0.0])] * 5,
        linkParentIndices=[0] * 5,
        linkJointTypes=[p.JOINT_FIXED] * 5,
        linkJointAxis=[[0.0, 0.0, 1.0]] * 5,
    )
    r = max(hx, hy) * 1.35
    return cid, (px, py, r)


def _spawn_realistic_tree(
    px: float,
    py: float,
    rng: random.Random,
    *,
    trunk_r: float,
    trunk_h: float,
    scale: float,
) -> tuple[float, list[int]]:
    """Conifer-style tree: tapered trunk + clustered foliage spheres. Returns (crown_radius, body_ids)."""
    ids: list[int] = []
    tr = trunk_r * scale
    th = trunk_h * scale
    tz = th * 0.5 + 0.02
    trunk_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=tr, height=th)
    trunk_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=tr * 1.02,
        length=th,
        rgbaColor=[0.28, 0.17, 0.10, 1.0],
    )
    trunk_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=trunk_col,
        baseVisualShapeIndex=trunk_vis,
        basePosition=[px, py, tz],
    )
    ids.append(trunk_id)

    base_z = tz + th * 0.48
    foliage_specs: list[tuple[float, float, float, float, list[float]]] = [
        (0.0, 0.0, base_z + 0.55 * scale, 0.42 * scale, [0.10, 0.42, 0.16, 1.0]),
        (0.22 * scale, 0.12 * scale, base_z + 0.38 * scale, 0.34 * scale, [0.12, 0.48, 0.18, 1.0]),
        (-0.20 * scale, -0.08 * scale, base_z + 0.40 * scale, 0.32 * scale, [0.11, 0.46, 0.17, 1.0]),
        (0.10 * scale, -0.24 * scale, base_z + 0.30 * scale, 0.30 * scale, [0.13, 0.50, 0.19, 1.0]),
        (-0.12 * scale, 0.22 * scale, base_z + 0.28 * scale, 0.28 * scale, [0.09, 0.40, 0.15, 1.0]),
        (0.18 * scale, -0.18 * scale, base_z + 0.55 * scale, 0.26 * scale, [0.14, 0.52, 0.20, 1.0]),
    ]

    crown_xyr: list[tuple[float, float, float]] = []
    for ox, oy, cz, rad, rgba in foliage_specs:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=rad)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=rad, rgbaColor=rgba)
        fid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[px + ox, py + oy, cz],
        )
        ids.append(fid)
        crown_xyr.append((ox, oy, rad))

    crown_r = max(math.hypot(ox, oy) + rad for ox, oy, rad in crown_xyr)

    return crown_r, ids


def _spawn_x500_style_quad() -> int:
    """Larger, square-motor quad silhouette (x500-style: hub + cross arms + props)."""
    hub_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.13, 0.13, 0.04])
    hub_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.13, 0.13, 0.04],
        rgbaColor=[0.14, 0.14, 0.16, 1.0],
    )
    L = 0.40
    arm_w = 0.032
    arm_t = 0.02
    arm_x_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[L, arm_w, arm_t],
        rgbaColor=[0.24, 0.24, 0.26, 1.0],
    )
    arm_y_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[arm_w, L, arm_t],
        rgbaColor=[0.24, 0.24, 0.26, 1.0],
    )
    motor_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.052,
        length=0.045,
        rgbaColor=[0.32, 0.32, 0.35, 1.0],
    )
    prop_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.21,
        length=0.01,
        rgbaColor=[0.62, 0.78, 0.92, 0.45],
    )
    skid_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.20, 0.016, 0.016],
        rgbaColor=[0.38, 0.38, 0.42, 1.0],
    )

    z_arm = 0.008
    z_motor = 0.038
    z_prop = 0.072
    z_skid = -0.065
    link_masses = [0.02] * 12
    link_collision = [-1] * 12
    link_visual = [
        arm_x_vis,
        arm_y_vis,
        motor_vis,
        motor_vis,
        motor_vis,
        motor_vis,
        prop_vis,
        prop_vis,
        prop_vis,
        prop_vis,
        skid_vis,
        skid_vis,
    ]
    link_positions = [
        [0.0, 0.0, z_arm],
        [0.0, 0.0, z_arm + 0.001],
        [L, 0.0, z_motor],
        [-L, 0.0, z_motor],
        [0.0, L, z_motor],
        [0.0, -L, z_motor],
        [L, 0.0, z_prop],
        [-L, 0.0, z_prop],
        [0.0, L, z_prop],
        [0.0, -L, z_prop],
        [0.14, 0.14, z_skid],
        [-0.14, -0.14, z_skid],
    ]
    link_orientations = [p.getQuaternionFromEuler([0.0, 0.0, 0.0])] * 12
    link_orientations[10] = p.getQuaternionFromEuler([0.0, 0.0, math.pi / 4.0])
    link_orientations[11] = p.getQuaternionFromEuler([0.0, 0.0, math.pi / 4.0])
    link_inertial_pos = [[0.0, 0.0, 0.0]] * 12
    link_inertial_orn = [p.getQuaternionFromEuler([0.0, 0.0, 0.0])] * 12
    link_parent = [0] * 12
    link_joint_types = [p.JOINT_FIXED] * 12
    link_joint_axis = [[0.0, 0.0, 1.0]] * 12

    drone_id = p.createMultiBody(
        baseMass=1.2,
        baseCollisionShapeIndex=hub_col,
        baseVisualShapeIndex=hub_vis,
        basePosition=[0.0, 0.0, 12.0],
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision,
        linkVisualShapeIndices=link_visual,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_inertial_pos,
        linkInertialFrameOrientations=link_inertial_orn,
        linkParentIndices=link_parent,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axis,
    )
    p.changeDynamics(drone_id, -1, linearDamping=0.0, angularDamping=0.0)
    return drone_id
