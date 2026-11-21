"""Coarse collision-aware routing over the supplied static arena map."""
from array import array
import heapq
import json
import math
import os
import sys
import time


class ArenaNavigator:
    resolution = 6.0
    clearance = 3.0
    max_expansions = 6000
    search_budget_s = 0.012

    def __init__(self, map_dir, mech_config):
        world_path = os.path.join(map_dir, "v2", "world.json")
        with open(world_path, encoding="utf-8") as source:
            world = json.load(source)

        self.bounds = world["bounds"]
        self.min_x = float(self.bounds["minX"])
        self.min_y = float(self.bounds["minY"])
        self.max_x = float(self.bounds["maxX"])
        self.max_y = float(self.bounds["maxY"])
        self.nx = max(1, math.ceil((self.max_x - self.min_x) / self.resolution))
        self.ny = max(1, math.ceil((self.max_y - self.min_y) / self.resolution))
        self.step_up = float(mech_config.get("step_up", 1.2))
        self.mech_height = float(mech_config.get("height", 9.0))
        self.max_slope = math.tan(math.radians(float(mech_config.get("max_slope_deg", 50.0))))

        terrain = world["terrain"]
        terrain_path = os.path.join(map_dir, "v2", terrain["file"])
        heights = array("h")
        with open(terrain_path, "rb") as source:
            heights.frombytes(source.read())
        if sys.byteorder != "little":
            heights.byteswap()
        if len(heights) != int(terrain["nx"]) * int(terrain["ny"]):
            raise ValueError("terrain grid length does not match world.json")

        self.terrain = heights
        self.terrain_nx = int(terrain["nx"])
        self.terrain_ny = int(terrain["ny"])
        self.terrain_min_x = float(terrain["minX"])
        self.terrain_min_y = float(terrain["minY"])
        self.terrain_step = float(terrain["step"])
        self.terrain_scale = float(terrain["scale"])
        self.elevations = [
            self._height_at(*self.cell_center(cell))
            for cell in range(self.nx * self.ny)
        ]
        self.blocked = bytearray(self.nx * self.ny)
        self._rasterize_obstacles(world.get("prisms", []))
        self._cache_goal = None
        self._cache_path = None
        self._cache_index = 0

    def _height_at(self, x, y):
        gx = (x - self.terrain_min_x) / self.terrain_step
        gy = (y - self.terrain_min_y) / self.terrain_step
        ix, iy = int(round(gx)), int(round(gy))
        if not (0 <= ix < self.terrain_nx and 0 <= iy < self.terrain_ny):
            return 0.0
        return self.terrain[iy * self.terrain_nx + ix] * self.terrain_scale

    def _cell(self, x, y):
        i = int((x - self.min_x) / self.resolution)
        j = int((y - self.min_y) / self.resolution)
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return None
        return j * self.nx + i

    def cell_center(self, cell):
        x, y = cell % self.nx, cell // self.nx
        return (
            self.min_x + (x + 0.5) * self.resolution,
            self.min_y + (y + 0.5) * self.resolution,
        )

    @staticmethod
    def _inside_ring(x, y, ring):
        inside = False
        count = len(ring) // 2
        if count < 3:
            return False
        j = count - 1
        for i in range(count):
            xi, yi = ring[2 * i], ring[2 * i + 1]
            xj, yj = ring[2 * j], ring[2 * j + 1]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _segment_distance_sq(x, y, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return (x - ax) ** 2 + (y - ay) ** 2
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
        px, py = ax + t * dx, ay + t * dy
        return (x - px) ** 2 + (y - py) ** 2

    def _near_ring(self, x, y, ring, margin_sq):
        count = len(ring) // 2
        for i in range(count):
            j = (i + 1) % count
            if self._segment_distance_sq(
                x, y, ring[2 * i], ring[2 * i + 1], ring[2 * j], ring[2 * j + 1]
            ) <= margin_sq:
                return True
        return False

    def _rasterize_obstacles(self, prisms):
        margin = self.clearance + self.resolution * 0.5
        margin_sq = margin * margin
        for prism in prisms:
            if prism.get("kind") not in ("building", "stadium", "bridge"):
                continue
            ring = prism.get("ring", [])
            if len(ring) < 6:
                continue
            holes = prism.get("holes", [])
            xs, ys = ring[0::2], ring[1::2]
            min_i = max(0, int((min(xs) - margin - self.min_x) / self.resolution))
            max_i = min(self.nx - 1, int((max(xs) + margin - self.min_x) / self.resolution))
            min_j = max(0, int((min(ys) - margin - self.min_y) / self.resolution))
            max_j = min(self.ny - 1, int((max(ys) + margin - self.min_y) / self.resolution))
            top, base = float(prism["top"]), float(prism["base"])
            for j in range(min_j, max_j + 1):
                for i in range(min_i, max_i + 1):
                    x = self.min_x + (i + 0.5) * self.resolution
                    y = self.min_y + (j + 0.5) * self.resolution
                    inside = self._inside_ring(x, y, ring)
                    in_hole = any(self._inside_ring(x, y, hole) for hole in holes)
                    if in_hole:
                        if not any(self._near_ring(x, y, hole, margin_sq) for hole in holes):
                            continue
                    elif not inside and not self._near_ring(x, y, ring, margin_sq):
                        continue

                    ground = self.elevations[j * self.nx + i]
                    if top <= ground + self.step_up or base >= ground + self.mech_height:
                        continue
                    self.blocked[j * self.nx + i] = 1

    def _nearest_open(self, cell, max_radius=8):
        if cell is not None and not self.blocked[cell]:
            return cell
        if cell is None:
            return None
        x, y = cell % self.nx, cell // self.nx
        for radius in range(1, max_radius + 1):
            candidates = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < self.nx and 0 <= yy < self.ny:
                        candidate = yy * self.nx + xx
                        if not self.blocked[candidate]:
                            candidates.append(candidate)
            if candidates:
                return min(
                    candidates,
                    key=lambda item: (item % self.nx - x) ** 2 + (item // self.nx - y) ** 2,
                )
        return None

    def _neighbors(self, cell):
        x, y = cell % self.nx, cell // self.nx
        here_z = self.elevations[cell]
        for dx, dy, base_cost in (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),
        ):
            xx, yy = x + dx, y + dy
            if not (0 <= xx < self.nx and 0 <= yy < self.ny):
                continue
            neighbor = yy * self.nx + xx
            if self.blocked[neighbor]:
                continue
            if dx and dy and (
                self.blocked[y * self.nx + xx] or self.blocked[yy * self.nx + x]
            ):
                continue
            dz = self.elevations[neighbor] - here_z
            horizontal = self.resolution * base_cost
            if dz > self.step_up + 0.45 or (dz > 0 and dz / horizontal > self.max_slope):
                continue
            slope_cost = 1.0 + min(1.5, abs(dz) / max(1.0, horizontal) * 1.4)
            yield neighbor, base_cost * slope_cost

    def _find_path(self, start, goal):
        start, goal = self._nearest_open(start), self._nearest_open(goal)
        if start is None or goal is None:
            return None
        if start == goal:
            return [start]

        start_time = time.perf_counter()
        goal_x, goal_y = goal % self.nx, goal // self.nx
        frontier = [(0.0, 0.0, start)]
        came_from, costs = {start: -1}, {start: 0.0}
        best, best_h = start, math.hypot(goal_x - start % self.nx, goal_y - start // self.nx)
        expansions, found = 0, False

        while frontier and expansions < self.max_expansions:
            if expansions and time.perf_counter() - start_time > self.search_budget_s:
                break
            _, cost, current = heapq.heappop(frontier)
            if cost != costs.get(current):
                continue
            x, y = current % self.nx, current // self.nx
            heuristic = math.hypot(goal_x - x, goal_y - y)
            if heuristic < best_h:
                best, best_h = current, heuristic
            if current == goal:
                best, found = current, True
                break
            expansions += 1
            for neighbor, edge_cost in self._neighbors(current):
                new_cost = cost + edge_cost
                if new_cost >= costs.get(neighbor, float("inf")):
                    continue
                costs[neighbor] = new_cost
                came_from[neighbor] = current
                nx, ny = neighbor % self.nx, neighbor // self.nx
                estimate = new_cost + math.hypot(goal_x - nx, goal_y - ny)
                heapq.heappush(frontier, (estimate, new_cost, neighbor))

        if not found and best == start:
            return None
        path = []
        current = best
        while current != -1:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _line_clear(self, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        count = max(1, math.ceil(math.hypot(dx, dy) / (self.resolution * 0.4)))
        previous = None
        start_cell = self._cell(*start)
        for i in range(count + 1):
            t = i / count
            cell = self._cell(start[0] + dx * t, start[1] + dy * t)
            if cell is None or (self.blocked[cell] and cell != start_cell):
                return False
            if previous is not None:
                x0, y0 = previous % self.nx, previous // self.nx
                x1, y1 = cell % self.nx, cell // self.nx
                if abs(x1 - x0) > 1 or abs(y1 - y0) > 1:
                    return False
                if x0 != x1 and y0 != y1:
                    if self.blocked[y0 * self.nx + x1] or self.blocked[y1 * self.nx + x0]:
                        return False
            previous = cell
        return True

    def direction(self, start, goal):
        start_cell, goal_cell = self._cell(*start), self._cell(*goal)
        if start_cell is None or goal_cell is None:
            return None
        goal_cell = self._nearest_open(goal_cell)
        if goal_cell is None:
            return None

        if goal_cell != self._cache_goal or not self._cache_path:
            self._cache_goal = goal_cell
            self._cache_path = self._find_path(start_cell, goal_cell)
            self._cache_index = 0
        path = self._cache_path
        if not path:
            return None

        nearest = min(
            range(self._cache_index, len(path)),
            key=lambda index: math.dist(self.cell_center(path[index]), start),
        )
        if (
            nearest == len(path) - 1
            and path[nearest] != goal_cell
            and math.dist(self.cell_center(path[nearest]), start) < self.resolution * 1.5
        ):
            self._cache_path = self._find_path(start_cell, goal_cell)
            self._cache_index = 0
            path = self._cache_path
            if not path:
                return None
            nearest = 0
        if math.dist(self.cell_center(path[nearest]), start) > self.resolution * 2.0:
            self._cache_path = self._find_path(start_cell, goal_cell)
            self._cache_index = 0
            path = self._cache_path
            if not path:
                return None
            nearest = 0
        self._cache_index = nearest

        waypoint = self.cell_center(path[nearest])
        for index in range(nearest + 1, len(path)):
            candidate = self.cell_center(path[index])
            if not self._line_clear(start, candidate):
                break
            waypoint = candidate
            self._cache_index = index
        dx, dy = waypoint[0] - start[0], waypoint[1] - start[1]
        if math.hypot(dx, dy) < 1.0:
            dx, dy = goal[0] - start[0], goal[1] - start[1]
        return [dx, dy]

    def constrain(self, start, direction):
        length = math.hypot(direction[0], direction[1])
        if length < 1e-8:
            return [0.0, 0.0]
        desired = [direction[0] / length, direction[1] / length]
        horizon = max(self.resolution * 1.5, 12.0)
        best, best_score = [0.0, 0.0], -float("inf")
        for step in range(24):
            angle = 2.0 * math.pi * step / 24
            candidate = [math.cos(angle), math.sin(angle)]
            endpoint = [start[0] + candidate[0] * horizon, start[1] + candidate[1] * horizon]
            if not self._line_clear(start, endpoint):
                continue
            score = candidate[0] * desired[0] + candidate[1] * desired[1]
            if score > best_score:
                best, best_score = candidate, score
        return best if best_score > -0.5 else [0.0, 0.0]
