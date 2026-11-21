"""Placement-focused mechfight/1 bot."""
import math
import mechfight as mf


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(x, y):
    length = math.hypot(x, y)
    if length < 1e-8:
        return [0.0, 0.0]
    return [x / length, y / length]


def angle_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def intercept_time(relative, velocity, speed):
    a = sum(v * v for v in velocity) - speed * speed
    b = 2 * sum(r * v for r, v in zip(relative, velocity))
    c = sum(r * r for r in relative)
    if c < 1e-8 or speed <= 0:
        return 0.0
    if abs(a) < 1e-8:
        return max(0.0, -c / b) if b < 0 else 0.0
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return 0.0
    root = math.sqrt(discriminant)
    times = [t for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)) if t > 0]
    return min(times) if times else 0.0


class Bot:
    name = "Builder-Apex"

    def __init__(self, hello: mf.Hello):
        self.config = hello["config"]
        self.memory_s = self.config["vision"].get("memory_s", 5.0)
        self.memory = {}
        self.last_attacker = None
        self.last_attacked_at = -100.0
        self.last_damaged_bearing = None
        self.last_damaged_at = -100.0
        self.weapon_value = {"gun": 1.0, "mortar": 2.0, "laser": 2.7, "lightning": 3.0}

    def remember(self, obs):
        now = obs["t"]
        for actor in obs["mechs"]:
            if actor["kind"] == "mech":
                self.memory[actor["id"]] = (actor["pos"][:], actor["vel"][:], now)
        self.memory = {
            key: value for key, value in self.memory.items()
            if now - value[2] <= self.memory_s
        }
        for event in obs["events"]:
            if event["type"] == "damaged":
                self.last_damaged_bearing = event.get("bearing")
                self.last_damaged_at = now
                if event.get("by") is not None:
                    self.last_attacker = event["by"]
                    self.last_attacked_at = now

    def closest_enemy(self, obs, me):
        visible_mechs = [a for a in obs["mechs"] if a["kind"] == "mech"]
        if visible_mechs:
            return min(
                visible_mechs,
                key=lambda a: distance(a["pos"], me["pos"]) - 0.01 * a.get("hp", 1000),
            )
        drones = [a for a in obs["mechs"] if a["kind"] == "npc"]
        if drones:
            return min(drones, key=lambda a: distance(a["pos"], me["pos"]))
        return None

    def enemy_count_near(self, obs, me, radius):
        return sum(
            1 for actor in obs["mechs"]
            if actor["kind"] == "mech" and distance(actor["pos"], me["pos"]) < radius
        )

    def evasion(self, obs, me):
        px, py = me["pos"][:2]
        threats = []
        for projectile in obs["projectiles"]:
            rx = projectile["pos"][0] - px
            ry = projectile["pos"][1] - py
            vx = projectile["vel"][0] - me["vel"][0]
            vy = projectile["vel"][1] - me["vel"][1]
            vv = vx * vx + vy * vy
            if vv < 1:
                continue
            time = max(0.0, min(1.5, -(rx * vx + ry * vy) / vv))
            miss = math.hypot(rx + vx * time, ry + vy * time)
            if time < 1.5 and miss < 8.0:
                threats.append(projectile)
        if not threats:
            return None

        speed = max(7.0, me["stats"].get("walk_speed", 11.7) * 0.85)
        best_direction = [0.0, 0.0]
        best_clearance = -1.0
        for step in range(16):
            angle = 2.0 * math.pi * step / 16
            dx, dy = math.cos(angle), math.sin(angle)
            clearance = float("inf")
            for projectile in threats:
                rx = projectile["pos"][0] - px
                ry = projectile["pos"][1] - py
                vx = projectile["vel"][0] - me["vel"][0] - speed * dx
                vy = projectile["vel"][1] - me["vel"][1] - speed * dy
                vv = vx * vx + vy * vy
                time = max(0.0, min(1.2, -(rx * vx + ry * vy) / vv)) if vv > 1e-8 else 0.0
                clearance = min(clearance, math.hypot(rx + vx * time, ry + vy * time))
            if clearance > best_clearance:
                best_clearance = clearance
                best_direction = [dx, dy]
        return best_direction

    def safe_destination(self, obs, me):
        zone = obs["zone"]
        pos = me["pos"]
        if not me["in_zone"]:
            return zone["center"][:]

        phase_has_started = obs["t"] >= zone["shrink_start"]
        if not phase_has_started:
            next_distance = distance(pos, zone["next_center"])
            next_staging_radius = zone["next_radius"] * 0.50
            distance_to_staging = max(0.0, next_distance - next_staging_radius)
            speed = me["stats"].get("walk_speed", 11.7)
            if me["energy"] > 25:
                speed = max(speed, me["stats"].get("boost_speed", speed) * 0.7)
            time_to_staging = distance_to_staging / max(1.0, speed)
            time_until_shrink = zone["shrink_start"] - obs["t"]
            phase_has_started = time_until_shrink <= time_to_staging + 10.0

        if not phase_has_started:
            center, radius = zone["center"], zone["radius"]
        else:
            center, radius = zone["next_center"], zone["next_radius"]

        if radius <= 1:
            return center[:]

        # Use the inner half of the active circle as a low-risk staging area.
        desired_radius = radius * (0.50 if radius < 80 else 0.58)
        dx, dy = pos[0] - center[0], pos[1] - center[1]
        current_radius = math.hypot(dx, dy)
        if current_radius <= desired_radius:
            return pos[:2]
        destination_radius = max(0.0, desired_radius - min(8.0, radius * 0.04))
        direction = unit(dx, dy)
        return [
            center[0] + direction[0] * destination_radius,
            center[1] + direction[1] * destination_radius,
        ]

    def safe_objective(self, point, obs, margin=0.72):
        zone = obs["zone"]
        if obs["t"] < zone["shrink_start"]:
            center, radius = zone["center"], zone["radius"]
        else:
            center, radius = zone["next_center"], zone["next_radius"]
        return distance(point, center) <= radius * margin

    def objective(self, obs, me, safe_goal, nearest_enemy):
        pos = me["pos"]
        if not me["in_zone"]:
            return safe_goal, "zone"
        danger_distance = distance(nearest_enemy["pos"], pos) if nearest_enemy else float("inf")
        if danger_distance > 55:
            nearby_cubes = [
                cube for cube in obs["cubes"]
                if distance(cube["pos"], pos) < 32
                and abs(cube["pos"][2] - pos[2]) < 12
                and self.safe_objective(cube["pos"], obs)
            ]
            if nearby_cubes:
                cube = min(nearby_cubes, key=lambda item: distance(item["pos"], pos))
                return cube["pos"][:2], "cube"

            candidate_drops = [
                drop for drop in obs["drops"]
                if drop["lands_in"] <= 0
                and (drop["holder"] is None or drop["holder"] == me["id"])
                and self.safe_objective(drop["pos"], obs, margin=0.85)
                and distance(drop["pos"], pos) < 75
            ]
            if candidate_drops:
                drop = min(candidate_drops, key=lambda item: distance(item["pos"], pos))
                if self.enemy_count_near(obs, me, 115) == 0:
                    return drop["pos"][:2], "drop"

            available_slots = [i for i, weapon in enumerate(me["weapons"]) if weapon is None]
            if available_slots:
                pickups = [
                    item for item in obs["pickups"]
                    if distance(item["pos"], pos) < 18
                    and abs(item["pos"][2] - pos[2]) < 10
                    and self.safe_objective(item["pos"], obs)
                ]
                if pickups:
                    pickup = max(
                        pickups,
                        key=lambda item: self.weapon_value.get(item["kind"], 0)
                        - 0.025 * distance(item["pos"], pos),
                    )
                    return pickup["pos"][:2], "weapon"
        return safe_goal, "zone"

    def pickup_action(self, obs, me):
        weapons = me["weapons"]
        kinds = [weapon["kind"] for weapon in weapons if weapon]
        enemies_near = self.enemy_count_near(obs, me, 40) > 0
        pickups = [
            item for item in obs["pickups"]
            if distance(item["pos"], me["pos"]) <= self.config["loot"]["pickup_radius"]
            and abs(item["pos"][2] - me["pos"][2]) <= 4
        ]
        if not pickups:
            return None
        pickup = max(
            pickups,
            key=lambda item: (
                self.weapon_value.get(item["kind"], 0),
                -distance(item["pos"], me["pos"]),
            ),
        )
        if pickup["kind"] in kinds:
            return None
        for slot, weapon in enumerate(weapons):
            if weapon is None:
                return {"take": pickup["id"]}
        if not enemies_near:
            replace = min(
                (i for i, weapon in enumerate(weapons) if i > 0 and weapon),
                key=lambda i: self.weapon_value.get(weapons[i]["kind"], 0),
                default=None,
            )
            if replace is not None and self.weapon_value.get(pickup["kind"], 0) > (
                self.weapon_value.get(weapons[replace]["kind"], 0) + 0.2
            ):
                return {"take": pickup["id"], "drop": replace}
        return None

    def upgrade_choice(self, obs, me):
        offer = obs["offer"]
        if not offer:
            return None
        held = {weapon["kind"] for weapon in me["weapons"] if weapon}
        definitions = {upgrade["id"]: upgrade for upgrade in self.config["upgrades"]}

        def score(choice):
            upgrade = definitions.get(choice["id"], {})
            text = (choice["id"] + " " + choice["name"] + " " + choice["desc"]).lower()
            requirement = upgrade.get("requires")
            if requirement and requirement not in held:
                return -100
            if any(word in text for word in ("armor", "damage taken", "health", "regen", "repair")):
                return 10
            if any(word in text for word in ("energy", "boost", "walk speed", "movement", "jet")):
                return 8
            if any(word in text for word in ("sensor", "vision", "sight")):
                return 6
            if any(word in text for word in ("damage", "fire rate", "cooldown", "reload")):
                return 5
            return 1

        return max(range(len(offer["choices"])), key=lambda i: score(offer["choices"][i]))

    def direct_aim(self, me, target, projectile_speed=None):
        pos, velocity = me["pos"], target.get("vel", [0.0, 0.0, 0.0])
        target_center = [target["pos"][0], target["pos"][1], target["pos"][2] + 4.5]
        muzzle = [pos[0], pos[1], pos[2] + me["stats"].get("muzzle_height", 7.0)]
        relative = [target_center[i] - muzzle[i] for i in range(3)]
        time = intercept_time(relative, velocity, projectile_speed) if projectile_speed else 0.0
        point = [target_center[i] + velocity[i] * time for i in range(3)]
        dx, dy, dz = point[0] - muzzle[0], point[1] - muzzle[1], point[2] - muzzle[2]
        return [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]

    def mortar_aim(self, me, target, stats):
        x, y = me["pos"][:2]
        speed = stats.get("speed", 72.0)
        gravity = stats.get("gravity", self.config["weapons"]["mortar"].get("gravity", 22.0))
        muzzle_z = me["pos"][2] + me["stats"].get("muzzle_height", 7.0) + 2.5
        base = target["pos"][:]
        velocity = target.get("vel", [0.0, 0.0, 0.0])
        flight = math.hypot(base[0] - x, base[1] - y) / max(1.0, speed)
        for _ in range(3):
            tx = base[0] + velocity[0] * flight
            ty = base[1] + velocity[1] * flight
            tz = base[2] + velocity[2] * flight
            dx, dy = tx - x, ty - y
            horizontal = max(1.0, math.hypot(dx, dy))
            vertical = tz + 4.5 - muzzle_z
            discriminant = speed ** 4 - gravity * (gravity * horizontal ** 2 + 2 * vertical * speed ** 2)
            if discriminant < 0:
                return None
            tangent = (speed ** 2 - math.sqrt(discriminant)) / (gravity * horizontal)
            pitch = math.atan(tangent)
            flight = horizontal / max(1.0, speed * math.cos(pitch))
        return [math.atan2(dy, dx), pitch]

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        self.remember(obs)
        action: mf.Action = {"tick": obs["tick"]}

        if obs["offer"]:
            action["pick"] = self.upgrade_choice(obs, me)

        pickup = self.pickup_action(obs, me)
        if pickup:
            action.update(pickup)

        enemy = self.closest_enemy(obs, me)
        safe_goal = self.safe_destination(obs, me)
        goal, objective_kind = self.objective(obs, me, safe_goal, enemy)
        dx, dy = goal[0] - me["pos"][0], goal[1] - me["pos"][1]
        goal_distance = math.hypot(dx, dy)
        move = unit(dx, dy) if goal_distance > 2.5 else [0.0, 0.0]

        threats = []
        for actor in obs["mechs"]:
            if actor["kind"] != "mech":
                continue
            ex, ey = actor["pos"][0] - me["pos"][0], actor["pos"][1] - me["pos"][1]
            d = math.hypot(ex, ey)
            if d >= 105:
                continue
            strength = (105.0 - d) / 105.0
            if actor.get("charging"):
                strength *= 1.6
            if actor["id"] == self.last_attacker and obs["t"] - self.last_attacked_at < 8:
                strength *= 1.8
            if me["hp"] < me["max_hp"] * 0.55:
                strength *= 1.5
            away = unit(-ex, -ey)
            threats.append((away, strength))

        if threats:
            flee = [
                sum(direction[axis] * weight for direction, weight in threats)
                for axis in (0, 1)
            ]
            flee = unit(flee[0], flee[1])
            danger = max(weight for _, weight in threats)
            zone_weight = 3.0 if not me["in_zone"] else (1.0 if goal_distance > 8 else 0.0)
            active_center = obs["zone"]["center"]
            active_radius = obs["zone"]["radius"]
            if obs["t"] >= obs["zone"]["shrink_start"]:
                active_center = obs["zone"]["next_center"]
                active_radius = obs["zone"]["next_radius"]
            edge_margin = active_radius - distance(me["pos"], active_center)
            if edge_margin < 35:
                zone_weight = max(zone_weight, 3.0)
            flee_weight = 1.1 + danger * (1.4 if me["hp"] < me["max_hp"] * 0.6 else 1.0)
            move = unit(move[0] * zone_weight + flee[0] * flee_weight,
                        move[1] * zone_weight + flee[1] * flee_weight)
            radial = unit(me["pos"][0] - active_center[0], me["pos"][1] - active_center[1])
            outward = move[0] * radial[0] + move[1] * radial[1]
            if edge_margin < 25 and outward > 0:
                move = unit(move[0] - radial[0] * outward, move[1] - radial[1] * outward)

        dodge = self.evasion(obs, me)
        if dodge:
            if me["in_zone"]:
                move = unit(move[0] * 0.35 + dodge[0] * 0.65,
                            move[1] * 0.35 + dodge[1] * 0.65)
            else:
                move = unit(move[0] * 0.75 + dodge[0] * 0.25,
                            move[1] * 0.75 + dodge[1] * 0.25)

        urgency = not me["in_zone"] or (enemy is not None and distance(enemy["pos"], me["pos"]) < 30)
        boost = bool(
            math.hypot(*move) > 0.1
            and me["energy"] > (12 if urgency else 35)
            and (urgency or goal_distance > 28)
        )
        action.update(move=move, boost=boost, jump=bool(dodge and me["grounded"]))

        target = enemy
        if target is None and self.memory:
            actor_id, (position, velocity, seen_at) = max(
                self.memory.items(), key=lambda item: item[1][2]
            )
            age = obs["t"] - seen_at
            if age <= self.memory_s:
                target = {
                    "id": actor_id, "kind": "mech",
                    "pos": [position[i] + velocity[i] * min(age, 1.0) for i in range(3)],
                    "vel": velocity, "weapons": [],
                }

        current_yaw, current_pitch = me["aim"]
        aim = [current_yaw, current_pitch]
        fire = [False, False]

        visible_target = enemy is not None
        target_distance = distance(target["pos"], me["pos"]) if target else float("inf")
        if visible_target and target:
            target_id = target["id"]
            recently_attacked = (
                target_id == self.last_attacker and obs["t"] - self.last_attacked_at < 8
            )
            danger_close = target_distance < (42 if target["kind"] == "mech" else 32)
            engage = danger_close or (recently_attacked and target_distance < 100)
            if target.get("charging") and target_distance < 150 and me["hp"] > me["max_hp"] * 0.35:
                engage = True
            if engage:
                weapons = me["weapons"]
                direct_weapon = next(
                    (
                        i for i, weapon in enumerate(weapons)
                        if weapon and weapon["kind"] == "lightning"
                        and target_distance <= weapon["stats"].get("range", 45)
                        and me["energy"] >= weapon["stats"].get("energy", 3.0)
                    ),
                    None,
                )
                if direct_weapon is None:
                    direct_weapon = next(
                        (i for i, weapon in enumerate(weapons) if weapon and weapon["kind"] == "gun"
                         and target_distance <= weapon["stats"].get("range", 230)),
                        None,
                    )
                if direct_weapon is None:
                    direct_weapon = next(
                        (i for i, weapon in enumerate(weapons) if weapon and weapon["kind"] == "laser"
                         and target_distance <= weapon["stats"].get("range", 280)),
                        None,
                    )
                if direct_weapon is None:
                    direct_weapon = next(
                        (i for i, weapon in enumerate(weapons) if weapon and weapon["kind"] == "mortar"
                         and 18 < target_distance < 180),
                        None,
                    )

                if direct_weapon is not None:
                    weapon = weapons[direct_weapon]
                    kind = weapon["kind"]
                    if kind == "gun":
                        aim = self.direct_aim(me, target, weapon["stats"].get("speed", 252.0))
                    elif kind == "mortar":
                        mortar_solution = self.mortar_aim(me, dict(target), weapon["stats"])
                        if mortar_solution:
                            aim = mortar_solution
                    else:
                        aim = self.direct_aim(me, target)

                    yaw_error = abs(angle_delta(aim[0], current_yaw))
                    pitch_error = abs(aim[1] - current_pitch)
                    aligned = yaw_error < 0.12 and pitch_error < 0.12

                    if kind == "gun":
                        if weapon.get("ammo", 0) > 0 and weapon.get("reload", 0) <= 0 and aligned:
                            fire[direct_weapon] = True
                        if not fire[direct_weapon] and weapon.get("reload", 0) <= 0 and (
                            weapon.get("ammo", 0) <= 0
                            or (weapon.get("ammo", 0) <= 2 and target_distance > 50)
                        ):
                            action["reload"] = True
                    elif kind == "laser":
                        charge = weapon.get("charge", 0.0)
                        full_charge = weapon["stats"].get("charge", 0.75)
                        if weapon.get("cooldown", 0) <= 0.02 and (
                            charge < full_charge * 0.88 or not aligned
                        ):
                            fire[direct_weapon] = True
                    elif kind == "lightning":
                        if aligned and me["energy"] >= weapon["stats"].get("energy", 3.0):
                            fire[direct_weapon] = True
                    elif kind == "mortar":
                        if aligned:
                            fire[direct_weapon] = True
            else:
                aim = self.direct_aim(me, target)
        elif self.last_damaged_bearing is not None and obs["t"] - self.last_damaged_at < 8:
            aim = [self.last_damaged_bearing, 0.0]
        else:
            recent_sounds = [
                sound for sound in obs["sounds"]
                if sound["kind"] in ("gun", "mortar", "explosion", "laser", "lightning", "npc_gun")
                and sound["age"] < 1.0
            ]
            if recent_sounds:
                sound = min(recent_sounds, key=lambda item: item["age"])
                aim = [sound["bearing"], 0.0]

        action["aim"] = aim
        action["fire"] = fire

        if target:
            mf.draw.line(
                [me["pos"][0], me["pos"][1], me["pos"][2] + 8],
                [target["pos"][0], target["pos"][1], target["pos"][2] + 4.5],
                "#ff5040",
            )
        mf.draw.point([goal[0], goal[1], me["pos"][2]], "#40c0ff", 6)
        mf.draw.label(
            "%s | hp %.0f | cubes %d | alive %d"
            % (objective_kind, me["hp"], me["cubes"], obs["alive"])
        )
        return action


mf.run(Bot)
