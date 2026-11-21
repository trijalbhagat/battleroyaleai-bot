"""Placement-oriented mechfight/1 bot with zone, cover, loot and combat policy."""
import math
import mechfight as mf
from navigation import ArenaNavigator


TAU = 2.0 * math.pi


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(x, y):
    length = math.hypot(x, y)
    return [0.0, 0.0] if length < 1e-8 else [x / length, y / length]


def wrap_angle(angle):
    return (angle + math.pi) % TAU - math.pi


def lead_position(origin, target, projectile_speed):
    if projectile_speed is None or projectile_speed <= 0:
        return target["pos"][:]
    relative = [target["pos"][i] - origin[i] for i in range(3)]
    velocity = target.get("vel", [0.0, 0.0, 0.0])
    a = sum(v * v for v in velocity) - projectile_speed * projectile_speed
    b = 2.0 * sum(relative[i] * velocity[i] for i in range(3))
    c = sum(v * v for v in relative)
    roots = []
    if abs(a) < 1e-9:
        if abs(b) > 1e-9 and -c / b > 0:
            roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0:
            root = math.sqrt(discriminant)
            roots.extend(
                t for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
                if t > 0
            )
    flight = min(roots) if roots else 0.0
    return [target["pos"][i] + velocity[i] * flight for i in range(3)]


class Bot:
    name = "Builder-Apex"

    def __init__(self, hello):
        self.config = hello["config"]
        self.slot = hello["slot"]
        self.seed = hello["seed"]
        self.memory_s = self.config["vision"].get("memory_s", 5.0)
        self.enemies = {}
        self.loot_memory = {}
        self.last_damage = -100.0
        self.last_damage_bearing = None
        self.last_attacker = None
        self.last_attacker_time = -100.0
        self.recovering = False
        self.mode = "roam"
        self.stuck = 0.0
        self.last_position = None
        self.previous_move = [0.0, 0.0]
        self.strafe_sign = 1.0 if self.slot % 2 else -1.0
        self.navigator = None
        try:
            self.navigator = ArenaNavigator(hello["map"]["dir"], self.config["mech"])
        except (OSError, KeyError, ValueError):
            pass

    def remember(self, obs):
        now = obs["t"]
        for enemy_id in list(self.enemies):
            if now - self.enemies[enemy_id]["seen"] > self.memory_s:
                del self.enemies[enemy_id]

        for actor in obs["mechs"]:
            if actor["kind"] == "mech":
                self.enemies[actor["id"]] = {
                    "pos": actor["pos"][:],
                    "vel": actor.get("vel", [0.0, 0.0, 0.0])[:],
                    "aim": actor.get("aim", [0.0, 0.0])[:],
                    "hp": actor.get("hp", self.config.get("mech", {}).get("hp", 1400)),
                    "max_hp": actor.get("max_hp", self.config.get("mech", {}).get("hp", 1400)),
                    "weapons": actor.get("weapons", [])[:],
                    "charging": actor.get("charging", False),
                    "seen": now,
                    "visible": True,
                }
            else:
                pass

        for event in obs["events"]:
            if event["type"] == "damaged":
                self.last_damage = now
                self.last_damage_bearing = event.get("bearing")
                if event.get("by") is not None:
                    self.last_attacker = event["by"]
                    self.last_attacker_time = now

        for item in obs["pickups"] + obs["cubes"]:
            key = ("pickup" if "kind" in item else "cube", item["id"])
            self.loot_memory[key] = (item["pos"][:], now)
        for key in list(self.loot_memory):
            if now - self.loot_memory[key][1] > 20:
                del self.loot_memory[key]

    def visible_mechs(self, obs):
        return [actor for actor in obs["mechs"] if actor["kind"] == "mech"]

    def target_score(self, enemy, me, now):
        distance_m = distance(enemy["pos"], me["pos"])
        hp = enemy.get("hp", 1400)
        max_hp = enemy.get("max_hp", 1400)
        hp_ratio = hp / max(1.0, max_hp)
        score = distance_m
        score += hp_ratio * 180
        if hp_ratio < 0.30:
            score -= 180
        if enemy["id"] == self.last_attacker and now - self.last_attacker_time < 8:
            score -= 80
        if enemy.get("charging"):
            score -= 100
        return score

    def choose_target(self, obs, me):
        visible = self.visible_mechs(obs)
        if visible:
            return min(
                visible,
                key=lambda enemy: self.target_score(enemy, me, obs["t"]),
            )
        return None

    def zone_state(self, obs, me):
        zone = obs["zone"]
        pos = me["pos"]
        now = obs["t"]
        current = zone["center"], zone["radius"]
        next_circle = zone["next_center"], zone["next_radius"]
        from_start = obs["t"] >= zone["shrink_start"]
        if from_start:
            fraction = 0.0
            if zone["shrink_end"] > zone["shrink_start"]:
                fraction = min(
                    1.0,
                    max(0.0, (now - zone["shrink_start"]) /
                        (zone["shrink_end"] - zone["shrink_start"]))
                )
            center = [
                zone["center"][0] + (zone["next_center"][0] - zone["center"][0]) * fraction,
                zone["center"][1] + (zone["next_center"][1] - zone["center"][1]) * fraction,
            ]
            radius = zone["radius"] + (zone["next_radius"] - zone["radius"]) * fraction
            return center, radius, True
        return next_circle[0], next_circle[1], False

    def safe_goal(self, obs, me):
        zone = obs["zone"]
        pos = me["pos"]
        center, radius, active = self.zone_state(obs, me)

        if not me["in_zone"]:
            return zone["center"][:]

        if active:
            margin = max(12.0, min(40.0, radius * 0.12))
            safe_radius = max(0.0, radius - margin)
        else:
            next_distance = distance(pos, zone["next_center"])
            safe_radius = max(0.0, zone["next_radius"] - max(15.0, min(35.0, zone["next_radius"] * 0.12)))
            walk = me["stats"].get("walk_speed", 11.7)
            boost = me["stats"].get("boost_speed", 19.5)
            travel_speed = boost if me["energy"] > 35 else walk
            travel = max(0.0, next_distance - safe_radius) / travel_speed
            before_start = zone["shrink_start"] - obs["t"]
            must_go = before_start <= travel + 8.0
            if not must_go:
                center, radius = zone["center"], zone["radius"]
                safe_radius = max(0.0, radius - max(15.0, min(35.0, radius * 0.10)))

        dx, dy = pos[0] - center[0], pos[1] - center[1]
        radial = math.hypot(dx, dy)
        if radial <= safe_radius:
            return pos[:2]
        scale = safe_radius / radial if radial > 1e-8 else 0
        return [center[0] + dx * scale, center[1] + dy * scale]

    def route_goal(self, goal, me):
        if self.navigator is None:
            return unit(goal[0] - me["pos"][0], goal[1] - me["pos"][1])
        routed = self.navigator.direction(me["pos"][:2], goal)
        if routed is None:
            return unit(goal[0] - me["pos"][0], goal[1] - me["pos"][1])
        move = unit(routed[0], routed[1])
        return self.navigator.constrain(me["pos"][:2], move)

    def upgrade_choice(self, obs, me):
        offer = obs["offer"]
        held = {weapon["kind"] for weapon in me["weapons"] if weapon}
        hp_ratio = me["hp"] / max(1.0, me["max_hp"])
        priorities = {
            "armor": 100,
            "plating": 96 if hp_ratio < 0.75 else 88,
            "nanites": 94 if hp_ratio < 0.80 else 84,
            "servos": 86,
            "laser_damage": 92,
            "laser_charge": 90,
            "gun_damage": 86,
            "gun_rate": 84,
            "gun_mag": 80,
            "gun_velocity": 78,
            "lightning_efficiency": 88 if "lightning" in held else 68,
            "lightning_range": 84 if "lightning" in held else 64,
            "mortar_reload": 76,
            "mortar_splash": 74,
            "reactor": 82 if "lightning" in held else 72,
            "capacitor": 70,
            "gyros": 78,
            "jets": 82,
            "sensors": 66,
            "laser_pierce": 60,
            "lightning_chain": 72,
            "mortar_velocity": 68,
        }
        definitions = {upgrade["id"]: upgrade for upgrade in self.config["upgrades"]}
        best = 0
        best_score = -1000
        for index, choice in enumerate(offer["choices"]):
            definition = definitions.get(choice["id"], {})
            required = definition.get("requires")
            if required and required not in held:
                continue
            score = priorities.get(choice["id"], 1)
            if score > best_score:
                best_score, best = score, index
        return best

    def weapon_pickup_action(self, obs, me):
        held = {weapon["kind"] for weapon in me["weapons"] if weapon}
        empty_slot = next((i for i, weapon in enumerate(me["weapons"]) if weapon is None), None)
        enemies_near = any(
            actor["kind"] == "mech" and distance(actor["pos"], me["pos"]) < 45
            for actor in obs["mechs"]
        )
        candidates = [
            item for item in obs["pickups"]
            if distance(item["pos"], me["pos"]) <= self.config["loot"]["pickup_radius"]
            and abs(item["pos"][2] - me["pos"][2]) <= 4
            and item["kind"] not in held
        ]
        if not candidates:
            return None
        value = {"laser": 30, "lightning": 25, "mortar": 10, "gun": 1}
        pickup = max(
            candidates,
            key=lambda item: value[item["kind"]] - 0.02 * distance(item["pos"], me["pos"]),
        )
        if empty_slot is not None:
            return {"take": pickup["id"]}
        if not enemies_near:
            for slot in (1, 0):
                weapon = me["weapons"][slot]
                if weapon is not None:
                    new_value = value[pickup["kind"]]
                    old_value = value[weapon["kind"]]
                    if new_value > old_value + 5:
                        return {"take": pickup["id"], "drop": slot}
        return None

    def incoming_dodge(self, obs, me):
        threats = []
        x, y = me["pos"][:2]
        for projectile in obs["projectiles"]:
            rx, ry = projectile["pos"][0] - x, projectile["pos"][1] - y
            vx, vy = projectile["vel"][0] - me["vel"][0], projectile["vel"][1] - me["vel"][1]
            speed_sq = vx * vx + vy * vy
            if speed_sq < 1:
                continue
            flight = max(0.0, min(1.4, -(rx * vx + ry * vy) / speed_sq))
            miss = math.hypot(rx + vx * flight, ry + vy * flight)
            if flight < 1.4 and miss < 8:
                threats.append((vx, vy))
        if not threats:
            return None
        vx, vy = threats[0]
        direction = unit(-vy, vx)
        return direction

    def crate_target(self, obs, me):
        if me["hp"] < me["max_hp"] * 0.65:
            return None
        visible_mechs = self.visible_mechs(obs)
        if visible_mechs:
            return None
        if obs["t"] - self.last_damage < 2.5:
            return None
        candidates = []
        for crate in obs["crates"]:
            d = distance(crate["pos"], me["pos"])
            if d < 55 and abs(crate["pos"][2] - me["pos"][2]) < 12:
                candidates.append((d, crate))
        if not candidates:
            return None
        d, crate = min(candidates, key=lambda item: item[0])
        return {
            "id": crate["id"],
            "kind": "crate",
            "pos": crate["pos"][:],
            "vel": [0.0, 0.0, 0.0],
            "hp": crate.get("hp", 300),
            "max_hp": 300,
            "weapons": [],
        }

    def direct_aim(self, me, target, gun_speed=None):
        target_pos = target["pos"][:]
        target_pos[2] += 4.5 if target["kind"] != "crate" else 1.2
        muzzle_z = me["pos"][2] + me["stats"].get("muzzle_height", 7)
        origin = [me["pos"][0], me["pos"][1], muzzle_z]
        point = lead_position(origin, {"pos": target_pos, "vel": target.get("vel", [0, 0, 0])}, gun_speed)
        dx, dy, dz = point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]
        return [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]

    def act(self, obs):
        me = obs["self"]
        self.remember(obs)
        action = {"tick": obs["tick"]}
        now = obs["t"]

        hp_ratio = me["hp"] / max(1.0, me["max_hp"])
        if hp_ratio < 0.55:
            self.recovering = True
        elif hp_ratio >= 0.88:
            self.recovering = False

        if obs["offer"]:
            action["pick"] = self.upgrade_choice(obs, me)
        pickup_action = self.weapon_pickup_action(obs, me)
        if pickup_action:
            action.update(pickup_action)

        enemies = self.visible_mechs(obs)
        target = self.choose_target(obs, me) if enemies else self.crate_target(obs, me)
        safe_goal = self.safe_goal(obs, me)
        target_distance = distance(target["pos"], me["pos"]) if target else 10000

        threats = []
        many = len(enemies) >= 3 or (len(enemies) >= 2 and hp_ratio < 0.65)
        under_fire = now - self.last_damage < 3.0
        for enemy in enemies:
            d = distance(enemy["pos"], me["pos"])
            if d < 105:
                ex = enemy["pos"][0] - me["pos"][0]
                ey = enemy["pos"][1] - me["pos"][1]
                away = unit(-ex, -ey)
                strength = (105 - d) / 105
                if enemy["id"] == self.last_attacker:
                    strength *= 1.8
                if enemy.get("charging"):
                    strength *= 1.6
                threats.append((away, strength))

        if self.recovering or many or under_fire and hp_ratio < 0.72:
            goal = safe_goal
            mode = "cover" if under_fire or many else "recover"
        else:
            goal = safe_goal
            mode = "zone"
        self.mode = mode

        if target and target["kind"] != "crate" and not self.recovering and not many:
            enemy_hp_ratio = target.get("hp", 1400) / max(1.0, target.get("max_hp", 1400))
            disadvantaged = hp_ratio + 0.12 < enemy_hp_ratio
            if under_fire and target_distance < 70 and not disadvantaged:
                mode = "fight"
            elif enemy_hp_ratio < 0.30 and target_distance < 130:
                mode = "finish"
            elif target_distance < 55:
                mode = "fight"

        cube = None
        if mode in ("zone", "roam") and target is None:
            candidates = [
                item for item in obs["cubes"]
                if distance(item["pos"], me["pos"]) < 30
                and abs(item["pos"][2] - me["pos"][2]) < 12
            ]
            if candidates:
                cube = min(candidates, key=lambda item: distance(item["pos"], me["pos"]))
                goal = cube["pos"][:2]
                mode = "cube"

        if mode not in ("fight", "finish") and not many:
            for pickup in obs["pickups"]:
                kind = pickup["kind"]
                d = distance(pickup["pos"], me["pos"])
                held = {weapon["kind"] for weapon in me["weapons"] if weapon}
                if kind not in held and kind in ("laser", "lightning") and d < 100:
                    goal = pickup["pos"][:2]
                    mode = "loot"
                    break
        self.mode = mode

        move = self.route_goal(goal, me)
        if threats:
            flee = [
                sum(direction[axis] * weight for direction, weight in threats)
                for axis in (0, 1)
            ]
            flee = unit(flee[0], flee[1])
            danger = max(weight for _, weight in threats)
            if mode in ("cover", "recover") or hp_ratio < 0.72:
                move = unit(move[0] + flee[0] * (1.2 + danger),
                            move[1] + flee[1] * (1.2 + danger))
            elif mode == "fight" and target_distance < 45:
                tangent = [
                    -unit(target["pos"][0] - me["pos"][0], target["pos"][1] - me["pos"][1])[1] * self.strafe_sign,
                    unit(target["pos"][0] - me["pos"][0], target["pos"][1] - me["pos"][1])[0] * self.strafe_sign,
                ]
                move = unit(move[0] * 0.45 + tangent[0] * 0.55,
                            move[1] * 0.45 + tangent[1] * 0.55)

        dodge = self.incoming_dodge(obs, me)
        if dodge:
            move = unit(move[0] * 0.55 + dodge[0] * 0.45,
                        move[1] * 0.55 + dodge[1] * 0.45)

        goal_distance = distance(goal, me["pos"])
        urgent = not me["in_zone"] or (mode in ("cover", "recover") and under_fire)
        boost = (
            math.hypot(move[0], move[1]) > 0.1
            and me["energy"] > (30 if urgent else 45)
            and (urgent or goal_distance > 50)
        )
        jump = bool(dodge and me["grounded"])
        action.update(move=move, boost=bool(boost), jump=jump)
        if self.last_position is not None:
            moved = distance(me["pos"], self.last_position)
            trying = math.hypot(*self.previous_move) > 0.5
            self.stuck = self.stuck + 0.1 if trying and moved < 0.13 else max(0.0, self.stuck - 0.2)
        self.last_position = me["pos"][:]
        self.previous_move = move[:]

        aim = [me["aim"][0], me["aim"][1]]
        fire = [False, False]
        if target:
            target_distance = distance(target["pos"], me["pos"])
            if mode in ("fight", "finish") or under_fire or target["kind"] == "crate":
                gun_slot = next(
                    (i for i, weapon in enumerate(me["weapons"])
                     if weapon and weapon["kind"] == "gun"),
                    None,
                )
                lightning_slot = next(
                    (i for i, weapon in enumerate(me["weapons"])
                     if weapon and weapon["kind"] == "lightning"),
                    None,
                )
                laser_slot = next(
                    (i for i, weapon in enumerate(me["weapons"])
                     if weapon and weapon["kind"] == "laser"),
                    None,
                )

                if target["kind"] == "crate":
                    gun = me["weapons"][gun_slot] if gun_slot is not None else None
                    if gun:
                        gun_speed = gun["stats"].get("speed", 252)
                        aim = self.direct_aim(me, target, gun_speed)
                        yaw_error = abs(wrap_angle(aim[0] - me["aim"][0]))
                        pitch_error = abs(aim[1] - me["aim"][1])
                        if gun.get("ammo", 0) > 0 and gun["reload"] == 0:
                            fire[gun_slot] = yaw_error < 0.12 and pitch_error < 0.12
                elif lightning_slot is not None and target_distance < 45:
                    weapon = me["weapons"][lightning_slot]
                    aim = self.direct_aim(me, target)
                    yaw_error = abs(wrap_angle(aim[0] - me["aim"][0]))
                    if yaw_error < math.radians(weapon["stats"].get("cone_deg", 30)) * 0.5:
                        if me["energy"] >= weapon["stats"].get("energy", 3):
                            fire[lightning_slot] = True
                elif laser_slot is not None and target_distance < 280:
                    weapon = me["weapons"][laser_slot]
                    aim = self.direct_aim(me, target)
                    yaw_error = abs(wrap_angle(aim[0] - me["aim"][0]))
                    pitch_error = abs(aim[1] - me["aim"][1])
                    aligned = yaw_error < 0.12 and pitch_error < 0.12
                    charge = weapon.get("charge", 0)
                    full_charge = weapon["stats"].get("charge", 0.75)
                    if weapon["cooldown"] <= 0.02:
                        fire[laser_slot] = charge < full_charge * 0.88 or not aligned
                elif gun_slot is not None and target_distance < 230:
                    gun = me["weapons"][gun_slot]
                    gun_speed = gun["stats"].get("speed", 252)
                    aim = self.direct_aim(me, target, gun_speed)
                    yaw_error = abs(wrap_angle(aim[0] - me["aim"][0]))
                    pitch_error = abs(aim[1] - me["aim"][1])
                    if gun.get("ammo", 0) > 0 and gun["reload"] == 0:
                        fire[gun_slot] = yaw_error < 0.12 and pitch_error < 0.12
                    if gun["reload"] == 0 and gun.get("ammo", 12) <= 2:
                        action["reload"] = True

        elif self.last_damage_bearing is not None and now - self.last_damage < 5:
            aim = [self.last_damage_bearing, 0.0]
        else:
            for sound in obs["sounds"]:
                if sound["kind"] in ("gun", "mortar", "explosion", "laser", "lightning"):
                    aim = [sound["bearing"], 0.0]
                    break

        action["aim"] = aim
        action["fire"] = fire
        mf.draw.point([goal[0], goal[1], me["pos"][2]], "#40c0ff", 5)
        mf.draw.label(
            "%s | hp %.0f | cubes %d | alive %d"
            % (mode, me["hp"], me["cubes"], obs["alive"])
        )
        if target:
            mf.draw.line(
                [me["pos"][0], me["pos"][1], me["pos"][2] + 8],
                [target["pos"][0], target["pos"][1], target["pos"][2] + 4.5],
                "#ff5040",
            )
        return action


mf.run(Bot)
