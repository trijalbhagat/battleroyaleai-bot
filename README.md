# Builder-Apex

Aggressive bot for the Builders Mech Royale track. Keep `Dockerfile`,
`example_bot.py`, and `mechfight.py` at the repository root. All custom bot
logic is implemented inside `example_bot.py`; `mechfight.py` is only the
official protocol transport wrapper.

The policy aggressively pursues visible opponents while healthy, with a
low-health recovery mode:

- It selects a nearby, vulnerable, or actively threatening opponent and
  boost-chases them while keeping the safe zone as a hard constraint.
- Below 42% health it switches to recovery movement and avoids threats; recovery
  ends at 72% health to prevent rapid mode-flipping.
- It continues returning fire during recovery, dodges incoming projectiles, and
  prioritizes zone entry when outside the safe zone.
- It builds a coarse walkability grid from the mounted arena map and routes
  around buildings and stadium walls instead of relying only on straight-line movement.
- It opportunistically collects nearby cubes, supply drops, and useful weapons
  when no opponent is visible.
- It picks upgrades using their protocol IDs, prioritizes survivability and mobility,
  tracks recent attackers, leads gun shots, and solves mortar trajectories.
- It uses weapon-specific range and aiming rules for gun, laser, lightning, and mortar.

The bot passed the track's protocol checker on its 300-observation test stream
with the arena map mounted. That confirms protocol and timing compliance, not
match performance; validate the image and compare results across practice
matches before submitting.

Build the Linux image from this folder:

```powershell
docker build --platform linux/amd64 -t builders-apex-bot .
```

The report [ADVANCED_RESEARCH_REPORT.md](ADVANCED_RESEARCH_REPORT.md) is the
architecture reference for this implementation. The submitted bot does not
depend on external custom Python modules or a downloaded model.
