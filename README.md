# Builder-Apex

Placement-oriented bot for the Builders Mech Royale track. Keep `Dockerfile`,
`example_bot.py`, `mechfight.py`, and `navigation.py` at the repository root.

The policy emphasizes survival over chasing kills:

- It avoids rushing to the center: it relocates away from the shared drop area,
  holds safe ground, and estimates travel time before staging for the next zone.
- It avoids nearby or actively threatening mechs, prioritizes returning to the zone, and
  sidesteps incoming projectiles by comparing predicted miss distances.
- It builds a coarse walkability grid from the mounted arena map and routes
  around buildings and stadium walls instead of relying only on straight-line movement.
- It opportunistically collects nearby cubes, supply drops, and useful
  weapons, while avoiding objectives when an enemy is close.
- It picks upgrades using their protocol IDs, prioritizes survivability and mobility,
  tracks recent attackers, leads gun shots, and solves mortar trajectories.
- It returns fire at useful ranges when attacked or aimed at, while avoiding
  unnecessary long-range exchanges.

The bot passed the track's protocol checker on its 300-observation test stream
with the arena map mounted. That confirms protocol and timing compliance, not
match performance; validate the image and compare results across practice
matches before submitting.

Build the Linux image from this folder:

```powershell
docker build --platform linux/amd64 -t builders-apex-bot .
```
