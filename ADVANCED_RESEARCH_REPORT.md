# Advanced Research Project: AI Battle-Royale Agent

**Scope:** autonomous, officially supported or simulator-based play only  
**Model caveat:** GPT-6 is not available to this Copilot session. This report applies the highest reasoning capability available in the session and is written without live-match or benchmark claims.

## Executive recommendation

The best first architecture is **not a monolithic neural policy**. It is a
**hybrid hierarchical system**:

1. a deterministic perception and state-estimation layer;
2. a behavior-tree or finite-state strategic layer;
3. a rule-based safety and legality layer;
4. hierarchical planners for zone, navigation, cover, loot, and combat;
5. one or more learned value or policy components that optimize bounded
   choices only after a strong baseline exists.

This recommendation follows the strongest engineering conclusion for
Battle-Royale domains: partial observability, irregular physics, large action
spaces, sparse placement rewards, and consequential mistakes make pure
end-to-end RL expensive and difficult to debug. A hybrid system provides:

- predictable safety;
- fast decisions;
- inspectable failures;
- strong performance with limited data;
- a natural insertion point for RL experiments.

The reinforcement-learning program should begin with PPO or SAC on
**small, fixed-scope subproblems**, such as dodge-heading selection or
short-horizon zone entry. It should not initially control the entire mech.

---

# Deliverable 1 — Research report

## 1. Problem formulation

A Battle-Royale agent operates in a discounted or episodic partially observable
Markov decision process:

\[
\mathcal P(s_{t+1},r_t\mid s_t,a_t),\qquad
a_t\sim\pi_\theta(a_t\mid o_{0:t})
\]

where:

- \(o_t\) is the locally observable state;
- \(s_t\) is the latent true state;
- \(a_t\) is a movement, aim, inventory, and combat action;
- \(r_t\) must reflect survival, zone safety, HP preservation, useful
  resources, and legitimate combat outcomes.

A mech, human, or AI-controlled character is a time-staged policy:

\[
\pi(o_t)=\text{SafetyGate}(
\text{ThreatMonitor},
\text{StatePolicy},
\text{Planner},
\text{LegalActionFilter})
\]

The environment is not fully observable. Enemy identities, positions,
inventory, intent, and plans are latent. The agent must track unknown state,
not merely react to the latest frame.

## 2. Existing research and technologies

### 2.1 Deep Q-Networks

Mnih et al. trained DQN to play Atari games directly from pixels and
demonstrated that a single model can learn many tasks from reward alone
([Nature, 2015](https://doi.org/10.1038/nature14236)).

Double DQN reduces maximization overestimation by separating action selection
from value estimation
([Hasselt, Guez, and Silver, 2015](https://arxiv.org/abs/1509.06461)).

DQN is useful for small or discretized action sets. It is a poor default for
continuous two-dimensional movement, continuous aiming, and action composition
with inventory and weapon state. Discretizing movement into, for example, 16
headings can make DQN practical for an isolated controller.

### 2.2 PPO

PPO uses clipped policy-gradient objectives to limit destructive policy updates
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)).

PPO is currently the most practical general-purpose choice for a continuous
or hybrid action-space Battle-Royale simulator:

- stable for many practical environments;
- easy to parallelize;
- supports continuous actions;
- has mature implementations;
- is robust to sparse but shaped rewards if the reward is carefully defined.

Its disadvantages are sample inefficiency and policy updates that can plateau
without curriculum or opponent diversity.

### 2.3 A2C

A2C synchronously combines a value baseline and a policy gradient. It is
conceptually simpler than PPO and can work for moderate action spaces.

A2C is usually weaker than PPO for final reliability in a production policy
because A2C offers less update conservatism. It may still be useful as a fast
research baseline.

### 2.4 SAC

SAC maximizes a combination of return and policy entropy
([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)).

SAC is usually attractive for:

- continuous movement;
- continuous aiming;
- small-to-medium state vectors;
- sample-efficient off-policy learning.

The tradeoffs are sensitivity to replay quality, high-dimensional action
stability, and safety. Pure SAC should not output raw movement that is allowed
to enter a wall or leave the zone. A legal-action projection or constrained
controller should sit above it.

### 2.5 Multi-agent reinforcement learning

MADDPG introduced centralized training with decentralized execution for
mixed cooperative-competitive environments
([Lowe et al., 2017](https://arxiv.org/abs/1706.02275)).

In a Battle-Royale simulator, each player is a single policy, but other-agent
behavior changes the transition distribution. This makes the environment
non-stationary. Single-agent PPO can still learn, but it should train against:

- static baseline policies;
- rule-based bots;
- earlier checkpoints;
- league samples;
- different map seeds;
- mixed cooperative squads if supported.

MADDPG-style centralized critics can help only if joint opponent information
is available during training and hidden at deployment.

The policy should not require opponent policy distributions at runtime.

### 2.6 Hierarchical reinforcement learning

Hierarchical methods decompose decisions into options or subpolicies. The
options framework defines temporally extended actions
([Sutton, Precup, and Singh, 1999](https://doi.org/10.1016/S0004-3702(99)00052-1)).

This maps naturally onto Battle-Royale AI:

- `Relocate`
- `Recover`
- `AcquireResource`
- `Engage`
- `Retreat`
- `Capture`
- `HoldPosition`

Each high-level mode can call one or more low-level controllers. This is
preferable to giving the policy 100 Hz movement plus all inventory decisions
at once.

### 2.7 Imitation learning

Behavioral cloning is the cheapest way to obtain a launch policy from
demonstration data, but it inherits demonstrator mistakes and struggles
with covariate shift.

GAIL learns a discriminator to distinguish expert from generated trajectories
([Ho and Ermon, 2016](https://arxiv.org/abs/1606.03417)).

Imitation is useful for:

- initialization;
- recording human or expert actions;
- generating a baseline in an existing engine;
- bootstrapping a strategic policy.

It is less suitable as the sole learning method because human demonstrations
often contain rare-death and multi-agent edge cases poorly.

### 2.8 Planning and learned models

MuZero learns latent dynamics and performs model-based tree planning
([Schrittwieser et al., 2020](https://arxiv.org/abs/1911.08265)).

For Battle Royale, full MuZero-style planning is likely too expensive:

- many visible and hidden agents;
- continuous state;
- stochastic loot;
- high branching factor;
- latency requirements.

A lightweight domain model is more appropriate:

- projectile constant-velocity predictions;
- short zone interpolations;
- shortest-path navigation;
- candidate resource utility;
- short-horizon survival simulations.

Monte Carlo rollouts can augment learned policies only if constrained to
milliseconds.

### 2.9 Game-AI precedents

AlphaStar combines deep learning, self-play, population-based training, and
strategic abstractions to reach Grandmaster-level StarCraft II play
([Nature, 2019](https://www.nature.com/articles/s41586-019-1714-7)).

AlphaStar’s lesson is not “train one network on everything.” It demonstrates
that:

- population diversity matters;
- long-horizon strategies must be decomposed;
- strategic and fine-control behavior can be separated;
- infrastructure is as important as the agent.

OpenAI’s hide-and-seek project showed emergent tool use and policy diversity
in a multi-agent environment
([arXiv:1909.07528](https://arxiv.org/abs/1909.07528)).

Emergent behavior is difficult to compare against placement KPIs and is not
automatically transferable to a fixed competition. It remains relevant for
curriculum and adversarial diversity.

---

# Deliverable 2 — System architecture

## 1. High-level architecture

```text
                         Official Supported API / Simulator
                                       |
                              Observation adapter
                                       |
                    +------------------+------------------+
                    |                                     |
          Perception normalizer                 Geometry interface
                    |                                     |
                    +------------------+------------------+
                                       |
                             State estimator / tracker
                     self, actors, loot, projectiles,
                     zone, damage, inventory, confidence
                                       |
                       +---------------+---------------+
                       |                               |
                 Threat estimator              Strategic model
                 survival, TTK,               mode and objective
                 cover, exposure              utility
                       |                               |
                       +---------------+---------------+
                                       |
                             Hierarchical policy
                    zone / recover / acquire / engage
                                       |
                       +---------------+---------------+
                       |                               |
                 Route planner                   Combat planner
                 A*/JPS/grid                    aim / weapon / dodge
                       |                               |
                       +---------------+---------------+
                                       |
                           Safety and legality filter
                       zone / wall / cooldown / LOS
                                       |
                              Executed action

          +------------------------------------------------+
          | Learning loop: replay / simulation / eval suite |
          | PPO/SAC subpolicy, opponent league, telemetry   |
          +------------------------------------------------+
```

## 2. Component definitions

### 2.1 Observation adapter

Responsibilities:

- validate protocol or simulator data;
- normalize field names;
- convert unknown or optional fields;
- enforce type and range constraints;
- attach a monotonic observation time.

It must not implement tactics. It creates a stable internal representation.

### 2.2 Perception normalizer

Responsibilities:

- convert world-frame coordinates;
- merge ego-relative and world-relative observations;
- normalize velocities and heights;
- add derived fields such as distance and relative yaw;
- mark unknown or stale observations.

Do not assume missing HP or inventory means zero. It means unknown.

### 2.3 Static geometry interface

Uses:

- terrain height;
- building and stadium prisms;
- bridges and holes;
- water and nonblocking scenery;
- trees and canopy metadata;
- navigation grid.

Core APIs:

```text
ground_height(x, y) -> height
traversable(p, radius, height) -> bool
line_of_sight(a, b) -> clear|blocked|unknown
shot_clear(a, b, time_budget) -> clear|blocked|unknown
shortest_route(start, goal) -> route
distance_to_cover(p, threat) -> float
```

Unknown geometry must not be interpreted as clear. Shooting and cover decisions
need different semantics:

- for a shot, unknown geometry should suppress firing;
- for cover, unknown geometry should suppress a “covered” label.

### 2.4 State estimator

Use separate trackers for:

- visible enemies;
- missed or stale enemies;
- projectiles;
- beams;
- damage events;
- loot;
- crates;
- supply drops;
- zone geometry;
- inventory;
- HP and energy.

For a known enemy, estimate:

\[
\hat p(t+\Delta)
=p_t+\bar v_t\Delta
\]

where velocity estimates should be smoothed and age-weighted.

Track confidence:

\[
c_e(t)=e^{-(t-t_{\text{seen}})/\tau}
\]

Confidence should decay by source and recall class. A current visible track
is more reliable than a noisy sound track.

### 2.5 Threat estimator

Outputs:

- imminent projectile collisions;
- charging or aimed enemies;
- outnumbering;
- line-of-sight exposure;
- estimated incoming damage;
- rear-threat exposure;
- zone damage.

The threat value of actor \(i\) can be:

\[
T_i=d_{zone}+d_{aim}+d_{weapon}+d_{hp}+d_{range}+d_{rear}
\]

It should be a vector or structured result, not a single opaque float.

### 2.6 Strategic policy

The strategic policy chooses:

```text
Relocate
Recover
Acquire
Engage
Finish
Hold
Capture
Retreat
```

It should use hysteresis:

\[
enterRecovery(H<0.55),\qquad exitRecovery(H\geq0.88)
\]

This prevents rapid switching whenever HP crosses a boundary.

### 2.7 Route planner

Use:

- Dijkstra or A* for static road coverage;
- JPS only after grid-level benchmarking;
- navigation-mesh in a 3D engine;
- receding-horizon replanning.

A route cost should include:

\[
C(route)=D+\lambda_zT_z+\lambda_tT_t+\lambda_eExposure
\]

where:

- \(D\) is distance;
- \(T_z\) is zone risk;
- \(T_t\) is threat exposure;
- \(E\) is expected damage.

### 2.8 Combat planner

Produce a weapon plan:

```text
WeaponPlan:
  weapon slot
  target id
  desired yaw/pitch
  intercept time
  release condition
  max feasible distance
  estimated TTK
  exposure score
  risk
```

Do not fire merely because a target is visible. Fire only if:

- the weapon is ready;
- aim error is acceptable;
- target ray is clear;
- target is in a useful range;
- the state policy permits engagement.

### 2.9 Safety filter

Safety must be final:

- protect zone deadline;
- reject out-of-bounds route samples;
- reject empty or stale action fields;
- prevent fire when cooldown/ammo/energy is invalid;
- prevent a laser from being dropped or swapped while charging;
- reject supply-drop capture in dangerous overlap;
- limit boost/jet energy reserve;
- emit finite values only.

### 2.10 Learning loop

```text
Simulator episodes
    -> trajectories
    -> reward/telemetry
    -> policy update
    -> checkpoint
    -> league evaluation
    -> curriculum selection
    -> repeat
```

The runtime bot should not learn from live matches unless the competition
explicitly permits online learning and the architecture can roll back unsafe
changes.

---

# Deliverable 3 — Technology selection

## 1. Technology comparison

| Technology | Strengths | Weaknesses | Recommended role |
|---|---|---|---|
| Python | Rapid development, ecosystem, easy debugging | Slow numerical loops unless vectorized | Main bot, rules, and lightweight RL |
| PyTorch | Flexible RL, strong ecosystem, TorchScript/ONNX export | GPU and packaging overhead | Neural policy and simulator training |
| Gymnasium | Standard environment API | Not enough for fixed 16-agent match semantics without wrappers | API base for custom environment |
| Stable-Baselines3 | Reliable PPO/SAC/DQN implementations and tooling | Less flexible distributed training | First RL implementation |
| SB3 Zoo | Reproducible scripts and benchmark conventions | Adds setup complexity | Hyperparameter and benchmark baseline |
| Unity ML-Agents | Editor-based agents, 3D physics, parallel environments | Engine integration and build complexity | Preferred visual 3D simulator |
| Unreal | High-fidelity 3D, strong assets | Larger infrastructure and Python integration cost | Optional large-scale 3D version |
| Ray RLlib | Distributed MARL and scalable policy training | Steeper setup, cluster dependencies | Later population/self-play stage |
| PettingZoo | Standard multi-agent environment API | Not a complete 3D engine | Shared MARL wrapper |
| ONNX/TorchScript | Lower-latency deployment | Export and control-flow limits | Runtime model deployment |
| NumPy | Fast state and route calculations | Python overhead in repeated loops | State and geometry preprocessing |

## 2. Python versus engine-native development

Python is appropriate for:

- state estimation;
- rule evaluation;
- Gymnasium interfaces;
- PPO/SAC training;
- offline analysis.

Python is not automatically appropriate for:

- the exact 50 ms competitive loop;
- high-frequency ray or rigid-body queries;
- large map geometry;
- multi-agent simulation at millions of steps.

The engine should own physics and perception. Python should own policy and
training unless an official 50 ms API makes this impossible.

## 3. Recommended stack

For a first implementation:

```text
Environment:
  Custom authoritative Python or Unity 3D simulator

RL interface:
  Gymnasium API

Algorithms:
  PPO for baseline
  SAC for continuous subpolicies
  Imitation/bootstrap optional

Training:
  Stable-Baselines3
  vectorized Gymnasium environments
  TensorBoard
  Seeds and YAML configuration

Later scale:
  Ray RLlib
  Population self-play
  Distributed rollouts

Runtime deployment:
  Pure rules + exported/quantized subpolicy
  TorchScript or ONNX only after latency tests
```

Official documentation:

- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Stable-Baselines3 JMLR paper](https://jmlr.org/papers/v22/20-1364.html)
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Unity ML-Agents](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/index.html)
- [PyTorch Inference](https://pytorch.org/serve/)

---

# Deliverable 4 — Development roadmap

## Milestone 0: specification freeze

Duration: 1–2 weeks.

Deliverables:

- formal observation/action schema;
- authoritative game rules;
- determinism and seed policy;
- test maps;
- protocol tests;
- baseline bots;
- replay format.

Exit criteria:

- same seed reproduces same match;
- every bot action is schema valid;
- no hidden state is available to the policy unless it is published to the
  observation.

## Milestone 1: deterministic baseline

Duration: 2–3 weeks.

Implement:

- finite-state or behavior-tree mode;
- zone pathfinding;
- resource collection;
- health recovery;
- simple leading aim;
- cooldown/ammo checks;
- stuck recovery;
- loss and safety logging.

Exit criteria:

- baseline completes matches without crashes;
- protocol test passes;
- placement score is reproducible across seed batches.

## Milestone 2: geometry and combat

Duration: 2–4 weeks.

Implement:

- static map cache;
- 2D and 3D line checks;
- short projectiles;
- cover candidates;
- weapon-specific aiming;
- target TTK estimate;
- 16/24-direction dodge candidates.

Exit criteria:

- no shot is fired into unknown geometry;
- cover and route plans are measurable;
- target selection adds placement value in simulation.

## Milestone 3: first RL subpolicy

Duration: 3–5 weeks.

Train one bounded policy:

- observation: local threat, zone vector, HP, cover score;
- action: 16 discrete headings or 2D continuous movement;
- reward: survival, safe-zone entry, avoided damage, and useful cover.

Use PPO or SAC. Mask invalid actions.

Exit criteria:

- policy never worsens on a fixed replay suite;
- policy beats a rule-based dodge baseline;
- latency budget remains within 10 ms at the policy level.

## Milestone 4: hierarchical policy

Duration: 4–6 weeks.

Train a high-level objective policy:

- safe-zone targeting;
- resource detour;
- engage or recover;
- endgame hold.

Keep low-level behavior deterministic.

Exit criteria:

- stable mode hysteresis;
- no thrashing;
- higher placement than the deterministic baseline in paired seeds.

## Milestone 5: opponent league

Duration: 4–8 weeks.

Create:

- easy stationary bot;
- aggressive hunter;
- cautious zone bot;
- weapon-specialist bots;
- evolving historical bots;
- mixed difficulty seeds.

Train against league opponents and sample teams of different strengths.

Exit criteria:

- Elo or pairing evaluation improves;
- no opponent-specific only policy;
- holdout league scores do not decay.

## Milestone 6: performance and robustness

Duration: 2–4 weeks.

Add:

- dropout or adversarial observation noise;
- state-aging tests;
- latency degradation tests;
- bad JSON and missing fields;
- packet queue handling;
- action replay;
- crash fallback;
- model rollback.

Exit criteria:

- worst-case decision latency below budget;
- no invalid action;
- deterministic replay within accepted numerical tolerance.

---

# Deliverable 5 — Implementation blueprint

## 1. Recommended folder structure

```text
battle-royale-ai/
├── README.md
├── pyproject.toml
├── Dockerfile
├── configs/
│   ├── baseline.yaml
│   ├── curriculum.yaml
│   ├── ppo.yaml
│   └── eval.yaml
├── src/
│   └── brbot/
│       ├── __init__.py
│       ├── protocol/
│       │   ├── adapter.py
│       │   ├── schema.py
│       │   └── action.py
│       ├── state/
│       │   ├── estimator.py
│       │   ├── enemy_track.py
│       │   ├── item_track.py
│       │   └── zone_track.py
│       ├── geometry/
│       │   ├── world.py
│       │   ├── grid.py
│       │   ├── astar.py
│       │   └── visibility.py
│       ├── strategy/
│       │   ├── state_machine.py
│       │   ├── mode_policy.py
│       │   ├── objectives.py
│       │   └── hysteresis.py
│       ├── combat/
│       │   ├── target.py
│       │   ├── intercept.py
│       │   ├── weapons.py
│       │   └── dodge.py
│       ├── resources/
│       │   ├── inventory.py
│       │   ├── loot.py
│       │   ├── crates.py
│       │   └── drops.py
│       ├── safety/
│       │   ├── legality.py
│       │   ├── fallback.py
│       │   └── watchdog.py
│       ├── rl/
│       │   ├── env.py
│       │   ├── reward.py
│       │   ├── wrappers.py
│       │   ├── masks.py
│       │   └── train.py
│       ├── league/
│       │   ├── opponents.py
│       │   ├── pairing.py
│       │   └── elo.py
│       └── bot.py
├── simulator/
│   ├── engine.py
│   ├── world.py
│   ├── actors.py
│   ├── combat.py
│   ├── zone.py
│   └── loot.py
├── tests/
│   ├── test_protocol.py
│   ├── test_geometry.py
│   ├── test_state_machine.py
│   ├── test_combat.py
│   ├── test_navigation.py
│   ├── test_rewards.py
│   └── test_regression.py
├── tools/
│   ├── replay.py
│   ├── trace.py
│   ├── league.py
│   └── benchmark.py
└── reports/
    ├── experiments/
    └── ablations/
```

## 2. Core interfaces

```text
ObservationAdapter.normalize(raw) -> Observation
StateEstimator.update(observation) -> WorldState
ThreatEstimator.predict(world_state) -> ThreatSet
StrategicPolicy.select(world_state, threats) -> Goal|Mode
NavigationController.plan(world_state, goal) -> Route
CombatController.plan(world_state, target) -> WeaponPlan
SafetyFilter.apply(world_state, command) -> Action
```

## 3. State representation

Use structured state:

```text
SelfState:
  position_xyz
  velocity_xyz
  actual_aim
  hp/max_hp
  energy/max_energy
  grounded
  in_zone
  inventory
  active_cooldowns

TrackedActor:
  id
  position
  velocity
  actual_aim
  hp/max_hp if known
  weapons if known
  seen_time
  confidence
  source_kind

Threat:
  source id
  distance
  bearing
  imminent impact time
  miss/overlap distance
  rear
  weapon
  confidence

ZoneState:
  center/radius now
  center/radius next
  shrink_start/end
  dps

Objective:
  kind
  position
  utility
  eta
  service_time
  zone_risk
  threat_risk
  expires
```

## 4. Reward function

Avoid a single dense reward. A component structure is easier to tune:

\[
R_t=
r_{\text{placement}}+
r_{\text{survival}}+
r_{\text{zone}}+
r_{\text{damage}}+
r_{\text{resource}}+
r_{\text{combat}}+
r_{\text{safety}}
\]

### Recommended components

**Placement and survival**

\[
R_{\text{survival}}=\alpha \Delta t_{\text{alive}}
\]

\[
R_{\text{placement}}=\beta(16-\text{place})
\]

Use placement at episode end as a primary metric, not as a single dense reward
if off-policy learning instability becomes too high.

**Zone safety**

\[
R_{\text{zone}}=-\gamma(1-me_{inzone})-\delta
\max(0,d-r_{safe})
\]

**Health**

\[
R_{\text{health}}=\epsilon(h_t-h_{t-1})
\]

**Combat**

\[
R_{\text{combat}}=w_hh_{dealt}-w_rf_{received}
\]

Do not reward kills more strongly than placement. Kills also create noisy
encouragement to leave safety.

**Resources**

\[
R_{\text{resource}}=w_c\Delta c+w_u\Delta u
\]

**Unsafe behavior**

\[
R_{\text{safety}}=-w_w\mathbb I[wall]-w_z\mathbb I[outzone]
-w_f\mathbb I[invalidfire]
\]

The implementation may normalize each component to prevent magnitude
domination.

## 5. Observation/action spaces

### 5.1 Recommended initial observation

Use a fixed vector plus an optional map feature tensor:

```text
ego:
  normalized hp, energy, position
  velocity
  aim
  inventory
  remaining cooldowns

zone:
  relative center
  radius
  shrink fraction
  next-circle relative center/radius
  dps

threat summary:
  nearest attacker distance/bearing
  charging flag
  aim confidence
  number nearby
  incoming projectiles

objective:
  next goal relative vector
  route distance
  objective type
  objective age

mode:
  one-hot high-level mode
  recovery state
```

### 5.2 Initial action space

A safe action interface:

```text
mode in [recover, relocate, acquire, engage, hold]
move_x, move_y
aim_yaw, aim_pitch
fire_slot_mask
pickup choice
upgrade choice
boost/jump
```

For PPO/SAC, the controller can output continuous desired movement and aim,
but the high-level strategic policy should remain discrete.

Action mask invalid choices. Never allow a model to own:

- illegal pickup;
- invalid weapon slot;
- out-of-range fire;
- zone-breaking decisions;
- model-generated raw debug data.

## 6. Curriculum

### Phase A

Environment:

- one AI versus stationary targets;
- no zone for 60 seconds;
- gun only;
- fixed map.

Learn:

- shooting;
- aim leading;
- movement;
- cover.

### Phase B

Environment:

- 1 player versus 3/6/12 weak bots;
- shrinking zone;
- standard weapon loot;
- random spawns.

Learn:

- threat avoidance;
- zone timing;
- resource tradeoffs.

### Phase C

Environment:

- full 16-player matches;
- mixed opponents;
- supply drops;
- upgrades;
- map variants.

Learn:

- placement;
- mode switching;
- late game.

### Phase D

Environment:

- population league;
- no reward leakage;
- hidden state perturbation;
- blocked geometry;
- high agent speed.

Learn:

- robustness;
- adaptation.

## 7. Self-play

Use:

- 50% current checkpoint;
- 25% earlier checkpoints;
- 15% rule-based opponents;
- 10% random or novelty opponents.

Store historical policies. Rank them using paired leagues over fixed seed
batches.

Do not select models based on one match. Repeated paired comparison reduces
seed and opponent variance.

## 8. Preventing overfitting

- seed all world randomness;
- train on multiple map seeds;
- randomize opponent mix;
- randomize HP and equipment only within legitimate rules;
- hold out a seed split;
- freeze a league test set;
- penalize policy-specific exploitability;
- include observation noise;
- test 20–30 rematches on identical seeds.

---

# Deliverable 6 — Evaluation strategy

## 1. Metrics

### Core placement metrics

- mean placement;
- median placement;
- win rate;
- placement by 16-point bracket;
- top-5 and top-10 rate;
- survival time;
- zone-death rate;
- HP at match end.

### Combat metrics

- damage dealt;
- damage received;
- damage per minute;
- shots fired versus effective hits;
- TTK distribution;
- finish conversion;
- rear-hit rate;
- weapon-specific efficiency.

### Resource metrics

- time to laser/lightning;
- cube collection rate;
- upgrade count;
- crate destruction;
- NPC kill efficiency;
- supply-drop capture success;
- average loot travel distance;
- loot deaths, i.e. resources collected immediately before death.

### Navigation metrics

- distance to goal;
- route replans;
- stuck time;
- walls crossed;
- zone deadline misses;
- cover-entry success;
- route cost versus straight-line cost.

### Decision metrics

- p50/p90/p99/max decision latency;
- invalid action rate;
- queue/drop count;
- timeout count;
- state mode changes per minute;
- mode dwell time;
- legal action mask violations.

### Adaptability metrics

- win or placement delta against a stronger opponent;
- paired seed comparison;
- performance on an unseen map;
- performance under observation delay;
- performance with partial or noisy visibility;
- recovery after HP and weapon loss.

## 2. Paired evaluation

For policy A and B:

1. generate fixed seed batches;
2. swap agent slots;
3. run A vs B against the same opponents;
4. report mean placement and confidence intervals;
5. analyze paired differences per seed;
6. reject models that improve against only one bot family.

Do not report only average placement. Include:

- 25th percentile;
- median;
- 75th percentile;
- catastrophic placement rate;
- kill and survival contributions.

## 3. Minimum test suite

Tests should include:

- gun intercept stationary/moving;
- gun reload and empty mag;
- laser charge/release;
- laser not fired during unsafe charge cancellation;
- lightning cone and line of sight;
- mortar low/high arc;
- mortar self-splash rejection;
- cube pickup and upgrade thresholds;
- weapon swap and empty-slot take;
- contested supply drops;
- supply drop capture and holder change;
- early, middle, late, and zero-radius zones;
- low HP hysteresis;
- target hp unknown;
- packed threats;
- stale enemy memory;
- zone damage with no bearing;
- rear damage bearing;
- route blocked, no route, and inaccessible goal;
- stuck recovery;
- observation queue;
- missing optional fields;
- 50 ms latency degradation.

## 4. Benchmark ladder

Recommended opponent tiers:

| Tier | Opponent |
|---|---|
| 0 | no-op or random |
| 1 | fixed shooter |
| 2 | zone-only bot |
| 3 | starter example |
| 4 | rule-based strong baseline |
| 5 | earlier bot checkpoint |
| 6 | current league population |
| 7 | strongest holdout bot |

An agent is ready for a new milestone only if it:

- wins or ties at Tier \(k-1\);
- does not catastrophically regress on Tier \(k-2\);
- passes the protocol/latency suite;
- passes unseen-seed tests.

---

# Comparative conclusion

## Rule-based systems

**Strengths**

- immediate;
- understandable;
- low latency;
- easy safety constraints;
- strong data efficiency.

**Weaknesses**

- many brittle thresholds;
- hard to generalize;
- state-machine size grows.

## Behavior trees

**Strengths**

- composable;
- clear priority logic;
- reusable subtrees;
- strong debugging.

**Weaknesses**

- still requires careful planning;
- conditions can conflict;
- poor at temporal credit assignment.

## Finite-state machines

**Strengths**

- simple;
- deterministic;
- minimal inference cost.

**Weaknesses**

- difficult to handle concurrent objectives;
- transitions require carefully tuned hysteresis.

## Reinforcement learning

**Strengths**

- adapts from experience;
- can optimize state-specific behaviors;
- useful for opponent adaptation.

**Weaknesses**

- sparse placement reward;
- non-stationarity;
- sample demand;
- reproducibility challenges;
- potentially unsafe exploration.

## Deep reinforcement learning

**Strengths**

- learns continuous mappings;
- compact representation;
- useful for motion subpolicies.

**Weaknesses**

- expensive;
- hard to explain;
- vulnerable to distribution shift;
- reward hacking.

## Hybrid architecture

**Strengths**

- safest practical choice;
- strongest first-milestone performance;
- clear observability;
- preserves human-debuggable decisions;
- allows isolated RL experiments.

**Weaknesses**

- engineering complexity;
- learned policy may be limited by abstraction;
- requires a reliable simulator.

## Final recommendation

Use a **hierarchical hybrid controller**.

The architecture should:

1. use a deterministic rule-based outer policy;
2. use A* or an equivalent navigation controller;
3. use the official zone and weapon rules exactly;
4. use a state estimator for enemies and loot;
5. use behavior-tree or state-machine hysteresis for strategy;
6. use PPO or SAC only for bounded subcontrollers;
7. train a self-play league only after the deterministic baseline is reliable;
8. evaluate every model with paired seeds and placement metrics.

This is more likely to produce a capable, stable bot than a monolithic neural
policy or a pure learned policy. It also provides the safest path for repeated
experimentation in an officially supported environment.

---

# References

1. Mnih, V. et al. “Human-level control through deep reinforcement learning,”
   *Nature* 518, 529–533 (2015).
   https://doi.org/10.1038/nature14236

2. van Hasselt, H., Guez, A., & Silver, D. “Deep Reinforcement Learning with
   Double Q-learning,” 2015.
   https://arxiv.org/abs/1509.06461

3. Schulman, J. et al. “Proximal Policy Optimization Algorithms,” 2017.
   https://arxiv.org/abs/1707.06347

4. Mnih, V. et al. “Asynchronous Methods for Deep Reinforcement Learning,”
   2016. https://arxiv.org/abs/1602.01783

5. Haarnoja, T. et al. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep
   Reinforcement Learning,” 2018.
   https://arxiv.org/abs/1801.01290

6. Lowe, R. et al. “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
   Environments,” 2017.
   https://arxiv.org/abs/1706.02275

7. Sutton, R. S., Precup, D., & Singh, S. “Between MDPs and semi-MDPs:
   A framework for temporal abstraction in reinforcement learning,”
   *Artificial Intelligence* 112 (1999).
   https://doi.org/10.1016/S0004-3702(99)00052-1

8. Ho, J., & Ermon, S. “Generative Adversarial Imitation Learning,” 2016.
   https://arxiv.org/abs/1606.03417

9. Schrittwieser, J. et al. “Mastering Atari, Go, Chess and Shogi by Planning
   with a Learned Model,” 2019.
   https://arxiv.org/abs/1911.08265

10. Vinyals, O. et al. “Grandmaster level in StarCraft II using multi-agent
    reinforcement learning,” *Nature* 575 (2019).
    https://www.nature.com/articles/s41586-019-1724-z

11. OpenAI. “Multi-agent emergence via entropy maximization?” The hide-and-seek
    project: https://arxiv.org/abs/1909.07528

12. Raffin, A. et al. “Stable-Baselines3: Reliable Reinforcement Learning
    Implementations,” JMLR 22 (2021).
    https://jmlr.org/papers/v22/20-1364.html

13. Stable-Baselines3 documentation:
    https://stable-baselines3.readthedocs.io/

14. Gymnasium documentation:
    https://gymnasium.farama.org/

15. Ray RLlib documentation:
    https://docs.ray.io/en/latest/rllib/index.html

16. Unity ML-Agents documentation:
    https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/index.html

17. PyTorch documentation:
    https://pytorch.org/docs/stable/index.html

18. PettingZoo documentation:
    https://pettingzoo.farama.org/

19. Official Mech Royale track:
    https://www.joinbuilders.org/buildathon/tracks/builders

20. Official Mech Royale protocol:
    https://www.joinbuilders.org/buildathon/tracks/builders/files/protocol.md
