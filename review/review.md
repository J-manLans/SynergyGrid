# Repository Review — Group 3

**Course:** Applied Datateknik, Mid Sweden University  
**Group:** 3  
**Project name:** SynergyGrid: A Benchmark for Testing Long-Term Credit Assignment in Agents  
**Repository:** https://github.com/J-manLans/SynergyGrid  
**Review date:** 2026-03-04  
**Reviewer:** GitHub Copilot (automated evidence-based review) Claude Sonnet 4.6 

---

## 1. Repository Summary

### Project purpose
SynergyGrid is a custom Gymnasium-compatible discrete grid-world environment designed as a benchmark for testing long-term credit assignment in single-agent reinforcement learning. An agent navigates a configurable 2-D grid, collecting resources that yield direct positive or negative rewards or delayed synergy rewards only when picked up in a specific tier order. The environment targets researchers and students who want to train and evaluate RL agents on problems requiring multi-step planning and credit assignment, without requiring GPU hardware.

### Technology stack
Python 3.10–3.12; Gymnasium ≥ 1.2 (RL environment API); NumPy ≥ 2.2; Pygame ≥ 2.6 (optional visual rendering); Matplotlib ≥ 3.10 (optional plotting); Stable-Baselines3 with PPO, DQN, and A2C (agent training, dev dependency); pytest ≥ 9.0; black (auto-formatter). Packaged with setuptools via `pyproject.toml`; installable with `pip install -e .[dev]`.

### Current development state
Intermediate-to-mature. The core Gymnasium environment (`SynergyGridEnv`), grid world, agent, direct resources (positive/negative), tier-synergy resources, Pygame renderer, observation handler, and agent-runner scripts are all implemented and functional. The `effects/` subfolder (intended for resources that modify agent perception or the environment) contains only a placeholder file, indicating that FR-13 through FR-15 are incomplete. No README.md exists in the repository root, which is a notable gap. The commit history shows 49 commits across two authors with feature-branch PRs.

---

## 2. Evidence-Based Checklist of Good Practices

Scale: **Yes / Partly / No / Unclear**

### 2.1 Structure and organization

**The repository has a clear and logical folder structure.**  
Assessment: Yes  
Evidence: Root contains `src/`, `tests/`, `dev/`, `docs/`, `assets/`, `pyproject.toml`, `pytest.ini`, `.gitignore`. Inside `src/synergygrid/` the package is divided into `core/`, `gymnasium/`, `rendering/`, `agentrunner/`, `config/`, and `plot/`.  
Comment: The separation of source, tests, development notes, and assets is clean and purposeful.

**Source code, tests, configuration, and documentation are separated appropriately.**  
Assessment: Yes  
Evidence: Source code under `src/synergygrid/`; tests under `tests/` (mirroring `core/` and `gymnasium/` sub-folders); configuration in `pyproject.toml`, `pytest.ini`, and `src/synergygrid/config/configs.py`; developer documentation in `dev/`; user-facing docs intended in `docs/` (currently a placeholder).  
Comment: No test or config files are mixed into source directories.

**File and folder names are meaningful and consistent.**  
Assessment: Yes  
Evidence: `base_resource.py`, `base_tier_resource.py`, `grid_world.py`, `observation_space.py`, `pygame_renderer.py`, `register_env.py`, `train.py`, `eval.py` — all clearly describe their content. Folders (`core/resources/direct/`, `core/resources/synergy/`, `agentrunner/`) follow the same pattern.  
Comment: Naming is consistently lowercase with underscores, matching Python conventions.

**The repository avoids unnecessary generated files or clutter.**  
Assessment: Yes  
Evidence: A comprehensive `.gitignore` (215 lines) excludes `__pycache__/`, `.egg-info/`, `dist/`, `.venv/`, IDE files, etc. No compiled files or virtual environment directories are tracked.  
Comment: The only minor anomaly is `core/effects/dummy_file` — a non-Python placeholder left in a stub directory.

### 2.2 Code quality

**The code appears correct and runnable.**  
Assessment: Partly  
Evidence: `SynergyGridEnv` properly subclasses `gym.Env` and implements `reset()`, `step()`, and `render()` with correct return signatures. `test_environment.py` uses Gymnasium's own `check_env()` validator. Agent movement, score tracking, and resource consumption logic look correct. However, `pyproject.toml` references a `README.md` that does not exist — `pip install .` may warn or fail. `core/effects/` is empty beyond a placeholder, so any code path invoking effect-type resources would fail.  
Comment: Core paths appear sound; the missing README and incomplete effects module are the main gaps.

**The code follows consistent style and coding conventions.**  
Assessment: Yes  
Evidence: Type hints throughout (e.g. `grid_rows: int`, `-> int`, `NDArray`, `Generator | None`). `black` is a dev dependency. Docstrings on every public class and method. Class sections delimited with `# === Section ===` banners (`Init`, `API`, `Helpers`, `Abstract`).  
Comment: Style is notably consistent and more disciplined than typical student projects.

**The code is readable and reasonably modular.**  
Assessment: Yes  
Evidence: `BaseResource` (ABC) → `BaseTierResource` → `{PositiveResource, TierResource, NegativeResource}`; `GridWorld` is separate from `SynergyGridEnv`; `ObservationHandler` is its own class; `PygameRenderer` is fully decoupled from game logic. Each file is focused and small (most under 200 lines).  
Comment: The inheritance hierarchy and separation of concerns show deliberate modular design.

**There are no major obvious code smells (duplication, overly large files, unclear naming).**  
Assessment: Partly  
Evidence: No large files; `pygame_renderer.py` is the longest at ~408 lines but is justified rendering logic. Inline TODO comments exist (in `tier.py` about removing experimental reward scaling; a Swedish-language TODO in `observation_space.py`). `_chained_tiers` is a class-level `Final[list]` on `BaseResource` — a shared mutable list across all instances, which is a potential bug.  
Comment: The shared mutable class attribute for `_chained_tiers` is a subtle design issue; inline TODOs should be resolved or tracked as issues.

### 2.3 Documentation

**The repository contains a clear README.**  
Assessment: No  
Evidence: No `README.md` file exists anywhere in the repository. `pyproject.toml` declares `readme = "README.md"` but the referenced file is absent.  
Comment: This is the most significant documentation gap. Without a README, a new user has no entry point to the project.

**The README explains how to install, run, and use the system.**  
Assessment: No  
Evidence: No README exists. Install instructions only appear as comments at the top of `pyproject.toml`. Run instructions are in `dev/running_instructions.md`.  
Comment: The relevant information exists but is scattered in developer-facing files rather than in an accessible README.

**The documentation would help a new developer understand and contribute to the project.**  
Assessment: Partly  
Evidence: `dev/` contains `workflow.md` (detailed two-person Git workflow with branch naming conventions), `running_instructions.md` (how to run packages), `SynergyGridTechStack.md` (tech stack with examples), and `terminology.md`. Docstrings on classes and methods explain intent.  
Comment: Once a developer opens the `dev/` folder the material is helpful; the barrier is that there is no README to point them there.

**Important design decisions or setup details are documented.**  
Assessment: Partly  
Evidence: `dev/decisions.md` exists but is empty. `dev/risks.md` contains a detailed write-up on the observation space design challenge. `dev/SynergyGridTechStack.md` documents technology rationale. Inline comments in `environment.py` and `observation_space.py` explain non-trivial Gymnasium choices.  
Comment: The risks and tech-stack docs are informative, but the decisions log being empty means key architectural choices (e.g., the tier-chain mechanism, flat vs. Dict observation space) are not formally recorded.

### 2.4 Testing

**The repository contains tests.**  
Assessment: Yes  
Evidence: `tests/core/test_agent.py` (121 lines, parametrized movement tests, boundary tests, fixture-based), `tests/core/test_grid_world.py`, `tests/core/resources/synergy/` (tier resource consume tests added in commit `3babcce`), `tests/gymnasium/test_environment.py` (uses `gymnasium.utils.env_checker.check_env`).  
Comment: Coverage of the main modules is present.

**Tests are relevant to the main functionality.**  
Assessment: Yes  
Evidence: `test_agent.py` tests movement in all four directions, boundary clamping, reset, and resource consumption with a `DummyResource` stub. `test_environment.py` passes the environment through Gymnasium's official validator. Tier resource tests verify the synergy/chain logic, which is the project's defining feature.  
Comment: Tests target the most critical behaviours; stub-based isolation is correctly applied.

**Tests can be executed with clear instructions.**  
Assessment: Yes  
Evidence: `pytest.ini` configures `testpaths = tests` and pattern matching. The file header includes examples: `pytest`, `pytest -v`, per-file invocation. `dev/running_instructions.md` explains the `python3 -m package` convention for running the package.  
Comment: A developer can run the full test suite with a single `pytest` command after installing the package.

**Tests appear to pass, or there is evidence that they have been run successfully.**  
Assessment: Unclear  
Evidence: No CI/CD configuration and no committed test-output. Commit `3babcce` ("Wrote some tests to verify consume for TierResource") and `e3ef6a3` ("Create tests for Agent") suggest tests were written alongside implementation. `test_environment.py` calls `gym.make("synergy_grid-v0")` which requires the package to be installed.  
Comment: Tests appear structurally sound but there is no automated or recorded evidence of a successful run.

### 2.5 Collaboration and development practices

**Commit history suggests incremental development.**  
Assessment: Yes  
Evidence: 49 commits ranging from initial scaffolding through environment setup, resource implementation, modularization refactor (#17), rendering (#15), direct resources (#19), tier synergy, observation space iterations, to human-playable controls (#28). Feature branches merged via numbered PRs.  
Comment: The history shows a clear progression from setup to working environment to feature additions.

**Commit messages are meaningful.**  
Assessment: Partly  
Evidence: Most messages are descriptive: "Feature/implement simple pygame visuals (#15)", "Refactor/modularize code base (#17)", "Tier synergy functionality complete…I think". Some are informal or vague: "Small fixes", "Seems to work, unsure about the obs space though, had to compromise so the models could handle it. I'll clean up the code tomorrow".  
Comment: PR-merged commits tend to have good messages; direct pushes to main late in development show relaxed commit discipline.

**There is evidence of collaboration between both students.**  
Assessment: Partly  
Evidence: `git shortlog -sn --all` shows Joel Lansgren: 40 commits, Lauri: 9 commits. Both are listed as authors in `pyproject.toml`. Commits `2510eab` ("Copied Lauris PR and modified it.") and `677f6a2` ("Copied Lauris changes and modified them a little.") reference Lauri's contributions being integrated.  
Comment: Both students contributed, but the commit split is approximately 80 / 20. The "copied" pattern also suggests some integration friction rather than direct shared-branch collaboration.

---

## 3. What Is Being Done Well

1. **Professional code quality.** Consistent type hints, docstrings on every public symbol, `black` as a dev dependency, and structured `# === Section ===` banners make the codebase unusually readable for a student project.
2. **Sound Gymnasium integration.** `SynergyGridEnv` properly subclasses `gym.Env`, implements the full contract (`reset`, `step`, `render`, `action_space`, `observation_space`), and is validated with Gymnasium's own `check_env()` in tests.
3. **Thoughtful modular architecture.** The inheritance chain `BaseResource → BaseTierResource → {PositiveResource, TierResource, NegativeResource}` cleanly separates concerns; the renderer, observation handler, and grid world are fully decoupled from each other.
4. **Mature developer workflow documentation.** `dev/workflow.md` provides a detailed two-person Git branching strategy with example commands; `dev/running_instructions.md` and `dev/SynergyGridTechStack.md` give clear onboarding information for contributors.
5. **Incremental commit history with PRs.** Feature branches merged via numbered PRs (e.g. #15, #17, #19, #28) demonstrate a deliberate, traceable development process.

---

## 4. What Needs Improvement

1. **No README.md.** The repository has no user-facing entry point. `pyproject.toml` references `README.md` but the file is absent, directly failing NFR-4 and hindering any new user.
2. **FR-13 to FR-15 not implemented.** The `core/effects/` directory contains only a `dummy_file`. Resources that modify agent interaction, perception, or the environment are absent, leaving three functional requirements unaddressed.
3. **Potential shared-state bug in `_chained_tiers`.** `BaseResource._chained_tiers` is a class-level `Final[list]`, meaning all resource instances share the same list. This is likely unintentional and could cause reward miscalculations in multi-episode or multi-environment runs.
4. **No CI pipeline and no test-run evidence.** Tests are present but there is no automated execution (no `.github/workflows/`) and no committed output showing they pass. This makes it impossible to verify correctness without manually installing and running.
5. **Imbalanced collaboration and informal commits late in development.** The commit split is ~80 / 20, and several late commits have vague messages ("Small fixes", "Seems to work…"). `dev/decisions.md` is entirely empty.

---

## 5. Evaluation Against Requirements

### 5.1 Functional Requirements

| Requirement | Description | Status | Evidence | Comment |
|---|---|---|---|---|
| FR-1 | Be compatible with the Gymnasium API. | Implemented | `SynergyGridEnv(gym.Env)` with correct `reset()`, `step()`, `render()`, `action_space`, `observation_space`; validated by `gymnasium.utils.env_checker.check_env` in `test_environment.py`. | Fully compliant Gymnasium interface. |
| FR-2 | Initialize each training session with a maximum resource tier for staged training. | Implemented | `GridWorld.__init__` accepts `max_tier: int = 3`; `SynergyGridEnv.__init__` accepts `max_active_resources`; `agentrunner/train.py` and `eval.py` support staged configuration. | Staged training infrastructure present. |
| FR-3 | Expose a discrete and grid-based environment for the model to explore. | Implemented | `GridWorld` with configurable `grid_rows`/`grid_cols`; `action_space = spaces.Discrete(len(AgentAction))` with four cardinal directions. | |
| FR-4 | Initialize each episode with a starting score for the agent. | Implemented | `SynergyAgent(starting_score=25)`; `reset()` restores `self.score = self._starting_score`. Test `test_initial_position_center` verifies starting score. | |
| FR-5 | Impose a movement cost so the agent loses one score per move. | Implemented | `SynergyAgent.perform_action()`: `self.score -= 1` after every move; score also decremented in agent and verified in test. | |
| FR-6 | Spawn resources at random intervals determined by the number of steps taken in the episode. | Implemented | Each resource has a `_LIFE_SPAN` and `_cool_down` timer; `GridWorld.perform_agent_action()` ticks timers and spawns when cooldown expires. | Interval is step-driven via per-resource timers. |
| FR-7 | Implement resources that yield direct negative or positive rewards upon interaction. | Implemented | `PositiveResource.consume()` returns `_TIER_BASE_REWARD = 2`; `NegativeResource.consume()` returns `-3`. Both in `core/resources/direct/`. | |
| FR-8 | Implement delayed rewards through multi-step resource synergies. | Implemented | `TierResource` in `core/resources/synergy/tier.py` requires resources to be collected in tier order (0→1→2→…); `BaseTierResource._resolve_tier_progression()` enforces the chain. | Core differentiating feature; appears fully implemented. |
| FR-9 | Terminate an episode when a termination condition is met. | Implemented | `step()`: `terminated = self._world._agent.score <= 0`; `truncated = self._observation_handler._step_count_down <= 0`. | Both hard termination and step-limit truncation handled. |
| FR-10 | Return the agent's current score upon episode termination. | Implemented | `step()` returns `reward` on every step; `terminated` signals end; score accessible via `self._world._agent.score` and displayed in HUD. | |
| FR-11 | Allow environment configuration via external configuration files. | Implemented | `src/synergygrid/config/configs.py` defines environment and algorithm parameters; `assets/paths.json` stores asset paths; environment constructor accepts all key parameters. | |
| FR-12 | Provide optional visual feedback of the environment state. | Implemented | `PygameRenderer` (~408 lines) renders the grid with sprites from `assets/`; activated by `render_mode="human"` or `control=True`. | Optional and cleanly separated from game logic. |
| FR-13 | Include resources that modify the agent's interaction with other resources. | Not implemented | `core/effects/` contains only `dummy_file`. No effect-type resource class exists. | Placeholder only. |
| FR-14 | Include resources that modify the agent's perception of the environment. | Not implemented | Same as FR-13 — `effects/` is empty beyond placeholder. | Not implemented. |
| FR-15 | Include resources that modify the environment itself. | Not implemented | Same as FR-13 — `effects/` is empty beyond placeholder. | Not implemented. |
| FR-16 | Enable dynamic resource spawning based on agent performance for curriculum training. | Partly implemented | `agentrunner/train.py` and `config/configs.py` support curriculum-style staged configurations. Dynamic per-step spawning based on live agent performance is not clearly evidenced in `GridWorld`. | Infrastructure exists; runtime dynamic adjustment unconfirmed. |

### 5.2 Non-Functional Requirements

| Requirement | Description | Status | Evidence | Comment |
|---|---|---|---|---|
| NFR-1 | Be lightweight enough to run on consumer hardware without GPU acceleration. | Implemented | `configs.py` sets `"device": "cpu"` for all algorithms; core dependencies are Gymnasium + NumPy only; Pygame/Matplotlib are optional extras. No GPU-accelerated library required. | |
| NFR-2 | Be easy to integrate with single-agent RL models using the Gymnasium API. | Implemented | `register_env.py` registers `"synergy_grid-v0"` with `gym.register()`; compatible with Stable-Baselines3 PPO/DQN/A2C as shown in `agentrunner/`. | |
| NFR-3 | Be installable via `pip install .` with dependencies resolved automatically. | Implemented | `pyproject.toml` with `setuptools` build backend specifies all dependencies. `pip install -e .[dev]` is documented in the `pyproject.toml` header comment. Note: missing `README.md` may trigger a build warning. | |
| NFR-4 | Be usable with no documentation beyond a README. | Not implemented | No `README.md` exists in the repository. A new user has no documented entry point for installation or usage. | Most critical missing deliverable. |
| NFR-5 | Produce consistent results given the same random seed when testing a trained agent. | Partly implemented | `reset(seed=seed)` calls `super().reset(seed=seed)` (Gymnasium contract), and `self.np_random` is passed to `GridWorld.reset()`. Seeding infrastructure is in place but no reproducibility test verifies it. | Plausible but unverified. |
| NFR-6 | Follow modular design principles so components can be extended or replaced easily. | Implemented | Abstract base class `BaseResource` (ABC), separate `PygameRenderer`, separate `ObservationHandler`, separate `agentrunner` scripts, and pip-installable package structure all support easy extension and replacement. | |
| NFR-7 | Include automated tests covering core components. | Implemented | Tests in `tests/core/` (agent, grid world, resources) and `tests/gymnasium/` (environment API via `check_env`). `pytest.ini` configures discovery. | No CI to run them automatically; test structure is otherwise solid. |

---

## 6. Overall Assessment

### Summary judgment
SynergyGrid is a well-architected, clearly coded reinforcement-learning benchmark that demonstrates strong software engineering practices: a proper Gymnasium contract, a clean inheritance hierarchy for resources, type hints throughout, and a documented branching workflow. The core mechanics (grid navigation, direct rewards, tier-synergy delayed rewards, optional rendering) are implemented and tested. The main shortcomings are the complete absence of a README, three unimplemented functional requirements (FR-13 to FR-15, the effect-type resources), and a shared mutable class attribute that could introduce subtle bugs. Overall the project is at an intermediate-to-solid state and is close to being genuinely publishable as a lightweight RL benchmark.

### Confidence in this review
High for structural, code-quality, and FR/NFR assessments. The repository is fully present and source code, tests, and commit history were examined directly. Lower confidence on whether tests actually pass (no CI, no recorded output) and on the exact split of Lauri's contribution beyond commit count.

### Limitations of this review
Tests were not executed in this review environment; pass/fail status is inferred from code inspection only. The full content of all files under `tests/core/resources/` was not read. Dynamic training behaviour (FR-16, NFR-5) cannot be verified from static analysis alone.

---

## 7. Suggested Improvements

1. **Add a README.md** at the repository root covering: project purpose, prerequisites, installation (`pip install -e .[dev]`), how to run the environment (`python3 -m synergygrid` or via the agent-runner scripts), and how to run tests (`pytest`). This alone would close NFR-4 and unblock first-time users.
2. **Implement FR-13 to FR-15** (effect-type resources). Replace `core/effects/dummy_file` with concrete resource classes — e.g., one that inverts the next resource interaction, one that limits the agent's visible range, one that blocks grid cells. These are defining features that remain undelivered.
3. **Fix the `_chained_tiers` shared class-level list.** Move it to an instance variable or to `GridWorld`, which owns the episode lifecycle. The current design will silently produce incorrect reward calculations if multiple environment instances coexist (e.g., vectorized training with SB3).
4. **Add a GitHub Actions CI workflow** to run `pytest` on every push. This would surface regressions immediately and provide the missing evidence of reproducible test success.
5. **Fill in `dev/decisions.md` and clean up TODOs.** Record the key architectural decisions (tier-chain design, flat observation space choice, `_chained_tiers` intent) and resolve or track the inline TODO comments to preserve knowledge for future contributors or reviewers.
