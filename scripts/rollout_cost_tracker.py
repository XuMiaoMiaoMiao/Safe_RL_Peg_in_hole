"""Rollout episode cost tracking — shared by train_sac.py and train_sac_lagrangian.py.

Extracted from train_sac_lagrangian.py so the native SAC baseline and Lagrangian
SAC measure rollout safety with the *exact same* code path. Without this, the two
algorithms would log rollout cost differently and the SAC-vs-LagSAC benchmark
comparison would not be apples-to-apples.

Problem being solved:
  VectorCore.callback_step(samples) receives (s, a, r, s', absorbing, last, ...)
  per vector-step, but NOT step_info (which carries the cost tensor). Meanwhile,
  VectorizedDataset.flatten() corrupts the `last` flag by setting
  last_padded[-1, :] = True for the final vector-step of every fit chunk,
  making it impossible to count true episode boundaries from a flattened dataset.

Solution: a thin env wrapper caches (cost, mask) into a shared bridge object
immediately after step_all() returns and before VectorCore fires callback_step.
The tracker callback reads from the bridge and accumulates per-env episode costs,
detecting episode ends from the raw `last` flag in samples (which is computed by
VectorCore as `absorbing | timeout` — the authoritative, pre-flatten signal).

Per episode the tracker reports two quantities, filled in lockstep:
  - cost SUM via drain()      → the Lagrangian λ-update signal
  - cost MAX via drain_max()  → the paper "Maximum Violation per Episode"
"""

from collections import deque

import numpy as np
import torch


class StepCostBridge:
    """Shared state between CostEnvWrapper and EpisodeCostTracker.

    CostEnvWrapper.step_all() writes here right after env.step_all() returns.
    EpisodeCostTracker.__call__() reads here when VectorCore fires callback_step.
    The two always run in the same thread, so no locking is needed.
    """

    def __init__(self):
        self.cost = None   # [num_envs] float tensor, set each vector-step
        self.mask = None   # [num_envs] bool tensor (which envs were active)


class CostEnvWrapper:
    """Thin wrapper around the environment that populates StepCostBridge.

    Intercepts step_all() to cache the cost tensor and active-env mask before
    returning. Everything else (info, number, reset_all, stop, render_all, ...)
    is forwarded to the underlying env via __getattr__, so VectorCore sees a
    fully transparent wrapper.

    Why not modify VectorCore instead? VectorCore.callback_step() only receives
    the samples tuple — step_info (containing cost) is consumed internally and
    never passed to the callback. The wrapper + bridge pattern sidesteps this
    without touching the Mushroom framework.
    """

    def __init__(self, env, bridge: StepCostBridge):
        self._env = env
        self._bridge = bridge

    def step_all(self, mask, action):
        next_state, rewards, absorbing, step_info = self._env.step_all(mask, action)
        # Populate bridge BEFORE returning. VectorCore._step() calls step_all()
        # and then immediately returns to _run(), which fires callback_step.
        # By the time the callback reads the bridge, it already has the current step.
        self._bridge.cost = step_info.get("cost", None)
        self._bridge.mask = mask
        return next_state, rewards, absorbing, step_info

    def __getattr__(self, name):
        return getattr(self._env, name)


class EpisodeCostTracker:
    """Accumulates per-env episode cost sums (and maxima) from the training rollout.

    Used as VectorCore.callback_step — called once per vector-step with:
        samples = (state, action, reward, next_state, absorbing, last, ...)
    where `last = absorbing | (episode_steps >= horizon)`. This is the raw,
    per-env episode-end signal from VectorCore._step(), NOT the Mushroom-flattened
    `last` (which sets last_padded[-1, :] = True for every fit chunk boundary,
    corrupting episode count if read from a flattened dataset).

    Lifecycle per training epoch
    ────────────────────────────
    1. reset_accum()      — called at epoch start; discards partial episode costs
                            left over from the previous epoch's rollout boundary.
    2. core.learn()       — tracker fires per vector-step; accumulates costs;
                            pushes a completed episode's total cost to _completed
                            whenever last[i] == True for an active env.
    3. ready() + drain()  — after core.learn(): if ≥ min_episodes completed,
                            drain() returns (mean_cost, n) and clears the buffer.
                            drain_max() is the companion for the per-episode max.
    4. active = False     — before core.evaluate(): suppresses the tracker so that
                            eval episodes (deterministic policy) don't enter the
                            rollout cost buffer.
    5. core.evaluate()    — tracker is a no-op; bridge may still be written by
                            the env wrapper, but tracker ignores it.
    6. active = True      — after eval: re-enable for next epoch.
    """

    def __init__(self, num_envs: int, bridge: StepCostBridge,
                 min_episodes: int = 1, maxlen: int = 1000):
        """
        Args:
            num_envs: number of parallel environments.
            bridge: shared bridge object populated by CostEnvWrapper.
            min_episodes: minimum completed episodes required before drain()
                reports results (ready() returns False below this threshold).
                Default 1: always update if any episode completed this epoch.
            maxlen: rolling buffer capacity. If more episodes complete without a
                drain() call (e.g., very short episodes), oldest entries are
                dropped (deque semantics). Default 1000 is conservative.
        """
        self._bridge = bridge
        self._num_envs = num_envs
        self._min_episodes = min_episodes
        # Per-env running total / running max for the current episode.
        # Initialised lazily on the env's native device (GPU) on first call.
        # Persists across fit chunks within an epoch; cleared by reset_accum().
        self._accum: torch.Tensor | None = None
        self._accum_max: torch.Tensor | None = None
        self._device = None
        # Completed per-episode cost sums / maxima for the current epoch window.
        # Both deques are filled in lockstep by __call__ (one entry per episode).
        self._completed: deque = deque(maxlen=maxlen)
        self._completed_max: deque = deque(maxlen=maxlen)
        # Flip to False during eval so eval episodes don't pollute the buffer.
        self.active: bool = True

    def _init_accum(self, device):
        """Lazy-init _accum / _accum_max on the env's device (GPU in IsaacSim)."""
        self._device = device
        self._accum = torch.zeros(self._num_envs, dtype=torch.float32, device=device)
        self._accum_max = torch.zeros(self._num_envs, dtype=torch.float32, device=device)

    def __call__(self, samples):
        """VectorCore callback_step interface — called after each vector-step.

        Performance: accumulation stays on the tensor's native device (GPU) so
        there are no per-step CPU syncs. A CPU transfer happens only when at
        least one episode ends in this step (infrequent), using a single
        .nonzero().tolist() call instead of three whole-tensor .cpu() copies.
        """
        if not self.active:
            return
        if self._bridge.cost is None or self._bridge.mask is None:
            return

        _, _, _, _, _absorbing, last, _, _ = samples

        # Keep on native device — no .cpu() that would flush the CUDA stream.
        cost = torch.as_tensor(self._bridge.cost, dtype=torch.float32)
        mask = torch.as_tensor(self._bridge.mask, dtype=torch.bool)
        last_t = torch.as_tensor(last, dtype=torch.bool)

        if self._accum is None:
            self._init_accum(cost.device)
        cost = cost.to(self._device)
        mask = mask.to(self._device)
        last_t = last_t.to(self._device)

        # Vectorized accumulation: one GPU in-place op, no Python loop.
        self._accum.add_(cost * mask.float())
        # Per-episode running MAX (paper "Maximum Violation per Episode").
        # cost >= 0 always (clearance/penetration violation, collision 0/1), so
        # masking inactive envs to 0 can never spuriously raise the max.
        torch.maximum(self._accum_max, cost * mask.float(), out=self._accum_max)

        # Only pay the CPU sync cost when at least one episode actually ended.
        ended = mask & last_t
        if not ended.any():
            return

        # .nonzero().tolist() is a single sync; we then loop over the small
        # set of envs that ended (typically 0–num_envs per step).
        for i in ended.nonzero(as_tuple=True)[0].tolist():
            # True episode end: absorbing termination OR horizon timeout.
            self._completed.append(float(self._accum[i]))
            self._completed_max.append(float(self._accum_max[i]))
            self._accum[i] = 0.0
            self._accum_max[i] = 0.0

    def reset_accum(self):
        """Discard partial episode costs, preparing for a fresh epoch rollout.

        Must be called at the start of each epoch (before core.learn()).
        Without this, the tail of the previous epoch's unfinished episodes
        would bleed into the first completed episode of the new epoch.
        """
        if self._accum is not None:
            self._accum.zero_()
        if self._accum_max is not None:
            self._accum_max.zero_()

    def ready(self) -> bool:
        """True if enough episodes have completed to justify a λ update."""
        return len(self._completed) >= self._min_episodes

    def drain(self):
        """Return (mean_episode_cost, n_episodes) and clear the completed buffer.

        This is the Lagrangian λ-update signal: mean per-episode cost SUM.
        Returns (nan, 0) if the buffer is empty (shouldn't happen after ready()).
        """
        if not self._completed:
            return float("nan"), 0
        costs = list(self._completed)
        self._completed.clear()
        return float(np.mean(costs)), len(costs)

    def drain_max(self):
        """Return (mean of per-episode max-cost, n_episodes) and clear the buffer.

        Companion to drain(): drain() reports the per-episode cost SUM (λ signal);
        drain_max() reports the per-episode cost MAXIMUM (the paper's "Maximum
        Violation per Episode"). The two buffers are filled in lockstep by
        __call__, so call drain_max() alongside drain() each epoch.
        Returns (nan, 0) if the buffer is empty.
        """
        if not self._completed_max:
            return float("nan"), 0
        maxes = list(self._completed_max)
        self._completed_max.clear()
        return float(np.mean(maxes)), len(maxes)

    @property
    def n_episodes(self) -> int:
        return len(self._completed)
