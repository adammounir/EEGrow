"""Grow eegrow models *during* braindecode ``EEGClassifier`` training, via a callback.

Why a callback (and not the manual loop in :mod:`eegrow.training.loop`)?
``EEGClassifier`` (a skorch ``NeuralNet``) owns the fit loop: forward, backward,
optimizer step, scoring, early stopping. Growth is an *event* between epochs --
compute gromo statistics, line-search the best expansion, ``apply_change`` -- after
which the optimizer must be rebuilt, because ``apply_change`` replaces the weight
tensors that grew with brand-new ``Parameter`` objects. A skorch ``Callback`` is the
right hook: ``on_epoch_end`` it (optionally) grows ``net.module_`` and refreshes
``net.optimizer_``, preserving the momentum of the untouched parameters.

Just add the callback to a plain ``EEGClassifier`` -- there is no wrapper to learn::

    from braindecode import EEGClassifier
    from eegrow import GromoGrowth, GrowingSCCNet

    model = GrowingSCCNet(..., n_spatial_filters=4, target_n_spatial_filters=16)
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,   # our models output raw logits
        callbacks=[("gromo", GromoGrowth(grow_every=6))],
    )

**Device.** gromo's optimal update uses ``torch.linalg.eigh``, which has a native
kernel on CPU and CUDA but **not on MPS** (Apple Silicon). The callback therefore runs
growth on the training device when that device supports ``eigh`` (cpu/cuda) and only
falls back to CPU on MPS, moving the module back afterwards. Pass ``growth_device`` to
force a device. (Recent torch can also route ``eigh`` off MPS via
``PYTORCH_ENABLE_MPS_FALLBACK=1``; we keep the hop explicit so it does not depend on an
env var.)
"""

from __future__ import annotations

import logging

import torch
from skorch.callbacks import Callback

from eegrow.training.loop import MIN_SINGULAR_RATIO, grow_step

logger = logging.getLogger(__name__)

# Device types with a native ``torch.linalg.eigh`` kernel (gromo's optimal update).
# MPS has none, so on Apple Silicon growth falls back to CPU.
_EIGH_DEVICES = frozenset({"cpu", "cuda"})


def _growth_device_for(train_device: torch.device, override) -> torch.device:
    """Pick the device to run the growth (``eigh``) on.

    ``override`` wins if given. Otherwise we stay on the **training device** when it
    can run ``eigh`` (cpu/cuda) -- no needless host<->GPU copy -- and fall back to
    ``cpu`` only for MPS, the one backend without a native kernel.
    """
    if override is not None:
        return torch.device(override)
    if train_device.type in _EIGH_DEVICES:
        return train_device
    return torch.device("cpu")


class GromoGrowth(Callback):
    """skorch callback that periodically grows the module via gromo.

    Parameters
    ----------
    grow_every : int
        Run a growth step every ``grow_every`` epochs (never on the final epoch).
    growth_device : str or torch.device, optional
        Force the device of the growth computation. ``None`` (default) auto-selects:
        the training device if it supports ``eigh`` (cpu/cuda), else ``cpu`` (MPS).
    holdout_frac : float
        Fraction of the epoch's batches held out for the line search, so the scaling
        factor of the new neurons is chosen on data they did **not** fit (curbing the
        amplification of neurons that only fit training noise). Default ``0.2``.
    min_holdout_batches : int
        Only hold out a slice when the epoch has at least this many batches; below it
        the held-out side would be too small to be meaningful, so the line search
        falls back to the training batches. Default ``4``.
    min_singular_ratio : float
        How many neurons a step may propose, as a floor on their singular value
        relative to the best candidate's. See ``loop.grow_step``: gromo's own cut is
        absolute and sits above the entire spectrum of half our arms, which pins them
        to one neuron per step regardless of the data.
    verbose : bool
        Log a one-line summary (``INFO``) at each growth.
    """

    def __init__(self, grow_every: int = 15, growth_device=None,
                 holdout_frac: float = 0.2, min_holdout_batches: int = 4,
                 min_singular_ratio: float = MIN_SINGULAR_RATIO,
                 verbose: bool = True):
        self.grow_every = grow_every
        self.growth_device = growth_device
        self.holdout_frac = holdout_frac
        self.min_holdout_batches = min_holdout_batches
        self.min_singular_ratio = min_singular_ratio
        self.verbose = verbose
        self.done_ = False

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        epoch = len(net.history)
        if (self.done_ or dataset_train is None
                or epoch % self.grow_every != 0
                or epoch >= net.max_epochs):
            return
        model = net.module_
        if not getattr(model, "_growable_layers", []):
            return  # frozen model (no target width): nothing to grow

        # gromo grows by a (data-dependent) amount per step and does NOT stop at the
        # target on its own, so we enforce a *hard* cap two ways: we stop once the
        # width reaches target_width, AND we pass max_added below so grow_step trims
        # the step (sub_select_optimal_added_parameters) to land exactly on the
        # target -- the width never overshoots.
        cap = getattr(model, "target_width", None)
        if cap is not None and model.growable_width >= cap:
            self.done_ = True
            return

        train_device = next(model.parameters()).device
        growth_device = _growth_device_for(train_device, self.growth_device)
        moved = growth_device != train_device
        width_before = model.growable_width
        max_added = None if cap is None else max(1, cap - width_before)

        if moved:  # only MPS needs the round-trip; cpu/cuda grow in place
            model.to(growth_device)
        # Materialise the epoch's batches once, then split them: statistics on the
        # bulk, the line search on a disjoint held-out slice. The *list* is what makes
        # the split stable and disjoint -- re-iterating a shuffled loader would not
        # partition the same way twice.
        #
        # They stay on the HOST, and that is the point. gromo's `compute_statistics`
        # and `evaluate_model` each do `x, y = x.to(device), y.to(device)` per batch,
        # so moving the whole epoch here bought nothing and cost the epoch's size in
        # device memory, held for the duration of the growth step on top of the model
        # and its statistics. That is what made a growing arm's device peak scale with
        # the size of the training fold rather than with the architecture -- and a peak
        # that depends on the data volume cannot be predicted by a probe that replaces
        # the data. It is why profile_grid_memory.py, measuring on 512 synthetic
        # samples, underestimated the real cells and cost the 2026-08-16 smoke run 18
        # cells to OOMs against their own declared ceiling while the card sat 6 GiB
        # free; the 43 GB once seen on schirrmeister2017 is the same mechanism at the
        # other end of the scale.
        #
        # The cost is host-to-device traffic: the held-out slice is now transferred
        # once per scaling factor of the line search (four) instead of once. That is
        # PCIe bandwidth on a step that runs every `grow_every` epochs, against device
        # memory that decided how many cells could share a card.
        batches = list(net.get_iterator(dataset_train, training=True))
        if not batches:
            # The training iterator yielded nothing this epoch -- e.g. ``drop_last``
            # (set by EEGClassifier) drops the only sub-``batch_size`` fold. With no
            # data to estimate the growth statistics on, skip this growth opportunity
            # rather than crash the fit; the next ``grow_every`` epoch may have data.
            if moved:
                model.to(train_device)
            return
        stats_loader, val_loader = self._split_holdout(batches)
        result = grow_step(model, stats_loader, growth_device,
                           val_loader=val_loader, max_added=max_added,
                           min_singular_ratio=self.min_singular_ratio)
        if moved:
            model.to(train_device)

        width_after = model.growable_width
        self._record(net, result)
        if not result["applied"]:
            # The line search picked s=0 and ``grow_step`` discarded the update rather
            # than splice in dead neurons. The width is unchanged, but that is an
            # *abstention*, not a cap: the statistics are re-estimated from scratch at
            # the next opportunity and may well find a direction worth paying for. So
            # ``done_`` stays False -- treating this as "reached the target" is how a
            # model would freeze at width 8 after one unlucky epoch. The optimizer is
            # untouched too: no parameter object changed.
            if self.verbose:
                logger.info("gromo epoch %d: abstained (s=0) at width %d", epoch,
                            width_before)
            return
        if width_after <= width_before:
            self.done_ = True  # reached the target cap: stop trying
            return
        self._refresh_optimizer(net)
        if self.verbose:
            logger.info("gromo epoch %d: width %d -> %d (s=%.2g, %s params)", epoch,
                        width_before, width_after, result["s"],
                        f"{sum(p.numel() for p in model.parameters()):,}")

    @staticmethod
    def _record(net, result) -> None:
        """Stamp the growth decision on the current epoch of ``net.history``.

        The scaling factor was the one quantity the whole step turns on and the one
        quantity nothing kept: it was chosen and dropped inside ``grow_step``, so no
        run of any campaign can say whether a growth event bought anything. Recorded
        here rather than in ``FitRecorder`` because only this callback sees it, and on
        the growth epoch only -- the reader must treat a missing key as "no growth
        opportunity this epoch", which is what ``FitRecorder``'s ``k in h`` filter and
        pandas' NaN both already do.

        Guarded on ``record`` existing: the tests drive ``on_epoch_end`` on a stub net
        whose ``history`` is a plain list, and instrumentation must not be the reason a
        test double stops working.
        """
        record = getattr(getattr(net, "history", None), "record", None)
        if record is None:
            return
        record("grow_s", result["s"])
        record("grow_applied", bool(result["applied"]))
        record("grow_width_after", result["width_after"])
        record("grow_n_proposed", result["n_proposed"])
        record("grow_n_kept", result["n_candidates"])
        record("grow_first_order_improvement", result["first_order_improvement"])
        record("grow_eig_sum", result["eigenvalues_extension_sum"])
        # The quantities the step is *decided* on, and the only ones that cannot be
        # recovered afterwards from anything on disk. `grow_losses` is the held-out
        # loss at every point of `loop.SCALING_GRID`, keyed by the factor, so a reader
        # sees why `grow_s` won rather than only that it did -- an abstention because
        # the whole curve was flat and one because it was steeply worse are different
        # failures with different fixes. The two spectra bracket the selection rule:
        # `grow_eig_proposed` is what gromo offered, `grow_eig_kept` what the relative
        # floor and the width cap left, so the sum alone stops being ambiguous between
        # one dominant direction and a flat tail. `grow_param_update_decrease`
        # separates the gain from the weights already there from the new neurons'.
        record("grow_select_loss", result["select_loss"])
        record("grow_losses", result["losses"])
        record("grow_eig_proposed", result["eig_proposed"])
        record("grow_eig_kept", result["eig_kept"])
        record("grow_param_update_decrease", result["parameter_update_decrease"])

    def _split_holdout(self, batches):
        """Split ``batches`` into (stats, line-search) loaders per ``holdout_frac``.

        Returns ``(batches, None)`` when there are too few batches to hold a slice
        out, in which case the line search uses the training batches.
        """
        n_hold = int(round(self.holdout_frac * len(batches)))
        if len(batches) >= self.min_holdout_batches and n_hold >= 1:
            cut = len(batches) - n_hold
            return batches[:cut], batches[cut:]
        return batches, None

    @staticmethod
    def _refresh_optimizer(net):
        """Rebuild ``net.optimizer_`` over the new params, keeping the momentum of
        the unchanged ones.

        gromo only swaps the tensors that actually grew; the other ``Parameter``
        objects (BatchNorms, classifier...) keep their identity, so their optimizer
        state can be carried over. ``module.to(device)`` mutates a parameter's
        ``.data`` in place without changing the object, so identity is preserved
        across the CPU round-trip.
        """
        old_state = dict(net.optimizer_.state)
        net.initialize_optimizer()  # fresh optimizer over the current module params
        for p in net.module_.parameters():
            if p in old_state:
                net.optimizer_.state[p] = old_state[p]
