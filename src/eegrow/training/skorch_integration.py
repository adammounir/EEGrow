"""Train growable eegrow models through braindecode's skorch ``EEGClassifier``.

Why a callback (and not the manual loop in :mod:`eegrow.training.loop`)?
``EEGClassifier`` (a skorch ``NeuralNet``) owns the fit loop: forward, backward,
optimizer step, scoring, early stopping. Growth is an *event* between epochs --
compute gromo statistics, line-search the best expansion, ``apply_change`` -- after
which the optimizer must be rebuilt, because ``apply_change`` replaces the weight
tensors that grew with brand-new ``Parameter`` objects. A skorch ``Callback`` is the
right hook: ``on_epoch_end`` it (optionally) grows ``net.module_`` and refreshes
``net.optimizer_``, preserving the momentum of the untouched parameters.

Growth runs on **CPU** (gromo's optimal update uses ``torch.linalg.eigh``, not
available on MPS), so the callback moves the module to CPU for the growth step and
back to the training device afterwards.
"""

from __future__ import annotations

import torch
from skorch.callbacks import Callback

from eegrow.training.loop import grow_step


def _xy_iter(iterator, device):
    """Yield ``(X, y)`` on ``device`` from a skorch iterator.

    braindecode window datasets yield ``(X, y, crop_inds)`` 3-tuples; gromo's
    statistics only need the input and target, so we keep the first two elements.
    """
    for batch in iterator:
        x, y = batch[0], batch[1]
        yield x.to(device), y.to(device)


class GromoGrowth(Callback):
    """skorch callback that periodically grows the module via gromo.

    Parameters
    ----------
    grow_every : int
        Run a growth step every ``grow_every`` epochs (never on the final epoch).
    growth_device : str
        Device for the growth computation. Must be ``"cpu"``: gromo's optimal-update
        relies on ``torch.linalg.eigh``, not implemented on MPS.
    verbose : bool
        Print a one-line summary at each growth.
    """

    def __init__(self, grow_every: int = 15, growth_device: str = "cpu",
                 verbose: bool = True):
        self.grow_every = grow_every
        self.growth_device = growth_device
        self.verbose = verbose
        self._done = False

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        epoch = len(net.history)
        if (self._done or dataset_train is None
                or epoch % self.grow_every != 0
                or epoch >= net.max_epochs):
            return
        model = net.module_
        if not getattr(model, "_growable_layers", []):
            return  # frozen model (no target width): nothing to grow

        # gromo grows by a (data-dependent) amount per step and does NOT stop at the
        # target on its own, so we enforce a *soft* cap: once the width has reached
        # target_width we stop growing. A single growth step may slightly overshoot
        # the cap (we cannot shrink the last step), which is fine.
        cap = getattr(model, "target_width", None)
        if cap is not None and model.growable_width >= cap:
            self._done = True
            return

        train_device = next(model.parameters()).device
        width_before = model.growable_width

        # gromo growth must run on CPU; move there and back.
        model.to(self.growth_device)
        iterator = net.get_iterator(dataset_train, training=True)
        grow_step(model, _xy_iter(iterator, self.growth_device), self.growth_device)
        model.to(train_device)

        width_after = model.growable_width
        if width_after <= width_before:
            self._done = True  # reached the target cap: stop trying
            return
        self._refresh_optimizer(net)
        if self.verbose:
            print(f"  [gromo] epoch {epoch}: width {width_before} -> {width_after}, "
                  f"{sum(p.numel() for p in model.parameters()):,} params")

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


def make_eeg_classifier(
    model,
    *,
    max_epochs: int = 100,
    grow_every: int = 15,
    lr: float = 6.25e-4,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    device: str = "cpu",
    growth_device: str = "cpu",
    extra_callbacks=None,
    **kwargs,
):
    """Wrap a growable eegrow model in a braindecode ``EEGClassifier`` that grows.

    The returned classifier trains like any braindecode ``EEGClassifier``
    (``clf.fit(X, y)`` / ``clf.predict(X)``), with a :class:`GromoGrowth` callback
    that grows the module every ``grow_every`` epochs. If the model is frozen (no
    target width), the callback is a no-op and you get a plain baseline.

    Notes
    -----
    The default criterion is ``CrossEntropyLoss`` (our models output raw logits, not
    log-probabilities), and ``AdamW`` is the optimizer -- matching
    :mod:`eegrow.training.loop`.
    """
    from braindecode import EEGClassifier

    callbacks = [("gromo", GromoGrowth(grow_every=grow_every,
                                       growth_device=growth_device))]
    if extra_callbacks:
        callbacks += list(extra_callbacks)

    return EEGClassifier(
        module=model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_split=None,
        callbacks=callbacks,
        device=device,
        **kwargs,
    )
