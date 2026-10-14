"""Step-0 "published reference" criterion: reproduce Schirrmeister et al. 2017 on BCIC IV 2a.

WHY THIS EXISTS
---------------
The criterion asks whether ``braindecode.models.ShallowFBCSPNet`` / ``Deep4Net`` -- the
two references the whole campaign is measured against -- are implemented and trainable
as published. Our benchmark grid cannot answer that: it is within-session, trial-wise,
8-32 Hz without standardisation, and the paper is none of those (PREREG § 6). So the
criterion is tested under the AUTHORS' protocol, taken from the authors' own code, not
from the paper's prose and not from memory:

  robintibor/braindecode (0.4) ``examples/bcic_iv_2a_cropped.py`` and the modules it
  calls (``experiments/experiment.py``, ``experiments/stopcriteria.py``,
  ``experiments/monitors.py``, ``datautil/iterators.py``, ``datautil/signalproc.py``,
  ``torch_ext/constraints.py``, ``datautil/trial_segment.py``), fetched 14/10/2026.

Published target (arXiv 1703.05051v5, Table 2), 4-38 Hz, cropped, mean of 9 subjects:
Shallow ConvNet 71.9 %, Deep ConvNet 70.1 %. 4 Hz is the script's ``low_cut_hz``
default and the band of braindecode's current cropped example.

PROTOCOL, line by line from that code
-------------------------------------
* data: session T (MOABB ``0train``) for training, E (``1test``) for test; 22 EEG
  channels; x 1e6; Butterworth order 3, 4-38 Hz, CAUSAL (``scipy.signal.lfilter``,
  ``bandpass_cnt(filtfilt=False)``); exponential running standardisation
  (factor_new 1e-3, init_block_size 1000, eps 1e-4) exactly as the 0.4 function,
  including its init block standardised across channels per sample;
* trials: ival [-500, 4000] ms around the cue = 1125 samples at 250 Hz;
* split: the LAST 20 % of the training session (in time order, no shuffle) is the
  validation set (``split_into_two_sets``);
* models: ``ShallowFBCSPNet(final_conv_length=30)`` / ``Deep4Net(final_conv_length=2)``
  on ``input_time_length=1000``, made dense; braindecode 1.7 modules (that is the
  point: the CURRENT implementations are under test);
* crops: every sample after the receptive field predicted once, the last crop aligned
  on the trial end (``_get_start_stop_blocks_for_trial``); batches of ~60 crops from
  ``get_balanced_batches`` with ``RandomState((2017, 6, 28))`` reset at each phase;
* loss: NLL of the time-averaged log-softmax over the crop's predictions;
* optimizer: ``torch.optim.Adam`` defaults (lr 1e-3, no weight decay);
* constraint after every step: max L2 norm 2 on every weight but BatchNorm, 0.5 on the
  classifier (``MaxNormDefaultConstraint``) -- applied here to every Conv/Linear
  weight, because the 1.7 modules nest their convolutions where the 0.4 constraint,
  which only walks top-level children, would miss them;
* stopping, two phases (``Experiment.run``): phase 1 on train until 800 epochs or no
  decrease of trial-wise ``valid_misclass`` for 80 epochs, remembering the best epoch
  (ties go to the LATER epoch, ``<=``); reset model AND optimizer to it; phase 2 on
  train+valid until ``valid_loss`` < the phase-1 ``train_loss`` at the best epoch or
  until 2 x best_epoch total epochs; score = trial-wise test accuracy of the LAST
  epoch. Epoch 0 (before any update) is logged in each phase, as ``log_0_epoch=True``.

What differs, and cannot be helped: MOABB serves the BNCI-horizon .mat files run by run
(6 runs per session), so filtering and standardisation restart at each run instead of
running across the whole GDF recording.

Usage::

    python step0_reference.py --model shallow --subject 1 --seed 0 --out ref.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

IVAL_MS = (-500, 4000)
INPUT_TIME_LENGTH = 1000
MAX_EPOCHS = 800
MAX_INCREASE_EPOCHS = 80
BATCH_SIZE = 60
LOW_CUT_HZ, HIGH_CUT_HZ = 4.0, 38.0
FACTOR_NEW, INIT_BLOCK_SIZE, EPS = 1e-3, 1000, 1e-4
VALID_FRACTION = 0.2
ITERATOR_SEED = (2017, 6, 28)
FINAL_CONV_LENGTH = {"shallow": 30, "deep": 2}
PUBLISHED = {"shallow": 0.719, "deep": 0.701}


# ---------------------------------------------------------------- preprocessing

def bandpass_causal(data, fs, low=LOW_CUT_HZ, high=HIGH_CUT_HZ, order=3):
    """0.4 ``bandpass_cnt`` with its default ``filtfilt=False`` (data: chans x time)."""
    import scipy.signal

    b, a = scipy.signal.butter(order, [low / (0.5 * fs), high / (0.5 * fs)],
                               btype="bandpass")
    return scipy.signal.lfilter(b, a, data, axis=1)


def exponential_running_standardize_04(data, factor_new=FACTOR_NEW,
                                       init_block_size=INIT_BLOCK_SIZE, eps=EPS):
    """0.4 ``exponential_running_standardize``, verbatim; ``data`` is time x chans."""
    import pandas as pd

    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = np.array(demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed))))
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(data[0:init_block_size], axis=other_axis, keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=other_axis, keepdims=True)
        standardized[0:init_block_size] = ((data[0:init_block_size] - init_mean)
                                           / np.maximum(eps, init_std))
    return standardized


def load_subject(subject):
    """(X_train_session, y, X_test_session, y) as lists of 22 x 1125 float32 trials."""
    import mne
    from braindecode.datasets import MOABBDataset

    ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject])
    out = {}
    for sub in ds.datasets:
        raw = sub.raw.copy().pick("eeg")
        if len(raw.ch_names) != 22:
            raise RuntimeError(f"{len(raw.ch_names)} EEG channels, expected 22")
        fs = raw.info["sfreq"]
        data = raw.get_data() * 1e6
        if not np.isfinite(data).all():
            raise RuntimeError("non-finite samples in the raw; the 0.4 loader would "
                               "have imputed them -- refuse rather than guess")
        data = bandpass_causal(data, fs)
        data = exponential_running_standardize_04(data.T).T
        start = int(np.floor(IVAL_MS[0] * fs / 1000.0))
        stop = int(np.ceil(IVAL_MS[1] * fs / 1000.0))
        events, event_id = mne.events_from_annotations(sub.raw, verbose=False)
        names = {v: k for k, v in event_id.items()}
        session = str(sub.description["session"])
        trials, ys = out.setdefault(session, ([], []))
        for i_sample, _, code in events:
            # Annotation onset = MOABB stim (trial start, t=0) + interval[0] (2 s) = cue.
            a, b = i_sample - sub.raw.first_samp + start, i_sample - sub.raw.first_samp + stop
            if a < 0 or b > data.shape[1]:
                raise RuntimeError(f"trial [{a},{b}) outside the run; the 0.4 code "
                                   "would have dropped it with a warning")
            trials.append(data[:, a:b].astype("float32"))
            ys.append(names[code])
    classes = ["left_hand", "right_hand", "feet", "tongue"]
    pack = {}
    for session, (trials, ys) in out.items():
        pack[session] = (np.stack(trials), np.array([classes.index(v) for v in ys]))
    return pack["0train"], pack["1test"], fs


# --------------------------------------------------------------------- crops

def start_stop_blocks(n_times, input_time_length, n_preds):
    """0.4 ``_get_start_stop_blocks_for_trial`` for one trial of ``n_times`` samples."""
    n_rf = input_time_length - n_preds + 1
    blocks, i_stop = [], n_rf - 1
    while i_stop < n_times:
        i_stop += n_preds
        adj = min(i_stop, n_times)
        blocks.append((adj - input_time_length, adj))
    return blocks


def balanced_batches(n, batch_size, rng, shuffle):
    """0.4 ``get_balanced_batches``."""
    n_batches = int(np.round(n / float(batch_size)))
    if n_batches > 0:
        min_size, n_extra = n // n_batches, n % n_batches
    else:
        n_batches, min_size, n_extra = 1, n, 0
    inds = np.arange(n)
    if shuffle:
        rng.shuffle(inds)
    out, i = [], 0
    for b in range(n_batches):
        size = min_size + (1 if b < n_extra else 0)
        out.append(inds[i:i + size])
        i += size
    return out


class CropIterator:
    def __init__(self, n_preds):
        self.n_preds = n_preds
        self.reset_rng()

    def reset_rng(self):
        self.rng = np.random.RandomState(ITERATOR_SEED)

    def batches(self, X, y, shuffle):
        blocks = [(i, a, b) for i in range(len(X))
                  for a, b in start_stop_blocks(X.shape[2], INPUT_TIME_LENGTH,
                                                self.n_preds)]
        for idx in balanced_batches(len(blocks), BATCH_SIZE, self.rng, shuffle):
            sel = [blocks[j] for j in idx]
            yield (np.stack([X[i][:, a:b] for i, a, b in sel]),
                   np.array([y[i] for i, _, _ in sel]))


def trial_predictions(crop_preds, n_times, n_preds, n_trials):
    """0.4 ``compute_preds_per_trial_from_crops``: drop overlapping predictions."""
    n_rf = INPUT_TIME_LENGTH - n_preds + 1
    need0 = n_times - n_rf + 1
    out, k = [], 0
    for _ in range(n_trials):
        need, parts = need0, []
        while need > 0:
            p = crop_preds[k][:, -need:]
            parts.append(p)
            need -= p.shape[1]
            k += 1
        out.append(np.concatenate(parts, axis=1))
    assert k == len(crop_preds)
    return out


# ------------------------------------------------------------------ training

def max_norm_(model):
    """``MaxNormDefaultConstraint`` on every Conv/Linear weight; 0.5 on the last."""
    import torch

    layers = [m for m in model.modules()
              if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        for m in layers:
            m.weight.data = torch.renorm(m.weight.data, 2, 0, maxnorm=2)
        layers[-1].weight.data = torch.renorm(layers[-1].weight.data, 2, 0, maxnorm=0.5)


def build_model(kind, n_chans, n_classes):
    from braindecode.models import Deep4Net, ShallowFBCSPNet

    cls = ShallowFBCSPNet if kind == "shallow" else Deep4Net
    model = cls(n_chans, n_classes, n_times=INPUT_TIME_LENGTH,
                final_conv_length=FINAL_CONV_LENGTH[kind])
    model.to_dense_prediction_model()
    return model


def crop_loss(out, target):
    import torch.nn.functional as F

    return F.nll_loss(F.log_softmax(out, dim=1).mean(dim=2), target)


class Run:
    """``Experiment`` of braindecode 0.4, cropped, two phases."""

    def __init__(self, model, sets, device, n_preds, max_epochs=MAX_EPOCHS,
                 max_increase=MAX_INCREASE_EPOCHS):
        import torch

        self.model, self.sets, self.device = model.to(device), dict(sets), device
        self.opt = torch.optim.Adam(model.parameters())
        self.it = CropIterator(n_preds)
        self.n_preds, self.max_epochs, self.max_increase = n_preds, max_epochs, max_increase
        self.rows = []

    def _monitor(self):
        import torch

        row = {}
        self.model.eval()
        for name, (X, y) in self.sets.items():
            preds, losses = [], []
            with torch.no_grad():
                for xb, yb in self.it.batches(X, y, shuffle=False):
                    out = self.model(torch.as_tensor(xb, device=self.device))
                    losses.append(crop_loss(out, torch.as_tensor(yb, device=self.device)).item())
                    preds.extend(torch.log_softmax(out, 1).cpu().numpy())
            per_trial = trial_predictions(preds, X.shape[2], self.n_preds, len(X))
            labels = np.array([np.argmax(p.mean(axis=1)) for p in per_trial])
            row[f"{name}_loss"] = float(np.mean(losses))
            row[f"{name}_misclass"] = float(1 - np.mean(labels == y))
        self.rows.append(row)

    def _epoch(self):
        import torch

        X, y = self.sets["train"]
        self.model.train()
        for xb, yb in self.it.batches(X, y, shuffle=True):
            self.opt.zero_grad()
            loss = crop_loss(self.model(torch.as_tensor(xb, device=self.device)),
                             torch.as_tensor(yb, device=self.device))
            loss.backward()
            self.opt.step()
            max_norm_(self.model)
        self._monitor()

    def _remember(self):
        v = self.rows[-1]["valid_misclass"]
        if v <= self.best_val:                       # RememberBest: later epoch wins ties
            self.best_val, self.best_epoch = v, len(self.rows) - 1
            self.best_state = (deepcopy(self.model.state_dict()),
                               deepcopy(self.opt.state_dict()))

    def run(self):
        # ---- phase 1
        self.best_val, self.best_epoch = float("inf"), None
        nd_best, nd_low = 0, float("inf")            # NoDecrease state

        def stop1():
            nonlocal nd_best, nd_low
            i = len(self.rows) - 1
            cur = self.rows[-1]["valid_misclass"]
            if cur < (1 - 1e-6) * nd_low:
                nd_best, nd_low = i, cur
            return (i >= self.max_epochs) or (i - nd_best >= self.max_increase)

        self._monitor()
        self._remember()
        self.it.reset_rng()
        while not stop1():
            self._epoch()
            self._remember()
        n_phase1 = len(self.rows)
        # ---- reset to best, phase 2 on train + valid
        del self.rows[self.best_epoch + 1:]
        self.model.load_state_dict(self.best_state[0])
        self.opt.load_state_dict(self.best_state[1])
        loss_to_reach = self.rows[-1]["train_loss"]
        Xt, yt = self.sets["train"]
        Xv, yv = self.sets["valid"]
        self.sets["train"] = (np.concatenate([Xt, Xv]), np.concatenate([yt, yv]))
        best_epoch = self.best_epoch

        def stop2():
            return (len(self.rows) - 1 >= best_epoch * 2
                    or self.rows[-1]["valid_loss"] < loss_to_reach)

        self._monitor()
        self._remember()
        self.it.reset_rng()
        while not stop2():
            self._epoch()
            self._remember()
        last = self.rows[-1]
        return {"test_acc": 1 - last["test_misclass"], "best_epoch_phase1": best_epoch,
                "n_epochs_phase1": n_phase1 - 1, "n_rows_final": len(self.rows),
                "loss_to_reach": loss_to_reach,
                "phase2_stop": ("loss" if last["valid_loss"] < loss_to_reach
                                else "max_epochs"),
                "rows": self.rows}


def main(argv=None) -> int:
    import torch
    from braindecode.util import set_random_seeds

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(FINAL_CONV_LENGTH), required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="smoke only")
    ap.add_argument("--max-increase", type=int, default=MAX_INCREASE_EPOCHS,
                    help="smoke only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    t0 = time.time()
    (Xtr, ytr), (Xte, yte), fs = load_subject(a.subject)
    if Xtr.shape[2] != 1125 or fs != 250:
        raise RuntimeError(f"trial length {Xtr.shape[2]} at {fs} Hz, expected 1125 @ 250")
    n_first = int(round(len(Xtr) * (1 - VALID_FRACTION)))
    sets = {"train": (Xtr[:n_first], ytr[:n_first]),
            "valid": (Xtr[n_first:], ytr[n_first:]), "test": (Xte, yte)}
    set_random_seeds(a.seed, cuda=a.device.startswith("cuda"))
    model = build_model(a.model, Xtr.shape[1], 4)
    with torch.no_grad():
        n_preds = model(torch.zeros(1, Xtr.shape[1], INPUT_TIME_LENGTH)).shape[2]
    res = Run(model, sets, a.device, n_preds, a.max_epochs, a.max_increase).run()
    uuid = None
    if a.device.startswith("cuda"):
        uuid = str(torch.cuda.get_device_properties(torch.device(a.device)).uuid)
    rec = {"model": a.model, "subject": a.subject, "seed": a.seed,
           "n_train": n_first, "n_valid": len(Xtr) - n_first, "n_test": len(Xte),
           "n_preds_per_input": int(n_preds), "max_epochs": a.max_epochs,
           "max_increase": a.max_increase, "device": a.device, "gpu_uuid": uuid,
           "seconds": round(time.time() - t0, 1), **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    print(f"{a.model} S{a.subject} seed{a.seed}: test acc {res['test_acc']:.3f} "
          f"(best ep {res['best_epoch_phase1']}, {rec['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
