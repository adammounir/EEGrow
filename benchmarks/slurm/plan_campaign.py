"""Cut a campaign grid into passes that fit, from measured memory. Both memories.

WHAT THIS REPLACES
------------------
``memory_plan.py``, which computed K from device memory alone. That model is not
merely incomplete, it is wrong in the direction that kills cells: the first packed
campaign lost 30 ``lee2019_mi`` cells to the cgroup OOM killer with the GPU idle, and
a device-only K reproduces exactly that. Two sources of truth for one K is how an
incident happens, so there is now one.

THE TWO BOUNDS, AND WHY THEY ARE NOT THE SAME SHAPE
---------------------------------------------------
Device memory is a *per-card* budget. K co-tenants share one GPU::

    K_gpu = floor(card_mib x (1 - HEADROOM) / peak_reserved)

Host RAM is a *per-node* budget. The allocation asks for ``--mem`` once and all
``G x K`` tenants on the node draw from it, so the node hosts at most::

    N_ram = floor(mem_mib x (1 - HEADROOM) / peak_rss)

tenants IN TOTAL, however many GPUs there are. Reading that as a per-card bound was
the missing step: with G=3 and K=10, a dataset costing 6 GiB per tenant asks for
180 GiB against a 120 GiB request -- impossible before a single epoch is read, and
invisible to any calculation that reasons one card at a time.

SO SOLVE FOR G TOO, NOT ONLY FOR K
----------------------------------
Given the two bounds, the tenants a node can run is ``N = min(G_max x K_gpu, N_ram)``,
and the GPUs needed to hold them is ``ceil(N / K_gpu)`` -- which is often fewer than
were asked for. ``schirrmeister2017`` costs 27 GiB of host RAM per tenant, so a node
runs 3 of them whatever else is true; spreading those 3 over 3 GPUs idles two cards,
while 7 of the 8 models on that dataset want under 1 GiB of device memory (bd_shallow
wants 126 MiB of a 10.8 GiB card). Putting them on one GPU costs nothing and hands
two cards back to the fleet. Measured over this grid, solving for G cuts the
serialised device work by 21 % (342 -> 270 card-fulls) at identical cell count.

Passes are therefore keyed by ``(G, K)``, and the emitted sbatch line overrides
``--gres`` and ``--cpus-per-task`` accordingly.

WHY PASSES ARE CUT ON MEMORY AND NOT ON MODEL
---------------------------------------------
``pack_run.sh`` takes one global K, so a grid must be homogeneous in K. But K is a
property of the *cell*, and the classes cut across models: ``grow_shallow`` admits 10
co-tenants on ``bnci2014_004`` and exactly 1 on ``schirrmeister2017`` (9990 MiB
measured). Grouping by model name would pin every ``grow_shallow`` cell to its worst
dataset and waste most of the fleet. Grouping by (G, K) wastes nothing and costs one
extra ``sbatch`` per class.

THE PER-TENANT CEILING IS ALSO DERIVED, NOT GUESSED
---------------------------------------------------
``EEGROW_CUDA_FRACTION`` defaulted to 0.20, i.e. 2166 MiB on a 2080 Ti. Against the
measured grid that ceiling is below what 13 of 102 cells actually need: they would
OOM deterministically, release their claim, be re-claimed, and land in MISSING after
MAX_SWEEPS -- the exact failure the runner's own comments describe. The ceiling exists
to catch a runaway, not to ration the normal case, so it is emitted per pass as the
equal share of the card (see ``ceiling_fraction``), which is above that pass's
measured worst cell by construction and never over-subscribes the device.

USAGE
-----
    python benchmarks/slurm/plan_campaign.py --grid /scratch/amounir/grid_full.tsv \\
        --outdir /scratch/amounir/passes
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
PROFILE = BENCH / "profile_mem"
CONFIG = BENCH / "config"

# 15 %: the CUDA context is ~300 MiB per process and the allocator fragments. On the
# host side the same margin covers the page cache and the allocator's own slack.
HEADROOM = 0.15

# Per-cell safety factor on the MEASURED device peak, applied when solving for K.
#
# HEADROOM protects the *card*; it does nothing for an individual cell, and the two are
# not interchangeable. Solving K from the bare peak admits a cell whose measured peak
# effectively equals its own share, and the 2026-08-16 smoke run showed what that buys:
# grow_shallow x bnci2014_001 measures 908.0 MiB, its K=10 pass grants
# 0.084 x 10830.8 = 909.78 MiB, and all 11 of its cells died. A margin of 1.78 MiB, 0.2 %.
#
# Two independent reasons it has to be far larger:
#
#   Fragmentation. The OOM was not the peak overflowing its share. torch held 704 MiB
#   reserved and asked for a fresh 206 MiB segment -> 910 > 909.78. Reaching a peak of P
#   needs a ceiling above P, because the allocator cannot pack segments perfectly.
#
#   Measurement variance. The same cell is not the same number twice:
#   grow_shallow x shin2017a measured 3068 MiB in v1 and 3668 MiB in v2, +20 % between
#   two identical profiling runs on the same card.
#
# 1.25 covers the observed variance with fragmentation room on top. It costs throughput
# -- K falls, so a pass spans more node-fulls -- and that cost is the point: the
# alternative is not a faster campaign but one whose cells die at their own ceiling,
# which is how the last two runs were spent.
CELL_MARGIN = 1.25


def ceiling_fraction(k: int, worst_mib: float, card_mib: float) -> tuple[float, bool]:
    """Per-tenant device ceiling for a pass of K co-tenants, as a fraction of the card.

    The obvious formula -- the pass's worst measured cell plus a safety margin -- is
    wrong, and the test that caught it is worth keeping in mind: a K=2 pass whose worst
    cell was 4388 MiB produced a ceiling of 0.51, i.e. two tenants each entitled to
    51 % of one card. A ceiling that over-commits does not protect anyone; it just
    moves the failure from the tenant that exceeded its declared share to whichever
    tenant happens to allocate second, at the driver level, which is the very
    behaviour the ceiling exists to remove.

    The correct quantity is the equal share of the usable card::

        cap = (1 - HEADROOM) / K

    This is >= the worst cell by construction, since K = floor(usable / worst) implies
    usable / K >= worst, and it satisfies ``K x cap <= 1 - HEADROOM`` by definition, so
    a pass can never over-subscribe its own card.

    Returned at 3 decimals rather than 2: truncating to 2 costs up to 108 MiB on this
    card, which is enough to fall back under a worst cell that sits just below the
    equal share.

    The ceiling is raised above the equal share when the worst cell needs it -- which
    happens legitimately at K=1, where the pass's single tenant may want more than
    ``1 - HEADROOM`` of the card and no co-tenant is there to be starved. The flag
    returned alongside marks only the case that is actually unsafe: ``K x cap > 1``,
    i.e. the pass entitles its tenants to more card than exists. Eating into the
    headroom is a soft cost; over-subscribing the device is not.
    """
    if k == 1:
        # No co-tenant, so there is nobody for a ceiling to protect -- and a ceiling
        # here does active harm. The old formula gave grow_shallow x schirrmeister2017
        # (9830 MiB measured) a cap of 0.908 = 9834 MiB: 4 MiB of margin on a card with
        # 1000 MiB still free, i.e. the same 0.2 %-margin failure that killed the K=10
        # pass, but self-inflicted and for no benefit. Each K=1 tenant owns its GPU via
        # CUDA_VISIBLE_DEVICES, so it gets the whole card.
        return (1.0, False)
    cap = int((1 - HEADROOM) / k * 1000) / 1000
    if cap * card_mib < worst_mib:
        cap = min(1.0, (int(worst_mib / card_mib * 1000) + 1) / 1000)
    return (cap, k * cap > 1.0)


# --------------------------------------------------------------- smoke mode
# A validation run is not a small campaign, and the difference is entirely in these
# two strings. Every one of them answers a failure this project has already paid for.
#
# RESULTS_DIR / LOGS / CLAIMS, all three, never one or two. A shared RESULTS_DIR makes
# the campaign skip cells the validation happened to finish; a shared LOGS erases the
# only evidence a validation run produces; and a shared CLAIMS is the mechanism that
# silently lost 32 cells of the first packed campaign -- reap_stale only reaps claims
# whose owner is on the current host, so a claim left behind by a validation job on one
# node is invisible to a campaign job on another, which skips that cell forever.
#
# CELL_TIMEOUT=7200 against the campaign's 172800. The measured whole-dataset load is
# 2451 s for lee2019_mi and 1119 s for schirrmeister2017 (profile_host_ram.py), so two
# hours clears the slowest load with an hour of training behind it, and gives the small
# datasets hours. Cells hitting this wall is the expected outcome, not a failure: the
# question asked here is whether K tenants coexist, and that is settled long before a
# cell finishes.
#
# MAX_SWEEPS=1 follows from that. With a deliberately short timeout most cells end
# without a CSV, so they release their claims, and the default 3 sweeps would re-run
# the same node-full three times to learn the same thing three times.
#
# EXTRA=++model.grow_every=1 is what makes a two-hour window able to see the peak at
# all. The device peak of a growing arm is the growth step, and its cost is set by the
# layer geometry rather than by how many epochs preceded it -- profile_grid_memory.py
# relies on the same deviation for the same reason. At the production grow_every=5 a
# truncated cell would stop while the layer is still narrow, and the run would report
# a peak no production cell will ever have.
#
# All three paths sit outside /scratch/amounir/eegrow, which the deploy rsync mirrors
# with --delete. Anything written inside that tree by a job is deleted by the next
# deploy unless an exclude matches it, and none matches `logs/`.
SMOKE_ENV = (",RESULTS_DIR=/scratch/amounir/results_smoke"
             ",LOGS=/scratch/amounir/logs/smoke"
             ",CLAIMS=/scratch/amounir/eegrow_claims_smoke"
             ",CELL_TIMEOUT=7200,MAX_SWEEPS=1"
             ",EXTRA=++model.grow_every=1")
# 3 h, not the header's 3 days: a validation run that hangs must not hold three GPUs
# for a weekend. It has to exceed CELL_TIMEOUT plus the sweep's own startup, and does.
SMOKE_SBATCH = (" --time=03:00:00 --job-name=eegrow_smoke"
                " --output=/scratch/amounir/logs/smoke_%j.log")

# The production campaign gets its own triplet too, for two reasons that are not
# cosmetic. pack_run.sh warns that RESULTS_DIR, CLAIMS and LOGS are overridden together
# or not at all, so they move as one.
#
# CLAIMS, and this is the sharp one: a claim is an atomic mkdir carrying the owner host,
# and reap_stale() only reaps claims whose owner matches the CURRENT host. A claim left
# by margpu022 is therefore invisible AND unreapable to every job that lands anywhere
# else -- its cell is silently never computed, and no log says so. The default claims
# directory still held 235 such entries from the 2026-08-14 campaign (40 with an owner
# on margpu020/022, 195 with no owner file at all). Starting from a virgin directory
# costs nothing and removes the whole class of failure; the old one is archived, not
# deleted, so the record of what that campaign claimed survives.
#
# RESULTS_DIR because the default, benchmarks/results, already holds 2160 CSVs from
# 2026-07-23..08-10 produced by earlier grids and by code predating today's fixes
# (the growth-device peak in skorch_integration.py, the int/float cache key). Verified
# before the switch: 0 of this campaign's 1280 cells matches an existing CSV, so nothing
# would have been wrongly skipped -- but a result set meant for a paper should not have
# to be disentangled from an older one after the fact.
#
# Same rule as the smoke's: all three sit outside /scratch/amounir/eegrow, which the
# deploy rsync mirrors with --delete.
# The suffix is the plan's, so a result can be traced back to the plan that placed it
# without consulting anything else. Bump all four together or none.
#
# PARAMETERISED SINCE 2026-08-28, and the tag is not decoration. These three paths
# were the string "v5" hard-coded, which is exactly wrong for the campaign that
# REPLACES v5: sharing RESULTS_DIR with it would make the runner skip every cell v5
# already produced -- silently adopting the undertrained scores this grid exists to
# throw away. A campaign names its own triplet or it inherits another's results.
def campaign_env(tag: str) -> str:
    return (f",RESULTS_DIR=/scratch/amounir/results_{tag}"
            f",LOGS=/scratch/amounir/logs/pack_{tag}"
            f",CLAIMS=/scratch/amounir/eegrow_claims_{tag}")

# Passes submitted to more than one allocation.
#
# WHY. A pass is bounded by the wall of the single job that runs it, and pack_run.sh's
# time guard only *prints* the comparison -- it does not refuse. A pass whose work
# exceeds the 3-day header is therefore killed mid-cell with no requeue, and the cells
# it had not reached are simply absent.
#
# grid_g3k1 is the one that gets close: 130 cells, 169 h of measured compute, 3 at a
# time (K=1, G=3) = 56 h against a 72 h wall. 22 % of margin on an estimate built from
# historical medians, for the passe that carries the heaviest cells in the grid.
#
# Three allocations, not a three-way split of the TSV. The claim mechanism is built for
# exactly this ("allocations cooperate: a node that finishes early picks up what a
# slower one has not claimed"), and it load-balances a pass whose cells run from 0.2 h
# to 15.2 h -- a static split by line order would hand one shard all the long ones.
# It also audits better: each allocation sweeps the FULL grid, so a cell orphaned by a
# dead job still appears in every survivor's `cells still missing` list.
#
# Keyed by TSV stem so the reason stays attached to the pass it is about.
REPLICAS = {"grid_g3k1": 3}


def _ml_models() -> set[str]:
    """Models that never touch the GPU, so no device bound applies to them."""
    import re
    out = set()
    for p in (CONFIG / "model").glob("*.yaml"):
        txt = p.read_text()
        if re.search(r"^kind:\s*ml\s*$", txt, re.M):
            out.add(p.stem)
    return out


# WHERE A FIXED CONTROL GETS ITS MEMORY NUMBER.
#
# The profiling campaign measured the six base arms; the three fixed controls were added
# afterwards and have no measurement of their own. The planner's response to a missing
# measurement is to EXCLUDE the cell, which for these would have dropped all 216 of them
# -- the entire ablation the paper's central claim rests on -- out of a run that still
# reported itself complete.
#
# Re-profiling is one option and borrowing is the other, and borrowing is sound here
# because fix_X is not merely similar to grow_X, it is grow_X held at its terminal
# geometry from epoch 0. Checked field by field against the model YAMLs:
#
#     fix_shallow  n_filters_time=40      grow_shallow  target_n_filters_time=40
#     fix_sccnet   n_spatial_filters=22   grow_sccnet   target_n_spatial_filters=22
#     fix_deepeeg  w2_in=32               grow_deep     target_w2=32
#
# grow_X's peak is reached AT that width and carries the growth machinery (the extension
# tensors, the line search's forward passes) on top of it. So the borrowed number is an
# upper bound, and on memory an upper bound is the safe direction: it can only lower K,
# i.e. cost throughput, never cause an OOM.
BORROW_GPU_FROM = {"fix_shallow": "grow_shallow",
                   "fix_sccnet": "grow_sccnet",
                   "fix_deepeeg": "grow_deep"}


class Cell(NamedTuple):
    """A grid cell with the two measurements that decide which pass it lands in.

    Named rather than positional: it went from four fields to nine when the alignment
    arm arrived, and the consequence of reading one index wrong here is not a crash but
    a pass file whose cells describe a different experiment than the one planned.
    """

    eval: str
    dataset: str
    model: str
    align: str
    seed: str
    stem: str
    reserved: float   # peak GPU MiB, measured
    rss: float        # peak host MiB, measured
    bound: str        # which of the two put the cell in this pass


def _parse_cell(line: str) -> tuple[str, str, str, str, str, str]:
    """One grid row -> ``(eval, dataset, model, align, seed, stem)``.

    Two layouts, and the difference is not cosmetic. Since the alignment arm exists the
    generator emits six fields, the last being the output stem that ``pack_run.sh``
    tests to decide a cell is done. This planner rewrites each pass as its own TSV, so
    a parser that read only the first four would DROP the alignment column -- and
    pack_run's own legacy fallback would then fill it back in as ``none``. The result is
    a complete, well-formed campaign in which every aligned cell silently ran raw and
    overwrote its raw twin. Failure with no error message and no missing file, which is
    why the shapes are distinguished here rather than assumed.

    The four-field layout stays readable so the older point files (ml_*.txt, the pilot
    grids) still plan.
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 6:
        ev, ds, m, align, seed, stem = parts[:6]
        return ev, ds, m, align, seed, stem
    if len(parts) == 4:
        ev, ds, m, seed = parts
        return ev, ds, m, "none", seed, f"{m}__seed{seed}"
    raise ValueError(f"grid row has {len(parts)} fields, expected 4 or 6: {line!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True,
                    help="TSV: eval/dataset/model/align/seed/stem (or the legacy "
                         "eval/dataset/model/seed)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gpu-json", default=str(PROFILE / "grid_memory.json"))
    ap.add_argument("--ram-json", default=str(PROFILE / "host_ram.json"))
    # Defaults describe a tau node as pack_run.sbatch actually requests it.
    # 10830.8, not the advertised 11264. This is what cuda's mem_get_info reports as
    # total on the 2080 Ti, identically across all 108 profiled cells -- and it is the
    # same quantity set_per_process_memory_fraction multiplies the fraction by. Using
    # the advertised figure would make the emitted ceiling ~4 % looser than intended.
    ap.add_argument("--card-mib", type=float, default=10830.8,
                    help="device total AS THE DRIVER REPORTS IT (2080 Ti)")
    ap.add_argument("--mem-mib", type=float, default=120 * 1024, help="--mem of the allocation")
    ap.add_argument("--gpus", type=int, default=3, help="G: GPUs per allocation")
    ap.add_argument("--max-k", type=int, default=10)
    # 24, per pack_run.sbatch's own reasoning: a tau node carries 3-4 GPUs against 32
    # cores and is rarely empty, so a request near 32 sits in PENDING behind a node
    # that will never have both at once. Tenants beyond this share cores, which is
    # what already happened at G=3 K=10 (30 processes, 24 cores) and is harmless with
    # OMP_NUM_THREADS=1 -- the work is GPU- and I/O-bound, not core-bound.
    ap.add_argument("--max-cpus", type=int, default=24)
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="write submit.sh even though some cells have no memory "
                         "measurement and will be missing from the campaign")
    ap.add_argument("--smoke", action="store_true",
                    help="emit one node-full of the heaviest cells per pass instead "
                         "of the whole pass; validates the derived ceilings under "
                         "real co-tenancy, which the solo-on-the-card profile cannot")
    # THE CHECKOUT IS AN ARGUMENT, NOT A CONSTANT. /scratch/amounir/eegrow is the
    # August tree: it carries the drop_last fix but NOT the s=0 abstention (5337c56),
    # so a growing arm run there still adds permanently dead neurons and still reports
    # them in growable_width. Defaulting to it is how a campaign silently reproduces a
    # bug it was launched to remove.
    ap.add_argument("--root", default="/scratch/amounir/eegrow",
                    help="checkout the jobs run from; /scratch/amounir/eegrow is the "
                         "August tree and is missing the s=0 fix")
    ap.add_argument("--tag", default="v5",
                    help="names RESULTS_DIR / LOGS / CLAIMS together; a campaign that "
                         "reuses another's tag inherits its results as 'already done'")
    # --gres OVERRIDES THE WRAPPER'S HEADER, so emitting a bare `gpu:N` silently
    # discards a card-type pin the wrapper set on purpose. Every headline number of the
    # final grid is a subject-paired grow_X vs bd_X difference; letting one arm land on
    # ampere and the other on turing puts a hardware difference inside the paired
    # difference, and nothing downstream can remove it. Empty keeps the old behaviour.
    ap.add_argument("--gres-type", default="",
                    help="card class to pin, e.g. 'turing'; must match the wrapper's "
                         "own --gres or the emitted line overrides it away")
    ap.add_argument("--wrapper", default="benchmarks/slurm/pack_run.sbatch",
                    help="sbatch wrapper, relative to --root. The wrapper is where the "
                         "Hydra protocol overrides live, so it selects the science")
    args = ap.parse_args()

    gpu = {(r["model"], r["dataset"]): r for r in json.loads(Path(args.gpu_json).read_text())}
    ram = {r["dataset"]: r for r in json.loads(Path(args.ram_json).read_text())}
    ml = _ml_models()

    card_usable = args.card_mib * (1 - HEADROOM)
    mem_usable = args.mem_mib * (1 - HEADROOM)

    rows = [_parse_cell(l) for l in
            Path(args.grid).read_text().splitlines() if l.strip()]

    passes: dict[tuple[int, int], list] = defaultdict(list)
    impossible: list[tuple] = []
    unmeasured: list[tuple] = []
    borrowed: set[tuple[str, str]] = set()

    for ev, ds, m, align, seed, stem in rows:
        # --- device bound -------------------------------------------------------
        if m in ml:
            reserved, k_gpu = 0.0, args.max_k
        else:
            g = gpu.get((m, ds))
            if g is None or g.get("error") or not g.get("peak_reserved_mib"):
                donor = BORROW_GPU_FROM.get(m)
                d = gpu.get((donor, ds)) if donor else None
                if d and not d.get("error") and d.get("peak_reserved_mib"):
                    g = d
                    borrowed.add((m, ds))
                else:
                    unmeasured.append((ev, ds, m, seed,
                                       "no GPU measurement" if g is None
                                       else (g.get("error") or "no peak")))
                    continue
            reserved = g["peak_reserved_mib"]
            # max(1, ...): a cell that PRODUCED a measurement ran alone on this exact
            # card without OOM -- that is what producing the number means. So K=1 is
            # feasible for it by construction, and no headroom argument can overrule
            # an observation. Without this, grow_shallow x schirrmeister2017 (9990 MiB
            # reserved, measured, no OOM) is excluded because 9990 > 0.85 x 10831,
            # even though the 15 % margin exists to cover CO-TENANT fragmentation and
            # there are no co-tenants at K=1. Cells that did not fit alone are not
            # measurements at all: they land in `unmeasured` with their OOM.
            k_gpu = max(1, int(card_usable // (reserved * CELL_MARGIN)))
        # --- host bound ---------------------------------------------------------
        r = ram.get(ds)
        if r is None or r.get("error") or not r.get("peak_rss_mib"):
            unmeasured.append((ev, ds, m, seed, "no RAM measurement"))
            continue
        rss = r["peak_rss_mib"]
        k_ram = int(mem_usable // (args.gpus * rss))

        # How many tenants the NODE can host at all, from RAM alone. This is the real
        # ceiling on a RAM-bound pass, and it does not care how many GPUs there are.
        n_ram = int(mem_usable // rss)
        # Tenants we will actually run on a node, and the GPUs needed to hold them.
        # Solving for G rather than only for K is what stops a RAM-bound pass from
        # idling cards: schirrmeister2017 admits 3 tenants per node whatever happens,
        # and bd_shallow there wants 126 MiB of a 10.8 GiB card -- spreading those 3
        # tenants over 3 GPUs wastes two devices that another pass could be using.
        # Putting them on one GPU costs nothing and returns two cards to the fleet.
        n_tot = min(args.gpus * min(args.max_k, k_gpu), n_ram)
        if n_tot < 1:
            k, g_use = 0, args.gpus
        else:
            g_use = max(1, -(-n_tot // min(args.max_k, k_gpu)))   # ceil
            k = -(-n_tot // g_use)                                # ceil
        k = min(args.max_k, k, k_gpu)
        # Which constraint actually decided, for the report. A cell held at max_k by
        # neither memory is labelled as such rather than credited to whichever raw
        # bound happens to be smaller -- otherwise the plan reads as RAM-bound in
        # places where RAM had nothing to do with it.
        bound = ("cap" if min(k_gpu, n_ram) > args.max_k * args.gpus
                 else "ram" if n_ram < args.gpus * min(args.max_k, k_gpu)
                 else "gpu")
        if k < 1:
            # Only RAM can land here now: a cell with a GPU measurement is feasible at
            # K=1 by construction (see above), so an infeasible cell is one whose
            # single tenant does not fit in the node's memory at all.
            impossible.append((ev, ds, m, seed,
                               f"ram {rss:.0f}>{mem_usable:.0f} MiB for one tenant"))
            continue
        passes[(g_use, k)].append(
            Cell(ev, ds, m, align, seed, stem, reserved, rss, bound))

    # ------------------------------------------------------------------ report
    print(f"grid={args.grid}  cells={len(rows)}")
    print(f"card={args.card_mib:.0f} MiB usable={card_usable:.0f}   "
          f"node mem={args.mem_mib:.0f} MiB usable={mem_usable:.0f}  G={args.gpus}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmds = []
    print(f"{'pass':>9s} {'cells':>6s} {'worst GPU':>10s} {'worst RSS':>10s} "
          f"{'cap':>6s} {'cap MiB':>8s} {'node GiB':>9s} {'fulls':>6s}  bound  datasets")
    print("-" * 120)
    tight = []
    total_fulls = 0.0
    for g_use, k in sorted(passes, reverse=True):
        cells = passes[(g_use, k)]
        # Computed on the FULL pass, before any smoke subsetting: the ceiling under
        # test has to be the one the real run will use, or the smoke run validates a
        # configuration that never executes.
        worst_gpu = max(c.reserved for c in cells)
        worst_rss = max(c.rss for c in cells)
        # Node-fulls: how many times this pass has to fill an allocation end to end.
        # This, not the cell count, is what the campaign's wall-clock is made of.
        fulls = len(cells) / (g_use * k)
        total_fulls += fulls
        if args.smoke:
            # Exactly one node-full (G x K), heaviest first. Fewer would never put K
            # tenants on one card at once, which is the only thing this run is for;
            # more would just be the campaign.
            cells = sorted(cells, key=lambda c: -(c.reserved + c.rss))[:g_use * k]
        # Fraction of the WHOLE card, which is what cap_cuda_fraction applies via
        # torch.cuda.set_per_process_memory_fraction.
        cap, forced = ceiling_fraction(k, worst_gpu, args.card_mib)
        if forced:
            tight.append((g_use, k))
        # What the pass actually demands of the node's --mem, so the number can be
        # checked against the request instead of trusted.
        node_gib = g_use * k * worst_rss / 1024
        bounds = {c.bound for c in cells}
        dss = sorted({c.dataset for c in cells})
        tsv = outdir / f"grid_g{g_use}k{k}.tsv"
        # Six fields, the same shape the generator emitted. Writing four here would drop
        # the alignment arm and pack_run.sh would refill it as `none` -- see _parse_cell.
        tsv.write_text("".join(
            f"{c.eval}\t{c.dataset}\t{c.model}\t{c.align}\t{c.seed}\t{c.stem}\n"
            for c in cells))
        print(f"G={g_use} K={k:<4d} {len(cells):6d} {worst_gpu:10.0f} {worst_rss:10.0f} "
              f"{cap:6.3f} {cap * args.card_mib:8.0f} {node_gib:9.1f} {fulls:6.1f}  "
              f"{'/'.join(sorted(bounds)):7s} {','.join(dss)[:36]}")
        # --gres and --cpus-per-task override the sbatch header, so one wrapper serves
        # every pass. Cores are sized to the tenants (one BLAS thread each) with a
        # floor of 4 for the loader; asking for the header's 24 on a 1-GPU pass would
        # queue behind a node that has cores free but not that many.
        gres = f"gpu:{args.gres_type}:{g_use}" if args.gres_type else f"gpu:{g_use}"
        # G IS EXPORTED, NOT LEFT TO SLURM. pack_run.sh reads `G=${G:-$SLURM_GPUS_ON_NODE}`,
        # and under `--exclusive` SLURM hands over the WHOLE node -- so SLURM_GPUS_ON_NODE
        # is the node's physical card count, never the `--gres` count asked for above. The
        # 2026-08-30 launch showed it: a `--gres=gpu:turing:3` pass reported `G=4` on
        # margpu019, while margpu[018,022] have 3 cards and margpu028 has 2.
        #
        # On the GPU-bound passes that extra card is merely free capacity. On the passes
        # this planner deliberately SHRANK to fit host RAM it is fatal, because G is
        # precisely the decision being discarded: tenants are G*K, so g1k9 on a 4-card
        # node runs 36 tenants of 11.5 GiB (414 GiB) and g1k3 runs 12 of 26.4 GiB
        # (325 GiB), against 187 GiB of real memory. `--mem` cannot stop it -- it is not
        # enforced here (the turing nodes report AllocMem=0), so the arbiter is the kernel
        # OOM killer, which uses SIGKILL, which no trap catches. That is the incident that
        # cost v5 85 cells, and solving for G is worth nothing if the answer is dropped
        # between this line and the runner.
        line = (f"sbatch{SMOKE_SBATCH if args.smoke else ''} --gres={gres} "
                f"--cpus-per-task={min(args.max_cpus, max(4, g_use * k))} "
                f"--export=ALL,GRID={tsv},G={g_use},K={k},EEGROW_CUDA_FRACTION={cap:.3f}"
                f"{SMOKE_ENV if args.smoke else campaign_env(args.tag)} "
                f"{args.wrapper}")
        # See REPLICAS. One allocation unless the pass is long enough to risk its wall;
        # never under --smoke, whose whole point is a single node-full.
        n_alloc = 1 if args.smoke else REPLICAS.get(tsv.stem, 1)
        if n_alloc > 1:
            cmds.append(f"# {tsv.stem}: {n_alloc} allocations cooperating on one grid")
        cmds.extend([line] * n_alloc)
    print(f"\n{total_fulls:.1f} node-fulls of serialised work across "
          f"{len(passes)} passes")
    if tight:
        print(f"\nWARNING: passes {tight} entitle their tenants to more than one whole "
              "card (K x cap > 1). Over-subscribed -- do not run as is.")

    if impossible:
        print(f"\ncells that fit at no K ({len(impossible)}):")
        for ev, ds, m, seed, why in impossible[:40]:
            print(f"  {ev:15s} {ds:19s} {m:14s} s{seed}  {why}")
        if len(impossible) > 40:
            print(f"  ... and {len(impossible) - 40} more")
        (outdir / "impossible.tsv").write_text(
            "".join(f"{a}\t{b}\t{c}\t{d}\t{e}\n" for a, b, c, d, e in impossible))
    if unmeasured:
        print(f"\ncells with no measurement, EXCLUDED from every pass ({len(unmeasured)}):")
        seen = {}
        for ev, ds, m, seed, why in unmeasured:
            seen.setdefault((m, ds, why), 0)
            seen[(m, ds, why)] += 1
        for (m, ds, why), n in sorted(seen.items()):
            print(f"  {m:14s} x {ds:19s} {n:3d} cells  {why}")
        (outdir / "unmeasured.tsv").write_text(
            "".join(f"{a}\t{b}\t{c}\t{d}\t{e}\n" for a, b, c, d, e in unmeasured))

    if borrowed:
        print(f"\nGPU measurement borrowed for {len(borrowed)} (model, dataset) pairs "
              "-- upper bounds, see BORROW_GPU_FROM:")
        for m, ds in sorted(borrowed):
            print(f"  {m:14s} x {ds:19s} <- {BORROW_GPU_FROM[m]}")

    placed = sum(len(v) for v in passes.values())
    print(f"\nplaced {placed}/{len(rows)} cells in {len(passes)} passes")

    # AN INCOMPLETE PLAN IS NOT A PLAN. Excluding unmeasured cells and reporting it in
    # the middle of a long printout is how 216 cells -- the whole fixed-control arm, on
    # which the paper's central claim rests -- were dropped from a campaign that then ran
    # to completion and looked finished. The exclusion is a legitimate operation; doing
    # it while still writing submit.sh is not, because the next step after this script
    # is `bash submit.sh` and nobody re-reads a screen of output that ended in success.
    if unmeasured and not args.allow_unmeasured:
        print(f"\nREFUSING to write submit.sh: {len(unmeasured)} of {len(rows)} cells "
              "have no memory measurement and would be silently missing from the "
              f"campaign.\n  see {outdir / 'unmeasured.tsv'}\n"
              "  fix: profile them (benchmarks/slurm/profile_grid_memory.sbatch), add "
              "them to BORROW_GPU_FROM if a measured arm has the same geometry, or pass "
              "--allow-unmeasured to plan the grid without them ON PURPOSE.")
        return 1

    script = outdir / "submit.sh"
    script.write_text("#!/bin/bash\n# Generated by plan_campaign.py -- do not edit.\n"
                      f"# root={args.root}  tag={args.tag}  wrapper={args.wrapper}\n"
                      f"set -euo pipefail\ncd {args.root}\n"
                      + "\n".join(cmds) + "\n")
    script.chmod(0o755)
    print(f"wrote {script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
