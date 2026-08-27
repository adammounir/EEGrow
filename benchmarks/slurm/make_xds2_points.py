"""Point lists for the two grids that turn the pilot's non-results into results.

The pilot (job 454639, target BNCI2014_001) answered nothing about pooling and nothing
about interpolation, for two different reasons. Both are addressed here, and the ordering
of each list is the stopping rule: the cells that decide the question come first, so a
grid can be cancelled early without losing the answer.

Grid A -- power (`xds2_cho.txt`)
-------------------------------
`pooled - within` came out at +0.036 with a per-subject sd of 0.094 over n = 9 subjects.
That design's minimum detectable effect is 0.101: it could only ever have found a ten-point
gain, and the 95% CI [-0.037, +0.109] excludes nothing anyone cares about. The observed
effect needs n = 56 at 80% power, so the fix is subjects, not epochs or seeds.

Cho2017 supplies 52, and it records all 22 target electrodes natively -- it is a `core`
dataset, so this grid does not depend on the interpolation question at all. With the
pilot's 9 that is n = 61.

The per-subject deltas are the reason this matters beyond a p-value: they ran from -0.060
to +0.226 while the seed-to-seed sd was only 0.021. The spread is four times the noise, so
the effect is real and *heterogeneous* -- some subjects gain a lot, some lose. n = 52 is
what makes "who gains" a question with an answer (delta against within-subject score),
which is more interesting than the group mean either way.

`euclidean` first, both arms: that is the power question. Then `scale` on `pooled`, to
replicate the one effect the pilot did establish (euclidean - scale = +0.037, 8/9 subjects,
p = 0.009 -- and n = 9 was already sufficient there, because that effect is 1.1x the
between-subject sd where pooling was 0.4x). Then `none`, completing the ladder.

`lodo` is left out. Zero-shot transfer was +0.012 (p = 0.71) and it is not the claim; if
`pooled` fails at n = 61 there is nothing for `lodo` to add, and if it succeeds `lodo` can
be run afterwards on the cells that matter.

Grid B -- the interpolation controls (`xds2_interp.txt`)
------------------------------------------------------
`core+interp` minus `core` was -0.010 / +0.001. That null is uninformative by
construction: the tier adds one dataset, Shin2017A, worth +4.7% trials and +11.7%
subjects, and 9 of the 22 target electrodes are recorded natively so no interpolation
happens on them at all. The measurement was "does 5% more data help", not "does
interpolation enable cross-montage transfer".

The regime where interpolation is load-bearing is the one the pilot skipped:

`core+lowrank`  Zhou2016, 14 electrodes. The projection is rank 14 in 22 columns, so the
                covariance is singular by construction. Deep nets invert nothing and are
                indifferent; this arm asks whether rank-deficient interpolated data is
                usable data.
`core+extrap`   BNCI2014_004, 3 electrodes, geometric guard switched off. 19 of 22
                channels come from 3 spatial degrees of freedom, the worst 10.1 cm from
                anything recorded. This is the negative control: if the grid cannot tell
                this apart from `core`, then the pooled arm is insensitive to what is in
                the pool and *every* interpolation result here is uninterpretable,
                including the pilot's null. That makes it the most informative cell of the
                two grids, so it is ordered first.

Both are run against `core` at align=euclidean, the arm where pooling is least handicapped.

    python benchmarks/slurm/make_xds2_points.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

MODELS = ("bd_shallow", "grow_shallow")
SEEDS = (0, 1, 2)


def grid_a() -> list[str]:
    """target arm model align seed tier -- euclidean/both arms first."""
    rows = []
    for align, arms in (("euclidean", ("within", "pooled")),
                        ("scale", ("pooled",)),
                        ("none", ("pooled",))):
        for arm in arms:
            for model in MODELS:
                for seed in SEEDS:
                    rows.append(f"cho2017 {arm} {model} {align} {seed} core")
    return rows


def grid_b() -> list[str]:
    """The negative control first: it qualifies every other interpolation number."""
    rows = []
    for tier in ("core+extrap", "core+lowrank"):
        for model in MODELS:
            for seed in SEEDS:
                rows.append(f"bnci2014_001 pooled {model} euclidean {seed} {tier}")
    return rows


def main() -> int:
    for name, rows in (("xds2_cho", grid_a()), ("xds2_interp", grid_b())):
        p = HERE / f"{name}.txt"
        p.write_text("\n".join(rows) + "\n")
        print(f"{p.name}: {len(rows)} points (array 0-{len(rows) - 1})")
        for r in rows[:4]:
            print(f"   {r}")
        print("   ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
