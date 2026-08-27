"""Run one benchmark cell without Hydra's CLI decorator.

`run_moabb_hydra.run` is wrapped in `@hydra.main`, which builds an argparse parser at
import time. On Python 3.14 that raises `ValueError: badly formed help string`:
argparse now type-checks `help=`, and hydra 1.3.5 passes a lazy object there. The
config composition itself is fine, so this driver composes the same config tree with
`hydra.compose` and calls `run_moabb_hydra.main` directly.

Same override syntax as the CLI::

    python run_cell.py model=bd_deep4 dataset=bnci2014_001 eval=within_session \
        train.lr_schedule=cosine 'dataset.subjects=[1]'

It is a transport, not a second source of truth: every default still comes from
config/, so a cell run here and a cell run under sbatch differ only in how argv was
parsed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import run_moabb_hydra

    with initialize_config_dir(config_dir=str(HERE / "config"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=argv)
    print(OmegaConf.to_yaml(cfg.train), flush=True)

    t0 = time.perf_counter()
    res = run_moabb_hydra.main(cfg)
    dt = time.perf_counter() - t0
    print(f"\n[run_cell] {len(res)} rows in {dt:.1f}s "
          f"({dt / max(len(res), 1):.1f}s per fold-row), "
          f"mean score {res['score'].mean():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main(sys.argv[1:]))
