#!/bin/bash
# Deploy this checkout to the cluster tree the final campaign runs from.
#
#     bash benchmarks/slurm/deploy_final.sh              # to /scratch/amounir/eegrow_budget
#     HOST=margaret02 DEST=/scratch/amounir/eegrow_budget bash ...deploy_final.sh
#
# WHY `git archive | tar -x` AND NOT rsync.
#
# rsync --delete once removed 276 subjects from the shared epoch cache, and the reason
# it could is structural: --delete makes the destination a *function* of the source, so
# anything the source does not know about is garbage by definition -- including results,
# caches and logs that only exist on the cluster. `tar -x` can only create or overwrite.
# It cannot remove. A file that was deleted from the repository therefore survives on
# the cluster as a stale copy; that is the price, and it is the right side of the trade
# (a stale module that nothing imports costs nothing, a deleted results tree costs the
# campaign). `check` below prints anything present there and absent here.
#
# WHY `git archive HEAD` AND NOT THE WORKING TREE.
#
# The tree stamps its own commit into `.eegrow_sha`, and every result row carries that
# value. Shipping the working tree would make the stamp a claim about a commit whose
# content was not what ran -- a wrong sha is worse than no sha, because nothing
# afterwards can tell it apart from a right one. So: refuse on a dirty tree, ship
# exactly HEAD, stamp exactly HEAD.
set -euo pipefail

HOST="${HOST:-margaret02}"
DEST="${DEST:-/scratch/amounir/eegrow_budget}"
cd "$(git rev-parse --show-toplevel)"

SHA=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ -n "$(git status --porcelain -- src benchmarks pyproject.toml)" ]; then
  echo "REFUSING: uncommitted changes under src/ or benchmarks/." >&2
  git status --short -- src benchmarks pyproject.toml >&2
  echo "Commit them first: the deployed tree stamps $SHA onto every result row, and" >&2
  echo "that stamp has to be true." >&2
  exit 1
fi

echo "deploying $BRANCH @ ${SHA:0:12} -> $HOST:$DEST"

# `--` matters: only these three paths, so a stray top-level artefact cannot ride along.
git archive --format=tar HEAD -- src benchmarks pyproject.toml \
  | ssh "$HOST" "mkdir -p '$DEST' && tar -x -C '$DEST' && \
                 printf '%s\n' '$SHA' > '$DEST/.eegrow_sha' && \
                 find '$DEST' -name __pycache__ -type d -prune -exec rm -rf {} + "

# VERIFY, because a deploy that half-arrived looks exactly like one that arrived. The
# guard in pack_run.sh catches the wrong *package*; this catches the wrong *files*, and
# it catches them now rather than at hour 3 of a seven-day allocation.
echo "--- verifying"
ssh "$HOST" "cd '$DEST' && \
  echo \"stamp: \$(cat .eegrow_sha)\" && \
  for f in src/eegrow/alignment.py benchmarks/aligned_paradigm.py \
           benchmarks/subject_stamp.py benchmarks/utils.py \
           benchmarks/config/align/euclidean.yaml \
           benchmarks/slurm/pack_run.sh benchmarks/slurm/final_grid.tsv; do
    [ -s \"\$f\" ] && echo \"  ok   \$f\" || { echo \"  MISSING \$f\"; exit 1; }
  done && \
  echo \"cells in grid: \$(wc -l < benchmarks/slurm/final_grid.tsv)\""

# Content identity, per file. A name existing is not the check that matters -- a
# truncated transfer leaves a name. Compared file by file rather than as one digest of
# the whole tree, because the two sides legitimately hold different *sets* of files
# (tar -x never deletes, so stale modules survive on the cluster) and a whole-tree
# digest would report a difference on every deploy and be ignored by the second one.
echo "--- content check"
MANIFEST=$(git ls-tree -r --name-only HEAD -- src/eegrow benchmarks/config \
           benchmarks/utils.py benchmarks/run_moabb_hydra.py \
           benchmarks/aligned_paradigm.py benchmarks/subject_stamp.py \
           benchmarks/pipelines.py benchmarks/slurm/pack_run.sh \
           benchmarks/slurm/final_grid.tsv)
DIFFS=$(
  join -j 1 -o 0,1.2,2.2 \
    <(while IFS= read -r f; do echo "$f $(git show "HEAD:$f" | shasum | cut -d' ' -f1)"; \
      done <<< "$MANIFEST" | sort) \
    <(ssh "$HOST" "cd '$DEST' && while IFS= read -r f; do \
        echo \"\$f \$(shasum \"\$f\" 2>/dev/null | cut -d' ' -f1)\"; done" \
      <<< "$MANIFEST" | sort) \
  | awk '$2 != $3 {print "  DIFFERS " $1}'
)
if [ -z "$DIFFS" ]; then
  echo "  all $(wc -l <<< "$MANIFEST" | tr -d ' ') tracked files identical on both sides"
else
  echo "$DIFFS" >&2
  echo "REFUSING to declare the deploy done." >&2
  exit 1
fi
echo "done. Next: plan_campaign.py, then read the submit.sh it writes."
