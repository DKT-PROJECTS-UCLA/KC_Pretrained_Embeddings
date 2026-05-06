#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

cat <<'EOF' > best_run_path.txt
knowledgetracing42-ucla/titest___assist2017__final_embeddings_assist2017_text_embedding_3_small/dnuv07dz
EOF

python - "$(<best_run_path.txt)" assist2017 <<'PY'
import wandb, pathlib, sys, re, shutil

RUN_PATH, DATASET = sys.argv[1:]
RUN_PATH = RUN_PATH.strip()

api = wandb.Api()
run = api.run(RUN_PATH)
cfg = run.config

base = pathlib.Path(f"models/dkt_tiaocan_{DATASET}/{DATASET}_dkt_qid_models")
parts = [
    str(cfg.get("seed")),
    str(cfg.get("fold", 0)),
    str(cfg.get("dropout")),
    str(cfg.get("emb_size")),
    str(cfg.get("learning_rate")),
]
pattern = ".*" + ".*".join(re.escape(p) for p in parts if p != "None") + ".*"

matches = [p for p in base.iterdir() if p.is_dir() and re.match(pattern, p.name)]
if not matches:
    print("No matching folder for best run")
    sys.exit(0)

best_folder = max(matches, key=lambda p: p.stat().st_mtime)
dst = pathlib.Path("best_ckpt")
dst.mkdir(exist_ok=True)

for fn in ("qid_model.ckpt", "config.json"):
    src = best_folder / fn
    if src.exists():
        shutil.copy(src, dst / fn)

print("best_ckpt prepared")
PY
