#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

DATASET="$1"
PROJECT_NAME="$2"
COUNT="$3"
PRETRAINED_PATH="$4"
ENTITY="${5:-your-wandb-entity}"
EMB_SIZE="${6:-NA}"

rm -rf best_ckpt/ all_wandbs/

export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""

###############################################################################
# 1) Generate base sweep YAML
###############################################################################

python generate_wandb.py \
  --dataset_names "$DATASET" \
  --model_names dkt \
  --project_name "$PROJECT_NAME"

TEMPLATE_YAML=$(ls all_wandbs/"${DATASET}"_dkt_qid_0.yaml)
PATCHED_YAML="all_wandbs/${DATASET}_dkt_qid_0.patched.yaml"

###############################################################################
# 2) Patch sweep YAML
###############################################################################

python - "$TEMPLATE_YAML" "$PATCHED_YAML" "$PRETRAINED_PATH" "$EMB_SIZE" <<'PY'
import sys, yaml, os
src, dst, pt, emb_size = sys.argv[1:5]

with open(src) as f:
    y = yaml.safe_load(f)

y.setdefault("parameters", {})
y["parameters"]["pretrained_emb_path"] = {"value": pt}

if emb_size != "NA":
    y["parameters"]["emb_size"] = {"value": int(emb_size)}

# Name the sweep after the embedding file
emb_tag = os.path.basename(pt).replace(".pt", "")
y["name"] = f"sweep_{emb_tag}"

with open(dst,"w") as f:
    yaml.safe_dump(y, f, sort_keys=False)

print(f"Patched -> {dst}")
PY

###############################################################################
# 3) Start sweep
###############################################################################

SWEEP_OUTPUT=$(wandb sweep --entity "$ENTITY" --project "$PROJECT_NAME" "$PATCHED_YAML" 2>&1 | tee sweep_log.txt)

SWEEP_PATH=$(echo "$SWEEP_OUTPUT" | awk '/Run sweep agent with:/ {print $NF}')

[[ -n "${SWEEP_PATH:-}" ]] || SWEEP_PATH=$(echo "$SWEEP_OUTPUT" | grep -oE "${ENTITY}/${PROJECT_NAME}/[a-z0-9]+" | tail -n1 || true)
[[ -n "${SWEEP_PATH:-}" ]] || { echo "Could not get sweep path"; exit 1; }

export WANDB_NOTES="pretrained_emb_path: $(basename "$PRETRAINED_PATH")"

wandb agent "$SWEEP_PATH" --count "$COUNT"

# Gracefully stop the sweep; ignore the benign "already finished" error
if ! STOP_OUTPUT=$(wandb sweep --stop "$SWEEP_PATH" 2>&1); then
  if [[ "$STOP_OUTPUT" == *"Sweep already finished."* ]]; then
    echo "[info] Sweep already finished on WANDB; continuing."
  else
    echo "$STOP_OUTPUT" >&2
    exit 1
  fi
fi

###############################################################################
# 4) Add test metrics + copy best ckpt
###############################################################################

python - "$SWEEP_PATH" "$DATASET" <<'PY'
import sys, wandb, pathlib, re, subprocess, json

SWEEP_PATH, DATASET = sys.argv[1:]
api = wandb.Api()
sweep = api.sweep(SWEEP_PATH)

base = pathlib.Path(f"models/dkt_tiaocan_{DATASET}/{DATASET}_dkt_qid_models")

for run in sweep.runs:
    cfg = run.config

    parts = [
        str(cfg.get("seed")),
        str(cfg.get("fold", 0)),
        str(cfg.get("dropout")),
        str(cfg.get("emb_size")),
        str(cfg.get("learning_rate")),
    ]
    pattern = ".*" + ".*".join(re.escape(p) for p in parts if p and p!="None") + ".*"

    if not base.exists():
        continue

    matches = [p for p in base.iterdir() if p.is_dir() and re.match(pattern, p.name)]
    if not matches:
        continue

    ckpt_dir = max(matches, key=lambda p: p.stat().st_mtime)

    res = subprocess.run(
        ["python","wandb_predict.py","--save_dir",str(ckpt_dir),"--use_wandb","0"],
        capture_output=True, text=True
    )

    jline=""
    for line in res.stdout.splitlines():
        s=line.strip()
        if s.startswith("{") and s.endswith("}"):
            jline=s

    if not jline:
        continue

    try:
        d=json.loads(jline.replace("'","\""))
        run.summary["testauc"]=float(d.get("testauc",0))
        run.summary["testacc"]=float(d.get("testacc",0))
        run.summary.update()
    except:
        pass

best = max(sweep.runs, key=lambda r: r.summary.get("best_valid_auc", 0))
best_path = f"{best.entity}/{best.project}/{best.id}"
print(best_path)
with open("best_run_path.txt","w") as fout:
    fout.write(best_path + "\n")
PY


BEST_RUN_PATH="$(cat best_run_path.txt)"

python - "$BEST_RUN_PATH" "$DATASET" <<'PY'
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

pattern = ".*" + ".*".join(re.escape(p) for p in parts if p!="None") + ".*"

matches = [p for p in base.iterdir() if p.is_dir() and re.match(pattern, p.name)]
if not matches:
    print("No matching folder for best run")
    sys.exit(0)

best_folder = max(matches, key=lambda p: p.stat().st_mtime)

dst = pathlib.Path("best_ckpt")
dst.mkdir(exist_ok=True)

for fn in ("qid_model.ckpt","config.json"):
    src = best_folder / fn
    if src.exists():
        shutil.copy(src, dst / fn)

print("best_ckpt prepared")
PY
