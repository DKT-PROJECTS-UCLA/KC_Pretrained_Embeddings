#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""

###############################################################################
# CONFIG — COMMENT / UNCOMMENT LIKE PROJECT A
###############################################################################

# ===================== ASSIST 2009 =====================
# DATASETS=("assist2009")

# MODELS_assist2009=(
#   "final_embeddings_assist2009_text_embedding_3_small"
#   # add more if needed
# )

# DATASET_CSV_assist2009="../pretrained_Embedding/data_subsets/assist2009/skill_builder_data_corrected_collapsed.csv"

# COUNT=100
# ENTITY="knowledgetracing42-ucla"
# PROJECT_PREFIX="PT_2009_idea4_unfrozen_"
# METRICS_CSV="./metrics_b.csv"
# EMB_ROOT="../pretrained_Embedding"


# ===================== ASSIST 2012 =====================
DATASETS=("assist2012")
# MODELS_assist2012=("final_embeddings_assist2012_text_embedding_3_large" "final_embeddings_assist2012_embed_v4.0")
MODELS_assist2012=("final_embeddings_assist2012_text_embedding_3_large" )
DATASET_CSV_assist2012="../pretrained_Embedding/data_subsets/assist2012/2012-2013-data-with-predictions-4-final.csv"
COUNT=12
ENTITY="knowledgetracing42-ucla"
PROJECT_PREFIX="PT_2012_DKVMN"
METRICS_CSV="./metrics_b.csv"
EMB_ROOT="../pretrained_Embedding"


# ===================== ASSIST 2017 =====================
# DATASETS=("assist2017")
# MODELS_assist2017=("final_embeddings_assist2017_text_embedding_3_large")
# DATASET_CSV_assist2017="../pretrained_Embedding/data_subsets/assist2017/anonymized_full_release_competition_dataset.csv"
# COUNT=100
# ENTITY="knowledgetracing42-ucla"
# PROJECT_PREFIX="PT_2017_idea4_unfreeze"
# METRICS_CSV="./metrics_b.csv"
# EMB_ROOT="../pretrained_Embedding"

###############################################################################
# INIT METRICS CSV
###############################################################################

[[ -f "$METRICS_CSV" ]] || echo "dataset,emb_dir,pt_file,emb_size,project,testauc,testacc" > "$METRICS_CSV"

###############################################################################
# FUNCTIONS
###############################################################################

get_emb_size() {
  python - "$1" <<'PY'
import torch,sys
pt=sys.argv[1]
obj=torch.load(pt,map_location="cpu")
emb=None
if isinstance(obj,dict):
  for k in ("embeddings","embedding","weight","emb","E","W"):
    if k in obj and hasattr(obj[k],"shape"):
      emb=obj[k]; break
  if emb is None:
    for v in obj.values():
      if hasattr(v,"shape"):
        emb=v; break
elif hasattr(obj,"shape"):
  emb=obj
print(emb.shape[-1] if emb is not None else "NA")
PY
}

preprocess_dataset() {
  local dataset="$1"; local csv_src="$2"
  local dest="../data/$dataset"
  mkdir -p "$dest"
  cp "$csv_src" "$dest/" || { echo "Missing CSV: $csv_src"; exit 1; }
  python data_preprocess.py --dataset_name="$dataset"
}

###############################################################################
# MAIN LOOP
###############################################################################

for ds in "${DATASETS[@]}"; do

  # choose dataset CSV + model list
  case "$ds" in
    assist2009)
      CSV_SRC="$DATASET_CSV_assist2009"
      MODELS_VAR="MODELS_assist2009[@]"
      ;;
    assist2012)
      CSV_SRC="$DATASET_CSV_assist2012"
      MODELS_VAR="MODELS_assist2012[@]"
      ;;
    assist2017)
      CSV_SRC="$DATASET_CSV_assist2017"
      MODELS_VAR="MODELS_assist2017[@]"
      ;;
    *) echo "Unknown dataset: $ds"; exit 1 ;;
  esac

  MODEL_LIST=("${!MODELS_VAR}")

  preprocess_dataset "$ds" "$CSV_SRC"

  for model_folder in "${MODEL_LIST[@]}"; do
    mdir="$EMB_ROOT/$model_folder"
    [[ -d "$mdir" ]] || { echo "Missing folder: $mdir"; continue; }

    echo "[$ds] $model_folder"

    # portable mapfile replacement
    pt_files=()
    while IFS= read -r line; do
      pt_files+=("$line")
    done < <(find "$mdir" -maxdepth 1 -type f -name "pipeline_*_final_dkt.pt" | sort)

    [[ ${#pt_files[@]} -gt 0 ]] || { echo "No pipeline_*_final_dkt.pt"; continue; }

    project="${PROJECT_PREFIX}__${ds}__${model_folder}"

    for pt in "${pt_files[@]}"; do
      pt_base="$(basename "$pt")"
      emb_size="$(get_emb_size "$pt")"

      echo "  -> $pt_base (emb_size=$emb_size)"

      ./run_sweep_gpu_dkvmn.sh "$ds" "$project" "$COUNT" "$pt" "$ENTITY" "$emb_size"

      predict_out=$(python wandb_predict.py --save_dir=best_ckpt --use_wandb=0 || true)
      json_line=$(printf "%s\n" "$predict_out" | tail -n 200 | awk '/^\{.*\}$/' | tail -n1)

      read -r AUC ACC < <(python - <<'PY'
import json,sys,re
s=sys.stdin.read().strip()
if not s: print("NA NA"); exit()
try:
  s=re.sub("'",'"',s)
  d=json.loads(s)
  print(d.get("testauc","NA"), d.get("testacc","NA"))
except:
  print("NA NA")
PY
<<<"$json_line")

      echo "$ds,$model_folder,$pt_base,$emb_size,$project,$AUC,$ACC" >> "$METRICS_CSV"
    done

  done

done

echo "Done. Metrics -> $METRICS_CSV"
