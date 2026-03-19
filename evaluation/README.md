# Evaluation
This folder contains a script to evaluate multi-turn image-editing results by querying an OpenAI-style VLM (via the `openai` Python client). The script:

- downloads a benchmark dataset (`leigangqu/MSE-Bench`) and experimental results (`leigangqu/MSE-Bench-results`) from Hugging Face
- saves input images and generated images to `./tmp_data/`
- prompts the VLM to score each turn for **prompt-following** and **consistency**
- saves per-example JSON outputs and an aggregated CSV report

---

## ✅ Requirements

1. Python 3.10+ (or compatible)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI-compatible API key in the environment:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
```

> **Note:** `compute_score.py` uses the `openai` Python client. The `--api_model` flag controls which model is used (e.g., `gpt4o`, `gpt-5-nano`, `gpt-5.4`).

---

## 🚀 Running evaluation

The simplest run uses the default benchmark (`leigangqu/MSE-Bench`) and downloads results from the Hugging Face split matching `--model_name`.

```bash
model_name="vincie_7b"

python3 compute_score.py \
  --model_name "$model_name" \
  --api_model gpt-5-nano \
  --num_workers 32 \
  --res_path ./tmp_data/results/"$model_name".json
```

### Common flags

- `--model_name` (required): corresponds to the split name in `leigangqu/MSE-Bench-results`.
- `--res_path` (required): where the aggregated JSON results will be written.
- `--api_model` (default `gpt-5-nano`): model used for evaluation. 
- `--num_workers` (default `32`): number of parallel workers.
- `--batch_size` (default `2`): per-worker batch size.
- `--success_threshold` (default `6`): minimum score (0–10) for both `prompt_following` and `consistency` to count as successful.
- `--hf_data_name` (default `leigangqu/MSE-Bench`): Hugging Face dataset used as the reference data.

---

## 📦 Output files

- **Image cache:** `./tmp_data/images/<model_name>/...` (downloaded/generated images)
- **Per-example eval JSONs:** `./tmp_data/results/<model_name>/<index>.json`
- **Aggregated results JSON:** the file you pass to `--res_path` (e.g. `./tmp_data/results/vincie_7b.json`)
- **CSV summary:** `./tmp_data/csv/<res_path_basename>.csv` (generated automatically)

---

## 🔍 Evaluation logic (what is scored)

For each example, the script evaluates the **last turn** of the multi-turn edit using two criteria:

1. **prompt_following** – how well the last edited image follows the last user instruction (0–10)
2. **consistency** – how well the last edited image keeps the unchanged parts consistent with the reference (0–10)

A per-turn pass/fail (`all`) is computed as:

- `all = 1` when both scores exceed `--success_threshold` (default `6`)

---

## 🗂️ Notes / Troubleshooting

- The script expects the Hugging Face results dataset to contain columns like `turn_0_image`, `turn_1_image`, etc.
- If the script fails due to rate limits or API errors, re-run the same command: it caches per-example JSON in `./tmp_data/results/<model_name>/<index>.json`.

---

If you want a custom evaluation (different dataset / prompt template), modify `evaluation/compute_score.py` directly.
