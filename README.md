# nanoGPT × CFG-Dataset

Two compact code components:

| folder | purpose |
|--------|---------|
| **cfg-dataset/** | build & analyse context-free-grammar derivation datasets |
| **nanoGPT/**    | Karpathy nanoGPT with customizations to train on those datasets |

No raw datasets or model checkpoints are stored in Git — only source code.

---

## Quick start (verified on Python 3.10 + CUDA 12.4)

```bash
git clone https://github.com/codewithsajid/LLMs-Reasoning.git
cd llms-reasoning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

### Build Dataset 

python cfg-dataset/build_dataset_parallel.py \
       --max-depth 30 --bucket-cap 10000 \
       --output-dir data/dyck_1 --gzip --workers 16

---

### Convert to Plain-text for NanoGPT

python nanoGPT/cfg_to_text_v2.py \
       --input "../../dataset/dyck_1/*.jsonl.gz" \
       --train_out data/cfg/train.txt

---

### Train 

cd nanoGPT
python train.py config/train_cfg.py
