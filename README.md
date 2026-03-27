## Unsupervised Elicitation of Language Models

## Setup

### From Source
```bash
git clone https://github.com/Taslim-M/Unsupervised-Elicitation
cd Unsupervised-Elicitation
conda env create -f env.yaml
conda activate UE
pip install -e .
pip install vllm==0.14.1 bitsandbytes

set -g mouse on
huggingface-cli login --token YOUR_HF_TOKEN

huggingface-cli download yidingp/icm_OpinionQA --repo-type dataset --local-dir ./data --local-dir-use-symlinks False
```


### API for Pretrained Base Models

llama 3.1 405B
```bash
python -m vllm.entrypoints.openai.api_server  --model meta-llama/Llama-3.1-405B  --tensor-parallel-size 4  --served-model-name meta-llama/Llama-3.1-8B  --quantization bitsandbytes  --load-format bitsandbytes  --dtype bfloat16  --gpu-memory-utilization 0.92  --host 0.0.0.0  --port 8000
```

llama 3.1 70B multi-gpu
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B \
  --served-model-name llama70b-gpu0 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 \
  --port 8000
```
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B \
  --served-model-name llama70b-gpu1 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 \
  --port 8001
```
```bash
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B \
  --served-model-name llama70b-gpu2 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 \
  --port 8002
```
```bash
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B \
  --served-model-name llama70b-gpu3 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 \
  --port 8003
```

### Secrets

If you set `LLAMA_API_BASE` in your shell environment (e.g. `export LLAMA_API_BASE=http://127.0.0.1:8001/v1`), it will override the value loaded from `SECRETS`. This is useful when running multiple vLLM instances on different ports.

You should create a file called SECRETS at the root of the repository with the following contents:


```
LLAMA_API_BASE=<your_api_base_url>
NYU_ORG=None
ARG_ORG=None
API_KEY=None
```

### Data Preparation

Download data from this [link](https://drive.google.com/file/d/1AJdFJO9IHfOnWHyIlGvInyndLu6EvcfV/view?usp=sharing).
Put it under the `data/` directory.
OpinionQA files live under `data/processed_OpinionQA/`, and persona folds live under `data/persona_tailor_icm/`.

## Run

```bash
./scripts/train.sh llama70b-gpu0 123456
```

```bash
./scripts/train_persona.sh llama70b-gpu0 123456
```

The main script is located in `src/experiments/ICM.py`
An example command for labeling truthfulQA data:
```bash
export LLAMA_API_BASE="http://127.0.0.1:8000/v1"
python src/experiments/ICM.py ...
```

```bash
export LLAMA_API_BASE="http://127.0.0.1:8001/v1"
python src/experiments/ICM.py ...
```

```bash
python ICM.py --testbed OpinionQA --alpha 50 --file_name preferences_POLPARTY_binary_noRefused_Democrat_part2of4.json  --K 500 --model llama70b-gpu0 --batch_size 128
```

```bash
python ICM.py --testbed OpinionQA --alpha 50 --file_name preferences_POLPARTY_binary_noRefused_Republican_part1of4.json  --K 500 --model meta-llama/Llama-3.1-70B --batch_size 128
```

```bash
python src/experiments/ICM.py --testbed persona --alpha 50 --file_name DSN_50_fold1.json --K 500 --model llama70b-gpu0 --batch_size 256
```

For persona folds, if `--batch_size` is larger than the fold size, ICM will automatically run on the full fold.

### Interactive Pipeline
```bash
export LLAMA_API_BASE="http://127.0.0.1:8000/v1"

python ICM.py --testbed OpinionQA --alpha 50 --file_name preferences_POLPARTY_binary_noRefused_Independent_part1of4.json2of4.json  --K 500 --model llama70b-gpu0 --batch_size 128 --continue_from_existing 1
```

Arguments:

- `--seed`: random seed
- `--alpha`: the coefficient for mutual predictability in our scoring function
- `--testbed`: name of the testbed, e.g., alpaca, truthfulqa, gsm8k, persona
- `--model`: name of the pretrained base model, e.g., meta-llama/Llama-3.1-70B
- `--batch_size`: size of a minibatch when running ICM on large datasets that cannot be fit in to the context all at once[^1]. 
[^1]: Since ICM relies on in-context learning, it might not be able to fix all datapoints in the context at once. In our experiments, we split the whole dataset into $N$ batches (e.g., each batch consists of 256 datapoints) based on the context limit and data length, and run ICM independently on each batch.
- `--num_seed`: number of randomly labeled datapoints in the beginning.
- `--K`: max iteration
- `--consistency_fix_K`: max iteration for consistencyfix
- `--decay`: decay rate for simulating annealing
- `--initial_T`: initial temprature for simulated annealing
- `--final_T`: final temperature for simulated annealing
- `--scheduler`: decay scheduler for simulated annealing

### Iterative Fine-tuning

Instead of using the initial pretrained model ($M_0$) to label all $N$ batches, we do iterative fine-tuning: 

- fine-tune the pretrained model on the first $j$ batches to obtain $M_j$

- use $M_j$ to label the $j+1$-th batch.

We use [axolotl](https://github.com/axolotl-ai-cloud/axolotl) for fine-tuning.
