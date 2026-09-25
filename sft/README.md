# SFT Stage (SLCA-GRPO)

We initialise every policy with a tool-calling aware SFT checkpoint obtained by
fine-tuning the backbone on the `toucan_toolcall_sft` split (≈42k samples, the
"4/7" portion of the post-filter Toucan-1.5M pool; see `data/README.md`).

Training uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with
DeepSpeed ZeRO-2, FlashAttention-2, `template=qwen`, `cutoff_len=16384`.

## 1. Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
conda create -n lf python=3.10 -y
conda activate lf
pip install -e ".[torch,metrics,deepspeed,flash-attn]"
cd -
```

## 2. Register the SFT dataset in LLaMA-Factory

Copy the JSON files from `data/` into `LLaMA-Factory/data/` and add two entries
to `LLaMA-Factory/data/dataset_info.json`:

```json
{
  "toucan_toolcall_sft": {
    "file_name": "toucan_toolcall_sft.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "tools": "tools"
    }
  },
  "toucan_toolcall_full": {
    "file_name": "toucan_toolcall_full.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "tools": "tools"
    }
  }
}
```

## 3. Launch training

| Config                               | Backbone                | Epochs | Data split                |
| :----------------------------------- | :---------------------- | :----: | :------------------------ |
| `qwen2_5_3b_split_sft.yaml`          | Qwen2.5-3B-Instruct     | 2      | 4/7 SFT split (≈42k)      |
| `qwen2_5_7b_split_sft.yaml`          | Qwen2.5-7B-Instruct     | 2      | 4/7 SFT split (≈42k)      |
| `qwen3_8b_base_split_sft.yaml`       | Qwen3-8B-Base           | 2      | 4/7 SFT split (≈42k)      |
| `qwen2_5_7b_full_sft.yaml`           | Qwen2.5-7B-Instruct     | 1      | Full SFT pool (≈74k)      |

```bash
# 8-GPU default
bash sft/train_sft.sh sft/qwen2_5_7b_split_sft.yaml

# smaller setup (e.g. 4 GPUs)
bash sft/train_sft.sh sft/qwen2_5_3b_split_sft.yaml 4
```

The shell wrapper `train_sft.sh` sets `PYTHONUNBUFFERED=1`, flips on NCCL P2P/IB
and calls `torchrun src/train.py <cfg>` inside the LLaMA-Factory directory.

## 4. What comes next

The resulting checkpoint is consumed by the RL stage in `rl/`; set
`MODEL_PATH` in `rl/slca_grpo/train_slca_grpo.sh` to the `output_dir` from the
SFT YAML, e.g. `./outputs/sft_split/qwen2_5_7b_toucan_toolcall`.
