
# Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs

**Code Implementation for Paper Submission #111 at NeurIPS 2025**

For the convenience of reviewers, the provided code allows for reproducing the results reported in our paper with a single-line command.

Please note that this code is intended solely for the purpose of peer review. We kindly request that it **NOT** be distributed or used for any purpose beyond the review process.

## Build Up Environment

Our code has been successfully tested on 4×80GB A100/H100 GPUs with CUDA 12.1. The following commands will create a Conda environment with all the required dependencies:

```bash
  conda create -n AR_Lopti python=3.9
  conda activate AR_Lopti
  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
  pip3 install vllm==0.6.3 ray
  pip3 install flash-attn --no-build-isolation
  pip install -e .
  pip install wandb IPython matplotlib
  pip install torchdata==0.8.0
  pip install pylatexenc
  pip install tensordict==0.5.0
```

## Run the Code

After setting up the environment, you can run the code with the following command:

* For GRPO Baseline
  ```bash
    bash scripts/train_kklogic_baseline_4x80GB.sh
  ```
* For GRPO with Advantage Reweighting
  ```bash
    bash scripts/train_kklogic_AR_4x80GB.sh
  ```
* For GRPO with Lopti
  ```bash
    bash scripts/train_kklogic_Lopti_4x80GB.sh
  ```
* For GRPO with Advantage Reweighting + Lopti
  ```bash
    bash scripts/train_kklogic_AR-Lopti_4x80GB.sh
  ```

The models will be continuously evaluated during training, and all experimental records will be automatically logged to the `wandb` platform.

Please note that the model to be trained can be modified in **Lines 4-5** of each bash script. The default setting is `Qwen/Qwen2.5-7B-Instruct-1M`, and another option is `Qwen/Qwen2.5-3B-Instruct`.

Additionally, the baseline algorithm can be adjusted in **Line 22** of each bash script. The default setting is `grpo`, with `reinforce_plus_plus` as an alternative option.

## Acknowledgements
* This repository is built on top of [verl](https://github.com/volcengine/verl). We extend our gratitude to the verl team for open-sourcing such a powerful RL4LLMs framework.
* We also sincerely acknowledge the datasets and corresponding reward function provided by [LogicRL](https://github.com/Unakar/Logic-RL), [DeepScaleR](https://github.com/agentica-project/rllm), [AdaRFT](https://github.com/limenlp/verl), and [ORZ](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero).