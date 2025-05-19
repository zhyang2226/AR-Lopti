set -x

# ! huggingface cli download Models
# MODEL_PATH='Qwen/Qwen2.5-3B-Instruct'
MODEL_PATH='Qwen/Qwen2.5-7B-Instruct-1M'

# * Logic KK Dataset
DATASET_TRAIN='data/kk/instruct/merge_34567ppl'
DATASET_VAL_O='['data/kk/instruct/3ppl/test_new.parquet', 'data/kk/instruct/4ppl/test_new.parquet', 'data/kk/instruct/5ppl/test_new.parquet', 'data/kk/instruct/6ppl/test_new.parquet', 'data/kk/instruct/7ppl/test_new.parquet']'
DATASET_VAL_N='['data/deepscaler/amc_dsr.parquet', 'data/deepscaler/aime_dsr.parquet']'
EXP_NAME="Qwen2.5-7B-Instruct-1M-kklogic_grpo_Lopti"
MAX_PROMPT=400
MICRO_BS=32
VAL_N=16
TRAIN_BS=64
ROLLOUT_N=8
EPOCH=5
UPDATE_BS=256
ROLLOUT_TEMP=0.7
CLIP_LOW=0.20
CLIP_HIGH=0.24
ALGO="grpo" # reinforce_plus_plus or grpo
LOSS_AGG_MODE="token-mean" # "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
LR=1e-6
IMPORTANCE_SAMPLING="on"  # "on" or "off" (True or False) or "off_for_pos"
OPTIMIZER="adamw"  # sgd or adamw or RMSProp
KL_COEF=0.001

OUTPUT_DIR="./outputs/verl_logic_kk_${EXP_NAME}"

# ! Advantage Reweighting (AR)
REWEIGHT=True
REWEIGHT_METHOD="both_linear_normal"  # negative_sigmoid or both_sigmoid with "balance1" or mask_prob_interval1 or both_linear_normal or RAFT_linear or mask_prob_lower0.25
REWEIGHT_K=0.0  # alpha in the paper; Note that alpha=0.0 means no reweighting
REWEIGHT_TAU=1.0   # (1 - alpha) in the paper
NEG_ADV_WEIGHT=1.0

# ! Low-Probability Token Isolation (Lopti)
SEPERATE_UPDT=True  # Wether to use separate updating for low-probability tokens
SEPERATE_PROB=0.4   # eta in the paper; Note that positive values means updating low-prob token first, negative values means updating high-prob tokens first.


mkdir -p ${OUTPUT_DIR}
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ALGO \
    data.train_files="${DATASET_TRAIN}" \
    data.val_files_testonce="${DATASET_VAL_O}" \
    data.val_files_testN="${DATASET_VAL_N}" \
    data.train_batch_size=$TRAIN_BS \
    data.val_batch_size=$((TRAIN_BS / VAL_N)) \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.type=$OPTIMIZER \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$UPDATE_BS \
    actor_rollout_ref.actor.ppo_micro_batch_size=$MICRO_BS \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.importance_sampling=$IMPORTANCE_SAMPLING \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.n_val=$VAL_N \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    algorithm.samples_reweight=$REWEIGHT \
    algorithm.reweight_method=$REWEIGHT_METHOD \
    algorithm.reweight_k=$REWEIGHT_K \
    algorithm.reweight_tau=$REWEIGHT_TAU \
    algorithm.neg_adv_weight=$NEG_ADV_WEIGHT \
    algorithm.seperate_updating=$SEPERATE_UPDT \
    algorithm.seperate_prob=$SEPERATE_PROB \
    algorithm.seperate_portion=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='AR-Lopti' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.math_test_freq=20 \
    trainer.total_epochs=$EPOCH $@ 2>&1 | tee ${OUTPUT_DIR}/training_process.log
