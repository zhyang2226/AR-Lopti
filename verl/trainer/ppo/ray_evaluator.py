      
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                                      eos_mask=response_mask,
                                                                                      gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_reward_metrics(batch):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    reward_metrics = {}
    reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
    # Calculate all_correct ratio (value == 3)

    if "kk" in batch.non_tensor_batch['data_source'][0]:
        all_correct = torch.sum(reward_tensor == 3).float() / reward_tensor.numel()
        reward_metrics["reward/all_correct_ratio"] = all_correct.detach().item()
        # Calculate format_error ratio (value == -1)
        format_error = torch.sum(reward_tensor == -1).float() / reward_tensor.numel()
        reward_metrics["reward/format_error_ratio"] = format_error.detach().item()
        # Calculate wrong answer ratio (value == -1)
        format_error = torch.sum(reward_tensor == -0.5).float() / reward_tensor.numel()
        reward_metrics["reward/wrong_answer_ratio"] = format_error.detach().item()
    else:
        all_correct = torch.sum(reward_tensor == 1).float() / reward_tensor.numel()
        reward_metrics["reward/all_correct_ratio"] = all_correct.detach().item()

    return reward_metrics

def compute_reweight_advantage(advantage, log_prob, reweight_method, reweight_k, reweight_tau, neg_adv_weight):
    if "negative_sigmoid" in reweight_method:
        # reweight negative advantage
        reweight_advantage = 2 * torch.sigmoid(reweight_k * (torch.exp(log_prob) - reweight_tau)) * advantage
        neg_advantage = torch.where(advantage > 0, 0.0, reweight_advantage)
        pos_advantage = torch.where(advantage > 0, advantage, 0.0)

        # TODO re-Balance the advantage 
        # * Method1 Balance adv*prob
        if "balance1" in reweight_method:
            pos_adv_sum = torch.sum(pos_advantage * torch.exp(log_prob))
            neg_adv_sum = - torch.sum(neg_advantage * torch.exp(log_prob))
            neg_advantage *= pos_adv_sum / neg_adv_sum
            neg_advantage *= neg_adv_weight
        
        reweight_advantage = pos_advantage + neg_advantage
        # re-Balance the advantage 
    
    elif "both_sigmoid" in reweight_method:
        reweight_advantage = 2 * torch.sigmoid(reweight_k * (torch.exp(log_prob) - reweight_tau)) * advantage
        neg_advantage = torch.where(advantage > 0, 0.0, reweight_advantage)
        pos_advantage = torch.where(advantage > 0, reweight_advantage, 0.0)
        if "balance1" in reweight_method:
            pos_adv_sum = torch.sum(pos_advantage * torch.exp(log_prob))
            neg_adv_sum = - torch.sum(neg_advantage * torch.exp(log_prob))
            # neg_advantage *= pos_adv_sum / neg_adv_sum
            neg_advantage *= neg_adv_weight
            reweight_advantage = pos_advantage + neg_advantage
    
    elif "both_linear" in reweight_method: # reweight_advantage = reweight_k * prob + reweight_tau
        reweight_advantage = (reweight_k * torch.exp(log_prob) + reweight_tau) * advantage
        neg_advantage = torch.where(advantage > 0, 0.0, reweight_advantage)
        pos_advantage = torch.where(advantage > 0, reweight_advantage, 0.0)
        if "normal" in reweight_method:
            neg_advantage *= neg_adv_weight
            reweight_advantage = pos_advantage + neg_advantage
        elif "balance1" in reweight_method: # only reweight positive samples
            neg_advantage = torch.where(advantage > 0, 0.0, advantage)
            neg_advantage *= neg_adv_weight
            reweight_advantage = pos_advantage + neg_advantage
    
    elif "mask_prob" in reweight_method:
        # interval = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        interval = [0, 0.25, 0.5, 0.75, 1.0]
        if "interval1" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > interval[0], 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= interval[1], 1.0, 0.0)
        elif "interval2" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > interval[1], 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= interval[2], 1.0, 0.0)
        elif "interval3" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > interval[2], 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= interval[3], 1.0, 0.0)
        elif "interval4" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > interval[3], 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= interval[4], 1.0, 0.0)
        elif "interval5" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > interval[4], 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= interval[5], 1.0, 0.0)
        elif "interval6" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > interval[5], 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= interval[6], 1.0, 0.0)
        elif "interval7" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > interval[6], 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= interval[7], 1.0, 0.0)
        elif "interval8" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > interval[7], 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= interval[8], 1.0, 0.0)
        elif "lower0.25" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > 0, 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= 0.25, 1.0, 0.0)
        elif "lower0.5" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > 0, 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= 0.5, 1.0, 0.0)
        elif "lower0.75" in reweight_method:
            mask1 = torch.where(torch.exp(log_prob) > 0, 1.0, 0.0)
            mask2 = torch.where(torch.exp(log_prob) <= 0.75, 1.0, 0.0)
        else:
            raise NotImplementedError(f"Reweight method {reweight_method} is not supported.")
        mask = mask1 * mask2
        reweight_advantage = advantage * mask
    elif "RAFT" in reweight_method:
        reweight_advantage = torch.where(advantage > 0, 1.0, 0.0)
        if "linear" in reweight_method:
            reweight_advantage = (reweight_k * torch.exp(log_prob) + reweight_tau) * reweight_advantage
    else:
        raise NotImplementedError(f"Reweight method {reweight_method} is not supported.")
    
    return reweight_advantage

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayEvaluator(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler
        from torchdata.stateful_dataloader import StatefulDataLoader
        shuffle_seed = 42
        generator_train = torch.Generator()
        generator_train.manual_seed(shuffle_seed)

        generator_val1 = torch.Generator()
        generator_val1.manual_seed(shuffle_seed)
        generator_val2 = torch.Generator()
        generator_val2.manual_seed(shuffle_seed + 1)

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='right')
        sampler = RandomSampler(data_source=self.train_dataset, generator=generator_train)
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)
        # self.train_dataloader = DataLoader(dataset=self.train_dataset,
        #                                    batch_size=self.config.data.train_batch_size,
        #                                    shuffle=True,
        #                                    drop_last=True,
        #                                    collate_fn=collate_fn,
        #                                    generator=generator_train)
        if self.config.data.val_files_testonce is not None and self.config.data.val_files_testonce != "None":
            self.val_dataset_Testonce = RLHFDataset(parquet_files=self.config.data.val_files_testonce,
                                        tokenizer=self.tokenizer,
                                        prompt_key=self.config.data.prompt_key,
                                        max_prompt_length=1024,
                                        filter_prompts=True,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation='right')
            self.val_dataloader_Testonce = DataLoader(dataset=self.val_dataset_Testonce,
                                            batch_size=256,
                                            shuffle=True,
                                            drop_last=False,
                                            collate_fn=collate_fn,
                                            generator=generator_val1)
        
        if self.config.data.val_files_testN is not None and self.config.data.val_files_testN != "None":
            self.val_dataset_TestN = RLHFDataset(parquet_files=self.config.data.val_files_testN,
                                        tokenizer=self.tokenizer,
                                        prompt_key=self.config.data.prompt_key,
                                        max_prompt_length=1024,
                                        filter_prompts=True,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation='right')
            self.val_dataloader_TestN = DataLoader(dataset=self.val_dataset_TestN,
                                            batch_size=32,
                                            shuffle=True,
                                            drop_last=False,
                                            collate_fn=collate_fn,
                                            generator=generator_val2)
        
        # self.amc_dataset = RLHFDataset(parquet_files=self.config.data.amc_files,
        #                                 tokenizer=self.tokenizer,
        #                                 prompt_key=self.config.data.prompt_key,
        #                                 max_prompt_length=self.config.data.max_prompt_length,
        #                                 filter_prompts=True,
        #                                 return_raw_chat=self.config.data.get('return_raw_chat', False),
        #                                 truncation='error')
        # self.amc_dataloader = DataLoader(dataset=self.amc_dataset,
        #                                  batch_size=len(self.amc_dataset),
        #                                  drop_last=False,
        #                                  collate_fn=collate_fn)
        # self.aime_dataset = RLHFDataset(parquet_files=self.config.data.aime_files,
        #                                  tokenizer=self.tokenizer,
        #                                  prompt_key=self.config.data.prompt_key,
        #                                  max_prompt_length=570,
        #                                  filter_prompts=True,
        #                                  return_raw_chat=self.config.data.get('return_raw_chat', False),
        #                                  truncation='error')
        # self.aime_dataloader = DataLoader(dataset=self.aime_dataset,
        #                                  batch_size=64,
        #                                  drop_last=False,
        #                                  collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        # assert len(self.val_dataloader_Testonce) >= 1
        # assert len(self.amc_dataloader) >= 1
        # assert len(self.aime_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        # print(f'Size of val dataloader (One Response for each prompt): {len(self.val_dataloader_Testonce)}')
        # print(f'Size of AMC-EVAL dataloader: {len(self.amc_dataloader)}')
        # print(f'Size of AIME-EVAL dataloader: {len(self.aime_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate_greedy(self):
        thinking_tokens = ["wait", "Wait", "verify", "yet", 
                           "rethink", "think", 
                           "consider", "reconsider",
                           "analyze", "examine", "evaluate", "re-evaluate", "reevaluate", "assess",
                           "again", "re-examine", "reconsider"]
        reward_tensor_lst = []
        data_source_lst = []
        test_batch_idx = 0
        for test_data in self.val_dataloader_Testonce:
            test_batch = DataProto.from_single_dict(test_data)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            test_batch = test_batch.union(test_output_gen_batch)
            reward_tensor = self.val_reward_fn(test_batch)
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

            log_prob = self.actor_rollout_wg.compute_old_log_prob(test_output_gen_batch)
            prompt_list = self.tokenizer.batch_decode(test_output_gen_batch.batch['prompts'], skip_special_tokens=True)
            response_list = self.tokenizer.batch_decode(test_output_gen_batch.batch['responses'], skip_special_tokens=True)
            reward_for_save = reward_tensor.sum(dim=1)
            output_str = []
            output_tensor = []
            for i in range(len(prompt_list)):
                store_str_dict = {
                    # "prompt": prompt_list[i],
                    "response": response_list[i],
                    "reward": reward_for_save[i].item(),
                    "data_source": test_batch.non_tensor_batch['data_source'][i],
                }
                store_tensor_dict = {
                    "prompt": prompt_list[i],
                    "response": response_list[i],
                    "reward": reward_for_save[i].item(),
                    "responses_ids": test_output_gen_batch.batch['responses'][i],
                    "log_prob": log_prob.batch["old_log_probs"][i],
                    "data_source": test_batch.non_tensor_batch['data_source'][i],
                }
                output_str.append(store_str_dict)
                output_tensor.append(store_tensor_dict)

            if not os.path.exists(self.config.actor_rollout_ref.model.eval_store_path):
                os.makedirs(self.config.actor_rollout_ref.model.eval_store_path)
            # Save output_str as json.
            import json
            with open(os.path.join(self.config.actor_rollout_ref.model.eval_store_path, f"test_output_{test_batch_idx}.json"), 'w') as f:
                json.dump(output_str, f, indent=4)
            # Save tensor as pt
            torch.save(output_tensor, os.path.join(self.config.actor_rollout_ref.model.eval_store_path, f"test_output_{test_batch_idx}.pt"))

            test_batch_idx += 1
            print('validation generation end')

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            if "kk" in data_source:
                max_reward_value =  3
                min_reward_value = -3
            else:
                max_reward_value = 1
                min_reward_value = 0
            count_equal_max = sum(1 for reward in rewards if reward == max_reward_value)
            total_count = len(rewards)
            metric_dict[f'val/test_greedy/{data_source}'] = count_equal_max / total_count if total_count > 0 else 0

        return metric_dict
    
    def _validate_sampling(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader_Testonce:
            test_batch = DataProto.from_single_dict(test_data)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = 1.0
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            if "kk" in data_source:
                max_reward_value =  3
                min_reward_value = -3
            else:
                max_reward_value = 1
                min_reward_value = 0
            count_equal_max = sum(1 for reward in rewards if reward == max_reward_value)
            total_count = len(rewards)
            metric_dict[f'val/test_sampling/{data_source}'] = count_equal_max / total_count if total_count > 0 else 0

        return metric_dict
    
    def _validate_Testonce(self):
        metric_dict = {}
        new_metric = self._validate_greedy()
        metric_dict.update(new_metric)
        # new_metric = self._validate_sampling()
        # metric_dict.update(new_metric)
        return metric_dict

    def _validate_TestN(self):
        reward_tensor_lst = []
        data_source_lst = []
        test_batch_idx = 0
        for test_data in self.val_dataloader_TestN:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if n_val_samples > 1:
                test_gen_batch_padded.meta_info['val_temperature'] = 1.0
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

            log_prob = self.actor_rollout_wg.compute_old_log_prob(test_output_gen_batch)
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(test_output_gen_batch)
            prompt_list = self.tokenizer.batch_decode(test_output_gen_batch.batch['prompts'], skip_special_tokens=True)
            response_list = self.tokenizer.batch_decode(test_output_gen_batch.batch['responses'], skip_special_tokens=True)
            reward_for_save = reward_tensor.sum(dim=1)
            output_str = []
            output_tensor = []
            for i in range(len(prompt_list)):
                store_str_dict = {
                    # "prompt": prompt_list[i],
                    "response": response_list[i],
                    "reward": reward_for_save[i].item(),
                    "data_source": test_batch.non_tensor_batch['data_source'][i],
                }
                store_tensor_dict = {
                    "prompt": prompt_list[i],
                    "response": response_list[i],
                    "reward": reward_for_save[i].item(),
                    "responses_ids": test_output_gen_batch.batch['responses'][i],
                    "log_prob": log_prob.batch["old_log_probs"][i],
                    "ref_log_prob": ref_log_prob.batch["ref_log_prob"][i],
                    "data_source": test_batch.non_tensor_batch['data_source'][i],
                }
                output_str.append(store_str_dict)
                output_tensor.append(store_tensor_dict)

            if not os.path.exists(self.config.actor_rollout_ref.model.eval_store_path):
                os.makedirs(self.config.actor_rollout_ref.model.eval_store_path)
            # Save output_str as json.
            import json
            with open(os.path.join(self.config.actor_rollout_ref.model.eval_store_path, f"test_output_{test_batch_idx}.json"), 'w') as f:
                json.dump(output_str, f, indent=4)
            # Save tensor as pt
            torch.save(output_tensor, os.path.join(self.config.actor_rollout_ref.model.eval_store_path, f"test_output_{test_batch_idx}.pt"))

            test_batch_idx += 1

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu().reshape(-1, n_val_samples)  # (batch_size, n_val_samples)
        data_sources = np.concatenate(data_source_lst, axis=0).reshape(-1, n_val_samples)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i][0] 
            if data_source == '':
                data_source = 'unknown'
            if f"{data_source}-avg@{n_val_samples}" not in data_source_reward:
                data_source_reward[f"{data_source}-avg@{n_val_samples}"] = []
                data_source_reward[f"{data_source}-pass@{n_val_samples}"] = []
            for k in range(n_val_samples):
                data_source_reward[f"{data_source}-avg@{n_val_samples}"].append(reward_tensor[i][k].item())
            data_source_reward[f"{data_source}-pass@{n_val_samples}"].append(reward_tensor[i].max().item())

        metric_dict = {}

        for data_source, rewards in data_source_reward.items():
            if "kk" in data_source:
                max_reward_value =  3
                min_reward_value = -3
            else:
                max_reward_value = 1
                min_reward_value = 0
            count_equal_max = sum(1 for reward in rewards if reward == max_reward_value)
            # count_equal_max = sum(1 for reward in rewards if reward > min_reward_value)
            # count_score = sum(rewards)
            total_count = len(rewards)
            metric_dict[f'val/test_sampling/{data_source}'] = count_equal_max / total_count if total_count > 0 else 0

        return metric_dict
    

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
    
    # ! NEW Save checkpoint
    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    # ! NEW Load checkpoint
    def _load_checkpoint(self):
        global_step_folder = self.config.actor_rollout_ref.model.eval_path
        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None or global_step_folder=="None":
                print('Evaluating the scratch Model')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=False)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=False)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None:

            if self.config.data.val_files_testN is not None and self.config.data.val_files_testN != "None":
                val_metrics_testN = self._validate_TestN()
                pprint(f'Initial validation metrics: {val_metrics_testN}')
                logger.log(data=val_metrics_testN, step=self.global_steps)

            if self.config.data.val_files_testonce is not None and self.config.data.val_files_testonce != "None":
                val_metrics_testonce = self._validate_Testonce()
                pprint(f'Initial validation metrics: {val_metrics_testonce}')
                logger.log(data=val_metrics_testonce, step=self.global_steps)

            # amc_metrics = self._test_amc()
            # pprint(f'Initial AMC metrics: {amc_metrics}')
            # logger.log(data=amc_metrics, step=self.global_steps)
            # aime_metrics = self._test_aime()
            # pprint(f'Initial AIME metrics: {aime_metrics}')
            # logger.log(data=aime_metrics, step=self.global_steps)

            if self.config.trainer.get('val_only', False):
                return

        # # we start from step 1
        # self.global_steps += 1

        # for epoch in range(self.config.trainer.total_epochs):
        #     for batch_dict in self.train_dataloader:
        #         print(f'epoch {epoch}, step {self.global_steps}')
        #         metrics = {}
        #         timing_raw = {}

        #         batch: DataProto = DataProto.from_single_dict(batch_dict)

        #         # pop those keys for generation
        #         gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

        #         with _timer('step', timing_raw):
        #             # generate a batch
        #             with _timer('gen', timing_raw):
        #                 gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        #             batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
        #                                                      dtype=object)
        #             # repeat to align with repeated responses in rollout
        #             batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        #             batch = batch.union(gen_batch_output)

        #             # balance the number of valid tokens on each dp rank.
        #             # Note that this breaks the order of data inside the batch.
        #             # Please take care when you implement group based adv computation such as GRPO and rloo
        #             self._balance_batch(batch, metrics=metrics)

        #             # compute global_valid tokens
        #             batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

        #             # ! Add a new condition (not self.config.algorithm.samples_retemp) to avoid recomputing ref_log_prob
        #             if self.use_reference_policy and not self.config.algorithm.samples_retemp:
        #                 # compute reference log_prob
        #                 with _timer('ref', timing_raw):
        #                     ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        #                     batch = batch.union(ref_log_prob)

        #             # compute values
        #             if self.use_critic:
        #                 with _timer('values', timing_raw):
        #                     values = self.critic_wg.compute_values(batch)
        #                     batch = batch.union(values)

        #             with _timer('adv', timing_raw):
        #                 # compute scores. Support both model and function-based.
        #                 # We first compute the scores using reward model. Then, we call reward_fn to combine
        #                 # the results from reward model and rule-based results.
        #                 if self.use_rm:
        #                     # we first compute reward model score
        #                     reward_tensor = self.rm_wg.compute_rm_score(batch)
        #                     batch = batch.union(reward_tensor)

        #                 # we combine with rule-based rm
        #                 reward_tensor = self.reward_fn(batch)
        #                 batch.batch['token_level_scores'] = reward_tensor

        #                 # compute rewards. apply_kl_penalty if available
        #                 if not self.config.actor_rollout_ref.actor.use_kl_loss:
        #                     batch, kl_metrics = apply_kl_penalty(batch,
        #                                                          kl_ctrl=self.kl_ctrl,
        #                                                          kl_penalty=self.config.algorithm.kl_penalty)
        #                     metrics.update(kl_metrics)
        #                 else:
        #                     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

        #                 # compute advantages, executed on the driver process
        #                 batch = compute_advantage(batch,
        #                                           adv_estimator=self.config.algorithm.adv_estimator,
        #                                           gamma=self.config.algorithm.gamma,
        #                                           lam=self.config.algorithm.lam,
        #                                           num_repeat=self.config.actor_rollout_ref.rollout.n)
                        
        #                 # * For re-weight Negative Samples
        #                 if self.config.algorithm.samples_reweight:
        #                     batch.batch['old_advantages'] = torch.clone(batch.batch['advantages'])
        #                     batch.batch['advantages'] = compute_reweight_advantage(batch.batch['advantages'],
        #                                                                          batch.batch['old_log_probs'],
        #                                                                          self.config.algorithm.reweight_method,
        #                                                                          self.config.algorithm.reweight_k,
        #                                                                          self.config.algorithm.reweight_tau,
        #                                                                          self.config.algorithm.neg_adv_weight)
        #             # * For plot_distribution Figure
        #             if self.config.algorithm.plot_dist and self.global_steps >= 1:
        #                 from verl.trainer.fig_plot import draw_distribution_fig, draw_prob_change_adv, draw_prob_change_pos
        #                 if self.config.algorithm.plot_fig_type == 'prob_histogram':
        #                     plot_dict = {}
        #                     prompt_tensor = batch.batch['prompts']
        #                     plot_matching_indices = torch.where((prompt_tensor == prompt_tensor[0]).all(dim=1))[0]
        #                     initial_log_probs = self.ref_policy_wg.compute_ref_log_prob(batch)
        #                     plot_dict['prompts'] = prompt_tensor[plot_matching_indices, :]
        #                     plot_dict['responses'] = batch.batch['responses'][plot_matching_indices, :]
        #                     plot_dict['ref_logprobs'] = initial_log_probs.batch['ref_log_prob'][plot_matching_indices, :]
        #                     plot_dict['logprobs'] = torch.clone(batch.batch['old_log_probs'][plot_matching_indices, :])
        #                 elif self.config.algorithm.plot_fig_type == 'prob_stat':
        #                     plot_dict = {}
        #                     plot_dict['logprobs'] = torch.clone(batch.batch['old_log_probs'])
        #                 else:
        #                     raise NotImplementedError(f'plot_fig_type {self.config.algorithm.plot_fig_type} is not supported.')
        #             # * For plot_distribution Figure END

        #             # ! Compute ref_log_prob if self.config.algorithm.samples_retemp
        #             # TODO: here is a confict: kl divergence require ref_log_prob, but we don't have it if we use samples_retemp
        #             # TODO: try to fix it in future implementations
        #             if self.config.algorithm.samples_retemp:
        #                 # We first re-asign the temperature according to the temperature
        #                 if self.config.algorithm.retemp_method == 'chosen_reject':
        #                     retemp = batch.batch['advantages'][:,0].reshape(-1,1)
        #                     retemp = torch.where(retemp > 0, 
        #                                         self.config.algorithm.chosen_samples_retemp_value, 
        #                                         self.config.algorithm.reject_samples_retemp_value)
        #                 elif self.config.algorithm.retemp_method == 'prob_threshold':
        #                     retemp = torch.clone(torch.exp(batch.batch['old_log_probs']))
        #                     retemp = torch.where(retemp > self.config.algorithm.prob_threshold, 
        #                                         self.config.algorithm.above_threshold_retemp_value, 
        #                                         self.config.algorithm.below_threshold_retemp_value)
        #                 elif self.config.algorithm.retemp_method == 'prob_threshold_chosen_reject':
        #                     retemp_prob_threshold = torch.exp(batch.batch['old_log_probs'])
        #                     retemp_prob_threshold = torch.where(retemp_prob_threshold > self.config.algorithm.prob_threshold, 
        #                                                         self.config.algorithm.above_threshold_retemp_value, 0)
        #                     retemp_chosen_reject = torch.where(batch.batch['advantages'] > 0, 
        #                                                        self.config.algorithm.chosen_samples_retemp_value, 
        #                                                        self.config.algorithm.reject_samples_retemp_value)
        #                     retemp = torch.where(retemp_prob_threshold != 0, retemp_prob_threshold, retemp_chosen_reject)
        #                 else:
        #                     raise NotImplementedError
        #                 batch.batch['retemp_temperature'] = retemp
        #                 # Then, we update old_log_prob according to the new temperature
        #                 old_log_prob = self.actor_rollout_wg.compute_old_log_prob(batch)
        #                 batch.batch['old_log_probs'] = old_log_prob.batch['old_log_probs']
        #             if self.use_reference_policy and self.config.algorithm.samples_retemp:
        #                 # compute reference log_prob
        #                 with _timer('ref', timing_raw):
        #                     ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        #                     batch = batch.union(ref_log_prob)

        #             # update critic
        #             if self.use_critic:
        #                 with _timer('update_critic', timing_raw):
        #                     critic_output = self.critic_wg.update_critic(batch)
        #                 critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
        #                 metrics.update(critic_output_metrics)

        #             # implement critic warmup
        #             if self.config.trainer.critic_warmup <= self.global_steps:
        #                 # update actor
        #                 with _timer('update_actor', timing_raw):
        #                     # TODO: MERGE NEW TEMP into training process (NOT FINISHED)
        #                     # ! if self.config.algorithm.seperate_updating: we update actor seperately according to the probability
        #                     seperate_record_dict = {"actor/sep_prob": 0.0, "actor/sep_portion": 0.0}
        #                     response_length = batch.batch['responses'].size(1)
        #                     if self.config.algorithm.seperate_updating:
        #                         updating_advantage = torch.clone(batch.batch['advantages'])
        #                         batch.batch['kl_entropy_mask'] = torch.clone(batch.batch['attention_mask'])
        #                         # * update first time
        #                         if self.config.algorithm.seperate_prob != 0.0: #  ~ Prob Threshold Method
        #                             if self.config.algorithm.seperate_prob > 0: # updating low_prob tokens first
        #                                 indication = torch.exp(batch.batch['old_log_probs']) < self.config.algorithm.seperate_prob
        #                             else:
        #                                 indication = torch.exp(batch.batch['old_log_probs']) >= -self.config.algorithm.seperate_prob
        #                             seperate_record_dict["actor/sep_prob"] = abs(self.config.algorithm.seperate_prob)
        #                             seperate_record_dict["actor/sep_portion"] = (torch.sum(indication)/torch.sum(batch.batch['attention_mask'][:, -response_length:])).item()
        #                         else: #  ~ Prob Portion Method, only activated when self.config.algorithm.seperate_prob == 0
        #                             real_probs = torch.exp(batch.batch['old_log_probs']) * batch.batch['attention_mask'][:, -response_length:]
        #                             real_probs = real_probs[real_probs > 0]
        #                             if self.config.algorithm.seperate_portion > 0: # filter out zero probs
        #                                 sep_prob_with_portion = torch.quantile(real_probs, self.config.algorithm.seperate_portion, dim=0, keepdim=True)
        #                                 indication = torch.exp(batch.batch['old_log_probs']) < sep_prob_with_portion
        #                             else:
        #                                 sep_prob_with_portion = torch.quantile(real_probs, -self.config.algorithm.seperate_portion, dim=0, keepdim=True)
        #                                 indication = torch.exp(batch.batch['old_log_probs']) >= sep_prob_with_portion
        #                             seperate_record_dict["actor/sep_prob"] = sep_prob_with_portion.item()
        #                             seperate_record_dict["actor/sep_portion"] = abs(self.config.algorithm.seperate_portion)
                                
        #                         batch.batch['advantages'] = torch.where(indication, updating_advantage, 0.0)
        #                         batch.batch['kl_entropy_mask'][:, -response_length:] = torch.where(indication, batch.batch['attention_mask'][:, -response_length:], 0)
        #                         actor_output = self.actor_rollout_wg.update_actor(batch)
        #                         # * update second time
        #                         batch.batch['advantages'] = torch.where(indication, 0.0, updating_advantage)
        #                         batch.batch['kl_entropy_mask'][:, -response_length:] = torch.where(indication, 0, batch.batch['attention_mask'][:, -response_length:])
        #                         actor_output_new = self.actor_rollout_wg.update_actor(batch)
        #                         # * merge two actor_output
        #                         for key in actor_output.meta_info['metrics']:
        #                             if isinstance(actor_output.meta_info['metrics'][key], float):
        #                                 actor_output.meta_info['metrics'][key] = (actor_output.meta_info['metrics'][key] + actor_output_new.meta_info['metrics'][key]) / 2
        #                             elif isinstance(actor_output.meta_info['metrics'][key], list):
        #                                 actor_output.meta_info['metrics'][key] = actor_output.meta_info['metrics'][key] + actor_output_new.meta_info['metrics'][key]
        #                         # * recover the original batch
        #                         batch.batch['advantages'] = updating_advantage
        #                     # ! else we update actor once together for all tokens
        #                     else:
        #                         actor_output = self.actor_rollout_wg.update_actor(batch)
        #                     # ! END
        #                 metrics.update(seperate_record_dict)
        #                 actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
        #                 metrics.update(actor_output_metrics)
                    
        #             # * For plot_distribution Figure
        #             if self.config.algorithm.plot_dist and self.global_steps >= 1:
        #                 if self.config.algorithm.plot_fig_type == 'prob_histogram':
        #                     batch.batch['retemp_temperature'] = self.config.actor_rollout_ref.rollout.temperature * torch.ones_like(batch.batch['advantages'])
        #                     new_log_probs = self.actor_rollout_wg.compute_old_log_prob(batch)
        #                     plot_dict['logprobs_step1'] = new_log_probs.batch['old_log_probs'][plot_matching_indices, :]
        #                     plot_dict['advantages'] = batch.batch['advantages'][plot_matching_indices, 0]
        #                     plot_dir = os.path.join(self.config.trainer.default_local_dir, f'plot_distribution/step_{self.global_steps}')
        #                     # draw_distribution_fig(plot_dir, 1, self.tokenizer, **plot_dict)
        #                     draw_distribution_fig(plot_dir, 1, self.tokenizer, **plot_dict)
        #                 elif self.config.algorithm.plot_fig_type == 'prob_stat':
        #                     batch.batch['retemp_temperature'] = self.config.actor_rollout_ref.rollout.temperature * torch.ones_like(batch.batch['advantages'])
        #                     new_log_probs = self.actor_rollout_wg.compute_old_log_prob(batch)
        #                     plot_dict['logprobs_step1'] = new_log_probs.batch['old_log_probs']
        #                     plot_dict['attention_mask'] = batch.batch['attention_mask'][:, -response_length:]
        #                     plot_dict['advantages'] = batch.batch['old_advantages'] if 'old_advantages' in batch.batch else batch.batch['advantages']
        #                     plot_dir = os.path.join(self.config.trainer.default_local_dir, f'plot_prob_change/step_{self.global_steps}_adv')
        #                     draw_prob_change_adv(plot_dir, **plot_dict)
        #                     # plot_dir = os.path.join(self.config.trainer.default_local_dir, f'plot_prob_change/step_{self.global_steps}_pos')
        #                     # draw_prob_change_pos(plot_dir, **plot_dict)
        #                     plot_dict["grad_norm"] = metrics["actor/grad_norm"]
        #                     value_dir = os.path.join(self.config.trainer.default_local_dir, f'plot_prob_change/step_{self.global_steps}_dict.pth')
        #                     torch.save(plot_dict, value_dir)
        #                 else:
        #                     raise NotImplementedError(f'plot_fig_type {self.config.algorithm.plot_fig_type} is not supported.')
        #             # * For plot_distribution Figure END

        #             # reward
        #             reward_metrics = compute_reward_metrics(batch)
        #             metrics.update(reward_metrics)

        #             # validate
        #             if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
        #                 self.global_steps % self.config.trainer.test_freq == 0:
        #                 with _timer('testing', timing_raw):
        #                     val_metrics_testN = self._validate_TestN()
        #                     pprint(f'Steps:{self.global_steps} - validation metrics: {val_metrics_testN}')

        #                     val_metrics_testonce = self._validate_Testonce()
        #                     pprint(f'Steps:{self.global_steps} - validation metrics: {val_metrics_testonce}')
        #                 metrics.update(val_metrics_testN)
        #                 metrics.update(val_metrics_testonce)
                    
        #             # if self.val_reward_fn is not None and self.config.trainer.math_test_freq > 0 and \
        #             #     self.global_steps % self.config.trainer.math_test_freq == 0:
        #                 # amc_metrics: dict = self._test_amc()
        #                 # metrics.update(amc_metrics)
        #                 # aime_metrics: dict = self._test_aime()
        #                 # metrics.update(aime_metrics)

        #             if self.config.trainer.save_freq > 0 and \
        #                     self.global_steps % self.config.trainer.save_freq == 0:
        #                 with _timer('save_checkpoint', timing_raw):
        #                     self._save_checkpoint()


        #         # collect metrics
        #         metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        #         metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        #         # TODO: make a canonical logger that supports various backend
        #         logger.log(data=metrics, step=self.global_steps)

        #         self.global_steps += 1

        #         if self.config.trainer.total_steps > 0 and self.global_steps > self.config.trainer.total_steps:
        #             break

        #         if self.global_steps >= self.total_training_steps:
        #             # 在所有训练步骤结束时保存最后一个检查点
        #             with _timer('save_checkpoint', timing_raw):
        #                 self._save_checkpoint()
        #             # 执行验证
        #             if self.val_reward_fn is not None:
        #                 val_metrics_testN = self._validate_TestN()
        #                 pprint(f'Steps:{self.global_steps} - validation metrics: {val_metrics_testN}')
        #                 logger.log(data=val_metrics_testN, step=self.global_steps)

        #                 val_metrics_testonce = self._validate_Testonce()
        #                 pprint(f'Steps:{self.global_steps} - validation metrics: {val_metrics_testonce}')
        #                 logger.log(data=val_metrics_testonce, step=self.global_steps)
        #             return

    