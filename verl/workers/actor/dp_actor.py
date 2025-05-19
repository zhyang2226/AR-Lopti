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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # ! add rollout temperature here
                if torch.is_tensor(temperature):
                    # print(f"rollout temperature is a tensor, now we reformulate it: {temperature.shape}, the input_ids shape: {input_ids.shape}")
                    if temperature.size(1) == 1:
                        temperature = temperature.expand(-1, input_ids.size(1))
                    else:
                        padding = temperature[:, -1].unsqueeze(1).expand(-1, input_ids.size(1) - temperature.size(1))
                        temperature = torch.cat([padding, temperature], dim=1)
                    # print(f"rollout temperature is a tensor, after reformulation: {temperature.shape}, the input_ids shape: {input_ids.shape}")
                    temperature = index_first_axis(rearrange(temperature.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                    indices).transpose(0, 1)
                # ! end rollout temperature

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)
                    # ! for custom_defined temperature
                    if torch.is_tensor(temperature):
                        temperature, _, _ = ulysses_pad_and_slice_inputs(temperature, None,
                                                                        self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                # ! for custom_defined temperature
                if torch.is_tensor(temperature):
                    temperature = temperature.transpose(0, 1)  # (total_nnz, vocab_size)
                    # print(f"logits_rmpad: {logits_rmpad.shape} and temperature: {temperature.shape}")
                logits_rmpad.div_(temperature)
                # if torch.is_tensor(temperature):
                #     print(f"logits_rmpad after div: {logits_rmpad.shape}")

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                # TODO: CHECK if this need to be changed for custom_defined temperature
                assert not torch.is_tensor(temperature), "temperature must be a float when not using rmpad"

                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        # ! original temperature is defined here
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        if 'retemp_temperature' in data.batch:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'retemp_temperature']
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                # ! re-define temperature here
                if 'retemp_temperature' in micro_batch:
                    # print(f"retemp_temperature size: {micro_batch['retemp_temperature'].shape}")
                    _, log_probs = self._forward_micro_batch(micro_batch, temperature=micro_batch['retemp_temperature'])
                else:
                    _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs
    
    def compute_entropy(self, data: DataProto) -> torch.Tensor:
        """Compute the entropy of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the entropy tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        # ! original temperature is defined here
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        if 'retemp_temperature' in data.batch:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'retemp_temperature']
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        entropies_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                # ! re-define temperature here
                if 'retemp_temperature' in micro_batch:
                    # print(f"retemp_temperature size: {micro_batch['retemp_temperature'].shape}")
                    entropy, _ = self._forward_micro_batch(micro_batch, temperature=micro_batch['retemp_temperature'])
                else:
                    entropy, _ = self._forward_micro_batch(micro_batch, temperature=temperature)
            entropies_lst.append(entropy)
        entropies = torch.concat(entropies_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == entropies.size(0), f"{len(indices)} vs. {entropies.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            entropies = entropies[revert_indices]

        return entropies

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        
        # ! original temperature is defined here
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        if 'retemp_temperature' in data.batch:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'retemp_temperature']
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']

        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if 'kl_entropy_mask' in data.batch:
            select_keys.append('kl_entropy_mask')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                # ! re-define temperature here
                if 'retemp_temperature' in data:
                    # print(f"retemp_temperature: {data['retemp_temperature']}")
                    temperature = data['retemp_temperature']

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                loss_agg_mode = self.config.loss_agg_mode
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                if self.config.importance_sampling == "on":
                    pass
                    # print("Using importance sampling for all samples")
                
                elif self.config.importance_sampling == "off_for_pos":
                    print("Using importance sampling only for negative samples")
                    # Only use importance sampling for negative samples
                    # Set old_log_prob to log_prob where advantages > 0
                    old_log_prob = torch.where(advantages > 0, log_prob.detach(), old_log_prob.detach())

                elif self.config.importance_sampling == "off":
                    print("Not using importance sampling, assign log_prob to old_log_prob")
                    old_log_prob = log_prob.detach().clone()
                else:
                    raise ValueError(f"Unknown importance_sampling mode: {self.config.importance_sampling}. "
                                        "It should be one of ['on', 'off', 'off_for_pos'].")

                # pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                #                                                               log_prob=log_prob,
                #                                                               advantages=advantages,
                #                                                               eos_mask=response_mask,
                #                                                               cliprange_low=clip_ratio_low,
                #                                                               cliprange_high=clip_ratio_high)

                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss(
                                                                                old_log_prob=old_log_prob,
                                                                                log_prob=log_prob,
                                                                                advantages=advantages,
                                                                                response_mask=response_mask,
                                                                                cliprange=clip_ratio,
                                                                                cliprange_low=clip_ratio_low,
                                                                                cliprange_high=clip_ratio_high,
                                                                                clip_ratio_c=clip_ratio_c)
            
                # compute entropy loss from entropy
                if 'kl_entropy_mask' in data:
                    entropy = entropy * data['kl_entropy_mask'][:, -response_length:]
                # entropy_loss = verl_F.masked_mean(entropy, response_mask)
                entropy_loss = core_algos.agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    if 'kl_entropy_mask' in data:
                        kld = kld * data['kl_entropy_mask'][:, -response_length:]
                    # kl_loss = masked_mean(kld, response_mask)
                    kl_loss = core_algos.agg_loss(loss_mat=kld,
                                                  loss_mask=response_mask,
                                                  loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                    'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                }
                append_to_dict(metrics, data)

            print(f"Batch {batch_idx}: Micro batch size = {responses.size(0)} and gradient_accumulation = {self.gradient_accumulation}")
            print(metrics)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
