import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import torch
import numpy as np
import textwrap
import os

def wrap_text(text, width):
    return '\n'.join(textwrap.wrap(text, width))

def find_first_eos_position(array, eos_token_id):
    eos_positions = np.where(array == eos_token_id)[0]
    if len(eos_positions) > 0:
        return eos_positions[0]
    else:
        return len(array)

def find_max_length(response_list, eos_token_id):
    positions = [find_first_eos_position(array, eos_token_id) for array in response_list]
    return max(positions)

# * This function draw two column figures,
# * Left column is the logprob of the responses,
# * Right column is the entropies of the responses.
# * It only draws the one with top2 advantages and lowest2 advantages.
def draw_distribution_fig(fig_plot_name, noptepochs, tokenizer, **kwargs):
    # first transfer all tensor to numpy
    for key in kwargs:
        if isinstance(kwargs[key], torch.Tensor):
            if kwargs[key].dtype == torch.bfloat16:
                kwargs[key] = kwargs[key].to(torch.float32).cpu().numpy()
            else:
                kwargs[key] = kwargs[key].cpu().numpy()

    if 'entropies' in kwargs:
        fig = plt.figure(figsize=(50, 32))
        gs = GridSpec(8, 6, figure=fig)
        # ax_image = fig.add_subplot(gs[:, :2])
        axes_bars = [fig.add_subplot(gs[i, :]) for i in range(8)]
    else:
        fig = plt.figure(figsize=(50, 16))
        gs = GridSpec(4, 6, figure=fig)
        # ax_image = fig.add_subplot(gs[:, :2])
        axes_bars = [fig.add_subplot(gs[i, :]) for i in range(4)]

    # # * Part 1: Draw the image (there is no image for LLM)
    # img = kwargs['images'][0].transpose(1, 2, 0)
    # img = (img - img.min()) / (img.max() - img.min())
    # ax_image.imshow(img)
    # ax_image.axis('off')
    # ax_image.set_title('Radiology Image', fontsize=16)

    # * Part 2: Draw the logprobs and entropies
    # ~ Step 2-1: Select the top2 and lowest2 advantages
    top2_idx = np.argsort(kwargs['advantages'])[-2:][::-1]
    top2_value = kwargs['advantages'][top2_idx]
    lowest2_idx = np.argsort(kwargs['advantages'])[:2][::-1]
    lowest2_value = kwargs['advantages'][lowest2_idx]
    plot_idx = np.concatenate([top2_idx, lowest2_idx], axis=0)
    plot_value = np.concatenate([top2_value, lowest2_value], axis=0)

    # ~ Step 2-2: Extract the corresponding probs and entropies
    responses = kwargs['responses'][plot_idx]
    plot_max_x = find_max_length([item for item in responses], tokenizer.eos_token_id)
    if plot_max_x > 256:
        plot_max_x = 256

    plot_content_dict = {}
    plot_content_dict['ref_probs'] = np.exp(kwargs['ref_logprobs'][plot_idx, :plot_max_x])
    plot_content_dict['probs'] = np.exp(kwargs['logprobs'][plot_idx, :plot_max_x])
    if 'entropies' in kwargs:
        plot_content_dict['ref_entropies'] = kwargs['ref_entropies'][plot_idx, :plot_max_x]
        plot_content_dict['entropies'] = kwargs['entropies'][plot_idx, :plot_max_x]
    for index in range(noptepochs):
        plot_content_dict[f'probs_step{index+1}'] = np.exp(kwargs[f'logprobs_step{index+1}'][plot_idx, :plot_max_x])
        if 'entropies' in kwargs:
            plot_content_dict[f'entropies_step{index+1}'] = kwargs[f'entropies_step{index+1}'][plot_idx, :plot_max_x]
    probs_cmap = cm.get_cmap('Blues', 3+noptepochs)
    probs_colors = [probs_cmap(i+1) for i in range(2+noptepochs)]
    if 'entropies' in kwargs:
        entropies_cmap = cm.get_cmap('Oranges', 3+noptepochs)
        entropies_colors = [entropies_cmap(i+1) for i in range(2+noptepochs)]

    if 'entropies' in kwargs:
        dominator = 2
    else:
        dominator = 1
    for i, ax in enumerate(axes_bars):
        plot_index = i // dominator
        if i % dominator == 0:  # Plot Probs
            bars = []
            report = tokenizer.decode(responses[plot_index], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            report = wrap_text(report, width=550)
            bars.append(ax.bar(np.arange(0, plot_max_x) * (2+noptepochs) + 0, plot_content_dict['ref_probs'][plot_index], color=probs_colors[0]))
            bars.append(ax.bar(np.arange(0, plot_max_x) * (2+noptepochs) + 1, plot_content_dict['probs'][plot_index], color=probs_colors[1]))
            for index in range(noptepochs):
                bars.append(ax.bar(np.arange(0, plot_max_x) * (2+noptepochs) + index+2, plot_content_dict[f'probs_step{index+1}'][plot_index], color=probs_colors[index+2]))
            ax.set_title(f'Probs of the {i//2+1}th Report (Advantage{plot_value[plot_index]:.2}) \n{report}')
            ax.set_yticks([0.0, 1.12])
            ax.set_xticks(np.arange(0, plot_max_x) * (2+noptepochs) + 0.5)
            tokens = [tokenizer.decode([idx]) for idx in responses[plot_index, :plot_max_x]]
            ax.set_xticklabels(tokens, rotation=90)

        else: 
            bars = []
            report = tokenizer.decode(responses[plot_index], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            report = wrap_text(report, width=550)
            bars.append(ax.bar(np.arange(0, plot_max_x) * (2+noptepochs) + 0, plot_content_dict['ref_entropies'][plot_index], color=entropies_colors[0]))
            bars.append(ax.bar(np.arange(0, plot_max_x) * (2+noptepochs) + 1, plot_content_dict['entropies'][plot_index], color=entropies_colors[1]))
            for index in range(noptepochs):
                bars.append(ax.bar(np.arange(0, plot_max_x) * (2+noptepochs) + index+2, plot_content_dict[f'entropies_step{index+1}'][plot_index], color=entropies_colors[index+2]))
            ax.set_title(f'Entropies of the {i//2+1}th Report (Advantage{plot_value[plot_index]:.2}) \n{report}')
            ax.set_yticks([0.0, plot_content_dict['ref_entropies'][plot_index].max() + 0.2])
            ax.set_xticks(np.arange(0, plot_max_x) * (2+noptepochs) + 0.5)
            tokens = [tokenizer.decode([idx]) for idx in responses[plot_index, :plot_max_x]]
            ax.set_xticklabels(tokens, rotation=90)

    # * Part 3: Save the figure
    fig_plot_dir = os.path.dirname(fig_plot_name)
    if not os.path.exists(fig_plot_dir):
        os.makedirs(fig_plot_dir, exist_ok=True)
    
    prompt = tokenizer.decode(kwargs['prompts'][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    prompt = wrap_text(prompt, width=550)
    fig.suptitle(f"Prompt: {prompt}")

    plt.tight_layout()
    fig.savefig(f"{fig_plot_name}.jpg", dpi=200)
    plt.close(fig)


def draw_prob_change_adv(fig_plot_name, **kwargs):
    # first transfer all tensor to numpy
    for key in kwargs:
        if isinstance(kwargs[key], torch.Tensor):
            if kwargs[key].dtype == torch.bfloat16:
                kwargs[key] = kwargs[key].to(torch.float32).cpu().numpy()
            else:
                kwargs[key] = kwargs[key].cpu().numpy()

    advantage_interval = [-10, -2, -1, -0.5, 0, 0.5, 1, 2, 10]
    prob_interval = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    try:
        advantage = kwargs['advantages']
        original_prob = np.exp(kwargs['logprobs'])
        prob_change = np.exp(kwargs['logprobs_step1']) - np.exp(kwargs['logprobs'])
    except KeyError as e:
        raise KeyError(f"Missing key in kwargs: {e}")
    plot_num = len(advantage_interval) - 1

    # we do not consider entropy for now
    fig = plt.figure(figsize=(32, 24))
    gs = GridSpec(3, 4, figure=fig)
    axes_bars = [fig.add_subplot(gs[i//4, i%4]) for i in range(plot_num)]
    axes_bars.append(fig.add_subplot(gs[2, :2]))
    axes_bars.append(fig.add_subplot(gs[2, 2:]))

    # * Part 1: Draw the Prob-change according to the Advantage
    for i in range(plot_num):
        indices = np.where((advantage >= advantage_interval[i]) & (advantage < advantage_interval[i+1]))
        if len(indices[0]) == 0:
            continue

        prob_changes = prob_change[indices[0], indices[1]]
        original_probs = original_prob[indices[0], indices[1]]

        prob_change_stats = []
        prob_count = []
        for j in range(len(prob_interval) - 1):
            prob_indices = np.where((original_probs > prob_interval[j]) & (original_probs <= prob_interval[j+1]))[0]
            if len(prob_indices) > 0:
                prob_change_stats.append(prob_changes[prob_indices])
                prob_count.append(len(prob_indices))
            else:
                prob_change_stats.append(np.array([0.]))
                prob_count.append(0)
        axes_bars[i].axhline(y=0, color='red', linestyle='--', linewidth=1, zorder=0)

        # ! BOX PLOT
        # axes_bars[i].boxplot(prob_change_stats, labels=[f'{prob_interval[j]}-{prob_interval[j+1]}\nTotal{prob_count[j]}' for j in range(len(prob_interval) - 1)], showmeans=True)
        # ! VIOLIN PLOT
        violin_parts = axes_bars[i].violinplot(prob_change_stats, showmeans=True, showmedians=True)
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_edgecolor('blue')
            pc.set_alpha(0.7)
        violin_parts['cmeans'].set_color('red')
        violin_parts['cmedians'].set_color('black')
        axes_bars[i].set_xticks(range(1, len(prob_interval)))
        axes_bars[i].set_xticklabels([f'{prob_interval[j]}-{prob_interval[j+1]}\nTotal{prob_count[j]}' for j in range(len(prob_interval) - 1)])
        # ! PLOT END

        axes_bars[i].set_title(f'Advantage Interval: {advantage_interval[i]} to {advantage_interval[i+1]}')
        axes_bars[i].set_ylim(-0.3, 0.3)
        axes_bars[i].set_xlabel('Original Probability Interval')
        axes_bars[i].set_ylabel('Probability Change')

    # * Part 2: Draw the Probability Change For all Advantage
    indices = np.where(advantage != 0)
    prob_changes = prob_change[indices[0], indices[1]]
    original_probs = original_prob[indices[0], indices[1]]

    prob_change_stats = []
    prob_count = []
    for j in range(len(prob_interval) - 1):
        prob_indices = np.where((original_probs > prob_interval[j]) & (original_probs <= prob_interval[j+1]))[0]
        if len(prob_indices) > 0:
            prob_change_stats.append(prob_changes[prob_indices])
            prob_count.append(len(prob_indices))
    if len(prob_change_stats) != 0:
        axes_bars[plot_num].axhline(y=0, color='red', linestyle='--', linewidth=1, zorder=0)
        # ! BOX PLOT
        # axes_bars[plot_num].boxplot(prob_change_stats, labels=[f'{prob_interval[j]}-{prob_interval[j+1]}\nTotal{prob_count[j]}' for j in range(len(prob_interval) - 1)], showmeans=True)
        # ! VIOLIN PLOT
        violin_parts = axes_bars[plot_num].violinplot(prob_change_stats, showmeans=True, showmedians=True)
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_edgecolor('blue')
            pc.set_alpha(0.7)
        violin_parts['cmeans'].set_color('red')
        violin_parts['cmedians'].set_color('black')
        axes_bars[plot_num].set_xticks(range(1, len(prob_interval)))
        axes_bars[plot_num].set_xticklabels([f'{prob_interval[j]}-{prob_interval[j+1]}\nTotal{prob_count[j]}' for j in range(len(prob_interval) - 1)])
        # ! PLOT END
        
        axes_bars[plot_num].set_title(f'All Advantage')
        axes_bars[plot_num].set_ylim(-0.3, 0.3)
        axes_bars[plot_num].set_xlabel('Original Probability Interval')
        axes_bars[plot_num].set_ylabel('Probability Change')

    # * Part 3: Draw the Probability Distribution Histogram
    axes_bars[plot_num+1].hist(original_prob.flatten(), bins=10, color='blue', alpha=0.7)
    axes_bars[plot_num+1].set_title('Probability Distribution Histogram')
    axes_bars[plot_num+1].set_xlabel('Probability')
    axes_bars[plot_num+1].set_ylabel('Frequency')
    axes_bars[plot_num+1].set_xticks(prob_interval)

    # * Part 4: Save the figure
    fig_plot_dir = os.path.dirname(fig_plot_name)
    if not os.path.exists(fig_plot_dir):
        os.makedirs(fig_plot_dir, exist_ok=True)

    plt.tight_layout()
    fig.savefig(f"{fig_plot_name}.jpg", dpi=200)
    plt.close(fig)


def draw_prob_change_pos(fig_plot_name, **kwargs):
    # First, transfer all tensors to numpy
    for key in kwargs:
        if isinstance(kwargs[key], torch.Tensor):
            if kwargs[key].dtype == torch.bfloat16:
                kwargs[key] = kwargs[key].to(torch.float32).cpu().numpy()
            else:
                kwargs[key] = kwargs[key].cpu().numpy()

    position_interval = [0, 16, 32, 64, 128, 256, 512, 1024, 2048]
    prob_interval = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    try:
        advantage = kwargs['advantages']
        original_prob = np.exp(kwargs['logprobs'])
        prob_change = np.exp(kwargs['logprobs_step1']) - np.exp(kwargs['logprobs'])
    except KeyError as e:
        raise KeyError(f"Missing key in kwargs: {e}")
    
    # Generate token positions using np.linspace
    # Assuming batch_size and seq_len are the dimensions of logprobs
    batch_size, seq_len = kwargs['logprobs'].shape
    positions = np.linspace(0, seq_len - 1, seq_len, dtype=np.int32)  # Token positions for one sequence
    positions = np.tile(positions, (batch_size, 1))  # Repeat for all sequences in the batch

    plot_num = len(position_interval) - 1

    # Create the figure and subplots
    fig = plt.figure(figsize=(32, 32))  # Adjusted figure size to accommodate more subplots
    gs = GridSpec(6, 4, figure=fig)  # Each subplot now needs two rows
    axes_bars = [fig.add_subplot(gs[i % 4, i // 4]) for i in range(plot_num * 2)]

    # * Part 1: Draw the Prob-change according to the Position Interval
    for i in range(plot_num):
        indices = np.where((positions >= position_interval[i]) & (positions < position_interval[i + 1]))
        if len(indices[0]) == 0:
            continue

        # Split into positive and negative advantage
        pos_indices = np.where(advantage[indices[0], indices[1]] > 0)
        neg_indices = np.where(advantage[indices[0], indices[1]] < 0)

        for idx_type, adv_indices, ax in zip(
            ['Positive Advantage', 'Negative Advantage'],
            [pos_indices, neg_indices],
            [axes_bars[2 * i], axes_bars[2 * i + 1]]
        ):
            if len(adv_indices[0]) == 0:
                continue

            prob_changes = prob_change[indices[0], indices[1]][adv_indices[0]]
            original_probs = original_prob[indices[0], indices[1]][adv_indices[0]]

            prob_change_stats = []
            prob_count = []
            for j in range(len(prob_interval) - 1):
                prob_indices = np.where((original_probs > prob_interval[j]) & (original_probs <= prob_interval[j + 1]))[0]
                if len(prob_indices) > 0:
                    prob_change_stats.append(prob_changes[prob_indices])
                    prob_count.append(len(prob_indices))
                else:
                    prob_change_stats.append(np.array([0.]))
                    prob_count.append(0)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, zorder=0)
            ax.boxplot(
                prob_change_stats,
                labels=[f'{prob_interval[j]}-{prob_interval[j + 1]}\nTotal{prob_count[j]}' for j in range(len(prob_interval) - 1)],
                showmeans=True
            )
            ax.set_title(f'Position {position_interval[i]}-{position_interval[i+1]}: {idx_type}')
            ax.set_ylim(-0.3, 0.3)
            ax.set_xlabel('Original Probability Interval')
            ax.set_ylabel('Probability Change')

    # * Part 2: Draw the Probability Change for All Positions
    indices = np.where(positions >= 0)  # Include all valid positions
    prob_changes = prob_change[indices[0], indices[1]]
    original_probs = original_prob[indices[0], indices[1]]

    for idx_type, adv_filter, ax in zip(
        ['Positive Advantage', 'Negative Advantage'],
        [advantage > 0, advantage < 0],
        [fig.add_subplot(gs[4, :2]), fig.add_subplot(gs[5, :2])]
    ):
        adv_indices = np.where(adv_filter)
        if len(adv_indices[0]) == 0:
            continue

        prob_changes = prob_change[adv_indices[0], adv_indices[1]]
        original_probs = original_prob[adv_indices[0], adv_indices[1]]

        prob_change_stats = []
        prob_count = []
        for j in range(len(prob_interval) - 1):
            prob_indices = np.where((original_probs > prob_interval[j]) & (original_probs <= prob_interval[j + 1]))[0]
            if len(prob_indices) > 0:
                prob_change_stats.append(prob_changes[prob_indices])
                prob_count.append(len(prob_indices))
        if len(prob_change_stats) != 0:
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, zorder=0)
            ax.boxplot(
                prob_change_stats,
                labels=[f'{prob_interval[j]}-{prob_interval[j + 1]}\nTotal{prob_count[j]}' for j in range(len(prob_interval) - 1)],
                showmeans=True
            )
            ax.set_title(f'All Positions: {idx_type}')
            ax.set_ylim(-0.3, 0.3)
            ax.set_xlabel('Original Probability Interval')
            ax.set_ylabel('Probability Change')

    # * Part 3: Draw the Probability Distribution Histogram
    ax_hist = fig.add_subplot(gs[4:, 2:])
    ax_hist.hist(original_prob.flatten(), bins=10, color='blue', alpha=0.7)
    ax_hist.set_title('Probability Distribution Histogram')
    ax_hist.set_xlabel('Probability')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_xticks(prob_interval)

    # * Part 4: Save the figure
    fig_plot_dir = os.path.dirname(fig_plot_name)
    if not os.path.exists(fig_plot_dir):
        os.makedirs(fig_plot_dir, exist_ok=True)

    plt.tight_layout()
    fig.savefig(f"{fig_plot_name}.jpg", dpi=200)
    plt.close(fig)