# raw_adv.py
import torch
from typing import Optional
from verl.trainer.ppo.core_algos import register_adv_est

# ---------- compute advantage ----------

@register_adv_est("raw_reward")   # 在 yaml 里会用到这个名字
def compute_raw_reward_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[object] = None,
    **kwargs,
):
    """
    目标：advantages = reward（不减均值/不除std/不白化）
    token_level_rewards: (bs, resp_len)
      - outcome reward 通常只在最后一个 token 非零；sum 后就是标量 reward
    response_mask: (bs, resp_len)
    """
    resp_len = token_level_rewards.shape[-1]

    # 标量 reward per sample
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # broadcast 成 token-level advantages
    advantages = scores.unsqueeze(-1).expand(-1, resp_len) * response_mask

    # returns 在 outcome reward 里无所谓，直接同 advantages
    returns = advantages
    return advantages, returns
