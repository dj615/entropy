# dynamic_beta_kl.py
import numpy as np
import torch
from collections import defaultdict
from typing import Sequence, Optional


class DynamicBetaKLController:
    """
    Implements the Dynamic β-Schedule in your paragraph.

    For each prompt-level group of G rollouts:
      r_i: scalar reward for rollout i  (pre-KL scores)
      g_i = r_i - mean_group(r)
      p   = (1/G) * sum 1[g_i > 0]
      R_i = relu(g_i) / sum relu(g_j)   (if positive mass exists)
      n_eff = 1 / sum R_i^2
      \tilde n = n_eff / G
      s_p = 4 p (1-p)
      s = s_p * \tilde n
      beta_t = beta_max - (beta_max - beta_min) * s
    """

    def __init__(
        self,
        beta_min: float,
        beta_max: float,
        eps: float = 1e-8,
        ema_alpha: Optional[float] = None,
    ):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.ema_alpha = ema_alpha  # if not None, do EMA smoothing
        self.value = self.beta_max  # start conservative

    @torch.no_grad()
    def update_with_stats(
        self,
        *,
        rewards: torch.Tensor,
        uids: Sequence,
    ):
        """
        Args:
            rewards: (bs,) scalar pre-KL rewards for each rollout.
                     In verl this can be token_level_scores.sum(-1).
            uids:    length bs, prompt-group id for each rollout.
                     Same uid => same prompt group (G rollouts).

        Sets self.value to beta_t (optionally EMA-smoothed).
        """
        if rewards.numel() == 0:
            return

        r = rewards.detach().float().cpu().numpy()
        uids_list = list(uids)

        # group indices by uid
        groups = defaultdict(list)
        for i, uid in enumerate(uids_list):
            groups[uid].append(i)

        s_list = []
        for uid, idxs in groups.items():
            rr = r[idxs]
            G = len(rr)
            if G == 0:
                continue

            mean_r = rr.mean()
            g = rr - mean_r                 # g_i
            p = (g > 0).mean()              # fraction above baseline

            pos = np.clip(g, 0.0, None)     # relu(g_i)
            pos_sum = pos.sum()

            if pos_sum <= self.eps:
                # no positive mass => unreliable
                tilde_n = 0.0
            else:
                R = pos / pos_sum
                n_eff = 1.0 / np.sum(R * R)     # 1 / sum R_i^2
                tilde_n = n_eff / G             # normalized eff. sample size in (0,1]

            s_p = 4.0 * p * (1.0 - p)       # bell-shaped in [0,1]
            s = s_p * tilde_n               # reliability score in [0,1]
            s_list.append(s)

        if len(s_list) == 0:
            s_batch = 0.0
        else:
            s_batch = float(np.mean(s_list))

        beta_t = self.beta_max - (self.beta_max - self.beta_min) * s_batch
        beta_t = float(np.clip(beta_t, self.beta_min, self.beta_max))

        if self.ema_alpha is not None:
            a = float(self.ema_alpha)
            self.value = a * self.value + (1.0 - a) * beta_t
        else:
            self.value = beta_t
