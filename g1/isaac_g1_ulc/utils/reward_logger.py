"""
Reward Breakdown Logger for Isaac Lab RL Training
==================================================
Drop-in utility for per-component reward logging.

Training mode: writes to self.extras → TensorBoard picks it up
Play mode: writes step-level CSV + console output for debugging

Usage in environment:
    self.reward_logger = RewardLogger(
        reward_names=["vx", "height", "orientation", ...],
        reward_weights={"vx": 3.0, "height": 3.0, ...},
        num_envs=self.num_envs,
        device=self.device,
    )

    # In _compute_rewards():
    self.reward_logger.record("vx", raw_vx_reward)
    self.reward_logger.record("height", raw_height_reward)
    total = self.reward_logger.compute_total()
    self.extras.update(self.reward_logger.get_extras())

Author: Turan (isaac-g1-ulc)
"""

import torch
import csv
import os
import json
from collections import OrderedDict
from typing import Optional, Dict, List


class RewardLogger:
    """Per-component reward breakdown logger."""

    def __init__(
        self,
        reward_names: List[str],
        reward_weights: Dict[str, float],
        num_envs: int,
        device: str = "cuda:0",
        play_mode: bool = False,
        log_dir: Optional[str] = None,
        console_every: int = 10,
    ):
        self.reward_names = reward_names
        self.reward_weights = reward_weights
        self.num_envs = num_envs
        self.device = device
        self.play_mode = play_mode
        self.console_every = console_every

        # Component buffers
        self.raw = OrderedDict()
        self.weighted = OrderedDict()

        # Step counter
        self._step = 0

        # CSV (play mode)
        self._csv_writer = None
        self._csv_file = None
        self._json_log = []

        if play_mode and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            # CSV for spreadsheet analysis
            csv_path = os.path.join(log_dir, "reward_breakdown.csv")
            self._csv_file = open(csv_path, "w", newline="")
            header = (
                ["step", "total_reward"]
                + [f"raw_{n}" for n in reward_names]
                + [f"wtd_{n}" for n in reward_names]
                + [f"pct_{n}" for n in reward_names]
            )
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(header)

            # JSON for visualization tool
            self._json_path = os.path.join(log_dir, "reward_breakdown.json")

            print(f"[RewardLogger] Logging to: {log_dir}")

    def record(self, name: str, raw_value: torch.Tensor):
        """Record a single reward component (unweighted, shape=[num_envs])."""
        self.raw[name] = raw_value.detach()
        w = self.reward_weights.get(name, 1.0)
        self.weighted[name] = raw_value.detach() * w

    def compute_total(self) -> torch.Tensor:
        """Sum all weighted components -> total reward tensor."""
        total = torch.zeros(self.num_envs, device=self.device)
        for name in self.reward_names:
            if name in self.weighted:
                total += self.weighted[name]
        return total

    def get_extras(self) -> dict:
        """
        Build TensorBoard extras dict with 3 views:
          RR/ = Raw Reward (unweighted)
          RW/ = Reward Weighted
          RB/ = Reward Budget (% of total absolute reward)
        """
        extras = {}

        total_abs = sum(
            self.weighted[n].abs().mean().item()
            for n in self.reward_names if n in self.weighted
        )
        total_abs = max(total_abs, 1e-8)

        total_val = sum(
            self.weighted[n].mean().item()
            for n in self.reward_names if n in self.weighted
        )
        extras["RW/total"] = total_val

        for name in self.reward_names:
            if name not in self.weighted:
                continue
            raw_m = self.raw[name].mean().item()
            wtd_m = self.weighted[name].mean().item()
            pct = abs(wtd_m) / total_abs * 100.0

            extras[f"RR/{name}"] = raw_m
            extras[f"RW/{name}"] = wtd_m
            extras[f"RB/{name}"] = pct

        return extras

    def step(self, env_id: int = 0):
        """Call once per env step in play mode."""
        self._step += 1
        if not self.play_mode:
            return

        self._log_csv(env_id)
        self._log_json(env_id)

        if self._step % self.console_every == 0:
            self.print_breakdown(env_id)

    def _log_csv(self, env_id: int):
        if self._csv_writer is None:
            return

        total = sum(
            self.weighted[n][env_id].item()
            for n in self.reward_names if n in self.weighted
        )
        total_abs = sum(
            abs(self.weighted[n][env_id].item())
            for n in self.reward_names if n in self.weighted
        )
        total_abs = max(total_abs, 1e-8)

        row = [self._step, f"{total:.4f}"]
        for name in self.reward_names:
            v = self.raw[name][env_id].item() if name in self.raw else 0.0
            row.append(f"{v:.6f}")
        for name in self.reward_names:
            v = self.weighted[name][env_id].item() if name in self.weighted else 0.0
            row.append(f"{v:.6f}")
        for name in self.reward_names:
            v = self.weighted[name][env_id].item() if name in self.weighted else 0.0
            row.append(f"{abs(v)/total_abs*100:.1f}")
        self._csv_writer.writerow(row)

    def _log_json(self, env_id: int):
        total = sum(
            self.weighted[n][env_id].item()
            for n in self.reward_names if n in self.weighted
        )
        total_abs = sum(
            abs(self.weighted[n][env_id].item())
            for n in self.reward_names if n in self.weighted
        )
        total_abs = max(total_abs, 1e-8)

        entry = {"step": self._step, "total": round(total, 4), "components": {}}
        for name in self.reward_names:
            if name not in self.weighted:
                continue
            entry["components"][name] = {
                "raw": round(self.raw[name][env_id].item(), 6),
                "weighted": round(self.weighted[name][env_id].item(), 6),
                "pct": round(abs(self.weighted[name][env_id].item()) / total_abs * 100, 1),
            }
        self._json_log.append(entry)

    def print_breakdown(self, env_id: int = 0):
        """Console-friendly reward breakdown table."""
        total = sum(
            self.weighted[n][env_id].item()
            for n in self.reward_names if n in self.weighted
        )
        total_abs = sum(
            abs(self.weighted[n][env_id].item())
            for n in self.reward_names if n in self.weighted
        )
        total_abs = max(total_abs, 1e-8)

        print(f"\n{'='*65}")
        print(f"  Step {self._step:5d} | Total: {total:+.4f}")
        print(f"{'='*65}")
        print(f"  {'Component':<18} {'Raw':>9} {'xW':>5} {'Weighted':>10} {'Budget':>7}")
        print(f"  {'-'*52}")

        sorted_components = sorted(
            [(n, self.weighted[n][env_id].item()) for n in self.reward_names if n in self.weighted],
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        for name, wtd_val in sorted_components:
            raw_val = self.raw[name][env_id].item()
            weight = self.reward_weights.get(name, 1.0)
            pct = abs(wtd_val) / total_abs * 100.0

            # Flag: dominant (>>), notable (>), or dead (!)
            if pct > 30:
                flag = ">>"
            elif pct > 15:
                flag = "> "
            elif pct < 1.0:
                flag = "!!"  # DEAD reward -- policy ignores this
            else:
                flag = "  "

            print(f"  {flag} {name:<16} {raw_val:+.5f} {weight:>4.1f}x {wtd_val:+.5f}  {pct:5.1f}%")

        print(f"{'='*65}")

    def close(self):
        """Flush and close files."""
        if self._csv_file:
            self._csv_file.close()
        if self._json_log and hasattr(self, '_json_path'):
            with open(self._json_path, "w") as f:
                json.dump({
                    "reward_names": self.reward_names,
                    "reward_weights": self.reward_weights,
                    "steps": self._json_log,
                }, f)
            print(f"[RewardLogger] JSON saved: {self._json_path}")
