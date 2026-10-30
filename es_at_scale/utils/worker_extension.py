import gc
import time
import torch
import random
import numpy as np


def _stateless_init_process_group(
    master_address, master_port, rank, world_size, device
):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, 
        port=master_port, 
        rank=rank, 
        world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    """

    def _set_seed(self, seed):
        # set a seed locally on the worker extension for reproducibility
        self.local_seed = seed

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_self_initial_weights(
        self,
    ):
        """Save a copy of itself in the CPU memory."""
        self.initial_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self.initial_weights[name] = p.detach().clone().cpu()
        print("Initial weights saved.")


    # ------------------------------------------------------------------------------------------------------------
    # IFS-ES options (inverse-metric-es fork). Defaults reproduce the upstream behaviour exactly.
    #   noise_iid      : seed each parameter tensor with seed + tensor index (upstream reuses one seed for all tensors)
    #   exact_restore  : restore = copy of the pre-perturbation weights (upstream subtracts the noise again in bf16)
    #   fp32_master    : engine 0 keeps fp32 master weights; updates are accumulated there (upstream adds in bf16)
    #   precond_mode   : 'none' | 'sample' (eps ~ N(0, c (F+lam I)^{2q}), placement A) | 'placeB' (isotropic eps,
    #                    estimate preconditioned by the same matrix); see es_at_scale/utils/kfac_precond.py
    # ------------------------------------------------------------------------------------------------------------
    def configure_es(self, opts):
        self.es_opts = dict(opts)
        self._base, self._master, self._fac, self._S = None, None, None, None
        info = {}
        mode = self.es_opts.get("precond_mode", "none")
        if mode != "none":
            from es_at_scale.utils.kfac_precond import load_factors, make_scales
            dev = next(self.model_runner.model.parameters()).device
            fac, meta = load_factors(self.es_opts["precond_path"], dev, torch.float32)
            names = {n for n, _ in self.model_runner.model.named_parameters()}
            missing = [n for n in fac if n not in names]
            assert not missing, ("factor names not found in vLLM model", missing[:5])
            shapes = {n: tuple(p.shape) for n, p in self.model_runner.model.named_parameters()}
            for n, d in fac.items():
                want = (sum(l.numel() for l in d["lG"]), sum(l.numel() for l in d["lA"]))
                assert shapes[n] == want, (n, shapes[n], want)
            self._S, info = make_scales(fac, float(self.es_opts["precond_q"]), float(self.es_opts["precond_gamma"]),
                                        self.es_opts.get("precond_norm", "kl"))
            self._fac = fac
            info["n_kfac_weights"] = len(fac)
            info["factor_meta"] = {k: v for k, v in meta.items() if isinstance(v, (int, float, str))}
        if self.es_opts.get("exact_restore"):
            self._base = {n: p.detach().clone() for n, p in self.model_runner.model.named_parameters()}
        torch.cuda.empty_cache()
        return info

    def _opt(self, key, default=None):
        return getattr(self, "es_opts", {}).get(key, default)

    def _draw(self, idx, p, seed):
        gen = torch.Generator(device=p.device)
        gen.manual_seed(int(seed) + (idx if self._opt("noise_iid") else 0))
        return torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)

    def _eps(self, idx, name, p, seed):
        """Perturbation direction for one tensor (p.dtype), shaped by K-FAC in 'sample' mode."""
        z = self._draw(idx, p, seed)
        if self._opt("precond_mode", "none") == "sample" and name in self._fac:
            from es_at_scale.utils.kfac_precond import sample_transform
            return sample_transform(self._fac[name], self._S[name], z.float()).to(p.dtype)
        return z

    def restore_self_weights(self, seed, sigma):
        if self._opt("exact_restore"):
            for name, p in self.model_runner.model.named_parameters():
                p.data.copy_(self._base[name])
            torch.cuda.synchronize()
            return True
        self._set_seed(seed)
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    
    def init_inter_engine_group(self, master_address: str, master_port: int,
                            engine_idx: int, num_engines: int):
            # TP rank within the engine (0..tp_size-1)
            tp_rank = getattr(self, "rank", None)
            tp_size = getattr(self, "world_size", None)

            if tp_rank is None or tp_size is None:
                tp_rank = getattr(self, "local_rank", 0)
                tp_size = getattr(getattr(self, "parallel_config", None), "tensor_parallel_size", None)
                if tp_size is None:
                    raise RuntimeError("Could not determine TP rank/size from vLLM worker attributes.")

            # IMPORTANT: create a group PER tp_rank across engines
            # So: rank is engine_idx, world_size is num_engines
            # Use a different port per tp_rank so each group has its own rendezvous
            port = int(master_port) + int(tp_rank)

            self.inter_pg = _stateless_init_process_group(
                master_address,
                port,
                rank=int(engine_idx),
                world_size=int(num_engines),
                device=self.device,
            )
            return True
    
    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(
                p, src=int(src_rank), stream=torch.cuda.current_stream()
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        """theta += alpha/N * sum_n coeff_n eps_n. In 'sample' mode eps_n = T(z_n) with T linear, so the sum is
        T(sum_n coeff_n z_n): one transform per tensor. In 'placeB' mode the isotropic estimate is preconditioned."""
        mode = self._opt("precond_mode", "none")
        fac, S = getattr(self, "_fac", None), getattr(self, "_S", None)
        if self._opt("fp32_master") and getattr(self, "_master", None) is None:
            self._master = {n: p.detach().float().clone() for n, p in self.model_runner.model.named_parameters()}
        master = getattr(self, "_master", None)
        for idx, (name, p) in enumerate(self.model_runner.model.named_parameters()):
            update_accumulator = torch.zeros_like(p.data, dtype=torch.float32)
            for i, seed in enumerate(seeds):
                noise = self._draw(idx, p, seed)
                update_accumulator.add_(noise.to(torch.float32) * coeffs[i])
                del noise
            if fac is not None and name in fac:
                from es_at_scale.utils.kfac_precond import sample_transform, precond_transform
                if mode == "sample":
                    update_accumulator = sample_transform(fac[name], S[name], update_accumulator)
                elif mode == "placeB":
                    update_accumulator = precond_transform(fac[name], S[name], update_accumulator)
            update_accumulator.mul_(float(alpha) / float(population_size))
            if master is not None:
                master[name].add_(update_accumulator)
                p.data.copy_(master[name].to(p.dtype))
            else:
                p.data.add_(update_accumulator.to(p.dtype))
            del update_accumulator
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        """
        Add noise(seed) scaled by sigma_or_scale * coeff (or subtract when negate=True).
        - For exploration:  perturb_self_weights(seed, SIGMA, 1.0, False)
          and restore with restore_self_weights(seed, SIGMA) as before.
        - For ES update:   perturb_self_weights(seed, 1.0, coeff, False)
          where coeff = ALPHA/POPULATION_SIZE * norm_reward.
        """
        self._set_seed(seed)
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0

        exact = self._opt("exact_restore")
        for idx, (name, p) in enumerate(self.model_runner.model.named_parameters()):
            if exact:
                self._base[name].copy_(p.data)
            noise = self._eps(idx, name, p, seed)
            p.data.add_(sign * scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if not self._opt("quiet"):
            print(f"Weights changed with: sign={sign}; scale={sign * scale}.")

    def save_self_weights_to_disk(self, filepath):
        """Save the current model weights to disk."""
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        print(f"Model weights saved to {filepath}.")

    def load_weights_from_disk(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(state_dict[name].to(self.device))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
