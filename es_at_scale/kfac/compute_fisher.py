"""True-Fisher K-FAC factors of a causal LM on Countdown prompts, saved under vLLM (fused) parameter names.

Protocol (Phase 3b): 500 task prompts = the 200 Countdown train prompts + 300 freshly generated Countdown problems
(same template; none equal to an eval prompt). One response per prompt is SAMPLED from the model itself
(temperature 1, top_p 1, top_k 0, max_new_tokens 512) so the Fisher is the true Fisher, not the empirical one.
For each sequence, L = sum over response tokens of log p(y_t | y_<t, x); hooks on every nn.Linear in the decoder
layers record the input a_t and the output gradient g_t = dL/ds_t at every position. K-FAC "expand":
    A = mean over all tokens of a a^T,   G = mean over sequences of sum_t g_t g_t^T,   F_W ~ A (x) G,
both accumulated block-diagonally (block <= --block) in fp32. Fused vLLM weights: qkv_proj gets the shared A and
G = blockdiag(G_q, G_k, G_v); gate_up_proj gets the shared A and G = blockdiag(G_gate, G_up).
Output: torch file {'factors': {vllm_name: {QA, lA, QG, lG}}, 'meta': ...} and a JSON spectrum summary.
Usage: python -m es_at_scale.kfac.compute_fisher --model Qwen/Qwen2.5-0.5B-Instruct --out PATH.pt
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPLATE = ("You are a helpful assistant. You first think about the reasoning process in your mind and then provide the "
            "user with the answer.Using the numbers [{nums}], create an equation that equals {target}. You can use basic "
            "arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> "
            "tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.Let "
            "me solve this step by step.\n<think>")


def gen_problems(n, seed, exclude):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        k = rng.choice([3, 4])
        nums = [rng.randint(1, 99) for _ in range(k)]
        vals = nums[:]
        rng.shuffle(vals)
        acc = vals[0]
        for v in vals[1:]:
            op = rng.choice('+-*/')
            if op == '/' and (v == 0 or acc % v):
                op = '+'
            acc = acc + v if op == '+' else acc - v if op == '-' else acc * v if op == '*' else acc // v
        if not (1 <= acc <= 200):
            continue
        ctx = TEMPLATE.format(nums=' '.join(map(str, nums)), target=acc)
        if ctx in exclude:
            continue
        exclude.add(ctx)
        out.append(ctx)
    return out


def blocks(n, b):
    k = math.ceil(n / b)
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--train-dataset', default='datasets/train/countdown')
    ap.add_argument('--eval-dataset', default='datasets/evaluation_suite/countdown')
    ap.add_argument('--n-prompts', type=int, default=500)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--block', type=int, default=1024)
    ap.add_argument('--gen-batch', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--weights', default=None, help='optional vLLM-named state dict (.pth) to load before computing (refresh)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    if a.weights:
        load_vllm_state(model, torch.load(a.weights, map_location='cpu'))
    model.eval()

    train = [r['context'] for split in load_from_disk(a.train_dataset).values() for r in split]
    evalp = set(r['context'] for split in load_from_disk(a.eval_dataset).values() for r in split)
    assert not (set(train) & evalp)
    prompts = train[:a.n_prompts] + gen_problems(max(0, a.n_prompts - len(train)), a.seed, set(train) | evalp)

    # 1) sample responses from the model (true Fisher)
    t0 = time.time()
    seqs = []
    for i in range(0, len(prompts), a.gen_batch):
        enc = tok(prompts[i:i + a.gen_batch], return_tensors='pt', padding=True).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
        P = enc['input_ids'].shape[1]
        for j in range(out.shape[0]):
            p_ids = enc['input_ids'][j][enc['attention_mask'][j].bool()]
            r_ids = out[j, P:]
            r_ids = r_ids[r_ids != tok.pad_token_id]
            seqs.append((p_ids.cpu(), r_ids.cpu()))
    t_gen = time.time() - t0
    print('sampled %d responses, mean len %.1f (%.0fs)' % (len(seqs), np.mean([len(r) for _, r in seqs]), t_gen), flush=True)

    # 2) K-FAC accumulation with hooks
    lin = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and '.layers.' in n}
    stats = {}
    for n, m in lin.items():
        ab, gb = blocks(m.in_features, a.block), blocks(m.out_features, a.block)
        stats[n] = dict(ab=ab, gb=gb, A=[torch.zeros(b, b, device=dev) for b in ab],
                        G=[torch.zeros(b, b, device=dev) for b in gb])
    cur = {}

    def fwd_hook(name):
        def h(mod, inp, out):
            x = inp[0].detach()[0].float()                       # (T, in)
            for k, Ab in enumerate(torch.split(x, stats[name]['ab'], 1)):
                stats[name]['A'][k].add_(Ab.T @ Ab)
            def gh(g):
                for k, gb in enumerate(torch.split(g.detach()[0].float(), stats[name]['gb'], 1)):
                    stats[name]['G'][k].add_(gb.T @ gb)
            out.register_hook(gh)
        return h

    hs = [m.register_forward_hook(fwd_hook(n)) for n, m in lin.items()]
    # exact diagonal of the (sequence-level) true Fisher for EVERY parameter: mean over sequences of (d log p / d theta)^2
    diag = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters()}
    n_tok = 0
    t0 = time.time()
    for p_ids, r_ids in seqs:
        if len(r_ids) == 0:
            continue
        ids = torch.cat([p_ids, r_ids]).unsqueeze(0).to(dev)
        logits = model(ids).logits.float()
        lp = torch.log_softmax(logits[0, len(p_ids) - 1:-1], -1)
        L = lp.gather(1, ids[0, len(p_ids):, None]).sum()
        L.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                diag[n].add_(p.grad.float() ** 2)
        model.zero_grad(set_to_none=True)
        n_tok += ids.shape[1]
    for h in hs:
        h.remove()
    n_seq = sum(1 for _, r in seqs if len(r))
    print('K-FAC accumulated over %d sequences, %d tokens (%.0fs)' % (n_seq, n_tok, time.time() - t0), flush=True)

    # 3) eigendecompose, fuse to vLLM names, save
    eig = {}
    for n, s in stats.items():
        eig[n] = dict(A=[torch.linalg.eigh(A / n_tok) for A in s['A']], G=[torch.linalg.eigh(G / n_seq) for G in s['G']])

    def pack(name_a, names_g):
        QA = [e.eigenvectors.cpu() for e in eig[name_a]['A']]
        lA = [e.eigenvalues.clamp_min(0).cpu() for e in eig[name_a]['A']]
        QG = [e.eigenvectors.cpu() for ng in names_g for e in eig[ng]['G']]
        lG = [e.eigenvalues.clamp_min(0).cpu() for ng in names_g for e in eig[ng]['G']]
        return dict(QA=QA, lA=lA, QG=QG, lG=lG)

    factors = {}
    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        pre = 'model.layers.%d.' % i
        factors[pre + 'self_attn.qkv_proj.weight'] = pack(pre + 'self_attn.q_proj', [pre + 'self_attn.%s_proj' % x for x in 'qkv'])
        factors[pre + 'self_attn.o_proj.weight'] = pack(pre + 'self_attn.o_proj', [pre + 'self_attn.o_proj'])
        factors[pre + 'mlp.gate_up_proj.weight'] = pack(pre + 'mlp.gate_proj', [pre + 'mlp.gate_proj', pre + 'mlp.up_proj'])
        factors[pre + 'mlp.down_proj.weight'] = pack(pre + 'mlp.down_proj', [pre + 'mlp.down_proj'])
    # q/k/v share their input and gate/up share theirs: verify the A factors agree (they are the same activations)
    for i in range(n_layers):
        pre = 'model.layers.%d.' % i
        for x, y in (('self_attn.q_proj', 'self_attn.k_proj'), ('mlp.gate_proj', 'mlp.up_proj')):
            dA = max(float((u - v).abs().max()) for u, v in zip(stats[pre + x]['A'], stats[pre + y]['A']))
            assert dA <= 1e-3 * max(float(u.abs().max()) for u in stats[pre + x]['A']), (pre, x, y, dA)

    summary = spectra_summary(factors)
    # diagonal Fisher under vLLM (fused) names, plus how it compares with K-FAC's own diagonal diag(G) (x) diag(A)
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    dfused = {}
    for n in list(diag):
        diag[n] /= n_seq
    for n, v in diag.items():
        for parts, fused in ((('q_proj', 'k_proj', 'v_proj'), 'qkv_proj'), (('gate_proj', 'up_proj'), 'gate_up_proj')):
            if any('.%s.' % x in n for x in parts):
                if '.%s.' % parts[0] in n:
                    dfused[n.replace(parts[0], fused)] = torch.cat([diag[n.replace(parts[0], x)] for x in parts], 0)
                break
        else:
            dfused[n] = v
    dfused = {k: v.cpu() for k, v in dfused.items()}
    dv = torch.cat([v.flatten() for v in dfused.values()]).double()
    ds = torch.sort(dv, descending=True).values
    dc = torch.cumsum(ds, 0) / ds.sum()
    dq = lambda p: float(dc[max(1, int(len(ds) * p)) - 1])
    kd_corr = []
    for name, d in factors.items():
        kd = torch.cat([torch.cat([torch.outer(QG.pow(2) @ lg, QA.pow(2) @ la) for QA, la in zip(d['QA'], d['lA'])], 1)
                        for QG, lg in zip(d['QG'], d['lG'])], 0)          # diag(G) (x) diag(A), blockwise
        x, y = torch.log(kd.flatten().double() + 1e-30), torch.log(dfused[name].flatten().double().cpu() + 1e-30)
        kd_corr.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
    summary['diag_fisher'] = dict(n_params=len(dv), frac_top0_1pct=dq(0.001), frac_top1pct=dq(0.01), frac_top10pct=dq(0.1),
                                  effrank_over_d=float(dv.sum() ** 2 / (dv ** 2).sum()) / len(dv),
                                  log_corr_kfac_diag_vs_exact_diag_median=float(np.median(kd_corr)))
    torch.save(dict(diag={k: v.to(torch.bfloat16).cpu() for k, v in dfused.items()}, meta=dict(model=a.model, n_seq=n_seq)),
               a.out.replace('.pt', '') + '_diag.pt')
    meta = dict(model=a.model, n_prompts=len(prompts), n_seq=n_seq, n_tok=n_tok, max_new_tokens=a.max_new_tokens,
                block=a.block, seed=a.seed, t_gen=t_gen, mean_resp_len=float(np.mean([len(r) for _, r in seqs])),
                weights=a.weights)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(dict(factors=factors, meta=meta), a.out)
    json.dump(dict(meta=meta, **summary), open(a.out.replace('.pt', '') + '_spectra.json', 'w'), indent=1)
    print(json.dumps(summary['global'], indent=1))


def spectra_summary(factors):
    """Per weight: cond / effective rank of A and G, and how concentrated the isotropic KL budget is:
    under isotropic noise the KL budget of eigendirection (i,j) is proportional to f_ij = lG_i lA_j."""
    per, allf = {}, []
    for name, d in factors.items():
        lA = torch.cat(d['lA']).double(); lG = torch.cat(d['lG']).double()
        f = torch.cat([torch.outer(g, a).flatten() for g in d['lG'] for a in d['lA']]).float()
        allf.append(f)
        er = lambda l: float(l.sum() ** 2 / (l ** 2).sum())
        fs = torch.sort(f, descending=True).values
        cum = torch.cumsum(fs.double(), 0) / fs.double().sum()
        per[name] = dict(dA=len(lA), dG=len(lG), condA=float(lA.max() / lA.clamp_min(1e-30).min()),
                         condG=float(lG.max() / lG.clamp_min(1e-30).min()),
                         effrank_A=er(lA), effrank_G=er(lG), trF=float(f.sum()),
                         kl_frac_top1pct=float(cum[max(1, len(fs) // 100) - 1]),
                         kl_frac_top10pct=float(cum[max(1, len(fs) // 10) - 1]))
    f = torch.cat(allf)
    fs = torch.sort(f, descending=True).values
    cum = torch.cumsum(fs.double(), 0) / fs.double().sum()
    q = lambda p: float(cum[max(1, int(len(fs) * p)) - 1])
    g = dict(n_params_kfac=len(f), mean_f=float(f.mean()), kl_frac_top0_1pct=q(0.001), kl_frac_top1pct=q(0.01),
             kl_frac_top10pct=q(0.1), effrank_over_d=float(f.double().sum() ** 2 / (f.double() ** 2).sum()) / len(f),
             median_f_over_mean=float(fs[len(fs) // 2] / f.mean()),
             frac_f_below_1em3_mean=float((f < 1e-3 * f.mean()).double().mean()),
             frac_f_below_1em2_mean=float((f < 1e-2 * f.mean()).double().mean()),
             frac_f_below_1em1_mean=float((f < 1e-1 * f.mean()).double().mean()))
    return dict(**{'global': g}, per_weight=per)


def load_vllm_state(model, sd):
    """Map a vLLM-named state dict (fused qkv / gate_up) into an HF Qwen2 model."""
    hf = dict(model.named_parameters())
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    nq, nkv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    with torch.no_grad():
        for k, v in sd.items():
            if 'qkv_proj' in k:
                q, kk, vv = torch.split(v, [nq, nkv, nkv], 0)
                for part, t in zip('qkv', (q, kk, vv)):
                    hf[k.replace('qkv_proj', part + '_proj')].copy_(t)
            elif 'gate_up_proj' in k:
                g, u = torch.chunk(v, 2, 0)
                hf[k.replace('gate_up_proj', 'gate_proj')].copy_(g)
                hf[k.replace('gate_up_proj', 'up_proj')].copy_(u)
            elif k in hf:
                hf[k].copy_(v)
            else:
                raise KeyError(k)


if __name__ == '__main__':
    main()
