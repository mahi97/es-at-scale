"""Kronecker-factored (K-FAC) noise shaping for ES perturbations (inverse-Fisher sampling, IFS-ES).

For a 2-D weight W (out x in) with K-FAC factors A = E[a a^T] (in x in) and G = E[g g^T] (out x out), both stored
block-diagonally (blocks <= 1024) as eigendecompositions A_b = Q_A,b diag(l_A,b) Q_A,b^T, G_b likewise, the Fisher
block is F ~ A (x) G with eigenvalues f_ij = l_G,i * l_A,j and eigenvectors (Q_A e_j) (x) (Q_G e_i).

Sampling ("A" placement) with covariance  Sigma = c (F + lam I)^{2q}  (q = -1/2: (F + lam I)^{-1}; q = -1/4: ^{-1/2};
q = +1/2: (F + lam I)^{+1}) is done in the eigenbasis:
    eps = Q_G ( Z * sqrt(c) s ) Q_A^T,  s_ij = (f_ij + lam)^q,  Z ~ N(0, I)
(Q_G^T Z Q_A is again N(0, I), so the inner rotation is skipped; eps is a deterministic linear function of Z, so the
ES update  sum_n z_n eps_n = T(sum_n z_n Z_n)  needs one transform per weight per iteration).
Placement "B" keeps Z isotropic for the perturbation and applies P = c (F + lam I)^{2q} to the estimate:
    U_B = Q_G ( (Q_G^T U Q_A) * c s^2 ) Q_A^T.
The damping lam = gamma * (global mean eigenvalue of F over all K-FAC'd weights). The constant c sets the budget:
    'euclid': sum c s^2 = sum 1           (E||eps||^2 equal to isotropic noise on the K-FAC'd weights)
    'kl'    : sum c f s^2 = sum f          (equal second-order KL budget, theory/kl_budget.tex)
Weights without factors (embedding, norms, biases) keep the isotropic noise unchanged.
"""
import torch


def load_factors(path, device, dtype=torch.float32):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    fac = {}
    for name, d in blob['factors'].items():
        fac[name] = {k: [t.to(device=device, dtype=dtype if k.startswith('Q') else torch.float32) for t in d[k]]
                     for k in ('QA', 'lA', 'QG', 'lG')}
    return fac, blob.get('meta', {})


def _f_blocks(d):
    """Fisher eigenvalues per (row block, col block): f[bi][bj] = outer(lG_bi, lA_bj)."""
    return [[torch.outer(lg, la) for la in d['lA']] for lg in d['lG']]


def make_scales(fac, q, gamma, norm):
    """Returns ({name: [[sqrt(c) * s]]}, info). s = (f + lam)^q with lam = gamma * mean(f) over all K-FAC'd weights."""
    lG_clamp = lambda t: t.clamp_min(0)
    tot_f, tot_n = 0.0, 0
    for d in fac.values():
        sg = sum(float(lG_clamp(l).sum()) for l in d['lG'])
        sa = sum(float(lG_clamp(l).sum()) for l in d['lA'])
        tot_f += sg * sa
        tot_n += sum(l.numel() for l in d['lG']) * sum(l.numel() for l in d['lA'])
    lam = gamma * tot_f / tot_n
    S, num_e, num_k, den_e, den_k = {}, 0.0, 0.0, 0.0, 0.0
    for name, d in fac.items():
        rows = []
        for fr in _f_blocks(d):
            row = []
            for f in fr:
                f = f.clamp_min(0)
                s = (f + lam) ** q
                num_e += f.numel(); den_e += float((s * s).sum())
                num_k += float(f.sum()); den_k += float((f * s * s).sum())
                row.append(s)
            rows.append(row)
        S[name] = rows
    c = num_e / den_e if norm == 'euclid' else num_k / den_k
    for rows in S.values():
        for row in rows:
            for j in range(len(row)):
                row[j] = row[j] * c ** 0.5
    # diagnostics: KL budget relative to isotropic, Euclidean budget relative to isotropic
    kl_rel = c * den_k / num_k
    eu_rel = c * den_e / num_e
    return S, dict(lam=lam, c=c, kl_budget_rel_iso=kl_rel, euclid_budget_rel_iso=eu_rel, q=q, gamma=gamma, norm=norm,
                   mean_f=tot_f / tot_n)


def _split(x, sizes, dim):
    return list(torch.split(x, sizes, dim=dim))


def sample_transform(d, s_rows, Z):
    """eps = Q_G (Z * s) Q_A^T, blockwise. Z: (out, in) fp32. Returns fp32."""
    gs = [l.numel() for l in d['lG']]
    as_ = [l.numel() for l in d['lA']]
    out = torch.empty_like(Z)
    r0 = 0
    for bi, (Zr, QG) in enumerate(zip(_split(Z, gs, 0), d['QG'])):
        c0 = 0
        for bj, (Zb, QA) in enumerate(zip(_split(Zr, as_, 1), d['QA'])):
            out[r0:r0 + gs[bi], c0:c0 + as_[bj]] = QG.to(Z.dtype) @ (Zb * s_rows[bi][bj].to(Z.dtype)) @ QA.to(Z.dtype).T
            c0 += as_[bj]
        r0 += gs[bi]
    return out


def precond_transform(d, s_rows, U):
    """U_B = Q_G ((Q_G^T U Q_A) * s^2) Q_A^T, blockwise (placement B: P applied to the isotropic ES estimate)."""
    gs = [l.numel() for l in d['lG']]
    as_ = [l.numel() for l in d['lA']]
    out = torch.empty_like(U)
    r0 = 0
    for bi, (Ur, QG) in enumerate(zip(_split(U, gs, 0), d['QG'])):
        c0 = 0
        for bj, (Ub, QA) in enumerate(zip(_split(Ur, as_, 1), d['QA'])):
            QGf, QAf = QG.to(U.dtype), QA.to(U.dtype)
            out[r0:r0 + gs[bi], c0:c0 + as_[bj]] = QGf @ ((QGf.T @ Ub @ QAf) * s_rows[bi][bj].to(U.dtype) ** 2) @ QAf.T
            c0 += as_[bj]
        r0 += gs[bi]
    return out


# ---- diagonal ablation (3PO / HiZOO-style structure): Sigma = c (diag(F) + lam)^{2q} on the same weights ----
def load_diag(path, names, device):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    return {n: blob['diag'][n].to(device=device, dtype=torch.float32) for n in names}


def make_diag_scales(diag, q, gamma, norm):
    tot_f = sum(float(d.sum()) for d in diag.values())
    tot_n = sum(d.numel() for d in diag.values())
    lam = gamma * tot_f / tot_n
    S, den_e, den_k = {}, 0.0, 0.0
    for n, f in diag.items():
        s = (f.clamp_min(0) + lam) ** q
        den_e += float((s * s).sum()); den_k += float((f * s * s).sum())
        S[n] = s
    c = tot_n / den_e if norm == 'euclid' else tot_f / den_k
    for n in S:
        S[n] = S[n] * c ** 0.5
    return S, dict(lam=lam, c=c, kl_budget_rel_iso=c * den_k / tot_f, euclid_budget_rel_iso=c * den_e / tot_n,
                   q=q, gamma=gamma, norm=norm, mean_f=tot_f / tot_n)
