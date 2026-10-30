"""Checks for es_at_scale/utils/kfac_precond.py against explicit dense Kronecker matrices (CPU, float64 where it matters).
Run: python tests/test_kfac_precond.py"""
import torch
from es_at_scale.utils.kfac_precond import make_scales, sample_transform, precond_transform

torch.manual_seed(0)


def rand_spd(n, cond):
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    return Q, torch.logspace(0, torch.log10(torch.tensor(float(cond))), n, dtype=torch.float64)


def make_weight(out_blocks, in_blocks):
    QG, lG, QA, lA = [], [], [], []
    for b in out_blocks:
        Q, l = rand_spd(b, 50); QG.append(Q); lG.append(l)
    for b in in_blocks:
        Q, l = rand_spd(b, 20); QA.append(Q); lA.append(l * 0.3)
    return dict(QG=QG, lG=lG, QA=QA, lA=lA)


def dense(d, which):
    """Dense block-diagonal factor matrices G (out x out), A (in x in)."""
    G = torch.block_diag(*[Q @ torch.diag(l) @ Q.T for Q, l in zip(d['QG'], d['lG'])])
    A = torch.block_diag(*[Q @ torch.diag(l) @ Q.T for Q, l in zip(d['QA'], d['lA'])])
    return G, A


fac = {'w1': make_weight([3, 4], [5]), 'w2': make_weight([6], [2, 3])}
ok = True
for q in (-0.5, -0.25, 0.5):
    for norm in ('euclid', 'kl'):
        S, info = make_scales(fac, q, 1e-2, norm)
        tot_e, tot_k, iso_e, iso_k = 0.0, 0.0, 0.0, 0.0
        for name, d in fac.items():
            G, A = dense(d, None)
            out, inn = G.shape[0], A.shape[0]
            F = torch.kron(G, A)                       # vec(W) row-major: W[i, j] -> i*inn + j, so F = G (x) A
            lam = info['lam']
            w, V = torch.linalg.eigh(F)
            Sigma_ref = info['c'] * V @ torch.diag((w.clamp_min(0) + lam) ** (2 * q)) @ V.T
            # empirical covariance of sample_transform
            n = 40000
            Z = torch.randn(n, out, inn, dtype=torch.float64)
            E = torch.stack([sample_transform(d, S[name], z) for z in Z[:4000]]).reshape(4000, -1)
            C = E.T @ E / 4000
            rel = float((C - Sigma_ref).norm() / Sigma_ref.norm())
            # exact covariance via the linear map: columns of T(e_k)
            T = torch.stack([sample_transform(d, S[name], torch.eye(out * inn, dtype=torch.float64)[k].reshape(out, inn)).flatten()
                             for k in range(out * inn)], 1)
            rel_exact = float((T @ T.T - Sigma_ref).norm() / Sigma_ref.norm())
            # placement B applies P = Sigma to the estimate
            U = torch.randn(out, inn, dtype=torch.float64)
            relB = float((precond_transform(d, S[name], U).flatten() - Sigma_ref @ U.flatten()).norm() / (Sigma_ref @ U.flatten()).norm())
            ok &= rel_exact < 1e-10 and relB < 1e-10 and rel < 0.1
            tot_e += float(torch.trace(Sigma_ref)); iso_e += out * inn
            tot_k += float(torch.trace(F @ Sigma_ref)); iso_k += float(torch.trace(F))
            print('q=%+.2f %-6s %s: cov exact rel.err %.1e, MC(4k) %.3f, placeB %.1e' % (q, norm, name, rel_exact, rel, relB))
        r = tot_e / iso_e if norm == 'euclid' else tot_k / iso_k
        ok &= abs(r - 1) < 1e-10
        print('   budget ratio vs isotropic (%s) = %.12f; info %s' % (norm, r, {k: round(v, 4) for k, v in info.items() if isinstance(v, float)}))
print('ALL PASS' if ok else 'FAIL')
