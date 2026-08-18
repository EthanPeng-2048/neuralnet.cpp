# ── 独立验证 CausalSelfAttention 梯度数学（纯 numpy，中心差分） ──────────
# 目标：给出 wq/wk/wv/wo/grad_x 的"教科书"解析梯度，并用中心差分交叉验证，
#       作为判断 attn_debug(暴力参考) 与 attn_gradcheck(中心差分) 谁对谁错的
#       独立基准。无任何引擎代码参与。
# ─────────────────────────────────────────────────────────────────────────
import numpy as np

EPS = 1e-5

def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=1, keepdims=True)

def causal_mask(seq):
    M = np.zeros((seq, seq))
    M[np.triu_indices(seq, 1)] = -np.inf
    return M

def attention_forward(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale):
    d = x.shape[0]
    dk = d // H
    Q = Wq @ x + bq          # (d, seq)
    K = Wk @ x + bk
    V = Wv @ x + bv
    concat = np.zeros_like(x)
    As = []
    for h in range(H):
        Qh = Q[h*dk:(h+1)*dk]      # (dk, seq)
        Kh = K[h*dk:(h+1)*dk]
        Vh = V[h*dk:(h+1)*dk]
        Sh = scale * (Qh.T @ Kh)   # (seq, seq)
        Sh = Sh + causal_mask(seq)
        Ah = softmax_rows(Sh)
        As.append(Ah)
        Oh = Vh @ Ah.T             # (dk, seq)
        concat[h*dk:(h+1)*dk] = Oh
    y = Wo @ concat + bo           # (d, seq)
    return y, concat, Q, K, V, As

def attention_backward(go, x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale):
    d = x.shape[0]
    dk = d // H
    y, concat, Q, K, V, As = attention_forward(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale)
    gc = Wq  # placeholder, will overwrite
    # grad wrt wo
    gw_o = go @ concat.T          # (d, d)
    gb_o = go.sum(axis=1, keepdims=True)
    gconcat = Wo.T @ go           # (d, seq)

    gQ = np.zeros_like(Q)
    gK = np.zeros_like(K)
    gV = np.zeros_like(V)
    for h in range(H):
        Vh = V[h*dk:(h+1)*dk]
        Kh = K[h*dk:(h+1)*dk]
        Qh = Q[h*dk:(h+1)*dk]
        gch = gconcat[h*dk:(h+1)*dk]
        Ah = As[h]
        gVh = gch @ Ah                            # (dk, seq): dL/dV = gc @ A (NOT A^T)
        gAh = gch.T @ Vh                          # (seq, seq): dL/dA = gc^T V
        # softmax bwd wrt S' (scaled+masked)
        dot = (Ah * gAh).sum(axis=1, keepdims=True)
        gS = Ah * (gAh - dot)                     # dL/dS'
        gQh = scale * (Kh @ gS.T)                 # dL/dQ = scale * K @ gS^T
        gKh = scale * (Qh @ gS)                   # dL/dK = scale * Q @ gS
        gQ[h*dk:(h+1)*dk] = gQh
        gK[h*dk:(h+1)*dk] = gKh
        gV[h*dk:(h+1)*dk] = gVh

    gw_q = gQ @ x.T
    gb_q = gQ.sum(axis=1, keepdims=True)
    gw_k = gK @ x.T
    gb_k = gK.sum(axis=1, keepdims=True)
    gw_v = gV @ x.T
    gb_v = gV.sum(axis=1, keepdims=True)
    gx = Wq.T @ gQ + Wk.T @ gK + Wv.T @ gV
    return dict(gw_q=gw_q, gb_q=gb_q, gw_k=gw_k, gb_k=gb_k,
                gw_v=gw_v, gb_v=gb_v, gw_o=gw_o, gb_o=gb_o, gx=gx,
                gQ=gQ, gK=gK, gV=gV)

def loss_fn(y, go):
    return np.sum(y * go)

def numerical_grads(x, go, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale, eps=EPS):
    d = x.shape[0]
    params = dict(wq=Wq, bq=bq, wk=Wk, bk=bk, wv=Wv, bv=bv, wo=Wo, bo=bo)
    out = {}
    for name, P in params.items():
        G = np.zeros_like(P)
        it = np.ndindex(P.shape)
        for idx in it:
            orig = P[idx]
            P[idx] = orig + eps
            yp, *_ = attention_forward(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale)
            lp = loss_fn(yp, go)
            P[idx] = orig - eps
            ym, *_ = attention_forward(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale)
            lm = loss_fn(ym, go)
            P[idx] = orig
            G[idx] = (lp - lm) / (2*eps)
        out[name] = G
    # grad_x
    Gx = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        yp, *_ = attention_forward(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale)
        lp = loss_fn(yp, go)
        x[idx] = orig - eps
        ym, *_ = attention_forward(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale)
        lm = loss_fn(ym, go)
        x[idx] = orig
        Gx[idx] = (lp - lm) / (2*eps)
    out['x'] = Gx
    return out

def main():
    rng = np.random.default_rng(123)
    d, H, seq = 16, 2, 8
    dk = d // H
    scale = 1.0 / np.sqrt(dk)
    x = rng.uniform(-1, 1, (d, seq))
    go = rng.uniform(-1, 1, (d, seq))
    Wq = rng.uniform(-1, 1, (d, d)); bq = rng.uniform(-1, 1, (d, 1))
    Wk = rng.uniform(-1, 1, (d, d)); bk = rng.uniform(-1, 1, (d, 1))
    Wv = rng.uniform(-1, 1, (d, d)); bv = rng.uniform(-1, 1, (d, 1))
    Wo = rng.uniform(-1, 1, (d, d)); bo = rng.uniform(-1, 1, (d, 1))

    ana = attention_backward(go, x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale)
    num = numerical_grads(x, go, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale)

    names = [('gw_q','wq'), ('gb_q','bq'), ('gw_k','wk'), ('gb_k','bk'),
             ('gw_v','wv'), ('gb_v','bv'), ('gw_o','wo'), ('gb_o','bo'), ('gx','x')]
    print("== 解析 vs 中心差分 (纯 numpy) ==")
    all_ok = True
    for ana_key, num_key in names:
        A, N = ana[ana_key], num[num_key]
        err = np.abs(A - N)
        max_err = err.max()
        idx = np.unravel_index(err.argmax(), err.shape)
        ok = max_err < 1e-4
        all_ok &= ok
        print(f"  {ana_key:6s}: {'OK ' if ok else 'FAIL'}  max_err={max_err:.3e}"
              f"  ana_max={np.abs(A).max():.3e} num_max={np.abs(N).max():.3e}"
              f"  @{idx} ana={A[idx]:.4e} num={N[idx]:.4e}")
    print("== 全部通过 ==" if all_ok else "== 存在失败 ==")

    # 逐步定位中间量：对 gQ/gK/gV 做中心差分
    print("\n== 中间量 gQ/gK/gV 验证 (中心差分) ==")
    # dL/dQ_ki: 扰动 Q_ki 会影响所有 S'_ij (j 列)，重算 loss
    _, _, Q, K, V, As = attention_forward(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, H, seq, scale)
    gconcat = Wo.T @ go
    gQ_ana = ana['gQ']; gK_ana = ana['gK']; gV_ana = ana['gV']
    dk = d // H
    for tag, gA_ana, M in [('gQ', gQ_ana, Q), ('gK', gK_ana, K), ('gV', gV_ana, V)]:
        G = np.zeros_like(M)
        for idx in np.ndindex(M.shape):
            orig = M[idx]
            M[idx] = orig + EPS
            # 重算 O 和 loss
            concat2 = np.zeros_like(x)
            for h in range(H):
                Qh = Q[h*dk:(h+1)*dk]; Kh = K[h*dk:(h+1)*dk]; Vh = V[h*dk:(h+1)*dk]
                Sh = scale * (Qh.T @ Kh) + causal_mask(seq)
                Ah = softmax_rows(Sh)
                concat2[h*dk:(h+1)*dk] = Vh @ Ah.T
            yp = Wo @ concat2 + bo
            lp = loss_fn(yp, go)
            M[idx] = orig - EPS
            concat2 = np.zeros_like(x)
            for h in range(H):
                Qh = Q[h*dk:(h+1)*dk]; Kh = K[h*dk:(h+1)*dk]; Vh = V[h*dk:(h+1)*dk]
                Sh = scale * (Qh.T @ Kh) + causal_mask(seq)
                Ah = softmax_rows(Sh)
                concat2[h*dk:(h+1)*dk] = Vh @ Ah.T
            ym = Wo @ concat2 + bo
            lm = loss_fn(ym, go)
            M[idx] = orig
            G[idx] = (lp - lm) / (2*EPS)
        err = np.abs(G - gA_ana)
        max_err = err.max()
        idx = np.unravel_index(err.argmax(), err.shape)
        ok = max_err < 1e-4
        print(f"  {tag}: {'OK ' if ok else 'FAIL'}  max_err={max_err:.3e}"
              f" ana={gA_ana[idx]:.4e} num={G[idx]:.4e} @{idx}")
        all_ok &= ok

if __name__ == '__main__':
    main()
