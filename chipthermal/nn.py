"""Small U-Net that reconstructs the full temperature map from sparse sensors (+ optional floorplan map)."""
import numpy as np
import torch
import torch.nn as nn

N, T_AMB, SCALE = 32, 25.0, 40.0


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    def __init__(self, cin, c=16):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(cin, c), Block(c, 2 * c), Block(2 * c, 4 * c)
        self.pool = nn.MaxPool2d(2)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), Block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), Block(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def make_input(T, sensors, dens, noise, rng, use_floorplan):
    """T: (B,1024) true maps; dens: (B,1024) design power density (normalized). Returns (B,C,32,32) tensor."""
    B = T.shape[0]
    read = T[:, sensors] + (rng.uniform(-noise, noise, size=(B, len(sensors))) if noise else 0.0)
    sparse = np.zeros((B, N * N), np.float32); mask = np.zeros((B, N * N), np.float32)
    sparse[:, sensors] = (read - T_AMB) / SCALE; mask[:, sensors] = 1.0
    ch = [sparse, mask] + ([dens] if use_floorplan else [])
    return torch.from_numpy(np.stack(ch, 1).reshape(B, len(ch), N, N).astype(np.float32))


def train_model(Ttr, Dtr, sensors, use_floorplan, Tva, Dva, epochs=40, bs=64, max_train_noise=1.0, seed=0, log=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = UNet(3 if use_floorplan else 2)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(Ttr) + bs - 1) // bs))
    ytr = torch.from_numpy(((Ttr - T_AMB) / SCALE).reshape(-1, 1, N, N).astype(np.float32))
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(Ttr))
        for k in range(0, len(Ttr), bs):
            idx = perm[k:k + bs]
            noise = rng.uniform(0, max_train_noise)
            xb = make_input(Ttr[idx], sensors, Dtr[idx], noise, rng, use_floorplan)
            loss = nn.functional.mse_loss(model(xb), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        va = rmse(predict(model, Tva, Dva, sensors, 0.0, np.random.default_rng(1), use_floorplan), Tva)
        if log:
            print(f"  {log} epoch {ep + 1}/{epochs} val RMSE {va:.3f}", flush=True)
        if va < best:
            best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, T, D, sensors, noise, rng, use_floorplan, bs=512):
    model.eval(); out = []
    for k in range(0, len(T), bs):
        out.append(model(make_input(T[k:k + bs], sensors, D[k:k + bs], noise, rng, use_floorplan)).numpy())
    return np.concatenate(out).reshape(len(T), -1) * SCALE + T_AMB


def rmse(est, T):
    return float(np.sqrt(np.mean((est - T) ** 2)))


def tmax_err(est, T):
    return float(np.mean(np.abs(est.max(1) - T.max(1))))


def noisy(T, sensors, noise, rng):
    return T[:, sensors] + (rng.uniform(-noise, noise, size=(len(T), len(sensors))) if noise else 0.0)


