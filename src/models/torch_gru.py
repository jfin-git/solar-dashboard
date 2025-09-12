# src/models/torch_gru.py
import argparse, yaml
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from src.utils.paths import data_dir, configs_dir, models_dir

# ---------- Data ----------
def make_sequences(df, history=48, horizon=24, feature_cols=None, target_col="y"):
    X_list, Y_list, ts_list = [], [], []
    vals = df[feature_cols + [target_col]].values
    idx = df.index
    for t in range(history, len(df) - horizon):
        past_feat = vals[t - history : t, :-1]         # [history, F]
        future_y  = vals[t : t + horizon, -1]          # [horizon]
        X_list.append(past_feat)
        Y_list.append(future_y)
        ts_list.append(idx[t])
    X = np.stack(X_list).astype(np.float32)            # [N, history, F]
    Y = np.stack(Y_list).astype(np.float32)            # [N, horizon]
    return X, Y, np.array(ts_list)

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)   # [N, T, F]
        self.Y = torch.from_numpy(Y)   # [N, H]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

# ---------- Model ----------
def pinball_loss(y, q_pred, taus):
    """
    y:      [B, H] or [B, H, 1]
    q_pred: [B, H, Q]
    taus:   list/tuple of quantiles
    """
    if y.ndim == 2:
        y = y.unsqueeze(-1)            # [B, H, 1]
    losses = []
    for i, tau in enumerate(taus):
        e = y - q_pred[..., i:i+1]     # [B, H, 1]
        losses.append(torch.maximum(tau*e, (tau-1)*e).mean())
    return torch.stack(losses).sum()

class GRUQuantile(pl.LightningModule):
    def __init__(self, input_size, horizon, hidden_size=64, num_layers=2, dropout=0.2,
                 lr=1e-3, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.save_hyperparameters()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=(dropout if num_layers > 1 else 0.0))
        # Project last hidden state → per-hour quantiles
        self.head = nn.Linear(hidden_size, horizon * len(quantiles))
        self.horizon = horizon
        self.quantiles = quantiles
        self.lr = lr

    def forward(self, x):
        """
        x: [B, history, F]
        returns q: [B, horizon, Q]
        """
        out, _ = self.gru(x)           # out: [B, history, H]
        h_last = out[:, -1, :]         # [B, H]
        q = self.head(h_last)          # [B, horizon*Q]
        B = x.size(0)
        q = q.view(B, self.horizon, len(self.quantiles))
        return q

    def training_step(self, batch, _):
        x, y = batch                    # x: [B, T, F], y: [B, H]
        q = self(x)                     # [B, H, Q]
        loss = pinball_loss(y, q, self.quantiles)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        q = self(x)
        loss = pinball_loss(y, q, self.quantiles)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=float(self.lr))


# ---------- Train entrypoint ----------
def train(site_name: str, cfg: dict):
    df = pd.read_parquet(data_dir("processed", f"{site_name}_features.parquet")).sort_index()
    feature_cols = [c for c in df.columns if c not in {"y", "pv_kw"}]

    hist = int(cfg["data"]["history_hours"])
    hor  = int(cfg["data"]["horizon_hours"])

    batch_size = int(cfg["training"]["torch"]["batch_size"])
    max_epochs = int(cfg["training"]["torch"]["max_epochs"])
    H = int(cfg["training"]["torch"]["hidden_size"])
    L = int(cfg["training"]["torch"]["num_layers"])
    D = float(cfg["training"]["torch"]["dropout"])
    LR = float(cfg["training"]["torch"]["learning_rate"])
    taus = tuple(float(t) for t in cfg["training"]["torch"]["quantiles"])


    X, Y, ts = make_sequences(df, history=hist, horizon=hor, feature_cols=feature_cols, target_col="y")
    ds = SeqDataset(X, Y)

    # 80/20 split (temporal-ish by not shuffling the dataset before split)
    n = len(ds)
    n_train = int(n * 0.8)
    n_val = n - n_train
    ds_train, ds_val = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(7))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = GRUQuantile(
        input_size=X.shape[-1], horizon=hor,
        hidden_size=H, num_layers=L, dropout=D, lr=LR, quantiles=taus
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=models_dir(), filename="gru-best", save_top_k=1, monitor="val_loss", mode="min"
    )
    es = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[ckpt, es], logger=False)
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)

    # Save P50 per-hour predictions for the whole dataset (for the app)
    model.eval()
    with torch.no_grad():
        q_all = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size])
            qb = model(xb)                         # [b, H, Q]
            q_all.append(qb[:, :, 1])             # take P50 index (1)
        p50 = torch.cat(q_all, dim=0).cpu().numpy()  # [N, H]

    # Write a simple per-sample summary: next 24h average P50 (to keep app unchanged)
    p50_avg24 = p50.mean(axis=1)                  # [N]
    pred_path = data_dir("processed", f"{site_name}_torch_preds.parquet")
    pd.DataFrame({"pred_p50_avg24h": p50_avg24}, index=ts).to_parquet(pred_path)
    print(f"Wrote torch predictions -> {pred_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site_name", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(configs_dir("train.yaml")))
    train(args.site_name, cfg)
