
import argparse, yaml, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.paths import data_dir, models_dir, configs_dir
from src.models.metrics import daylight_mask, mae, rmse, mape

def make_sequences(df, history=48, horizon=24, feature_cols=None, target_col="y"):
    X_list, Y_list, ts_list = [], [], []
    vals = df[feature_cols + [target_col]].values
    idx = df.index
    for t in range(history, len(df)-horizon):
        past = vals[t-history:t, :-1]
        future_y = vals[t:t+horizon, -1]
        X_list.append(past); Y_list.append(future_y); ts_list.append(idx[t])
    return np.stack(X_list), np.stack(Y_list), np.array(ts_list)

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

class GRUModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, lr=1e-3, quantiles=(0.1,0.5,0.9)):
        super().__init__()
        self.save_hyperparameters()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, 3)
        self.lr = lr
        self.quantiles = quantiles
    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        q = self.head(h)
        return q
    def pinball(self, y, q_pred, tau):
        return torch.mean(torch.maximum(tau*(y - q_pred), (tau-1)*(y - q_pred)))
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_next = y.mean(dim=1, keepdim=True)
        q = self(x)
        loss = 0.0
        for i, tau in enumerate(self.quantiles):
            loss = loss + self.pinball(y_next, q[:, i:i+1], tau)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train(site_name, cfg):
    df = pd.read_parquet(data_dir("processed", f"{site_name}_features.parquet")).sort_index()
    feature_cols = [c for c in df.columns if c not in {"y","pv_kw"}]
    H = cfg["training"]["torch"]["hidden_size"]
    L = cfg["training"]["torch"]["num_layers"]
    D = cfg["training"]["torch"]["dropout"]
    LR = cfg["training"]["torch"]["learning_rate"]
    hist = cfg["data"]["history_hours"]
    hor = cfg["data"]["horizon_hours"]

    X, Y, ts = make_sequences(df, history=hist, horizon=hor, feature_cols=feature_cols, target_col="y")
    n = len(X); n_train = int(n*0.8)
    dl_train = DataLoader(SeqDataset(X[:n_train], Y[:n_train]), batch_size=cfg["training"]["torch"]["batch_size"], shuffle=True)
    dl_val = DataLoader(SeqDataset(X[n_train:], Y[n_train:]), batch_size=cfg["training"]["torch"]["batch_size"])

    model = GRUModel(input_size=X.shape[-1], hidden_size=H, num_layers=L, dropout=D, lr=LR)
    ckpt_cb = ModelCheckpoint(dirpath=models_dir(), save_top_k=1, monitor="val_loss", mode="min", filename="gru-{epoch:02d}")
    es_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = pl.Trainer(max_epochs=cfg["training"]["torch"]["max_epochs"], callbacks=[ckpt_cb, es_cb], logger=False)

    # Lightweight manual training/validation (simplified to avoid logger wiring)
    for epoch in range(cfg["training"]["torch"]["max_epochs"]):
        model.train()
        for xb, yb in dl_train:
            loss = model.training_step((xb, yb), 0)
            loss.backward()
            torch.optim.Adam(model.parameters(), lr=LR).step()
            model.zero_grad()

    # Save p50-like predictions (mean of next 24h) for app demo
    with torch.no_grad():
        X_all = torch.tensor(X, dtype=torch.float32)
        q_all = model(X_all)
        p50 = q_all[:, 1].numpy()
    pred_path = data_dir("processed", f"{site_name}_torch_preds.parquet")
    pd.DataFrame({"pred_p50_avg24h": p50}, index=ts).to_parquet(pred_path)
    print(f"Wrote torch predictions -> {pred_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site_name", required=True)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(configs_dir("train.yaml")))
    train(args.site_name, cfg)
