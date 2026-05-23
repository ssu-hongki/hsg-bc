import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ProcessedDemoDataset(Dataset):
    def __init__(self, data_dir="data_processed", normalize=True):
        self.data_dir = data_dir
        self.normalize = normalize

        paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if len(paths) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        states_list = []
        subgoals_list = []
        actions_list = []
        subgoal_ids_list = []

        for path in paths:
            data = np.load(path, allow_pickle=True)

            if "states" not in data.files:
                raise KeyError(f"{path} does not contain 'states'")
            if "subgoals" not in data.files:
                raise KeyError(f"{path} does not contain 'subgoals'")
            if "actions" not in data.files:
                raise KeyError(f"{path} does not contain 'actions'")

            states = data["states"].astype(np.float32)
            subgoals = data["subgoals"].astype(np.float32)
            actions = data["actions"].astype(np.float32)

            if "subgoal_ids" in data.files:
                subgoal_ids = data["subgoal_ids"].astype(np.int64)
            else:
                subgoal_ids = np.argmax(subgoals, axis=1).astype(np.int64)

            n = min(len(states), len(subgoals), len(actions), len(subgoal_ids))

            if n == 0:
                continue

            states_list.append(states[:n])
            subgoals_list.append(subgoals[:n])
            actions_list.append(actions[:n])
            subgoal_ids_list.append(subgoal_ids[:n])

        if len(states_list) == 0:
            raise RuntimeError("No valid samples loaded.")

        self.states = np.concatenate(states_list, axis=0).astype(np.float32)
        self.subgoals = np.concatenate(subgoals_list, axis=0).astype(np.float32)
        self.actions = np.concatenate(actions_list, axis=0).astype(np.float32)
        self.subgoal_ids = np.concatenate(subgoal_ids_list, axis=0).astype(np.int64)

        self.inputs = np.concatenate([self.states, self.subgoals], axis=1).astype(np.float32)

        if self.normalize:
            self.input_mean = self.inputs.mean(axis=0, keepdims=True)
            self.input_std = self.inputs.std(axis=0, keepdims=True)
            self.input_std[self.input_std < 1e-6] = 1.0

            self.action_mean = self.actions.mean(axis=0, keepdims=True)
            self.action_std = self.actions.std(axis=0, keepdims=True)
            self.action_std[self.action_std < 1e-6] = 1.0

            self.inputs_norm = (self.inputs - self.input_mean) / self.input_std
            self.actions_norm = (self.actions - self.action_mean) / self.action_std
        else:
            self.input_mean = np.zeros((1, self.inputs.shape[1]), dtype=np.float32)
            self.input_std = np.ones((1, self.inputs.shape[1]), dtype=np.float32)
            self.action_mean = np.zeros((1, self.actions.shape[1]), dtype=np.float32)
            self.action_std = np.ones((1, self.actions.shape[1]), dtype=np.float32)

            self.inputs_norm = self.inputs
            self.actions_norm = self.actions

        self.inputs_tensor = torch.tensor(self.inputs_norm, dtype=torch.float32)
        self.actions_tensor = torch.tensor(self.actions_norm, dtype=torch.float32)
        self.subgoal_ids_tensor = torch.tensor(self.subgoal_ids, dtype=torch.long)

        print("========================================")
        print(f"Loaded files: {len(paths)}")
        print(f"Total samples: {len(self.inputs_tensor)}")
        print(f"State dim: {self.states.shape[1]}")
        print(f"Subgoal dim: {self.subgoals.shape[1]}")
        print(f"Input dim: {self.inputs.shape[1]}")
        print(f"Action dim: {self.actions.shape[1]}")
        print(f"Normalize: {self.normalize}")
        print("Subgoal distribution:")
        for i in range(self.subgoals.shape[1]):
            count = int((self.subgoal_ids == i).sum())
            print(f"  {i:02d}: {count}")
        print("========================================")

    def __len__(self):
        return len(self.inputs_tensor)

    def __getitem__(self, idx):
        return self.inputs_tensor[idx], self.actions_tensor[idx], self.subgoal_ids_tensor[idx]


class MLPLowLevelPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.05):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)
            losses.append(loss.item())

    if len(losses) == 0:
        return 0.0

    return float(np.mean(losses))


def train():
    set_seed(42)

    data_dir = "data_processed"
    save_path = "mlp_low_level_policy_final.pth"

    batch_size = 128
    epochs = 200
    lr = 1e-4
    val_ratio = 0.15
    hidden_dim = 256
    dropout = 0.05
    normalize = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    dataset = ProcessedDemoDataset(
        data_dir=data_dir,
        normalize=normalize,
    )

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    input_dim = dataset.inputs.shape[1]
    output_dim = dataset.actions.shape[1]
    state_dim = dataset.states.shape[1]
    subgoal_dim = dataset.subgoals.shape[1]

    model = MLPLowLevelPolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
        val_loss = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.8f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "state_dim": state_dim,
                    "subgoal_dim": subgoal_dim,
                    "output_dim": output_dim,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "normalize": normalize,
                    "input_mean": dataset.input_mean.astype(np.float32),
                    "input_std": dataset.input_std.astype(np.float32),
                    "action_mean": dataset.action_mean.astype(np.float32),
                    "action_std": dataset.action_std.astype(np.float32),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                save_path,
            )

    print("========================================")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Saved: {save_path}")
    print("========================================")


if __name__ == "__main__":
    train()