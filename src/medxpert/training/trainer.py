from __future__ import annotations
import json, logging, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.medxpert import MEDXPERT, ForwardAux
from .losses import compute_losses
from .schedulers import build_optimizer_scheduler


log = logging.getLogger("medxpert.trainer")


@dataclass
class TrainState:
    best_metric: float
    best_epoch: int
    history: List[Dict[str, float]]


class Trainer:
    """Minimal, config-driven trainer.

    Research-integrity contract:
      - `train_loader` drives optimizer updates.
      - `val_loader` drives checkpoint selection, early stopping, and all
        reported in-training metrics.
      - The TEST loader is NEVER passed to this class. Final test evaluation
        lives in `scripts/evaluate.py` and loads the frozen checkpoint.

    Writes under out_dir:
        best.pt, last.pt, history.json, training.log, val_predictions.npz
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        out_dir: Path | str,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

        tr = cfg["training"]
        self.epochs = int(tr.get("epochs", 1))
        self.grad_clip = float(tr.get("gradient_clip", 0.0))
        self.use_amp = bool(tr.get("mixed_precision", False)) and device.type == "cuda"
        self.monitor = str(tr.get("monitor_metric", "f1_macro"))
        self.monitor_mode = str(tr.get("monitor_mode", "max"))
        self.patience = int(tr.get("early_stopping_patience", 0))

        self.losses_cfg = cfg.get("losses", [{"name": "cross_entropy", "weight": 1.0}])
        self.base_loss_cfg = dict(cfg.get("loss", {}))

        self.optimizer, self.scheduler = build_optimizer_scheduler(
            self.model, cfg, steps_per_epoch=max(1, len(train_loader))
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Keep the latest val logits/labels on disk so post-hoc calibration
        # (temperature scaling) in evaluate.py can fit without ever touching
        # the test set.
        self._last_val_logits: Optional[np.ndarray] = None
        self._last_val_labels: Optional[np.ndarray] = None

    def _setup_logging(self) -> None:
        import sys
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        class FlushingFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        fh = FlushingFileHandler(self.out_dir / "training.log", mode="w", encoding="utf-8")
        fh.setFormatter(formatter)

        fh_txt = FlushingFileHandler(self.out_dir / "training_live.txt", mode="w", encoding="utf-8")
        fh_txt.setFormatter(formatter)

        fh_root = FlushingFileHandler(Path("training_live.txt"), mode="w", encoding="utf-8")
        fh_root.setFormatter(formatter)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        log.handlers = [fh, fh_txt, fh_root, sh]
        log.setLevel(logging.INFO)

    def _step(self, batch):
        imgs, labels, _paths = batch
        imgs = imgs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logits, aux = self.model(imgs)
            parts = compute_losses(self.losses_cfg, logits, labels, aux,
                                   self.model, self.base_loss_cfg)
        return logits, aux, labels, parts

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n = 0
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # log every ~10% of epoch
        for step, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad(set_to_none=True)
            logits, aux, labels, parts = self._step(batch)
            loss = parts["total"]
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            if isinstance(self.model, MEDXPERT) and aux is not None:
                self.model.update_memory(aux, labels)
            batch_sz = labels.size(0)
            total_loss += float(loss.detach().cpu()) * batch_sz
            n += batch_sz

            if step % log_interval == 0 or step == total_batches:
                curr_lr = self.optimizer.param_groups[0]["lr"]
                avg_loss = total_loss / max(1, n)
                log.info(
                    f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                    f"Step [{step:03d}/{total_batches:03d}] ({100.0 * step / total_batches:5.1f}%) | "
                    f"Batch Loss: {float(loss.detach().cpu()):.4f} | "
                    f"Running Avg: {avg_loss:.4f} | LR: {curr_lr:.2e}"
                )
        return {"train_loss": total_loss / max(1, n)}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        from ..evaluation.metrics import compute_classification_metrics
        self.model.eval()
        all_logits: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        total_loss = 0.0
        n = 0
        for batch in self.val_loader:
            logits, _aux, labels, parts = self._step(batch)
            total_loss += float(parts["total"].detach().cpu()) * labels.size(0)
            n += labels.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        self._last_val_logits = logits_np
        self._last_val_labels = labels_np
        metrics = compute_classification_metrics(logits_np, labels_np)
        metrics["val_loss"] = total_loss / max(1, n)
        return metrics

    def fit(self) -> TrainState:
        best = -float("inf") if self.monitor_mode == "max" else float("inf")
        best_epoch = -1
        history: List[Dict[str, float]] = []
        stale = 0
        log.info(f"=== Starting Training: {self.epochs} Epochs on {self.device} ===")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            log.info(f"--- Epoch [{epoch:02d}/{self.epochs:02d}] Starting ---")
            tr_metrics = self._train_epoch(epoch)
            val_metrics = self._validate()
            elapsed = time.time() - t0
            val_acc = val_metrics.get("accuracy", 0.0)
            val_f1 = val_metrics.get("f1_macro", 0.0)
            val_loss = val_metrics.get("val_loss", 0.0)
            row = {"epoch": epoch, **tr_metrics, **val_metrics,
                   "elapsed_s": round(elapsed, 2)}
            history.append(row)

            is_best = False
            if val_metrics and self.monitor in val_metrics:
                v = float(val_metrics[self.monitor])
                improved = (v > best) if self.monitor_mode == "max" else (v < best)
                if improved:
                    best = v
                    best_epoch = epoch
                    stale = 0
                    is_best = True
                    self._save(self.out_dir / "best.pt", epoch, row)
                else:
                    stale += 1

            best_tag = f" [NEW BEST {self.monitor}={best:.4f}]" if is_best else ""
            log.info(
                f"--- Epoch [{epoch:02d}/{self.epochs:02d}] Finished in {elapsed:.1f}s | "
                f"Train Loss: {tr_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}{best_tag} ---"
            )
            self._save(self.out_dir / "last.pt", epoch, row)
            if self.patience and stale >= self.patience:
                log.info(f"Early-stopping at epoch {epoch} (no improvement for {stale} epochs).")
                break
        with (self.out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=float)
        # Save frozen val logits+labels for post-hoc calibration fitting.
        if self._last_val_logits is not None:
            np.savez(self.out_dir / "val_predictions.npz",
                     logits=self._last_val_logits,
                     labels=self._last_val_labels)
        return TrainState(best_metric=best if best != -float("inf") else float("nan"),
                          best_epoch=best_epoch, history=history)

    def _save(self, path: Path, epoch: int, row: Dict[str, float]) -> None:
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": row,
        }, path)
