import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from load_encoder import sliding_window_embedding_inference, extract_volume_embeddings
import monai.transforms as transforms
from models.Uniformer import SSLEncoder


def setup_logger(log_file="training.log", log_to_console=True):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def coral_loss(logits, targets, K):
    """
    CORAL loss for ordinal classification with K discrete levels (0..K-1).
    logits: [N, (K-1)]  raw (no sigmoid)
    targets: [N]        integer in [0..K-1]
    We interpret logits[i] as "is class >= i+1?"
    Then do BCEWithLogits comparing to a 0/1 matrix:
      T[n,i] = 1 if targets[n] >= i+1 else 0
    """
    device = logits.device
    N = targets.size(0)
    # Build target matrix T => shape [N,(K-1)]
    T = torch.zeros((N, K - 1), device=device, dtype=torch.float)
    for i in range(K - 1):
        T[:, i] = (targets >= (i + 1)).float()

    bce = nn.BCEWithLogitsLoss()
    loss = bce(logits, T)
    return loss


def coral_predict(logits):
    probs = torch.sigmoid(logits)
    passed = (probs >= 0.5).sum(dim=1)  # how many thresholds were exceeded
    return passed


def ce_predict(logits):
    probs = torch.softmax(logits, dim=1)  # => [N, (K-1)]
    preds = torch.argmax(probs, dim=1)  # => [N]
    return preds


def mse_loss(logits, labels):
    """
    Builds a multi-hot [B, L, 27] then does MSE between
    sigmoid(logits) and the labels. Returns a scalar.
    """
    B, L, Q = logits.shape
    if Q != 27:
        raise ValueError(f"Expected 27 quadrants, got Q={Q}.")
    B_, L_, Q_ = labels.shape
    if Q_ != 27:
        raise ValueError(f"(labels) Expected 27 quadrants, got Q={Q_}.")

    # Convert logits -> probabilities
    probs = torch.sigmoid(logits)  # shape [B, L, 27]

    # Compute standard MSE
    loss_mse = F.mse_loss(probs, labels)
    return loss_mse


def bce_loss(logits, labels):
    B, L, Q = logits.shape
    if Q != 11:
        raise ValueError(f"(logits) Expected 11 regions, got Q={Q}.")
    B_, L_, Q_ = labels.shape
    if Q_ != 11:
        raise ValueError(f"(labels) Expected 11 regions, got Q={Q_}.")

    # Compute BCE with logits
    loss_bce = F.binary_cross_entropy_with_logits(logits, labels)
    return loss_bce


def dice_loss(logits, labels, eps: float = 1e-6):
    """
    Soft Dice loss for multi-label map of shape [B, L, Q]:
      logits: raw predictions           (no sigmoid)
      labels: binary {0,1} targets

    Returns the average Dice loss over all Q channels.
    """
    # apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)

    # sum over batch & sequence dims → per-channel totals
    # probs, labels: [B, L, Q]
    intersection = (probs * labels).sum(dim=(0, 1))  # [Q]
    cardinality = probs.sum(dim=(0, 1)) + labels.sum(dim=(0, 1))  # [Q]

    dice_score = (2. * intersection + eps) / (cardinality + eps)  # [Q]
    dice_loss = 1. - dice_score  # [Q]

    return dice_loss.mean()


def ce_loss(logits, labels):
    loss_ce = F.cross_entropy(logits, labels)
    return loss_ce


def compute_aux_loss(
    area_logits, shape_logits, satellite_logits, region_logits,
    area_targets, shape_targets, satellite_targets, region_targets,
    K_area=10, K_shape=7, K_satellite=5, keep_only_region=False,
    region_loss="bce"
    ):
    B = area_logits.size(0)

    area_2d = area_logits.view(B*4, (K_area-1))
    area_tgt_1d = area_targets.view(B*4)
    area_loss = coral_loss(area_2d, area_tgt_1d, K_area)

    shape_2d = shape_logits.view(B * 4, K_shape)
    shape_tgt_1d = shape_targets.view(B * 4)
    shape_loss = ce_loss(shape_2d, shape_tgt_1d)

    # solidity => [B*4,(K_solidity-1)]
    satellite_2d = satellite_logits.view(B * 4, K_satellite)
    satellite_tgt_1d = satellite_targets.view(B * 4)
    satellite_loss = ce_loss(satellite_2d, satellite_tgt_1d)

    if region_loss == "bce":
        region_loss = bce_loss(region_logits, region_targets)
    elif region_loss == "dice":
        region_loss = dice_loss(region_logits, region_targets)
    elif region_loss == "bce_dice":
        region_loss = bce_loss(region_logits, region_targets) + dice_loss(region_logits, region_targets)
    else:
        raise ValueError(f"Unknown bbox_loss: {region_loss}")

    if keep_only_region:
        total_loss = region_loss
    else:
        total_loss = area_loss + shape_loss + satellite_loss + region_loss
    loss_dict = {
        "area_loss": area_loss.item(),
        "shape_loss": shape_loss.item(),
        "satellite_loss": satellite_loss.item(),
        "region_loss": region_loss.item()
    }
    return total_loss, loss_dict


class AuxVisionDataset(Dataset):
    """Dataset for BraTS auxiliary tasks (e.g., tumor classification)"""
    def __init__(self, json_path, mode="train", transform=None, num_regions=11):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.num_regions = num_regions

        self.labels_order = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue",
            "Resection Cavity"
        ]
        # Load dictionary
        with open(json_path, "r") as f:
            self.data_list = json.load(f)

        self.samples = []
        for datum_dict in self.data_list:
            self.samples.append({
                "seg_file": datum_dict["seg_file"],
                "label_info": datum_dict["labels"]
            })
        self.transform = transforms.Compose([
            transforms.LoadImaged(keys=['image']),
            transforms.EnsureChannelFirstd(keys=['image']),
            transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True), 
            transforms.Orientationd(keys=['image'], axcodes='RAS'), 
            transforms.Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),  
            transforms.CropForegroundd(keys=['image'], source_key='image', margin=1),
            transforms.CenterSpatialCropd(keys=['image'], roi_size=(128, 128, 128)), # Add these transforms for consistent 128×128×128 volumes
            transforms.SpatialPadd(keys=['image'], spatial_size=(128, 128, 128), mode='constant', constant_values=0)
        ])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = self.samples[idx]
        seg_file = data["seg_file"]
        label_info = data["label_info"]

        # Load 4 modalities
        mod_t1c_file = seg_file.replace("seg", "t1c")
        mod_t1n_file = seg_file.replace("seg", "t1n")
        mod_t2f_file = seg_file.replace("seg", "t2f")
        mod_t2w_file = seg_file.replace("seg", "t2w")

        mod_t1c = self.transform({"image": mod_t1c_file})["image"]
        mod_t1n = self.transform({"image": mod_t1n_file})["image"]
        mod_t2f = self.transform({"image": mod_t2f_file})["image"]
        mod_t2w = self.transform({"image": mod_t2w_file})["image"]

        # Prepare area, extent, solidity, bbox
        # area, extent, solidity => [4], each label an integer in correct range
        area_vals = torch.zeros(4, dtype=torch.long)
        shape_vals = torch.zeros(4, dtype=torch.long)
        satellite_vals = torch.zeros(4, dtype=torch.long)
        region_vals = torch.zeros((4, self.num_regions), dtype=torch.float)

        for i, lbl in enumerate(self.labels_order):
            if lbl in label_info:
                metrics = label_info[lbl]
                area_vals[i] = metrics["area"]
                shape_vals[i] = metrics["shape"]
                satellite_vals[i] = metrics["satellite"]
                region_list = metrics["region"]
                for q in region_list:
                    if q < self.num_regions:
                        region_vals[i,q] = 1.0

        return {
            "t1c": mod_t1c,
            "t1n": mod_t1n,
            "t2f": mod_t2f,
            "t2w": mod_t2w,
            "area_targets": area_vals,         # [4]
            "shape_targets": shape_vals,     # [4]
            "satellite_targets": satellite_vals, # [4]
            "region_targets": region_vals,         # [4,regions]
            "seg_file": seg_file
        }


class BrainMVPClassifier(nn.Module):
    """BrainMVP encoder with classification head"""
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim=512,
        num_modalities=4,
        area_levels=8,
        shape_levels=7,
        satellite_levels=5,
        num_regions=11,
        reduce_embedding_dim=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.reduce_embedding_dim = reduce_embedding_dim
        
        # Calculate original embedding dimensions
        original_embedding_size = self.hidden_dim * num_modalities * (8*8*8)
        
        # Add optional dimensionality reduction layer
        if reduce_embedding_dim is not None:
            self.embedding_reducer = nn.Linear(original_embedding_size, reduce_embedding_dim)
            classifier_input_dim = reduce_embedding_dim
            print(f"Added embedding dimension reduction: {original_embedding_size} -> {reduce_embedding_dim}")
        else:
            self.embedding_reducer = None
            classifier_input_dim = original_embedding_size

        # Classification heads with updated input dimension
        self.area_head = nn.Linear(classifier_input_dim, 4 * (area_levels - 1))
        self.shape_head = nn.Linear(classifier_input_dim, 4 * shape_levels)
        self.satellite_head = nn.Linear(classifier_input_dim, 4 * satellite_levels)
        self.region_head = nn.Linear(classifier_input_dim, 4 * num_regions)

        self.area_levels = area_levels
        self.shape_levels = shape_levels
        self.satellite_levels = satellite_levels
        self.num_regions = num_regions

    def forward(self, mod1, mod2, mod3, mod4):
        B = mod1.size(0)

        # Extract features from each modality
        feats1 = extract_volume_embeddings(mod1, self.encoder)
        feats2 = extract_volume_embeddings(mod2, self.encoder)
        feats3 = extract_volume_embeddings(mod3, self.encoder)
        feats4 = extract_volume_embeddings(mod4, self.encoder)
        # Apply dimensionality reduction if specified
        if self.embedding_reducer is not None:
            feats1 = self.embedding_reducer(feats1)
            feats2 = self.embedding_reducer(feats2)
            feats3 = self.embedding_reducer(feats3)
            feats4 = self.embedding_reducer(feats4)


        feats = torch.cat([feats1, feats2, feats3, feats4], dim=1)  # [B, 512*4, 8*8*8]
        feats = feats.view(B, -1)
        area_feats = feats
        shape_feats = feats
        satellite_feats = feats
        region_feats = feats

        # area
        area_raw = self.area_head(area_feats)
        area_logits = area_raw.view(B, 4, (self.area_levels - 1))
        # shape
        shape_raw = self.shape_head(shape_feats)
        shape_logits = shape_raw.view(B, 4, self.shape_levels)
        # satellite
        satellite_raw = self.satellite_head(satellite_feats)
        satellite_logits = satellite_raw.view(B, 4, self.satellite_levels)
        # region
        region_raw = self.region_head(region_feats)
        region_logits = region_raw.view(B, 4, self.num_regions)

        return area_logits, shape_logits, satellite_logits, region_logits


@dataclass
class TrainingArguments:
    model_path: str = field(
        default="BrainMVP_uniformer.pt",
        metadata={"help": "Path to the pretrained BrainMVP checkpoint."}
    )
    freeze_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze BrainMVP encoder weights."}
    )
    
    batch_size: int = field(default=4, metadata={"help": "Batch size for training."})
    num_epochs: int = field(default=10, metadata={"help": "Number of training epochs."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    output_dir: str = field(default="./brats_aux_output", metadata={"help": "Output directory."})
    gpu: int = field(default=0, metadata={"help": "GPU ID to use."})
    tag: str = field(default="", metadata={"help": "Additional tag for output directory."})
    embed_dim: int = field(default=512, metadata={"help": "Embedding dimension."})
    
    # New parameter for dimensionality reduction
    reduce_embedding_dim: int = field(
        default=None, 
        metadata={"help": "Reduce embedding dimension to this size (e.g., 64). None for no reduction."}
    )


def main():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    output_dir = args.output_dir + f"_freeze_{args.freeze_encoder}_epochs_{args.num_epochs}" + args.tag
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(
        log_file=os.path.join(output_dir, "training.log"),
        log_to_console=True
    )
    train_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_aux_updated_v2_seed0.json"
    val_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_aux_updated_v2_seed0.json"
    test_file = "/local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_aux_updated_v2_seed0.json"

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")

    # Load BrainMVP model (following testing.py approach)
    logger.info(f"Loading BrainMVP encoder from {args.model_path}")

    # load BrainMVP encoder
    encoder = SSLEncoder(num_phase=1, initial_checkpoint=args.model_path)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    encoder_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '')
        if 'encoder.' in clean_key:
            encoder_key = clean_key.replace('encoder.', '')
            encoder_state_dict[encoder_key] = v
    encoder.load_state_dict(encoder_state_dict)
    print("model loaded!")
    
    # Freeze encoder if requested
    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen for training.")
    
    encoder = encoder.to(device)
    # Build classifier on top of BrainMVP
    model = BrainMVPClassifier(
        encoder=encoder,
        hidden_dim=args.embed_dim,
        reduce_embedding_dim=args.reduce_embedding_dim,  # Add this line
    ).to(device)

    logger.info(f"Model architecture:\n{model}")

    # Build datasets
    train_dataset = AuxVisionDataset(train_file)
    val_dataset = AuxVisionDataset(val_file)
    test_dataset = AuxVisionDataset(test_file)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # -----------------------------------------------------------
    # 3) Optimizer
    # -----------------------------------------------------------
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_val_score = -float("inf")  # higher = better
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # -----------------------------------------------------------
    # 4) Training Loop
    # -----------------------------------------------------------
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        area_loss = 0.0
        shape_loss = 0.0
        satellite_loss = 0.0
        region_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            mod1 = batch["t1c"].to(device)
            mod2 = batch["t1n"].to(device)
            mod3 = batch["t2f"].to(device)
            mod4 = batch["t2w"].to(device)

            area_targets = batch["area_targets"].to(device)         # [B,4]
            shape_targets = batch["shape_targets"].to(device)     # [B,4]
            satellite_targets = batch["satellite_targets"].to(device) # [B,4]
            region_targets = batch["region_targets"].to(device)         # [B,4,Q]

            optimizer.zero_grad()
            area_logits, shape_logits, satellite_logits, region_logits = model(mod1, mod2, mod3, mod4)

            loss, loss_dict = compute_aux_loss(
                area_logits, shape_logits, satellite_logits, region_logits,
                area_targets, shape_targets, satellite_targets, region_targets,
                K_area=8, K_shape=7, K_satellite=5)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            area_loss += loss_dict["area_loss"]
            shape_loss += loss_dict["shape_loss"]
            satellite_loss += loss_dict["satellite_loss"]
            region_loss += loss_dict["region_loss"]

        avg_train_loss = total_loss / len(train_loader)
        area_loss /= len(train_loader)
        shape_loss /= len(train_loader)
        satellite_loss /= len(train_loader)
        region_loss /= len(train_loader)

        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Area Loss: {area_loss:.4f} - Shape Loss: {shape_loss:.4f} - Satellite Loss: {satellite_loss:.4f} - Region Loss: {region_loss:.4f}")

        # Validation
        val_loss = 0.0
        area_val_loss = 0.0
        shape_val_loss = 0.0
        satellite_val_loss = 0.0
        region_val_loss = 0.0
        model.eval()
        running = {
            "loss": 0.0,
            "area_loss": 0.0, "shape_loss": 0.0,
            "sat_loss": 0.0,  "reg_loss": 0.0,
            # prediction buffers
            "area_pred": [], "area_tgt": [],
            "shape_pred": [], "shape_tgt": [],
            "sat_pred": [],   "sat_tgt": [],
            "iou_sum": 0.0,   "iou_cnt": 0
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                mod1, mod2, mod3, mod4 = (
                    batch["t1c"].to(device),
                    batch["t1n"].to(device),
                    batch["t2f"].to(device),
                    batch["t2w"].to(device)
                )
                area_tgt = batch["area_targets"].to(device)
                shape_tgt = batch["shape_targets"].to(device)
                sat_tgt   = batch["satellite_targets"].to(device)
                reg_tgt   = batch["region_targets"].to(device)

                area_logits, shape_logits, sat_logits, reg_logits = model(
                    mod1, mod2, mod3, mod4
                )

                # ------- loss
                loss, ldict = compute_aux_loss(
                    area_logits, shape_logits, sat_logits, reg_logits,
                    area_tgt, shape_tgt, sat_tgt, reg_tgt,
                    K_area=8, K_shape=7, K_satellite=5
                )
                running["loss"]       += loss.item()
                running["area_loss"]  += ldict["area_loss"]
                running["shape_loss"] += ldict["shape_loss"]
                running["sat_loss"]   += ldict["satellite_loss"]
                running["reg_loss"]   += ldict["region_loss"]

                # ------- predictions
                B = area_logits.size(0)

                # area (ordinal) → MAE
                area_pred = coral_predict(area_logits.view(B*4,7)).cpu()
                running["area_pred"].append(area_pred)
                running["area_tgt"].append(area_tgt.view(-1).cpu())

                # shape / satellite → accuracy
                shape_pred = ce_predict(shape_logits.view(B*4,7)).cpu()
                sat_pred   = ce_predict(sat_logits.view(B*4,5)).cpu()
                running["shape_pred"].append(shape_pred)
                running["shape_tgt"].append(shape_tgt.view(-1).cpu())
                running["sat_pred"].append(sat_pred)
                running["sat_tgt"].append(sat_tgt.view(-1).cpu())

                # IoU @ 0.5
                reg_pred_bin = (torch.sigmoid(reg_logits) >= 0.5)
                iou = hard_iou(reg_pred_bin.bool(), reg_tgt.bool())
                running["iou_sum"] += iou.sum().item()
                running["iou_cnt"] += iou.numel()

        # -------- aggregate metrics
        n_batches = len(val_loader)
        val_losses = {k: running[k] / n_batches for k in
                      ["loss", "area_loss", "shape_loss", "sat_loss", "reg_loss"]}

        area_mae   = mean_absolute_error(
                        torch.cat(running["area_tgt"]),
                        torch.cat(running["area_pred"])
                     )
        shape_acc  = (torch.cat(running["shape_tgt"]) ==
                      torch.cat(running["shape_pred"])).float().mean().item()
        sat_acc    = (torch.cat(running["sat_tgt"]) ==
                      torch.cat(running["sat_pred"])).float().mean().item()
        mean_iou   = running["iou_sum"] / running["iou_cnt"]

        # ------ combined “average score”
        area_score = 1 - area_mae / 7.0          # 7 = max ordinal gap
        val_score  = (area_score + shape_acc + sat_acc + mean_iou) / 4.0

        area_val_loss /= len(val_loader)
        shape_val_loss /= len(val_loader)
        satellite_val_loss /= len(val_loader)
        region_val_loss /= len(val_loader)
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} - Area Loss: {area_val_loss:.4f} - Shape Loss: {shape_val_loss:.4f} - Satellite Loss: {satellite_val_loss:.4f} - Region Loss: {region_val_loss:.4f}")

        logger.info(
            f"Epoch {epoch+1} - ValLoss {val_losses['loss']:.4f} | "
            f"MAE {area_mae:.3f} | ShapeAcc {shape_acc:.3f} | "
            f"SatAcc {sat_acc:.3f} | IoU@0.5 {mean_iou:.3f} | "
            f"AvgScore {val_score:.3f}"
        )

        # ----- save best on AvgScore
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best avg-score {val_score:.3f}  ➜  saved to {best_model_path}")

    logger.info("Training complete.")

    # -----------------------------------------------------------
    # 5) Test Evaluation
    # -----------------------------------------------------------
    logger.info("Evaluating on the test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_area_preds = []
    all_area_tgts  = []
    all_shape_preds = []
    all_shape_tgts  = []
    all_satellite_preds = []
    all_satellite_tgts  = []
    thresh = [-.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, .9, 1.0, 1.1]
    thresh_iou_list = [dict() for _ in range(4)]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            mod1 = batch["t1c"].to(device)
            mod2 = batch["t1n"].to(device)
            mod3 = batch["t2f"].to(device)
            mod4 = batch["t2w"].to(device)

            area_targets = batch["area_targets"].to(device)         # [B,4]
            shape_targets = batch["shape_targets"].to(device)     # [B,4]
            satellite_targets = batch["satellite_targets"].to(device) # [B,4]
            region_targets = batch["region_targets"].to(device)         # [B,4,Q]

            area_logits, shape_logits, satellite_logits, region_logits = model(mod1, mod2, mod3, mod4)

            B = area_logits.size(0)
            area_2d = area_logits.view(B*4, 7)              # => [B*4,9]
            area_pred_1d = coral_predict(area_2d)     # => [B*4]
            area_tgt_1d = area_targets.view(-1)             # => [B*4]
            all_area_preds.append(area_pred_1d.cpu())
            all_area_tgts.append(area_tgt_1d.cpu())

            shape_2d = shape_logits.view(B*4, 7)
            shape_pred_1d = ce_predict(shape_2d)  # => [B*4]
            shape_tgt_1d = shape_targets.view(-1)
            all_shape_preds.append(shape_pred_1d.cpu())
            all_shape_tgts.append(shape_tgt_1d.cpu())

            satellite_2d = satellite_logits.view(B*4, 5)
            satellite_pred_1d = ce_predict(satellite_2d)
            satellite_tgt_1d = satellite_targets.view(-1)
            all_satellite_preds.append(satellite_pred_1d.cpu())
            all_satellite_tgts.append(satellite_tgt_1d.cpu())

            # 4) BBox => IoU
            region_prob = torch.sigmoid(region_logits)  # shape [B,4,Q], in [0..1]
            for thresh_ in thresh:
                region_pred = (region_prob >= thresh_).float()  # hard threshold -> 0/1
                intersection = (region_pred * region_targets).sum(dim=2)  # [B,4]
                union = (region_pred + region_targets - region_pred * region_targets).sum(dim=2)  # [B,4]
                iou = (intersection + 1e-7)/ (union + 1e-7)  # [B,4]
                for label_index in range(4):
                    if thresh_ not in thresh_iou_list[label_index]:
                        thresh_iou_list[label_index][thresh_] = []
                    thresh_iou_list[label_index][thresh_].append(iou.cpu()[:, label_index])

    # stack predictions
    area_preds = torch.cat(all_area_preds).numpy()
    area_tgts  = torch.cat(all_area_tgts).numpy()
    shape_preds = torch.cat(all_shape_preds).numpy()
    shape_tgts  = torch.cat(all_shape_tgts).numpy()
    satellite_preds = torch.cat(all_satellite_preds).numpy()
    satellite_tgts  = torch.cat(all_satellite_tgts).numpy()
    thresh_mean_iou = [dict() for _ in range(4)]
    for label_index in range(4):
        for thresh_ in thresh:
            iou_tensor = torch.cat(thresh_iou_list[label_index][thresh_], dim=0) # shape [N*B]
            thresh_mean_iou[label_index][thresh_] = iou_tensor.mean().item()
    # Simple metrics: Mean Absolute Error for ordinal
    area_mae = mean_absolute_error(area_tgts, area_preds)
    shape_acc = (shape_tgts == shape_preds).mean()
    satellite_acc = (satellite_tgts == satellite_preds).mean()

    logger.info("========== TEST RESULTS ==========")
    logger.info(f"Area MAE:     {area_mae:.4f}")
    logger.info(f"Shape Acc:   {shape_acc:.4f}")
    logger.info(f"Satellite Acc: {satellite_acc:.4f}")
    logger.info(f"@0.50 Mean IoU: {np.mean([thresh_mean_iou[label_index][0.5] for label_index in range(4)]):.4f}")
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()