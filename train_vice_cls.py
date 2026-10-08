from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from utils.GLOBAL import ROOT_PATH

from dataset.vllm import EVQA, EIC, VLKEB, WaterBird, WaterBirdV2

IN_SCOPE_ROLES = {"reliability", "text_rephrase", "image_rephrase"}

_DATASET_CFG = {
    "EVQA": {
        "cls": EVQA,
        "train_json": "data/easy-edit-mm/vqa/vqa_train.json",
        "eval_json":  "data/easy-edit-mm/vqa/vqa_eval.json",
        "img_dir":    "data/easy-edit-mm/images",
        "cache_dir":  "data/easy-edit-mm/vqa",
    },
    "EIC": {
        "cls": EIC,
        "train_json": "data/easy-edit-mm/caption/caption_train_edit.json",
        "eval_json":  "data/easy-edit-mm/caption/caption_eval_edit.json",
        "img_dir":    "data/easy-edit-mm/images",
        "cache_dir":  "data/easy-edit-mm/caption",
    },
    "VLKEB": {
        "cls": VLKEB,
        "train_json": "data/VLKEB/train.json",
        "eval_json":  "data/VLKEB/eval.json",
        "img_dir":    "data/VLKEB/mmkb_images",
        "cache_dir":  "data/VLKEB",
    },
    "WATERBIRD": {
        "cls": WaterBird,
        "train_json": "data/WaterBird/edit_annotations_truelabel_balanced.json",
        "eval_json":  "data/WaterBird/edit_annotations_truelabel_balanced.json",
        "img_dir":    "data/WaterBird",
        "cache_dir":  "data/WaterBird",
        "split":      "train",
        "eval_split": "test",
    },
    "WATERBIRDV2": {
        "cls": WaterBirdV2,
        "train_json": "data/WaterBird-v2.0/edit_annotations.json",
        "eval_json":  "data/WaterBird-v2.0/edit_annotations.json",
        "img_dir":    "",
        "cache_dir":  "data/WaterBird-v2.0",
        "split":      "train",
        "eval_split": "test",
    },
}


def add_example(examples: List[Dict], group: List[int], item: Dict, sample_id: int, label: int, role: str):
    group.append(len(examples))
    image = item.get("image")
    image_path = item.get("image_path") if item.get("image_path") is not None else (image if isinstance(image, str) else None)
    examples.append({
        "image": image,
        "image_path": image_path,
        "prompt": item["prompt"],
        "sample_id": sample_id,
        "label": label,
        "role": role,
    })


def build_scope_examples(data) -> Tuple[List[Dict], List[List[int]]]:
    data_items = getattr(data, "data_with_img", data)
    examples, groups = [], []
    next_locality_label = len(data_items)

    for sample_id, data_item in enumerate(data_items):
        group = []
        add_example(examples, group, data_item["requests"][0], sample_id, sample_id, "reliability")

        for role in ("text_rephrase", "image_rephrase"):
            for item in data_item.get("generality", {}).get(role, []):
                add_example(examples, group, item, sample_id, sample_id, role)

        for role in ("text_loc", "image_loc"):
            for item in data_item.get("locality", {}).get(role, []):
                add_example(examples, group, item, sample_id, next_locality_label, role)
                next_locality_label += 1

        groups.append(group)
    return examples, groups


@torch.no_grad()
def encode_scope_features(examples: List[Dict], device: str, batch_size: int, clip_model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    feature_chunks, label_chunks = [], []
    for start in tqdm(range(0, len(examples), batch_size), desc="Encoding CLIP scope features"):
        batch = examples[start:start + batch_size]
        texts = [x["prompt"] for x in batch]
        text_inputs = processor(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        images, positions = [], []
        for idx, example in enumerate(batch):
            image = example["image"]
            if image is None:
                continue
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            images.append(image)
            positions.append(idx)

        image_features = torch.zeros(len(batch), text_features.shape[-1], dtype=text_features.dtype, device=device)
        if images:
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            encoded_images = model.get_image_features(**image_inputs)
            encoded_images = encoded_images / encoded_images.norm(p=2, dim=-1, keepdim=True)
            for idx, image_feature in zip(positions, encoded_images):
                image_features[idx] = image_feature

        feature_chunks.append(torch.cat([text_features, image_features], dim=-1).cpu())
        label_chunks.append(torch.tensor([x["label"] for x in batch], dtype=torch.long))

    return torch.cat(feature_chunks), torch.cat(label_chunks)


def build_projection_head(feature_dim: int, projection_dim: int, device: str):
    return torch.nn.Sequential(
        torch.nn.Linear(feature_dim, feature_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(feature_dim, projection_dim),
    ).to(device)


def supervised_contrastive_loss(reps: torch.Tensor, labels: torch.Tensor, temperature: float):
    logits = reps @ reps.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, -1e9)

    pos_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = (pos_mask.float() * log_prob).sum(dim=1)[valid] / pos_counts[valid]
    return -loss.mean()


def anchor_retrieval_loss(reps_norm: torch.Tensor, group_ids: torch.Tensor,
                          anchor_mask: torch.Tensor, in_scope_mask: torch.Tensor,
                          temperature: float):
    """
    CE over anchors: each in-scope (non-anchor) example must point to its own anchor.

    reps_norm:   [B, D] L2-normalized features
    group_ids:   [B]    group label (0..n_edits-1 for in-scope, >=n_edits for locality)
    anchor_mask: [B]    True for reliability anchor examples
    in_scope_mask:[B]   True for in-scope examples (reliability + rephrases)
    """
    valid = in_scope_mask & ~anchor_mask
    if not valid.any():
        return reps_norm.sum() * 0.0

    anchor_feats = reps_norm[anchor_mask]          # [n_anchors, D]
    anchor_groups = group_ids[anchor_mask]         # [n_anchors]

    sim = reps_norm[valid] @ anchor_feats.T / temperature  # [n_valid, n_anchors]
    # Target: for each valid example, which anchor column has the matching group_id
    target = (anchor_groups[None, :] == group_ids[valid, None]).float().argmax(dim=1)
    return torch.nn.functional.cross_entropy(sim, target)


def locality_margin_loss(reps_norm: torch.Tensor, anchor_mask: torch.Tensor,
                        in_scope_mask: torch.Tensor, margin: float):
    """
    Hinge loss: push locality max_sim to any anchor below `margin`.
    """
    locality_mask = ~in_scope_mask
    if not locality_mask.any():
        return reps_norm.sum() * 0.0

    anchor_feats = reps_norm[anchor_mask]               # [n_anchors, D]
    sim = reps_norm[locality_mask] @ anchor_feats.T     # [n_loc, n_anchors]
    max_sim = sim.max(dim=1).values                     # [n_loc]
    return torch.clamp(max_sim - margin, min=0).mean()


def train_vice_scope_classifier(
    features: torch.Tensor, labels: torch.Tensor, groups: List[List[int]],
    save_path: str, device: str,
    clip_model_name: str = "openai/clip-vit-base-patch32", epochs: int = 5, batch_size: int = 32,
    lr: float = 1e-3, temperature: float = 0.07, projection_dim: int = 256,
    loc_margin: float = 0.5, loc_lambda: float = 1.0,
):
    """
    Train projection head for VICE scope retrieval.

    Loss = anchor_retrieval_loss + loc_lambda * locality_margin_loss
      - anchor_retrieval_loss: CE over anchors — in-scope samples must retrieve own anchor
      - locality_margin_loss: hinge — locality max_sim to any anchor is pushed below `loc_margin`
    """
    projection = build_projection_head(features.shape[-1], projection_dim, device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(1993)

    # Precompute anchor mask: reliability example (first in each group) is the anchor
    anchor_mask_all = torch.zeros(len(features), dtype=torch.bool)
    for g in groups:
        anchor_mask_all[g[0]] = True
    in_scope_mask_all = labels < len(groups)

    last_ret_loss, last_loc_loss = 0.0, 0.0
    for epoch in range(epochs):
        order = torch.randperm(len(groups), generator=generator).tolist()
        ret_losses, loc_losses = [], []
        progress = tqdm(range(0, len(order), batch_size), desc=f"Training epoch {epoch + 1}/{epochs}")
        for start in progress:
            batch_indices = []
            for group_idx in order[start:start + batch_size]:
                batch_indices.extend(groups[group_idx])

            batch_indices_t = torch.tensor(batch_indices, dtype=torch.long)
            batch_feats = features[batch_indices].to(device)
            batch_group_ids = labels[batch_indices].to(device)
            batch_anchor = anchor_mask_all[batch_indices_t].to(device)
            batch_in_scope = in_scope_mask_all[batch_indices_t].to(device)

            reps = projection(batch_feats)
            reps_norm = reps / reps.norm(p=2, dim=-1, keepdim=True)

            ret_loss = anchor_retrieval_loss(reps_norm, batch_group_ids, batch_anchor,
                                             batch_in_scope, temperature)
            loc_loss = locality_margin_loss(reps_norm, batch_anchor, batch_in_scope, loc_margin)
            total_loss = ret_loss + loc_lambda * loc_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            last_ret_loss = float(ret_loss.item())
            last_loc_loss = float(loc_loss.item())
            ret_losses.append(last_ret_loss)
            loc_losses.append(last_loc_loss)
            progress.set_postfix(ret=f"{last_ret_loss:.4f}", loc=f"{last_loc_loss:.4f}")

        if ret_losses:
            print(f"Epoch {epoch + 1}: ret={sum(ret_losses)/len(ret_losses):.4f}  "
                  f"loc={sum(loc_losses)/len(loc_losses):.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "projection_state_dict": projection.state_dict(),
        "feature_dim": features.shape[-1],
        "projection_dim": projection_dim,
        "temperature": temperature,
        "loc_margin": loc_margin,
        "loc_lambda": loc_lambda,
        "clip_model_name": clip_model_name,
        "num_examples": int(features.shape[0]),
        "num_edit_samples": len(groups),
        "last_ret_loss": last_ret_loss,
        "last_loc_loss": last_loc_loss,
    }, save_path)
    print(f"Saved VICE scope classifier to {save_path}")
    return projection


def evaluate_reliability_anchor(examples: List[Dict], features: torch.Tensor, groups: List[List[int]], projection, device: str) -> Dict:
    """
    以每个编辑样本的 reliability 为 anchor，检验：
    - in-scope（reliability/text_rephrase/image_rephrase）: 与所有 reliability anchor 的相似度中，最大的那个应该是自己的 anchor
    - locality（text_loc/image_loc）: 与所有 anchor 的相似度都应该低（统计分布，不判对错）
    """
    projection.eval()
    with torch.no_grad():
        reps = projection(features.to(device))
        reps = reps / reps.norm(p=2, dim=-1, keepdim=True)

    # 构建 example_index -> group_id 映射
    n_examples = len(examples)
    example_to_group = torch.full((n_examples,), -1, dtype=torch.long, device=device)
    for gid, indices in enumerate(groups):
        for idx in indices:
            example_to_group[idx] = gid
    assert (example_to_group >= 0).all(), "Some examples not assigned to any group"

    # reliability anchor: 每个 group 的第一个 example
    rel_indices = [g[0] for g in groups]
    rel_features = reps[rel_indices]                        # [n_groups, feat_dim]
    rel_features = rel_features / rel_features.norm(p=2, dim=-1, keepdim=True)

    # 相似度矩阵: [n_examples, n_groups]
    sim = reps @ rel_features.T
    max_sim, pred_group = sim.max(dim=-1)                   # 每个 example 最像哪个 anchor
    own_sim = sim[torch.arange(n_examples), example_to_group]  # 与自己 anchor 的相似度

    # ——— in-scope: 判定正确性（最大相似度是否指向自己的 anchor） ———
    in_scope_mask = torch.tensor([x["role"] in IN_SCOPE_ROLES for x in examples], device=device)
    in_scope_correct = torch.zeros(n_examples, dtype=torch.bool, device=device)
    in_scope_correct[in_scope_mask] = (pred_group[in_scope_mask] == example_to_group[in_scope_mask])

    # ——— locality: 不判对错，只统计相似度分布 ———
    locality_max_sim = max_sim[~in_scope_mask]               # locality 与最近 anchor 的相似度
    p95 = float(locality_max_sim.kthvalue(max(2, int(0.95 * locality_max_sim.numel())), dim=0).values.item()) if locality_max_sim.numel() > 1 else 0.0
    p90 = float(locality_max_sim.kthvalue(max(2, int(0.90 * locality_max_sim.numel())), dim=0).values.item()) if locality_max_sim.numel() > 1 else 0.0
    p99 = float(locality_max_sim.kthvalue(max(2, int(0.99 * locality_max_sim.numel())), dim=0).values.item()) if locality_max_sim.numel() > 1 else 0.0

    # 按角色统计
    role_metrics = {}
    for role in sorted({x["role"] for x in examples}):
        idx = [i for i, x in enumerate(examples) if x["role"] == role]
        info = {
            "n": len(idx),
            "own_sim_mean": float(own_sim[idx].mean().item()),
            "max_sim_mean": float(max_sim[idx].mean().item()),
        }
        if role in IN_SCOPE_ROLES:
            info["acc"] = float(in_scope_correct[[i for i, x in enumerate(examples) if x["role"] == role]].float().mean().item()) if [i for i, x in enumerate(examples) if x["role"] == role] else 0.0
        role_metrics[role] = info

    # 总体统计
    in_scope_sim = own_sim[in_scope_mask]
    locality_sim = own_sim[~in_scope_mask]
    return {
        "in_scope_acc": float(in_scope_correct[in_scope_mask].float().mean().item()),
        "in_scope_own_sim_mean": float(in_scope_sim.mean().item()) if in_scope_sim.numel() else 0.0,
        "locality_own_sim_mean": float(locality_sim.mean().item()) if locality_sim.numel() else 0.0,
        "margin": float((in_scope_sim.mean() - locality_sim.mean()).item()) if in_scope_sim.numel() and locality_sim.numel() else 0.0,
        "locality_max_sim_p90": p90,
        "locality_max_sim_p95": p95,
        "locality_max_sim_p99": p99,
        "locality_max_sim_mean": float(locality_max_sim.mean().item()) if locality_max_sim.numel() else 0.0,
        "role_metrics": role_metrics,
    }


def print_scope_eval(name: str, result: Dict):
    print(f"{name} | in_scope_acc={result['in_scope_acc']:.4f}")
    print(f"       own_sim: in_scope={result['in_scope_own_sim_mean']:.4f} | locality={result['locality_own_sim_mean']:.4f} | margin={result['margin']:.4f}")
    print(f"       locality max_sim: mean={result['locality_max_sim_mean']:.4f} | p90={result['locality_max_sim_p90']:.4f} | p95={result['locality_max_sim_p95']:.4f} | p99={result['locality_max_sim_p99']:.4f}")
    for role, m in result["role_metrics"].items():
        if "acc" in m:
            print(f"       {role}: n={m['n']} | acc={m['acc']:.4f} | own_sim={m['own_sim_mean']:.4f} | max_sim={m['max_sim_mean']:.4f}")
        else:
            print(f"       {role}: n={m['n']} | own_sim={m['own_sim_mean']:.4f} | max_sim={m['max_sim_mean']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train VICE scope classifier")
    parser.add_argument("--data_name", type=str, default="evqa", choices=["evqa", "eic", "vlkeb", "waterbird", "waterv2"],
                        help="Dataset name")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=80)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--loc_margin", type=float, default=0.5)
    parser.add_argument("--loc_lambda", type=float, default=1.0)
    args = parser.parse_args()

    # Base config
    data_name = args.data_name.upper()
    if data_name == "WATERV2":
        data_name = "WATERBIRDV2"
    device = f"cuda:{args.device}"
    clip_model_name = args.clip_model_name
    feature_batch_size = args.feature_batch_size

    # Scope classifier config
    projection_dim = args.projection_dim

    # Scope classifier training config
    train_epochs = args.train_epochs
    train_batch_size = args.train_batch_size
    train_lr = args.train_lr
    temperature = args.temperature
    loc_margin = args.loc_margin
    loc_lambda = args.loc_lambda

    # 参数编码到 checkpoint 名称，参数不变时自动跳过训练
    param_tag = (f"ep{train_epochs}_bs{train_batch_size}_lr{str(train_lr).replace('.','')}"
                 f"_tmp{str(temperature).replace('.','')}_proj{projection_dim}"
                 f"_m{str(loc_margin).replace('.','')}_lm{str(loc_lambda).replace('.','')}")
    cls_ckpt_path = os.path.join(ROOT_PATH, "checkpoints", f"vice_scope_cls_{data_name}_{param_tag}.pt")

    # ========== 1. 加载数据 ==========
    print(f"Step 1/5: load {data_name} train and test data")

    cfg = _DATASET_CFG[data_name]
    DatasetCls = cfg["cls"]
    split_kwargs = {}
    if "split" in cfg:
        split_kwargs = {"split": cfg["split"], "data_n": None}
        eval_split_kwargs = {"split": cfg.get("eval_split", cfg["split"]), "data_n": None}
    else:
        split_kwargs = {"data_n": None}
        eval_split_kwargs = {"data_n": None}
    train_data = DatasetCls(os.path.join(ROOT_PATH, cfg["train_json"]), os.path.join(ROOT_PATH, cfg["img_dir"]), **split_kwargs)
    test_data  = DatasetCls(os.path.join(ROOT_PATH, cfg["eval_json"]),  os.path.join(ROOT_PATH, cfg["img_dir"]), **eval_split_kwargs)

    # ========== 2. 构建 scope examples ==========
    print("Step 2/5: build scope examples")
    train_examples, train_groups = build_scope_examples(train_data)
    test_examples, test_groups = build_scope_examples(test_data)
    assert train_examples and test_examples, "Train/test scope examples must be non-empty."

    # ========== 3. CLIP 编码（带缓存） ==========
    print("Step 3/5: load or encode CLIP scope features")
    cache_dir = os.path.join(ROOT_PATH, cfg["cache_dir"])
    train_cache = os.path.join(cache_dir, "vice_scope_features_train.pt")
    test_cache = os.path.join(cache_dir, "vice_scope_features_eval.pt")

    if os.path.exists(train_cache):
        train_features, train_labels = torch.load(train_cache, map_location="cpu")
        print(f"Loaded CLIP feature cache: {train_cache}")
    else:
        train_features, train_labels = encode_scope_features(train_examples, device, feature_batch_size, clip_model_name)
        torch.save((train_features, train_labels), train_cache)
        print(f"Saved CLIP feature cache: {train_cache}")

    if os.path.exists(test_cache):
        test_features, test_labels = torch.load(test_cache, map_location="cpu")
        print(f"Loaded CLIP feature cache: {test_cache}")
    else:
        test_features, test_labels = encode_scope_features(test_examples, device, feature_batch_size, clip_model_name)
        torch.save((test_features, test_labels), test_cache)
        print(f"Saved CLIP feature cache: {test_cache}")

    # ========== 4. 训练或加载 ==========
    if os.path.exists(cls_ckpt_path):
        print(f"Step 4/5: load classifier from {cls_ckpt_path}")
        ckpt = torch.load(cls_ckpt_path, map_location="cpu")
        projection = build_projection_head(ckpt["feature_dim"], ckpt["projection_dim"], device)
        projection.load_state_dict(ckpt["projection_state_dict"])
    else:
        print("Step 4/5: train (anchor-CE + locality margin)")
        projection = train_vice_scope_classifier(
            train_features, train_labels, train_groups, cls_ckpt_path, device,
            clip_model_name, train_epochs, train_batch_size, train_lr, temperature, projection_dim,
            loc_margin, loc_lambda,
        )

    # ========== 5. 评估 ==========
    print("Step 5/5: evaluate train and test")
    print_scope_eval("Train", evaluate_reliability_anchor(train_examples, train_features, train_groups, projection, device))
    print_scope_eval("Test", evaluate_reliability_anchor(test_examples, test_features, test_groups, projection, device))


if __name__ == "__main__":
    main()
