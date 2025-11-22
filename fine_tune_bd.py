import os
import time
import torch

from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from eval import evaluate
from modeling import ImageEncoder, ImageClassifier
from utils import cosine_lr, LabelSmoothing
from heads import get_classification_head


def finetune(args):
    print("[TRACE] Starting finetune()")

    train_dataset = args.train_dataset
    ckpdir = os.path.join(
        args.save,
        train_dataset,
        args.attack_method,
        f"poison_rate_{args.poison_rate}"
    )
    p = 1  # poison enabled

    # ------------------------------
    # Load or build model
    # ------------------------------
    if args.load is not None and isinstance(args.load, str) and args.load.endswith(".pt"):
        print(f"[TRACE] Loading encoder checkpoint from {args.load}")
        image_encoder = ImageEncoder.load(args.load)
    else:
        print("[TRACE] Building new image encoder...")
        image_encoder = ImageEncoder(args, keep_lang=False)

    print("[TRACE] Creating classification head...")
    classification_head = get_classification_head(args, train_dataset, poison=p)

    print("[TRACE] Building ImageClassifier model...")
    model = ImageClassifier(image_encoder, classification_head)

    print("[TRACE] Freezing classification head...")
    model.freeze_head()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRACE] Using device = {device}")

    model = model.to(device)
    preprocess_fn = model.train_preprocess
    print_every = 20

    # ------------------------------
    # Dataset loading
    # ------------------------------
    print("[TRACE] Creating poisoned dataset...")
    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        model=args.model,
        poison=p,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        poison_name=args.attack_method,
        portion=args.poison_rate,
        trigger_type=args.trigger_type,
        target_label=args.target_label,
    )
    num_batches = len(dataset.train_loader)
    print(f"[TRACE] Dataset created. {num_batches} batches per epoch.")

    # ------------------------------
    # Training setup
    # ------------------------------
    print("[TRACE] Creating loss/optimizer/scheduler...")
    loss_fn = LabelSmoothing(args.ls) if args.ls > 0 else torch.nn.CrossEntropyLoss()

    params = [param for param in model.parameters() if param.requires_grad]
    print(f"[TRACE] Trainable parameters: {sum(p.numel() for p in params)}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches
    )

    # ------------------------------
    # Save zero-shot encoder
    # ------------------------------
    print(f"[TRACE] Creating checkpoint directory: {ckpdir}")
    os.makedirs(ckpdir, exist_ok=True)

    zs_path = os.path.join(ckpdir, "zeroshot.pt")
    print(f"[TRACE] Saving zero-shot encoder → {zs_path}")
    model.image_encoder.save(zs_path)

    # ------------------------------
    # TRAINING LOOP
    # ------------------------------
    print("[TRACE] ENTERING TRAINING LOOP\n")

    for epoch in range(args.epochs):
        print(f"[TRACE] Starting epoch {epoch + 1}/{args.epochs}")
        model.train()

        print("[TRACE] Creating DataLoader for this epoch...")
        data_loader = get_dataloader(
            dataset,
            is_train=True,
            args=args,
            image_encoder=None,
            num_workers=args.num_workers,
            pin_memory=True
        )

        print("[TRACE] DataLoader ready. Beginning batches...")

        for i, batch in enumerate(data_loader):
            if i == 0:
                print("[TRACE] First batch successfully loaded.")

            step = i + epoch * num_batches
            scheduler(step)

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if i % print_every == 0:
                pct = 100.0 * i / num_batches
                print(f"[TRACE] Epoch {epoch + 1} | Batch {i}/{num_batches} ({pct:.1f}%) | Loss = {loss.item():.4f}")

        print(f"[TRACE] Finished epoch {epoch + 1}")

    # ------------------------------
    # EVALUATION
    # ------------------------------
    print("[TRACE] Training finished. Starting evaluation...")
    evaluate(model.image_encoder, args)

    # ------------------------------
    # SAVE FINAL MODEL
    # ------------------------------
    if args.trigger_type == 0:
        ft_path = os.path.join(ckpdir, "finetuned.pt")
    else:
        ft_path = os.path.join(ckpdir, "finetuned1.pt")

    print(f"[TRACE] Saving fine-tuned encoder → {ft_path}")
    model.image_encoder.save(ft_path)

    print("[TRACE] Finetune() complete.")
    return zs_path, ft_path


# =====================================================================
# MAIN SCRIPT
# =====================================================================

if __name__ == "__main__":
    print("[TRACE] Starting main()")

    data_location = os.path.expanduser("~/Datasets")

    models = ["ViT-B-32"]
    datasets = ["CIFAR100"]
    attack_methods = ["badnet"]
    poison_rates = [1.0, 0.5]
    trigger_types = [0, 1]

    epochs = {
        "CIFAR100": 5
    }

    for tp in trigger_types:
        for attack in attack_methods:
            for pr in poison_rates:
                for model_name in models:
                    for dataset in datasets:
                        print("\n" + "=" * 100)
                        print(f"[TRACE] Finetuning {model_name} on {dataset} | poison={pr} | trigger={tp}")
                        print("=" * 100)

                        args = parse_arguments()
                        args.lr = 1e-5
                        args.epochs = epochs[dataset]
                        args.data_location = data_location
                        args.train_dataset = dataset
                        args.batch_size = 64
                        args.model = model_name
                        args.poison_rate = pr
                        args.attack_method = attack
                        args.trigger_type = tp
                        args.save = f"checkpoints_poison/{model_name}"

                        # dataloader workers
                        args.num_workers = 8

                        # NOTE: args.target_label comes from CLI, default=0
                        finetune(args)
