"""
Transfer learning script for TransMod
Transfer to Ride-Hailing with frozen components

NOTE: Requires a pre-trained checkpoint from train_pretrain.py and real RH data.
Run from the project root:  python scripts/train_transfer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
import torch.optim as optim
import argparse
from tqdm import tqdm

from config import Config
from models import TransModFramework
from utils import compute_metrics, print_metrics
from utils.data_loader import get_dataloaders, save_checkpoint, load_checkpoint


def train_one_epoch(model, train_loader, optimizer, device, config):
    """
    Train for one epoch on RH data
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        # Move to device
        rh_data = {
            'spatial_features': batch['spatial_features'].to(device),
            'zone_embeddings': batch.get('zone_embeddings', None)
        }

        zone_centers = batch['zone_centers'].to(device)
        target = batch['target_demand'].to(device)

        # Forward pass
        pred = model.forward_transfer(rh_data, zone_centers)

        # Compute loss (MSE)
        loss = torch.nn.functional.mse_loss(pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, device, config):
    """
    Validation on RH data
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            rh_data = {
                'spatial_features': batch['spatial_features'].to(device),
                'zone_embeddings': batch.get('zone_embeddings', None)
            }

            zone_centers = batch['zone_centers'].to(device)
            target = batch['target_demand'].to(device)

            # Forward pass
            pred = model.forward_transfer(rh_data, zone_centers)

            # Compute loss
            loss = torch.nn.functional.mse_loss(pred, target)

            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            num_batches += 1

    avg_loss = total_loss / num_batches

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)

    return avg_loss, metrics


def main(args):
    """Main transfer training function"""
    # Configuration
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    print("\n" + "="*60)
    print("Loading RH data...")
    print("="*60)
    train_loader, val_loader, test_loader = get_dataloaders(config, mode='transfer')
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Model
    print("\n" + "="*60)
    print("Loading pre-trained model...")
    print("="*60)

    num_bike_stations = 1000  # Should match pre-training
    num_metro_stations = 1200
    num_zones = 800

    model = TransModFramework(
        config=config,
        num_bike_stations=num_bike_stations,
        num_metro_stations=num_metro_stations,
        num_zones=num_zones
    ).to(device)

    # Load pre-trained checkpoint
    if args.pretrain_checkpoint:
        checkpoint_path = args.pretrain_checkpoint
    else:
        checkpoint_path = config.pretrain_checkpoint

    load_checkpoint(model, None, checkpoint_path, device)
    print(f"Loaded pre-trained model from {checkpoint_path}")

    # Freeze components for transfer
    model.freeze_for_transfer()

    # Optimizer (only for trainable parameters)
    optimizer = optim.Adam(
        model.get_trainable_params(),
        lr=config.transfer_lr
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\n" + "="*60)
    print("Starting transfer learning...")
    print("="*60)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.transfer_epochs):
        print(f"\nEpoch {epoch+1}/{config.transfer_epochs}")
        print("-" * 60)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, device, config)

        # Print results
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix="Val")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                config.transfer_checkpoint
            )
            print(f"[OK] Best model saved (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*60)
    print("Transfer learning completed!")
    print("="*60)

    # Load best model and evaluate on test set
    load_checkpoint(model, None, config.transfer_checkpoint, device)
    test_loss, test_metrics = validate(model, test_loader, device, config)

    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, prefix="Test")

    # Print comparison with baseline (if available)
    print("\n" + "="*60)
    print("Transfer Learning Summary")
    print("="*60)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        if key == 'MAPE':
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransMod Transfer Learning")
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                       help='Path to pre-trained checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')

    args = parser.parse_args()
    main(args)
