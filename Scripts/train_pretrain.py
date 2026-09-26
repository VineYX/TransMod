"""
Pre-training script for TransMod
Train on Bike-sharing and Metro data

NOTE: This script requires real bike-sharing + metro data.
Run from the project root:  python scripts/train_pretrain.py
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
    Train for one epoch
    """
    model.train()
    total_losses = {
        'total': 0.0,
        'pred': 0.0,
        'mmd': 0.0,
        'infonce': 0.0,
        'kl': 0.0,
        'div': 0.0
    }

    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        # Move to device
        # NOTE: User needs to adapt this based on actual data format
        bike_data = {
            'features': batch['bike_features'].to(device),
            'coords': batch['bike_coords'].to(device),
            'od_flows': batch.get('bike_od_flows', None),
            'current_t': batch.get('current_t', 0)
        }

        metro_data = {
            'features': batch['metro_features'].to(device),
            'coords': batch['metro_coords'].to(device),
            'od_flows': batch.get('metro_od_flows', None),
            'current_t': batch.get('current_t', 0)
        }

        zone_centers = batch['zone_centers'].to(device)
        time_features = batch['time_features'].to(device)
        target = batch['target_demand'].to(device)

        # Forward pass
        pred, losses = model.forward_pretrain(
            bike_data, metro_data, zone_centers, time_features
        )

        # Compute total loss
        total_loss, loss_dict = model.compute_total_loss(
            pred, target, losses, stage='pretrain'
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1

    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


def validate(model, val_loader, device, config):
    """
    Validation
    """
    model.eval()
    total_losses = {
        'total': 0.0,
        'pred': 0.0
    }

    all_preds = []
    all_targets = []
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device (same as training)
            bike_data = {
                'features': batch['bike_features'].to(device),
                'coords': batch['bike_coords'].to(device),
                'od_flows': batch.get('bike_od_flows', None),
                'current_t': batch.get('current_t', 0)
            }

            metro_data = {
                'features': batch['metro_features'].to(device),
                'coords': batch['metro_coords'].to(device),
                'od_flows': batch.get('metro_od_flows', None),
                'current_t': batch.get('current_t', 0)
            }

            zone_centers = batch['zone_centers'].to(device)
            time_features = batch['time_features'].to(device)
            target = batch['target_demand'].to(device)

            # Forward pass
            pred, losses = model.forward_pretrain(
                bike_data, metro_data, zone_centers, time_features
            )

            # Compute loss
            total_loss, loss_dict = model.compute_total_loss(
                pred, target, losses, stage='pretrain'
            )

            total_losses['total'] += loss_dict['total']
            total_losses['pred'] += loss_dict['pred']

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

            num_batches += 1

    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)

    return total_losses, metrics


def main(args):
    """Main training function"""
    # Configuration
    config = Config()
    if args.config:
        # Load custom config if provided
        pass

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data loaders
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    train_loader, val_loader, test_loader = get_dataloaders(config, mode='pretrain')
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Model
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)

    # NOTE: User needs to provide actual numbers
    num_bike_stations = 1000  # Placeholder
    num_metro_stations = 1200  # Placeholder
    num_zones = 800  # Placeholder

    model = TransModFramework(
        config=config,
        num_bike_stations=num_bike_stations,
        num_metro_stations=num_metro_stations,
        num_zones=num_zones
    ).to(device)

    # Initialize geography-based assignment
    # NOTE: User needs to provide actual coordinates
    bike_coords = torch.randn(num_bike_stations, 2)
    metro_coords = torch.randn(num_metro_stations, 2)
    zone_centers = torch.randn(num_zones, 2)
    model.initialize_geography(bike_coords, metro_coords, zone_centers)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    print("\n" + "="*60)
    print("Starting pre-training...")
    print("="*60)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.pretrain_epochs):
        print(f"\nEpoch {epoch+1}/{config.pretrain_epochs}")
        print("-" * 60)

        # Train
        train_losses = train_one_epoch(model, train_loader, optimizer, device, config)

        # Validate
        val_losses, val_metrics = validate(model, val_loader, device, config)

        # Print results
        print(f"\nTrain Loss: {train_losses['total']:.4f} "
              f"(Pred: {train_losses['pred']:.4f}, "
              f"MMD: {train_losses['mmd']:.4f}, "
              f"InfoNCE: {train_losses['infonce']:.4f}, "
              f"KL: {train_losses['kl']:.4f}, "
              f"Div: {train_losses['div']:.4f})")

        print(f"Val Loss: {val_losses['total']:.4f} "
              f"(Pred: {val_losses['pred']:.4f})")
        print_metrics(val_metrics, prefix="Val")

        # Learning rate scheduling
        scheduler.step(val_losses['total'])

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                config.pretrain_checkpoint
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
    print("Pre-training completed!")
    print("="*60)

    # Load best model and evaluate on test set
    load_checkpoint(model, None, config.pretrain_checkpoint, device)
    test_losses, test_metrics = validate(model, test_loader, device, config)

    print("\nTest Results:")
    print(f"Test Loss: {test_losses['total']:.4f}")
    print_metrics(test_metrics, prefix="Test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransMod Pre-training")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()
    main(args)
