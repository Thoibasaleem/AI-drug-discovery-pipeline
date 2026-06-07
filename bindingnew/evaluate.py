import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import os
import json
from config import Config
from model import get_model
from dataset import get_dataloaders

def evaluate():
    device = Config.DEVICE
    
    # Load model
    model = get_model()
    checkpoint = torch.load(os.path.join(Config.MODEL_DIR, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    _, _, test_loader = get_dataloaders()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            drug = batch['drug'].to(device)
            protein = batch['protein'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(drug, protein)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(labels.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    pearson, p_value = pearsonr(actuals, predictions)
    
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test Samples: {len(actuals)}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Pearson Correlation: {pearson:.4f} (p-value: {p_value:.2e})")
    print("=" * 50)
    
    # Save metrics
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson': float(pearson),
        'p_value': float(p_value)
    }
    
    with open(os.path.join(Config.RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5, s=20)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual KIBA Affinity')
    plt.ylabel('Predicted KIBA Affinity')
    plt.title(f'DTI Prediction Performance\nR²={r2:.3f}, Pearson={pearson:.3f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'scatter_plot.png'), dpi=300)
    print(f"Plot saved to {Config.RESULTS_DIR}/scatter_plot.png")
    
    # Plot training history if available
    if os.path.exists(os.path.join(Config.RESULTS_DIR, 'history.json')):
        with open(os.path.join(Config.RESULTS_DIR, 'history.json'), 'r') as f:
            history = json.load(f)
        
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Training History')
        plt.grid(True)
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'training_history.png'), dpi=300)
        print(f"Training history saved to {Config.RESULTS_DIR}/training_history.png")

if __name__ == "__main__":
    evaluate()