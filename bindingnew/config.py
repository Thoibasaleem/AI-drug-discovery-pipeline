import os

class Config:
    # Paths
    DATA_PATH = "kiba.txt"  # Your uploaded file
    PROCESSED_DIR = "processed_data"
    MODEL_DIR = "models"
    RESULTS_DIR = "results"
    
    # Create dirs if not exist
    for d in [PROCESSED_DIR, MODEL_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Data preprocessing
    MAX_SEQ_LENGTH = 1000  # Truncate protein sequences
    MORGAN_RADIUS = 2
    MORGAN_BITS = 2048     # Drug fingerprint size
    AA_FEATURES = 20       # Amino acid composition (20 standard AAs)
    
    # Model architecture
    HIDDEN_DIM = 512
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    
    # Device
    DEVICE = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    
    # Split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1