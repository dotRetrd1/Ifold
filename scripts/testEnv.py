import hashlib
from pathlib import Path

def get_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read the file in chunks so it doesn't overload memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().upper()

def compare_weights():
    # 1. Setup paths assuming this script is in .../IFold/scripts/
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # 2. Define the two paths to compare
    models_weight = project_root / "data" / "trainingData" / "models" / "ifold_weights.pth"
    training_weight = script_dir / "training" / "ifold_weights.pth"
    
    print("-" * 40)
    print("Weight File Comparator")
    print("-" * 40)
    
    # 3. Verify files actually exist
    missing_files = False
    if not models_weight.exists():
        print(f"❌ Missing file: {models_weight}")
        missing_files = True
    if not training_weight.exists():
        print(f"❌ Missing file: {training_weight}")
        missing_files = True
        
    if missing_files:
        print("\nCannot compare. Please fix the missing files above.")
        return

    # 4. Hash and compare
    print("Hashing file in 'models' folder...")
    models_hash = get_file_hash(models_weight)
    print(f"Hash: {models_hash}\n")
    
    print("Hashing file in 'scripts/training' folder...")
    training_hash = get_file_hash(training_weight)
    print(f"Hash: {training_hash}\n")
    
    if models_hash == training_hash:
        print("✅ RESULT: The files are IDENTICAL.")
    else:
        print("⚠️ RESULT: The files are DIFFERENT.")

if __name__ == "__main__":
    compare_weights()