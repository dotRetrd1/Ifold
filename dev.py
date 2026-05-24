import argparse
import subprocess
import sys
import yaml
import webbrowser
from pathlib import Path
import os

os.environ["MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK"] = "0"
os.environ["MIOPEN_LOG_LEVEL"] = "1"  #Suppress MIOpen warnings

project_root = Path(__file__).parent
config_path = project_root / "config.yaml"

if not config_path.exists():
    raise FileNotFoundError(f"Could not find {config_path}. Please ensure it is in the root directory.")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_dir = project_root / config['paths']['scripts_datascripts']
train_dir = project_root / config['paths']['scripts_training']
results_dir = project_root / config['paths']['inference_out']

def run_script(script_path):
    if not script_path.exists():
        print(f"[!] Error: Could not find script at {script_path}")
        return False
    
    print(f"\n{'='*50}\nExecuting: {script_path.name}\n{'='*50}")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Process failed with exit code {e.returncode}")
        return False

def build_data(args):
    #Runs the dataset building and preparation pipeline.
    print("Starting Data Pipeline...")
    step1 = data_dir / "buildDataset.py"
    step2 = data_dir / "prepData.py"
    
    if run_script(step1):
        run_script(step2)

def train_model(args):
    #Runs the PyTorch training loop.
    train_script = train_dir / "train.py"
    run_script(train_script)

def render_result(args):
    #opens a previously saved render in browser
    target_folder = results_dir / args.folder_name
    html_file = target_folder / "interactive_plot.html"
    
    if not html_file.exists():
        print(f"[!] Error: Could not find a saved plot at {html_file}")
        print(f"Check your {results_dir.name} directory.")
        return
        
    print(f"[*] Opening {html_file.name} in browser")
    webbrowser.open(f"file://{html_file.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="iFold Developer CLI: Manage data, training, and results.")
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    #Data
    parser_data = subparsers.add_parser("data", help="Run buildDataset.py and prepData.py in sequence.")
    parser_data.set_defaults(func=build_data)

    #Train
    parser_train = subparsers.add_parser("train", help="Start the iFold training loop.")
    parser_train.set_defaults(func=train_model)

    #Render
    parser_render = subparsers.add_parser("render", help="Open a previously saved 3D interactive render.")
    parser_render.add_argument("folder_name", type=str, help="The name of the result folder (e.g., 1A3H_Prediction)")
    parser_render.set_defaults(func=render_result)

    args = parser.parse_args()
    
    #Execute the function associated with the chosen sub-command
    args.func(args)

if __name__ == "__main__":
    main()