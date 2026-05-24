lightweight "mini AlphaFold" pipeline that predicts the 3D structure of a protein based on its amino acid sequence. 

## Setup
1. Clone this repo.
2. Create your own virtual environment and install the requirements (PyTorch, BioPython, Plotly, etc.).
3. Set up your paths in `config.yaml`. (Make sure your `.pth` weights file is in the folder specified in the config).


## To Run (User Inference)

If you just want to predict a structure, you don't need to mess with any of the training scripts. Use the `predict.py` script at the root of the project.

**Option 1: I have a sequence**
Drop in any amino acid sequence. The model will translate it into a 2D map and render a 3D interactive HTML plot.
`python predict.py --sequence "[amino acid sequence]"`

**Option 2: I have a PDB ID**
If you pass a PDB ID, the script will automatically reach out to the RCSB PDB, fetch the exact sequence, run the prediction, and *also* download the actual ground truth 3D structure to render them side-by-side for comparison!
`python predict.py --pdb_id [PDB ID]`

*Note: All predictions, including a `.pdb` file and an interactive HTML visualizer, are automatically saved to your `results/` folder*


## To Develop & Train

If you are working on the model itself, use the `dev.py` CLI to run the pipeline. No need to dig into the subfolders.

**1. Data Preparation**
To download targets from the CATH/RCSB databases and process them into Nx4 feature matrices and NxN distance matrices:
`python main.py data`
*(Just follow the prompts in the terminal to choose how many proteins to download).*

**2. Training**
Make sure your hyperparameters (epochs, batch size, learning rate) are set in `config.yaml`, then run:
`python main.py train`
*(This will output backup checkpoints every 10 epochs to your models directory. With 32GB of RAM, you can push the batch size decently high).*

**3. Review Results**
if you want to look at a 3D plot you generated before without re-running the model then pass the name of the folder saved in your `results/` directory:
`python main.py render [folder_name]`