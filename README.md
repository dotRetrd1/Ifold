## Setup
1. Clone this repo.
2. Create your own virtual environment and install the requirements.
    - PyTorch
    - BioPython
    - Plotly
    - NumPy
    - Matplotlib
    - PyYAML
3. Set up your paths and hyperparams in `config.yaml`.
4. Place your trained `.pth` weights file inside the models directory specified in the config.


## To Run (User Inference)

If you just want to predict a structure: Use the `predict.py` script at the root of the project.
`python predict.py`

**Option 1: Predict from amino acid sequence**
Drop in any amino acid sequence. The model will translate it into a 2D map and render a 3D plot.
`python predict.py --sequence "[amino acid sequence]"`

**Option 2: I have a PDB ID**
If you pass a PDB ID, the script will automatically reach out to the RCSB PDB, fetch the exact sequence, run the prediction, and *also* download the actual ground truth 3D structure to render them side-by-side for comparison
`python predict.py --pdb_id [PDB ID]`

**Output**
Predictions can be saved into the results/ folder and include:
- predicted coordinates
- .pdb structure file
- interactive HTML visualization
- metadata about the prediction


## To Develop & Train

If you are working on the model itself, use the `dev.py` CLI to run the pipeline.

**1. Data Preparation**
To download targets from the CATH/RCSB databases:
`python main.py data`
*(Just follow the prompts in the terminal to choose how many proteins to download).*

**2. Training**
Make sure your hyperparameters (epochs, batch size, learning rate, etc..) are set in `config.yaml`, then run:
`python main.py train`
*This will output backup checkpoints every x [deafult=10] epochs to your models directory.*

**3. Review Results**
**Evaluate Distance Maps**
To inspect raw 2D predictions before reconstruction:
`python scripts/training/eval.py`
this will display:
 - predicted distance map
 - ground truth distance map
 - absolute error heatmap
*note this is more useful for debugging than youd think, the 3D is really prone to errors*

**Render previous prediction (in 3D)**
To reopen a saved prediction without rerunning inference:
`python main.py render [folder_name]`

**Current Architecture**
The model currently:
- predicts pairwise Cα distance maps
- uses a dilated residual CNN
- symmetrizes outputs
- reconstructs coordinates using classical MDS

Loss function combines:
- masked L1 distance loss
- local residue consistency loss
- triangle inequality penalty
- contact-map auxiliary loss

**Notes**
- Larger proteins are substantially harder than short proteins.
- The 3D reconstruction stage is approximate and may introduce artifacts even when the 2D prediction is good.
- Beta barrels are especially difficult due to long-range strand pairing constraints.
- Training on the general protein dataset before beta-barrel fine-tuning generally produces better results than training on beta barrels alone.