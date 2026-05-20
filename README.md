Okay so:

latest model is currently stored in the scripts folder alongside the train, eval and all other scripts basically

currently there isnt a fully trained model saved.

to train:
    1. Make sure you have the data you want to train on in data\trainingData\ca_coords
        (if you dont)
        - run buildDataset.py --> then run prepData.py (just follow the messages that pop up when you run)
    2. Make sure you decide on the params in train.py (phys lambda triangle, epochs to run, batch size etc)
    3. let it run for a while

to evaluate: (random protein)
    1. make sure the .pth weights file you want to evaluate (on a 2D map) is in scripts\training
    2. check that the weights is named the same as in eval.py
    3. run eval .py

to run: (Ill update this when I finish it such that you can choose what protein or sequence you want the model to predict the shape of, currently its not that (itll give you the illusion of choice and just choose from the coords I have saved))
    1. -