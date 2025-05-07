# global_vs_local

Comparison of a global model and local models to predict diagnosis from cortical sulci on three binary classification tasks :
* healthy control (HC) vs autism spectrum disorder (ASD)
* helathy control (HC) vs bipolar disorder (BD)
* healthy control (HC) vs schizophrenia (SCZ).

## Global model

1. Pre-training an encoder on UKBiobank dataset (~20 000 subjects) with the Barlow Twins framework
2. Trainig of a MLP head on the freeze pre-trained encoder for the three downstream tasks

## Local models

The local model is based on Champollion foundation model ([see the paper](https://link.springer.com/chapter/10.1007/978-3-031-78761-4_8))
1. Get embeddings for each sulcus area with the Champollion V0 sulcus encoder
2. Training a MLP from the embeddings for each sulcus area for the three downstream tasks
3. Weighted average of the local model predictions to get the final predictions (weights are learned with a linear regression)

## Differences between models

| Model | Global | Local |
| --- | --- | --- |
| Image resolution (mm³) | 1.5 | 2 |
| Data augmentations | cutout | sulcus specific (see Champollion paper) |

## Results

<table>
    <thead>
        <tr>
            <th>Task</th>
            <th>Test set</th>
            <th>Global</th>
            <th>Local</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>HC vs ASD</td>
            <td>internal</td>
            <td>63.0 ±1.1</td>
            <td>58.3 ±2.8</td>
        </tr>
        <tr>
            <td>external</td>
            <td>66.5 ±1.0</td>
            <td>56.7 ±3.2</td>
        </tr>
        <tr>
            <td rowspan=2>HC vs BD</td>
            <td>internal</td>
            <td>68.7 ±1.3</td>
            <td>61.1 ±2.7</td>
        </tr>
        <tr>
            <td>external</td>
            <td>59.7 ±1.5</td>
            <td>59.3 ±1.4</td>
        </tr>
        <tr>
            <td rowspan=2>HC vs SCZ</td>
            <td>internal</td>
            <td>66.3 ±0.6</td>
            <td>63.0 ±1.4</td>           
        </tr>
        <tr>
            <td>external</td>
            <td>57.6 ±0.4</td>
            <td>52.9 ±2.2</td>
        </tr>
    </tbody>
</table>

# Code

To train a model (global model for instance):
``` bash
# create virtual environment
python3 -m venv globalvslocal
# install librairies
python3 -m pip install -r requirements.txt
# enter to the right folder
cd global
# edit configuration to update path and parameters
gedit config.py
# train model
python train.py --help
```

The script are organized identically to train global and local models:
* ```dataset.py```: dataset to load data
* ```datamanager.py```: manager that load dataset correctly
* ```model.py```: torch module with fit and test method
* ```train.py```: script to load data and train models
* ```test.py```: script to test models (included in train for local)
* ```log.py```: function to set up logs and logger class
* ```classifier.py mlp.py densenet.py```: deep neural network architectures
* ```config.py```: path to directories and default values for parameters

Specific scripts for local models:
* ```make_dataset.py```: create array and dataframe from Champollion embeddings
* ```make_pca.py```: make a dimension reduction of Champollion embeddings with an ACP

Specific scripts for global models:
* ```loss.py```: implementation of the BarlowTwins loss for the original paper
* ```explain.py```: patch occlusion XAI method
* ```data_augmentation.py```: data augmentations for the pre-training
