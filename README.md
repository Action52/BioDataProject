# BioDataProject: Protein Function Prediction
* * *
This repository contains our solution to the final project in the Biological Data 
course from UniPd, under supervision of Prof. Damiano Piovesan.

## Installation
* * *
You will need to install the required Python packages in a clean environment.

```bash
conda create --name biodata python=3.11 --no-default-packages
conda activate biodata
pip install -r requirements.txt
```

## Train
* * *
To re-train, execute the simple_ff-preprocess.ipynb notebook, taking into consideration the paths of the files to use.


## Test
* * *
To run the testing pipeline, you can execute the main.py script.
```python
python main.py --config config.yaml
```

where config.yaml will contain the paths to the corresponding files to run inference on the models.  

The output will consist on a file with probabilities that can be fed to the CAFA evaluator. 
By default, you can find the output in <code>outputs/prediction.tsv</code>.

## Extra Information
* * *
For more information about the solution, procedure, etc, please check the report pdf file.

## License
* * *
- <a href="LICENSE">MIT License</a>

