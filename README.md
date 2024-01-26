# BioDataProject: Protein Function Prediction
* * *
This repository contains our solution to the final project in the Biological Data 
course from UniPd, under supervision of Prof. Damiano Piovesan.

## Installation
* * *
You will need to install the required Python packages in a clean environment.

```python
conda create --name biodata python=3.11 --no-default-packages
conda activate biodata
pip install -r requirements.txt
```

Also, our solution requires the protein embeddings to be stored inside a ChromaDB
database. You need to do this the first time you run the code. For this, set the ingestion
configuration to true in the config file.
```yaml
ingestion:
  execute: true
```
When running, the script will detect that you wish to ingest again the embeddings
and proceed to do it.

## Train
* * *

## Test
* * *

## Extra Information
* * *
For more information about the solution, procedure, etc, please check the report pdf file.

## License
* * *
- <a href="LICENSE">MIT License</a>

