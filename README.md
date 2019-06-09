# FENESTRA-Fake-News-Structure-and-Threat-Assessment
This Github includes the source code of our system, FENESTRA, an automated pipeline for the discovery and description of the narrative framework of conspiracy theories that circulate on social media and other forums.

---

## Data
Bridgegate and Pizzagate data can be accessed from this [link](https://oneshare.cdlib.org/stash/dataset/doi:10.5068/D1V665).


## Dependencies
This work consists of two main components:
 1. Relation Extraction pipeline: Uses Dependency Trees and Semantic Role Labeling Techniques to provide a high recall of Open Information Extraction tuples. This work is based on python 2.7.
 2. Relationship Aggregation pipeline: Uses contextualized embeddings techniques to group entities/relationships. This work is based on python 3.
 
Note: In an on-going work, we are transferring all the implementations to python 3.
 
### Installation -- First Component: Relation Extraction:
 
Required python: python 2.7

Step-1: Install required packages:
```
pip install networkx pandas matplotlib nxpd ipython nltk pycorenlp
```

Step-2: [Download](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip) the Stanford CoreNLP package, unzip it, and run the following command inside the unzipped folder:

```bash
java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
```

Step-3: [Download](https://github.com/biplab-iitb/practNLPTools/archive/master.zip) practNLPTools package, and run "sudo python setup.py install" in the downloaded folder.

Step-4: Modify the parameters in "relEx_parse_tree.py" according to the data your input data.
Feel free to take a look at the settings we used for Bridgegate and Pizzagate experiments.

Step-5: run

```bash
python relEx_parse_tree.py "path_to_input_file" "output_directory"
```


## Run

```bash
./run.sh
```

## Parameters
Name of input dataset:

```bash
DATASET="bridgegate"
```

## Reference
```bash
paper bib ref
```