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
 
Required python version: python 2.7

Step-1: Install required packages:

```bash
pip install -r requirements_python2.txt
```

Or alternatively install the following packages:
```
pip install networkx pandas matplotlib nxpd nltk pycorenlp enum34 tqdm
```
[Download](https://github.com/biplab-iitb/practNLPTools/archive/master.zip) practNLPTools package, and run "sudo python setup.py install" in the downloaded folder.


Step-2: [Download](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip) the Stanford CoreNLP package, unzip it, and run the following command inside the unzipped folder:

```bash
java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
```

Step-3: Modify the parameters in "relEx_parse_tree.py" according to the data your input data.
Feel free to take a look at the settings we used for Bridgegate and Pizzagate experiments.

Step-4: Run the following command to extract relationships:

```bash
python relEx_parse_tree.py "path_to_input_file" "output_directory"
```

### Installation -- Second Component: Aggregation:

Required python version: python 3.6+

Step-1: Install required packages:
```bash

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

## FAQ
Question 1: 
- I ran the relation extraction pipeline with the pronoun resolution task, and the output is empty, with no errors!
- Did you run the StanfordCoreNLP server on port 9000?
- Yes
- Did you?
- Yes
- Really?
- Sorry, I forgot! Thanks!