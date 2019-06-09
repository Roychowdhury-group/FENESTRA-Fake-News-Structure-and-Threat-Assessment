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
 
### Installation -- First Component: Relation Extraction
 
Required python version: python 2.7

**Step-1**: Install required packages:

```bash
pip install -r requirements_python2.txt
```

Or alternatively install the following packages:
```
pip install networkx pandas matplotlib nxpd nltk pycorenlp enum34 tqdm
```
Then [download](https://github.com/biplab-iitb/practNLPTools/archive/master.zip) practNLPTools package, and run "sudo python setup.py install" in the downloaded folder.


**Step-2**: [Download](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip) the Stanford CoreNLP package, unzip it, and run the following command inside the unzipped folder:

```bash
java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
```

**Step-3**: Modify the parameters in "relEx_parse_tree.py" according to the data your input data.
Feel free to take a look at the settings we used for Bridgegate and Pizzagate experiments.

**Step-4**: Run the following command to extract relationships:

```bash
python2.7 relEx_parse_tree.py "path_to_input_file" "output_directory"
```

**Outputs**
1. A csv file with sentences and their relationships

### Installation -- Second Component: Entity Extraction

Required python version: python 3.6+

Step-1: Install required packages:
```bash
pip install -r requirements_python3.txt
```

**Step-2**: Set Parameters

**Step-3**: run the following command:

```bash
python3.6 main.py 
```

**Outputs**
- Ranking of Named entities with their type, average over the confidence score of their mentions
- Ranking of the head words in the arguments
- A final ranking which combines the above ranking 
- Assign a contextualized embedding (average over BERT embeddings of their mentions) for top N actants in the final entity ranking
- Visualize top ranked entities in a 2D plot projected by PCA
- Store embeddings (and their meta data) in csv files align with TensorBoard's input formats
- Generate a time plot which visualizes how many entities are introduced to the story daily
- Provide a supplementary csv file for the time-plot which shows the entity names that got introduced daily, their associate post, and the sentence in which we detected their first presence.


[comment]: ### Installation -- Third Component: Aggregation

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

## Notes
**Note-1**
- If you run the relation extraction pipeline with the pronoun resolution task, and the output is empty, with no errors! Then, make sure you are running the StanfordCoreNLP server on port 9000.
