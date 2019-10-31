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
Step-1.1 [download](https://github.com/biplab-iitb/practNLPTools/archive/master.zip) practNLPTools package, and run "sudo python setup.py install" in the downloaded folder.
Step-1.2 Please use the NLTK Downloader to obtain the nltk-punkt resource which is used for sentence tokenizing. 
```bash
python -m nltk.downloader punkt
```

**Step-2**: [Download](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip) the Stanford CoreNLP package, unzip it, and run the following command inside the unzipped folder:

```bash
java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
```

**Step-3**: Modify the parameters in "relEx_parse_tree.py" according to your input data.
Feel free to take a look at the settings we used for Bridgegate and Pizzagate experiments. For example you need to set the "INPUT_DELIMITER" parameter to be "," when you feed a csv file, and set it to "\n" when feeding a txt file which the inputs are separated by new lines. 

**Step-4**: Run the following command to extract relationships:

```bash
python2.7 relEx_parse_tree.py "path_to_input_file" "output_directory"
```

**Outputs**
1. A csv file with sentences, relationships, relationship types, and more metadata (This is the main file which will be used in the next components) -- Output Name: INPUT_NAME_relations_MAX_ITERATION.csv 
2. A txt file with ranking of the relationships (based on simple exact matching) -- Output Name: INPUT_NAME_top_rels_MAX_ITERATION.txt
3. A csv file with sentences and their annotations (to be used for ) -- Output Name: INPUT_NAME_sentences_and_annotations_MAX_ITERATION.csv
4. A txt file with pairwise relationships of selected entities (you can hand pick entities of interest by populating the "entity_versions" dictionary in "utility_functions.py") -- Output Name: INPUT_NAME_pairwise_rels_selected_MAX_ITERATION_DATASET.txt
5. A json file with pairwise relationships formatted to be visualized in d3 js library -- Output Name: INPUT_NAME_g_arg_selected_MAX_ITERATION_EndTimeTimestamp.json
6. A gexf file with pairwise relationships formatted to be visualized in Gephi -- Output Name: INPUT_NAME_g_arg_selected_MAX_ITERATION_EndTimestamp.gexf
### Installation -- Second Component: Entity Extraction

Required python version: python 3.6+

**Step-1**: Install required packages:
```bash
pip3 install -r requirements_python3.txt
```

**Step-2**:  Modify the parameters in "main.py". Information about the parameters are commented in the code.

**Step-3**: run the following command:

```bash
python3.6 main.py 
```

**Outputs**
1. Ranking of Named entities with their type, average over the confidence score of their mentions
2. Ranking of the head words in the arguments
3. A final ranking which combines the above ranking 
4. Assign a contextualized embedding (average over BERT embeddings of their mentions) for top N actants in the final entity ranking
5. Visualize top ranked entities in a 2D plot projected by PCA
6. Store embeddings (and their meta data) in csv files align with TensorBoard's input formats
7. Generate a time plot which visualizes how many entities are introduced to the story daily -- In this experiment we consider the fullname of entities (which follows the format of entity_first_name + space + entity_last_name) because first names and last names could be common among multiple persons, however, fullnames are most likely unique.  
8. Provide a supplementary csv file for the time-plot which shows the entity names that got introduced daily, their associate post, and the sentence in which we detected their first presence.


## Reference

Tangherlini, Timothy; Roychowdhury, Vwani; Shahsavari, Shadi; Shahbazi, Behnam; Ebrahimzadeh, Ehsan (2019), An Automated Pipeline for the Discovery of Conspiracy and Conspiracy Theory Narrative Frameworks -- processed data, v2, UC Los Angeles Dash, Dataset, https://doi.org/10.5068/D1V665

## Notes
**Note-1**
- If you run the relation extraction pipeline with the pronoun resolution task, and the output is empty, with no errors! Then, make sure you are running the StanfordCoreNLP server on port 9000.
