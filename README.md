# FENESTRA-Fake-News-Structure-and-Threat-Assessment
This Github includes the source code of our system, FENESTRA, an automated pipeline for the discovery and description of the narrative framework of conspiracy theories that circulate on social media and other forums.

---

## Data

Data can be accessed from this [link](https://oneshare.cdlib.org/stash/dataset/doi:10.5068/D1V665).

- Bridgegate:
    - Input: Raw Text: XX news article from Northjersey and Hoffington news resources.
    - 

- Pizzagate
    - Input: Raw Text: XX 


## Dependencies
* python 3.6
* pandas, numpy, networkx, etc
* Flair, nltk, ...

## Notes
relation extraction - required packages:

1. 'practnlptools'


## Installation

---
Approach 1 - revising the original practNLPTools
1. Download "practNLPTools" -- revised version for python3 from [here]()
Go to the downloaded folder, and run:
```bash
sudo python setup.py install
```

---
Approach 2 - adopting to a newer version

---
Approach 3 - Let's start fresh!
Step-1: Install required packages:
```
pip install networkx pntl pandas matplotlib nxpd ipython nltk pycorenlp
```

To download required filed for pntl tools, open a terminal and run:  
```bash
sudo pntl -I true
```



Step-2: Download Stanford CoreNLP package, and run it as a server locally.

```bash
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
cd stanford-corenlp-full-2018-10-05
java -mx8g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```

Step-3: Download practNLPTools package from [here]() and just run "sudo python setup.py install" in its folder.

---


Step-4: Add the pntl folder into your path

Note: pntl fails when quatation is present (so we remove " from the data)


1. Install required packages
```bash
$ pip install ...
```
2. Download Stanford-corenlp package from this [link]().
3. Start a local server for co-reference resolution by running the following command:
```bash
Java -m ...
```

### Notes
Note-1: notes related to pntl package 
If you get "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed" error, then if you are using Python 3.6 on Mac OS, you can fix it by running the following command:
/Applications/Python\ 3.6/Install\ Certificates.command
Read more about this error at this [StackOverflow Thread](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error):


## Run

```bash
java -mx4g ...
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

## Installation Notes
If you are using Python 3.6 on Mac OS, you may need to run the following command