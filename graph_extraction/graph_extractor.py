import pandas as pd
import numpy
from bert_serving.client import BertClient

import pandas as pd
from collections import defaultdict
import re
import numpy as np
import sys
from sklearn.cluster import KMeans
import numpy
import nltk
import pickle
from bert_serving.client import BertClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.cluster import KMeans
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
stemmer = SnowballStemmer('english', ignore_stopwords=False)
vectorizer_stem_u = StemmedTfidfVectorizer(stemmer=stemmer, sublinear_tf=True)
vectorizer = TfidfVectorizer()
import os


###Parameters:
#1. Bert server needs to be set up
bc = BertClient(check_length=False)
delim = "\n"
#1. Prefix name of output files 
name='fake_news'
#2. Main text file of data (i.e. posts)
main_txt_file_path="/Users/user/data.txt"
#3. Supernodes list 
super_node_path="supernodes.csv"
#4. relations extracted csv file
relations_path="/Users/user/rels.csv"
#5. number of initial subnodes 
n=10
#6. threshold for deleting subnodes
threshold=1
#7. Number of words in each subnode for label creation 
m=4
def read_df_rel(based_dir, file_input_name):
    file_input = based_dir + file_input_name    
    ff = open(file_input)
    delim=","
    df = pd.read_csv(file_input,delimiter=delim,header=0)        
    return df
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
class StemmedTfidfVectorizer(TfidfVectorizer):
    
    def __init__(self, stemmer, *args, **kwargs):
        super(StemmedTfidfVectorizer, self).__init__(*args, **kwargs)
        self.stemmer = stemmer
        
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(word) for word in analyzer(doc.replace('\n', ' ')))
def getheadWord(s):
    res=str(s).split('{')
    if len(res)==1:
        return res[0].split('}')[0]
    else:
        return res[1].split('}')[0]
def getVerifiedVersion(rel):
    tmp=getheadWord(rel)
    if 'not' in rel:
        tmp='not_'+rel
    return tmp
def getEmbeedings(s):
    s_embeddings=[]
    if s:
        s_embeddings=bc.encode(s)
    return s_embeddings

def is_any_entities_present(sent, entity_list):
    for ent in entity_list:
        if ent.lower() in sent.lower():
            return True
    return False
from nltk.corpus import stopwords
import collections
def pickTop2(d,numT,threshold=0.5,printAll=False):
    num=0
    tmp=list(d.values())
    d_tmp2=defaultdict(int)
    tmp.sort()
    tmp=tmp[::-1]
    seen=set()
    for i in range(len(tmp)):
        for w, score in d.items():
            if score == tmp[i] and w not in seen:
                if score>0 and w in word2tfidf:
                    if w not in list(stopwords.words('english')):
                        d_tmp2[w]=score*word2tfidf[w]                   
                seen.add(w)
                break
    tmp=list(d_tmp2.values())
    tmp.sort()
    tmp=tmp[::-1]
    seen=set()
    res = collections.OrderedDict()
    last_score=-1
    for i in range(len(tmp)):
        for w, score in d_tmp2.items():
            if score == tmp[i]  and w not in seen:
                if score>0 and w in word2tfidf:
                    if w not in list(stopwords.words('english')):
                        if score>threshold*last_score:
                            last_score=score
                            if printAll:
                                print(w,score)
                            num+=1
                            res[w]=score
                            if num>numT:
                                if printAll:
                                    print("=====================")
                                return res
                        else:
                            if printAll:
                                print("=====================")
                            return res
                seen.add(w)
                break
    if printAll:
        print("=====================")
    return res
    
def findNodeConnections(wordlist,df_rels,n,PRINT=False):
    s=[]
    d=[]
    d_h=[]
    r=[]
    s_h=[]
    r_h=[]
    arrow=[]
    ids=[]
    for ind, row in df_rels.iterrows():
        if not row['isDup'] and len(str(row['arg1']))<100:
            if len(str(row['arg2']))<100:
                if is_any_entities_present(str(row['arg1']), wordlist):
                    s.append(str(row['arg1']).replace('{','').replace('}',''))
                    d.append(str(row['arg2']).replace('{','').replace('}',''))
                    r.append(row['rel'].replace('{','').replace('}','')) 
                    d_h.append(getheadWord(row['arg2']))
                    s_h.append(getheadWord(row['arg1']))
                    r_h.append(getheadWord(row['rel']))
                    arrow.append(0)
                    ids.append(ind)
                elif is_any_entities_present(str(row['arg2']), wordlist):
                    s.append(str(row['arg2']).replace('{','').replace('}',''))
                    d.append(str(row['arg1']).replace('{','').replace('}',''))
                    r.append(row['rel'].replace('{','').replace('}','')) 
                    d_h.append(getheadWord(row['arg1']))
                    s_h.append(getheadWord(row['arg2']))
                    r_h.append(getheadWord(row['rel']))
                    arrow.append(1)
                    ids.append(ind)  
    s_embeddings=getEmbeedings(s)
    nodes=numpy.concatenate([s_embeddings])
    m=min(n,len(nodes))
    if len(nodes)==0:
        print("no found!",wordlist)
        return [], [],[],[],[],[]
    kmeans = KMeans(n_clusters=m, random_state=0).fit(nodes)
    supernodes_in=[]
    r_in=[]
    supernodes_out=[]
    supernodes_self=[]
    r_out=[]
    supernodes_self_ids=[]
    for j in range(n):
        ins=[]
        r_i=[]
        outs=[]
        r_o=[]
        selfs=[]
        ids_selfs=set()
        for i in range(len(kmeans.labels_)):  
            if kmeans.labels_[i]==j:
                if arrow[i]==0:
                    outs.append(d[i])
                    r_o.append(r[i]+'-'+str(ids[i]))
                    
                else:
                    ins.append(d[i])
                    r_i.append(r[i]+'-'+str(ids[i]))
                selfs.append(s[i])
                ids_selfs.add(ids[i])
        supernodes_self.append(selfs)
        supernodes_self_ids.append(ids_selfs)
        r_in.append(r_i)
        r_out.append(r_o)
        supernodes_in.append(ins)
        supernodes_out.append(outs)
    return supernodes_in, r_in,supernodes_out,r_out,supernodes_self,supernodes_self_ids
import nltk
from nltk.corpus import stopwords
stps=set(stopwords.words('english'))
def getsetForSubnode(i,n,m):
    tmp=supernode_names_print[i][n]
    res=set()
    tmp=tmp.split(' ,')
    for i in range(len(tmp)):
        if i<m:
            if len(tmp[i])>1 and tmp[i] not in stps:
                res.add(tmp[i])
        else:
            return res
    return res
def findLabelForClass(set_names):
    x=list(set_names)
    x = sorted(x) 
    x = ' '.join(x)
    return x
def createApposRelation(r):
    r=r.lower().replace('{','').replace('}','')
    for ent in ent_set:
        r=r.replace(ent,'')
    return r
df = pd.read_csv(main_txt_file_path,delimiter=delim,header=0,error_bad_lines=False)
df_rels = read_df_rel("",relations_path )
df_tmp=pd.read_csv(super_node_path)
w=df_tmp.values.tolist()
entities=[]
for a in w:
    tmp=[]
    for t in a:
        if str(t)!='nan':
            tmp.append(str(t))
    entities.append(tmp)
all_sents=[]
for ind, row in df.iterrows():
    text=row['text']
    sentences=nltk.sent_tokenize(text)
    for s in sentences:
        all_sents.append(s)
X = vectorizer.fit_transform(list(df['text']))
X = vectorizer_stem_u.fit_transform(list(df['text']))
word2tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

supernodes_in=[]
r_in=[]
supernodes_out=[]
r_out=[]
supernodes_self=[]
supernodes_self_ids=[]
for ent in entities:
    supernodes_in_t, r_in_t,supernodes_out_t,r_out_t,supernodes_self_t,supernodes_self_ids_t=findNodeConnections(ent,df_rels,n)
    supernodes_in.append(supernodes_in_t)
    r_in.append(r_in_t)
    supernodes_out.append(supernodes_out_t)
    r_out.append(r_out_t)
    supernodes_self.append(supernodes_self_t)
    supernodes_self_ids.append(supernodes_self_ids_t)
#if you want to save:
# to_be_saved={}
# to_be_saved['supernodes_in']=supernodes_in
# to_be_saved['r_in']=r_in
# to_be_saved['r_out']=r_out
# to_be_saved['supernodes_self']=supernodes_self
# to_be_saved['supernodes_self_ids']=supernodes_self_ids

# save_obj(to_be_saved, "data_pickle" )
#Code for creating the node highest scores words
from nltk.stem import WordNetLemmatizer 
import nltk
lemmatizer = WordNetLemmatizer() 
supernode_names=[]
supernode_names_print=[]
for i in range(len(entities)):
    node_names=[]
    node_names_print=[]

    for ii in range(len(supernodes_self[i])):
        d=defaultdict(int)
        for s in supernodes_self[i][ii]:
            pcs=nltk.word_tokenize(s)
            for pc in pcs:
                d[lemmatizer.lemmatize(pc.lower())]+=1
        res=pickTop2(d,5,0.7,False)
        node_names.append(res)
        st=""
        for w in res:
            st+=w+" ,"
        node_names_print.append(st)
    supernode_names.append(node_names)
    supernode_names_print.append(node_names_print)
j=0
k=0
supernodes_self_status=[]
for i in range(len(supernodes_self)):
    tmp=[]
    for n in range(len(supernodes_self[i])):
        tmp.append(len(supernodes_self[i][n]))
        k+=1
    tmp_l=[]
    for n in range(len(supernodes_self[i])):
        if len(supernodes_self[i][n])/numpy.mean(tmp)>threshold:
            j+=1
            tmp_l.append(True)
        else:
            tmp_l.append(False)
    supernodes_self_status.append(tmp_l)
# print(i,j,k)

d_int_set={}
d_names={}
supernode_names_revised=[]
for i in range(len(supernodes_self)):
    nodes_rv=[]
    for n in range(len(supernodes_self[i])):
        tmp=getsetForSubnode(i,n,m)
        if not tmp:
            supernodes_self_status[i][n]=False
        nodes_rv.append(findLabelForClass(tmp))
    supernode_names_revised.append(nodes_rv)
res_tmp=set()
for i in range(len(supernodes_self)):
    for n in range(len(supernodes_self[i])):
        if supernodes_self_status[i][n]:
            res_tmp.add(supernode_names_revised[i][n])

ent_set=set()
for entl in entities:
    for ent in entl:
        ent_set.add(ent.lower())

import networkx as nx
g = nx.MultiDiGraph()
for i1 in range(len(entities)):
    for i2 in range(i1+1,len(entities)):
        target1=entities[i1]
        target2=entities[i2]
        supernodes1=supernodes_self_ids[i1]
        nodes1=supernodes_self[i1]
        nodes1_names=supernode_names_revised[i1]
        nodes2=supernodes_self[i2]
        nodes2_names=supernode_names_revised[i2]
        j=-1
        outward=r_out[i2]
        inward=r_in[i2]
        for k in range(len(supernodes1)):
            node2=supernodes1[k]
            for i in range(len(outward)):
                for j in range(len(outward[i])):
                    rel=outward[i][j]
                    number=int(rel.split('-')[1])
                    if number in node2:
                        if supernodes_self_status[i2][i]:# and supernodes_self_status[i2][k]:
                            tmp_rel=getVerifiedVersion(df_rels['rel'][number])
                            if df_rels['type'][number]=='appos':
                                g.add_edge(nodes2_names[i],nodes1_names[k],label=createApposRelation(df_rels['arg2'][number]))
                            else:
                                g.add_edge(nodes2_names[i],nodes1_names[k],label=tmp_rel)

        for k in range(len(supernodes1)):
            node2=supernodes1[k]
            for i in range(len(inward)):
                for j in range(len(inward[i])):
                    rel=inward[i][j]
                    number=int(rel.split('-')[1])
                    if number in node2:
                        if supernodes_self_status[i1][k]:# and supernodes_self_status[i2][i]:
                            tmp_rel=getVerifiedVersion(df_rels['rel'][number])
                            if df_rels['type'][number]=='appos':
                                g.add_edge(nodes1_names[k],nodes2_names[i],label=createApposRelation(df_rels['arg2'][number]))
                            else:
                                g.add_edge(nodes1_names[k],nodes2_names[i],label=tmp_rel)


gr=g.copy()
# os.remove('graph_'+name+'.txt')
with open('graph_'+name+'.txt', 'a') as the_file:
    nodes = gr.nodes()
    for n1 in nodes:
        for n2 in nodes:
            l = gr.get_edge_data(n1,n2)
            if l and len(l)>0:
                s=""
                for ll in l:
                    s+=" "+str(l[ll]['label'])
                the_file.write (str(n1)+"\t"+s+"\t"+str(n2))
                the_file.write('\n')
        the_file.write('\n')
        the_file.write("======================")
        the_file.write('\n')
the_file.close()
i=0
idis={}
for n in gr.nodes():
    idis[i]=n
    i+=1
# os.remove('node_'+name+'.txt')
with open('nodes_'+name+'.txt', 'a') as the_file:
    the_file.write('nodeID nodeLabel')
    the_file.write('\n')
    for j in range(i):
        the_file.write(str(j)+' '+str(idis[j]))
        the_file.write('\n')
the_file.close()
i=0
idis={}
nodes = gr.nodes()
all_edges={}
all_edge_idis={}
jj=0
for n1 in nodes:
    for n2 in nodes:
        l = gr.get_edge_data(n1,n2)
        if l and len(l)>0:
            s=""
            for ll in l:
                s+=" "+l[ll]['label']
            if s not in all_edges:
                all_edges[s]=jj
                all_edge_idis[jj]=s
                jj+=1
# os.remove('edge_idis_'+name+'.txt')
with open('edge_idis_'+name+'.txt', 'a') as the_file:
    the_file.write('edgeID edgeLabel')
    the_file.write('\n')
    for j in range(len(all_edges)):
        the_file.write(str(j)+' '+str(all_edge_idis[j]))
        the_file.write('\n')
the_file.close()
i=0
idis={}
for n in gr.nodes():
    idis[n]=i
    i+=1
# os.remove('edge_list_supernodes_bridgegate.txt')
with open('edge_list_'+name+'.txt', 'a') as the_file:
    nodes = gr.nodes()
    for n1 in nodes:
        for n2 in nodes:
            l = gr.get_edge_data(n1,n2)
            if l and len(l)>0:
                s=""
                for ll in l:
                    s+=" "+l[ll]['label']
                the_file.write (str(idis[n1])+" "+str(idis[n2])+" "+str(all_edges[s]))
                the_file.write('\n')
the_file.close()


