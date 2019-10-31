import pandas as pd
from collections import defaultdict
import re
import numpy as np
import sys
from sklearn.cluster import KMeans
import numpy
import nltk
from bert_serving.client import BertClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.cluster import KMeans
from nltk import SnowballStemmer
stemmer = SnowballStemmer('english', ignore_stopwords=False)

import nltk
from nltk.corpus import stopwords
######### Relationships .csv file path"
relations_path="/Users/user/Documents/StoryMiner/df_extractions_with_ner.csv"
######### Named Entity List output .csv file path"
ner_path="/Users/user/Documents/StoryMiner/goodreads/df_ent_final_ranking.csv"


######### Number of words per superNode
m=4 
def getheadWord(s):
    res=str(s).split('{')
    if len(res)==1:
        return res[0].split('}')[0]
    else:
        return res[1].split('}')[0]
def getEmbeedings(s,r,d):
    s_embeddings=[]
    d_embeddings=[]
    r_embeddings=[]
    if s:
        s_embeddings=bc.encode(s)
    if d:
        d_embeddings=bc.encode(r)
    if r:
        r_embeddings=bc.encode(d)
    return s_embeddings,d_embeddings,r_embeddings
def is_any_entities_present(sent, entity_list):
    for ent in entity_list:
        if ent.lower() in sent.lower():
            return True
    return False
def read_df_rel(based_dir, file_input_name):
    file_input = based_dir + file_input_name    
    ff = open(file_input)
    delim=","
    df = pd.read_csv(file_input,delimiter=delim,header=0)        
    return df
from nltk.corpus import stopwords
def getSecondBest(wordlist,d_tmp):
    tmp=list(d_tmp.values())
    tmp.sort()
    tmp=tmp[::-1]
    seen=set()
    for i in range(len(tmp)):
        for w, score in d_tmp.items():
            if score == tmp[i] and w not in wordlist and w not in seen:
                if score>0:
                    if w not in list(stopwords.words('english')):
                        print(w,score)
                        return w
                seen.add(w)
                break     
def is_any_entities_present2(sent, entity_list):
    for ent in entity_list:
        sent=sent.lower()
        tmp=word_tokenize(sent)
        if ent and ent.lower() in tmp:
            return True
    return False
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
    s_embeddings,d_embeddings,r_embeddings=getEmbeedings(s,r,d)
    nodes=numpy.concatenate([s_embeddings])
    m=min(n,len(nodes))
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
df_rels = read_df_rel("", relations_path)
df_ent = pd.read_csv(ner_path)
stps=set(stopwords.words('english'))
cnds=defaultdict(int)
versions=defaultdict(set)
for ind,row in df_ent.iterrows():
    c=nltk.word_tokenize(str(row['entity']))
    for cc in c:
        cnds[stemmer.stem(cc)]+=row['frequency_score_sum_NER_arg']
        versions[stemmer.stem(cc)].add(cc)
keys=set()
dups=set()
for ind,row in df_ent.iterrows():
#     if row['type']=='PERSON': #Apply this if you want to choose power network
        if row['frequency_score_sum_NER_arg']>5:
            c=nltk.word_tokenize(row['entity'])
            if len(c)>1:
                for cc in c:
                    if len(cc)>2:
                        if cc in keys:
                            keys.remove(cc)
                            dups.add(cc)
                        elif cc not in keys and cc not in keys:
                            keys.add(cc)


tmp=list(cnds.values())
tmp.sort()
tmp=tmp[::-1]
seen=set()
candidates=[]
for i in range(len(tmp)):
    for w, score in cnds.items():
        if score == tmp[i] and w not in seen:
            seen.add(w)
            if score>50 and len(w)>2:
                candidates.append(w) 
wordlists=[]
words_app=defaultdict(int)
seen=set()
for c in candidates:
    wordlist=[]
    for cc in versions[c]:
        if cc not in seen:
            wordlist.append(cc)
    if wordlist:
        for i in range(m):
            prev_post_num=-1
            prev_sentence_num=-1
            d_tmp=defaultdict(int)
            for ind, row in df_rels.iterrows():
                    prev_post_num= row['post_num']
                    prev_sentence_num= row['sentence_num']
#                     if  len(str(row['arg1']))<100:
#                         if len(str(row['arg2']))<100:
                            if is_any_entities_present2(str(row['arg1']), wordlist):
                                s=str(row['arg1']).replace('{','').replace('}','')
                                pcs=nltk.word_tokenize(s)
                                for pc in pcs:
                                    d_tmp[lemmatizer.lemmatize(pc.lower())]+=1
                                for wrd in nltk.word_tokenize(str(row['arg2'])):
                                    if wrd not in seen:
                                        words_app[wrd]+=1
                            elif is_any_entities_present2(str(row['arg2']), wordlist):
                                s=str(row['arg2']).replace('{','').replace('}','')
                                pcs=nltk.word_tokenize(s)
                                for pc in pcs:
                                    d_tmp[lemmatizer.lemmatize(pc.lower())]+=1
                                for wrd in nltk.word_tokenize(str(row['arg1'])):
                                    if wrd not in seen:
                                        words_app[wrd]+=1
            cwrd=getSecondBest(wordlist,d_tmp)
            if cwrd not in seen:
                wordlist.append(cwrd)
                seen.add(cwrd)
            else:
                break
        wordlists.append(wordlist)
wordlist_pd = pd.DataFrame(wordlists)

wordlist_pd.to_csv('supernodes.csv', index=False, header=False)
