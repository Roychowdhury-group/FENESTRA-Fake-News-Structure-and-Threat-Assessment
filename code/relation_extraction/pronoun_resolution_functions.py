import sys
sys.path.append('/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/pronoun_resolution/pycorenlp-parent/')
import nltk
from pycorenlp import StanfordCoreNLP
import json
import string
from nltk import tokenize
import numpy as np
import re
from nltk import word_tokenize

nlp = StanfordCoreNLP('http://localhost:9000')


def replace_pronoun_given_stanford_output(output, slding_length, isFirst):

    keys_length = len(output['corefs'])
    all_keys = output['corefs'].keys()
    
    break_list = ['i','myself','himself','herself','themselves','my']
    
    replacement_array = [['monster', 'monsters', 'monsterous'],['frankenstein','frankensteins','victor','victors', 'victor_frankenstein', 'victor_frankensteins', 'victor frankenstein'],['creature','creatures'],\
                        ['creation','creations','creationism'], ['mary_shellei', 'mary_shelley','mary','shellei','shelley','mary shellei', 'mary shelley'],\
                        ['creator','creators'],['man','mans'],['novel','novelization','stori','stories','tale'],['human','humane','humanity','humans','humanness'],\
                        ['life','lifes'],['death','deaths'],['revenge','reveng'],['letter','letters']]

    replacement_count = np.zeros(len(replacement_array))
    
    for i in range(keys_length):
        coref_length = len(output['corefs'][all_keys[i]])
        if (coref_length > 1):
            for j in range(coref_length):
                    
                if(output['corefs'][all_keys[i]][j]['isRepresentativeMention']):
                    rep_mention = output['corefs'][all_keys[i]][j]['text'].encode('ascii','ignore')
                    
                    rep_mention_tok = word_tokenize(rep_mention)
                    rep_mention_list = nltk.pos_tag(rep_mention_tok)

                    for p in range(len(rep_mention_list)):
                        if ((rep_mention_list[p][1] == 'IN') or (rep_mention_list[p][0] in ['who','which','where',','])):
                            new_list = []
                            for q in range(p):
                                new_list.append(rep_mention_list[q][0])
                            rep_mention = ' '.join(new_list)
                            break
                

                    find_index = rep_mention.find("'s")
                    if (find_index != -1):
                        if (rep_mention[find_index - 1] == " "):
                            rep_mention = rep_mention[:find_index - 1] + rep_mention[find_index:]
            
#             print(rep_mention.lower())
#             print(rep_mention.lower() in break_list)

            if (rep_mention.lower() in break_list):
                break
            
            for j in range(len(replacement_array)):
                if (rep_mention.lower() in replacement_array[j]):
                    replacement_count[j] += coref_length - 1
            
            for j in range(coref_length):       
                if not (output['corefs'][all_keys[i]][j]['isRepresentativeMention']):
                    start_index = output['corefs'][all_keys[i]][j]['startIndex']
                    end_index = output['corefs'][all_keys[i]][j]['endIndex']
                    sent_Num = output['corefs'][all_keys[i]][j]['sentNum'] - 1
                    
#                     print(rep_mention)
#                     print(output['sentences'][sent_Num]['tokens'][start_index-1]['word'])
#                     print(output['sentences'][sent_Num]['tokens'][start_index-1]['pos'] )
                    if not ("PRP" in output['sentences'][sent_Num]['tokens'][start_index-1]['pos']):
#                         print("True")
                        continue
                    
                    if (sent_Num < slding_length and not isFirst):
                        continue
                        
                    if (output['corefs'][all_keys[i]][j]['text'].lower() in break_list):
                        continue
                    
                    if (output['sentences'][sent_Num]['tokens'][start_index-1]['pos'] == 'PRP$'):
                        if ("'s" not in rep_mention):
                            output['corefs'][all_keys[i]][j]['text'] = rep_mention + "'s"
                            output['sentences'][sent_Num]['tokens'][start_index-1]['word'] = rep_mention + "'s"
                        else:
                            output['corefs'][all_keys[i]][j]['text'] = rep_mention
                            output['sentences'][sent_Num]['tokens'][start_index-1]['word'] = rep_mention

                        for k in range(start_index,end_index-1):
                            output['sentences'][sent_Num]['tokens'][k]['word'] = ""
                    else:
                        if (output['sentences'][sent_Num]['tokens'][start_index-1]['pos'] != 'PRP$'):
                            rep_mention_new = rep_mention.replace("'s","")
                            output['corefs'][all_keys[i]][j]['text'] = rep_mention_new
                            output['sentences'][sent_Num]['tokens'][start_index-1]['word'] = rep_mention_new
                        else:
                            output['corefs'][all_keys[i]][j]['text'] = rep_mention
                            output['sentences'][sent_Num]['tokens'][start_index-1]['word'] = rep_mention
                    for k in range(start_index,end_index-1):
                        output['sentences'][sent_Num]['tokens'][k]['word'] = ""
        
    return output, replacement_count



def get_text_with_replaced_pronouns(text):

    #f = open("Hello.txt","r")
    # f = open("The Hobbit (Middle-Earth Universe)_raw_en.txt","r")
    # new_file = open("Hobbit_pronoun_resolution.txt","w")
    # new_file = open("Frankenstein_pronoun_resolution.txt","w")
    #new_file = open("hello_resolution.txt","w")

    #text = re.split('\n\n',f.read())
    total_replacement_count = np.zeros(13)
    slding_length = 3
    num_sent = 6

    final_res = ""
    for idx in range(len(text)):
        tokenized_sent = tokenize.sent_tokenize(text[idx].decode('utf-8'))

        #print("input:")
        #print(tokenized_sent)

        k = 0
        isFirst = True
        n = 0
        #while(k < len(tokenized_sent)-1): -> this is wrong??
        while(k < len(tokenized_sent)):
            three_sentences = ' '.join(tokenized_sent[k:k+num_sent])

            three_sentences = re.sub(r'[^\x00-\x7F][\bs\b]+',"'s", three_sentences)
            three_sentences = re.sub(r'[^\x00-\x7F]+'," ", three_sentences)


            output = nlp.annotate(three_sentences.encode('ascii','replace'), properties={'annotators': 'tokenize,ssplit,pos,depparse,parse,coref', 'outputFormat': 'json'})

            output, replacement_count = replace_pronoun_given_stanford_output(output, slding_length, isFirst)
            total_replacement_count += replacement_count
            isFirst = False

            sent_len = len(output['sentences'])

            for i in range(sent_len):
                tempList =[]
                for j in range(len(output['sentences'][i]['tokens'])):     

                    if (output['sentences'][i]['tokens'][j]['word'] in string.punctuation and len(tempList) != 0):
                        tempList[-1] += output['sentences'][i]['tokens'][j]['word']
                    elif(output['sentences'][i]['tokens'][j]['word'] in ["'s","n't","'m","'ve","'d","'ll"] and len(tempList) != 0):
                        tempList[-1] += output['sentences'][i]['tokens'][j]['word']
                    elif(output['sentences'][i]['tokens'][j]['word'] == '-LRB-'):
                        tempList.append('(')
                    elif(output['sentences'][i]['tokens'][j]['word'] == '-RRB-'):
                        tempList[-1] += ')'
                    elif(output['sentences'][i]['tokens'][j]['word'] == '-LSB-'):
                        tempList.append('[')    
                    elif(output['sentences'][i]['tokens'][j]['word'] == '-RSB-'):
                        tempList[-1] += ']'

                    elif(output['sentences'][i]['tokens'][j]['word'] == '``'):
                        if (len(tempList) != 0 and tempList[-1] == '('):
                            tempList[-1] += '"'
                        else:
                            tempList.append('"')
                    elif(output['sentences'][i]['tokens'][j]['word'] == "''"):
                        if (len(tempList) != 0):
                            tempList[-1] += '"'
                        else:
                            tempList.append('"')
                    elif((len(tempList) != 0 and tempList[-1] == '"') | (len(tempList) != 0 and tempList[-1] == '("')):
                        tempList[-1] += output['sentences'][i]['tokens'][j]['word']
                    elif(len(tempList) != 0 and tempList[-1] == '('):
                        tempList[-1] += output['sentences'][i]['tokens'][j]['word']   
                    else:
                        tempList.append(output['sentences'][i]['tokens'][j]['word'])

                tokenized_sent.insert(n,' '.join(tempList))
                n += 1
            del tokenized_sent[n:n+num_sent]
            n -= slding_length

    #         print(tokenized_sent)
    #         print('\n')

            k = k + sent_len - slding_length
            if (k >= len(tokenized_sent) - slding_length):
                break

        final_res += ' '.join(tokenized_sent).encode('ascii','ignore')    
        #new_file.write(' '.join(tokenized_sent).encode('ascii','ignore'))
        #new_file.write('\n\n')
    #new_file.close()
    #print(total_replacement_count/len(text))
    return final_res