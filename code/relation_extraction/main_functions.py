from RE_init import *
from utility_functions import *
from pycorenlp import StanfordCoreNLP
from collections import Counter
import collections
import ast
import json
from networkx.readwrite.json_graph import node_link_data
from collections import OrderedDict
import unicodedata
from tqdm import tqdm


nlp = StanfordCoreNLP('http://localhost:9000')
def word_to_node_id(verbs_list, verb_ind, annotation):
    word = verbs_list[verb_ind]
    if word == "ROOT":
        return "ROOT-NNP-0"
    verb_occurance_num = verbs_list[0:verb_ind+1].count(word)
    i = -1
    for j in xrange(verb_occurance_num):
        i = annotation['words'].index(word, i+1)
    w_ind = i 
    return word+"-"+annotation['pos'][w_ind][1]+"-"+str(w_ind+1) 

def word_id_get_index(word_id):
    return int(word_id.split('-')[2])

def word_id_get_pos(word_id):
    # word_id syntax : (NAME-POS-INDEX ex. test-nnp-3)
    return word_id.split('-')[1]

def word_id_get_word(word_id):
    return word_id.split('-')[0]

def sort_word_ids(word_ids, head_word_ind):
    #if head_word_ind == -1:
    #    return "",""
    word_sorted_str = ""
    word_with_pos_sorted_str = ""
    
    word_ids_tuple = map(lambda x: (word_id_get_index(x),x), word_ids)
    word_ids_tuple.sort(key = lambda x: x[0]) #sort based on the first element - which is word's index
    for w in word_ids_tuple:
        if int(w[0]) == int(head_word_ind):
            word_sorted_str += "{" + word_id_get_word(w[1]) + "}" + " "
            word_with_pos_sorted_str += w[1] + " "            
        else:
            word_sorted_str += word_id_get_word(w[1]) + " "
            word_with_pos_sorted_str += w[1] + " "
        
        
    # removing the extra space at the end
    return word_sorted_str.strip(), word_with_pos_sorted_str.strip()

def isPronoun(word):
    tagged,tag = nltk.pos_tag([word])[0]
    if "PRP" in tag:
        return True
    return False

def get_verbs(annotation):
    verbs_list = [x[0] for x in annotation["pos"] if "VB" in x[1]]
    return verbs_list
def get_preps(annotation):
    prep_list = [x[0] for x in annotation['chunk'] if "PP" in x[1]]
    return prep_list

def get_nouns(annotation):
    return [x for x in re.findall(r'\(NN\w* (\w+)\)', annotation['syntax_tree'])]

def get_adjectives(annotation):
    return [x for x in re.findall(r'\(JJ (\w+)\)', annotation['syntax_tree'])]

def get_expansion_list(node_id, node_type="arg", rel_type="SVO"):
    arg_expand_list = ['nn', 'amod', 'det', 'neg', 'prep_of', 'num', 'quantmod', 'poss'] 
    arg_expand_non_nnp_list = ['infmod', 'partmod', 'ref', 'prepc_of', "dobj", "iobj"] #'rcmod' -> it's alwasys connected to nsubj # dobj/iobj for prepc_ cases
    rel_expand_list = ['advmod', 'mod', 'aux', 'auxpass', 'cop', 'prt','neg', 'poss']
    rel_expand_non_SVO_list = ['dobj', 'iobj']

    if node_type == "arg":
        if word_id_get_pos(node_id) != "NNP":
            return arg_expand_list + arg_expand_non_nnp_list
        else:
            return arg_expand_list

    if node_type == "rel":
        if rel_type != "SVO":
            return rel_expand_list + rel_expand_non_SVO_list
        else:
            return rel_expand_list
            
    return []

def expand_node_id(node_id, g_dir, node_type="arg", rel_type="SVO",EXTRACT_NESTED_PREPOSITIONS_RELS=True):
    res_expanded = defaultdict(list)
    expansion_list = get_expansion_list(node_id, node_type = node_type, rel_type = rel_type)
    g_dir_v = None
    node_extra_ids = []
    node_extra_ids.append(node_id)
    node_head_id_ind = word_id_get_index(node_id)
    try:
        g_dir_v = g_dir[node_id]
    except:
        ##print "Faild to get adjacency network of ", node_id, " while expanding it."
        return
    for word_id, e in g_dir_v.iteritems():
        if e["rel"] in expansion_list:
            if e["rel"] == "prep_of" or e["rel"] == "prepc_of":
                new_word_id = "of " + word_id.split("-")[0] + "-" + word_id.split("-")[1] + "-" + word_id.split("-")[2]
                node_extra_ids.append(new_word_id)
            else:
                node_extra_ids.append(word_id)
            
    node_final_word_str, node_final_with_pos_str = sort_word_ids(node_extra_ids, node_head_id_ind)
    
    res_expanded['rel_prepositions'] = ""
    if EXTRACT_NESTED_PREPOSITIONS_RELS:
        res_expanded['rel_prepositions'] = get_nested_preposition_rels(g_dir, node_id, expansion_list)
    
    res_expanded["final_word_str"] = node_final_word_str
    res_expanded["final_word_with_pos_str"] = node_final_with_pos_str
    
    return res_expanded
    
def expand_rel(rel, g_dir, EXTRACT_NESTED_PREPOSITIONS_RELS,ADD_EXTRAS=False):
    '''
    Expands arguments by adding extra related words, such as nn, amod, and so on. 
    Expands relations by adding extra related words, such as adverbs, and so on.
    Returns the expanded relation.
    Input: 
    '''
    '''
    Added by Arash:
    acomp: adjectival complement > very beautiful
    advmod > arg: Genetically modified food , rel: it really was a mistake
    amod > red meat
    xcomp for jj > I was ready to go, This is sad to hear
    acl, [mark] > he is the man you love, it was a day to remember
    ccomp, [dobj] > I am certain that he did it
    I saw the man you love
    It is all I know
    possessive: Bill's clothes
    
    # cc
    # add mwe to cc
    '''    
    def dfs_rel(word_id,extras,after,ADD_EXTRAS1):
        if ADD_EXTRAS1:
            if  word_id in g_dir:
                for item in g_dir[word_id]:
                    if item not in extras:
                        if after:
                            extras.append(item)
                            dfs_rel(item,extras,after,ADD_EXTRAS1)
                        else:
                            dfs_rel(item,extras,after,ADD_EXTRAS1)
                            extras.append(item)
    arg1 = rel['arg1']
    arg2 = rel['arg2']
    r = rel['rel']
    verb_with_id=r[:]
    arg_expand_list = ['nn', 'amod', 'det', 'neg', 'prep_of', 'num', 'quantmod', 'poss']
    arg_expand_non_nnp_list = ['infmod', 'partmod', 'ref', 'prepc_of', "dobj", "iobj"] #'rcmod' -> it's alwasys connected to nsubj
    rel_expand_list = ['advmod', 'mod', 'aux', 'auxpass', 'cop', 'prt','neg', 'poss']
    rel_expand_non_SVO_list = ['dobj', 'iobj'] 
    
    if rel["type"] == "SVCop":
        arg_expand_list = arg_expand_list + ['acomp', 'advmod', 'xcomp', 'acl', 'ccomp', 'aux']
        rel_expand_non_SVO_list = rel_expand_non_SVO_list + ['aux']
    
    
    arguments_names = ['arg1', 'arg2']
    '''
    EXPAND ARGUMENTS
    '''
    arg_extra_ids = defaultdict(list)
    arg_head_ind = defaultdict(list)
    for ind, arg_name in enumerate(arguments_names):            
        arg = rel[arg_name]
        if arg == "": # for SV relationships
            continue
        # add current argument to the extended version
        arg_extra_ids[arg_name].append(arg)
        if arg_name=='arg2':
            dfs_rel(arg,arg_extra_ids[arg_name],True,ADD_EXTRAS)
        arg_head_ind[arg_name] = word_id_get_index(arg)
        v_arg_id = arg
        # if it is not a proper noun -> expand more to cover rcmod, infmod, and so on.        
        if word_id_get_pos(v_arg_id) != "NNP": 
            arg_expand_list_final = arg_expand_list + arg_expand_non_nnp_list
        else:
            arg_expand_list_final = arg_expand_list

        g_dir_v = None
        try:
            g_dir_v = g_dir[v_arg_id]
        except:
            ##print "Faild to get adjacency network of ", v_arg_id, " while expanding it."
            continue
        for word_id, e in g_dir_v.iteritems():
            if e["rel"] in arg_expand_list_final:
                if e["rel"] == "prep_of" or e["rel"] == "prepc_of":
                    new_word_id = "of " + word_id.split("-")[0] + "-" + word_id.split("-")[1] + "-" + word_id.split("-")[2]
                    arg_extra_ids[arg_name].append(new_word_id)
                    if arg_name=='arg2':
                        dfs_rel(word_id,arg_extra_ids[arg_name],True,ADD_EXTRAS)
                else:     
                    if word_id not in arg_extra_ids[arg_name]:
                        arg_extra_ids[arg_name].append(word_id)
                        if arg_name=='arg2':
                            dfs_rel(word_id,arg_extra_ids[arg_name],True,ADD_EXTRAS)
    
    '''
    EXPAND THE RELATIONSHIP
    '''
    rel_extra_ids = []
    v_rel_id = r
    rel_extra_ids.append(v_rel_id)
    rel_head_ind = word_id_get_index(v_rel_id)
    if "SVO" not in rel["type"]:
    #if rel["type"] != "SVO":
        rel_expand_list_final = rel_expand_list + rel_expand_non_SVO_list
    else:
        rel_expand_list_final = rel_expand_list
    g_dir_v = None
    try:
        g_dir_v = g_dir[v_rel_id]
    except:
        pass
        ##print "Faild to get adjacency network of ", v_rel_id, " while expanding it."
    if g_dir_v is not None:
        for word_id, e in g_dir_v.iteritems():
            if e["rel"] in rel_expand_list_final:
                rel_extra_ids.append(word_id)
        

                
            
    arg1_final_word_str, arg1_final_with_pos_str = sort_word_ids(arg_extra_ids[arguments_names[0]], arg_head_ind[arguments_names[0]])
    arg2_final_word_str, arg2_final_with_pos_str = sort_word_ids(arg_extra_ids[arguments_names[1]], arg_head_ind[arguments_names[1]])
    rel_final_word_str, rel_final_with_pos_str = sort_word_ids(rel_extra_ids, rel_head_ind)
    
    
    
    rel_expanded = rel
    rel_expanded['arg1'] = arg1_final_word_str
    rel_expanded['arg2'] = arg2_final_word_str
    rel_expanded['rel'] = rel_final_word_str
    
    rel_expanded['arg1_with_pos'] = arg1_final_with_pos_str
    rel_expanded['arg2_with_pos'] = arg2_final_with_pos_str
    rel_expanded['rel_with_pos'] = rel_final_with_pos_str
    rel_expanded['rel_with_id']=verb_with_id

    if EXTRACT_NESTED_PREPOSITIONS_RELS:
        expansion_list = arg_expand_list # is it correct? how about get_expansion_list?
        rel_expanded['rel_prepositions'] = get_nested_preposition_rels(g_dir, r, expansion_list)
        rel_expanded['arg1_prepositions'] = get_nested_preposition_rels(g_dir, arg1, expansion_list)
        rel_expanded['arg2_prepositions'] = get_nested_preposition_rels(g_dir, arg2, expansion_list)
    
    return rel_expanded


def get_nested_preposition_rels(g_dir, main_word_id, expansion_list):
    if main_word_id == "":
        return ""
    nested_rels = []
    g_dir_v = None
    try:
        g_dir_v = g_dir[main_word_id]
    except:
        ##print "Faild to get adjacency network of ", main_word_id, " while expanding it."
        pass
    if g_dir_v is not None:                
        # extract the head words of nested relations (the ones connected with prepositions)

        for word_id, e in g_dir_v.iteritems():
            single_nested_rel = defaultdict(list)
            if "prep_" in e["rel"] or "prepc_" in e["rel"]:
                if e["rel"] == "prep_of" or e["rel"] == "prepc_of": #skip the preposition of
                    continue
                single_nested_rel_ids = []
                single_nested_rel_ids.append(word_id)
                #reason: type of preposition
                single_nested_rel["reason"] = e["rel"]
                nested_rel_head_id = word_id
                nested_rel_head_ind = word_id_get_index(word_id)
                
                g_dir_v_nested = None
                try:
                    g_dir_v_nested = g_dir[nested_rel_head_id]
                except:
                    continue
                    #print "Faild to get adjacency network of ", nested_rel_head_id, " while extracting nested relations."
                
                for word_id_nested, e_nested in g_dir_v_nested.iteritems():
                    if e_nested["rel"] in expansion_list:
                        single_nested_rel_ids.append(word_id_nested)
                nested_rel_final_word_str, nested_rel_final_with_pos_str = sort_word_ids(single_nested_rel_ids, nested_rel_head_ind)
                #print "nested rel : ", nested_rel_final_word_str
                single_nested_rel["text"] = nested_rel_final_word_str
                nested_rels.append(single_nested_rel)          
           
    nested_rels_str = ""
    for item in nested_rels:
        nested_rels_str += "REASON: " + item["reason"] + " TEXT: " + item["text"] + " -- "
    nested_rels_str.strip()
    
    return nested_rels_str
               
                
def create_node_attributes(n, annotation):
    '''
    This function takes a node (node_id) and returns its attributes 
    '''
    if n is None:
        return None
    # ROOT does not appear in the tree
    n_att = {}
    if n == "ROOT-NNP-0":
        n_word = "ROOT"
        n_att["word"] = n_word
        n_att["id"] = "ROOT-NNP-0"
        return n_att
    try:
        # extract attributes
        n_word, n_pos, n_ind = n.split('-')[0], n.split('-')[1], n.split('-')[2]        
        n_ind = int(n_ind) - 1 # make it 0 base - ROOT becomes "-1"
    except:
        #print error_msg(error_type="tokenizer")
        return None
    #n_pos = annotation['pos'][n_ind][1]
    
    n_att["word"] = n_word
    n_att["ind"] = n_ind
    n_att["pos"] = n_pos
    n_att["id"] = n
    
    return n_att

def dp_str_to_node_id(w_ind_str,pos):
    if w_ind_str == "ROOT-0":
        return "ROOT-NNP-0"
    word = w_ind_str.split('-')[0]
    try:
        word_ind = int(w_ind_str.split('-')[1])-1
        res = word+"-" + pos[word_ind][1] + "-" + str(word_ind+1)
    except:
        #print error_msg(error_type="tokenizer")
        return
    return res


#Added by Shadi
#How to find an expression?
def createMappDictionary1(annotation,t_orig):
    d = defaultdict(set)
    idis={}
    dep_parse = annotation['dep_parse']
    if dep_parse == '':
        return None
    dp_list = dep_parse.split('\n')
    pattern = re.compile(r'.+?\((.+?), (.+?)\)')
    for dep in dp_list:
        m = pattern.search(dep)
        n1 = dp_str_to_node_id(m.group(1),annotation['pos'])
        n2 = dp_str_to_node_id(m.group(2),annotation['pos'])
        n1_att = create_node_attributes(n1, annotation)
        n2_att = create_node_attributes(n2, annotation)
        if n1_att is not  None:
            if n1_att['word'] != 'ROOT':
                d[n1_att['word']].add(int(n1_att['id'].split('-')[2]))
                idis[int(n1_att['id'].split('-')[2])]=n1_att['id']
        if n2_att is not None:
            if n2_att['word'] != 'ROOT':
                d[n2_att['word']].add(int(n2_att['id'].split('-')[2]))
                idis[int(n2_att['id'].split('-')[2])] = n2_att['id']
    if t_orig:
        wordlist=nltk.word_tokenize(t_orig)
        j=1
        while j-1<len(wordlist):
            # if wordlist[j-1] not in d:
            d[wordlist[j-1]].add(j)
            j+=1
    return d, idis


def findOutcomes(f, words, res=[]):
    if not f:
        return res
    if not res:
        for node in f[0]:
            res.append([node])
        return findOutcomes(f[1:], words, res)
    new = []
    for path in res:
        for node in f[0]:
            if int(node) > int(path[-1]):
                path.append(node)
                new.append(path[:])
                path.pop()
    return findOutcomes(f[1:], words, new)

# def fixRelation(rel_ex,t_annotated,t_orig,g_dir):
#     exps = rel_ex.split(' , ')
#     for ex in exps:
#         ex = ex.replace('{', '')
#         ex = ex.replace('}', '')  # add path check from verb to other nodes.
#         ans = findWords(t_annotated,  ,t_orig)
#         print ex, ans
#         if ans:
#             print "The header is          :", findHeader(ans[0], g_dir)
#             print

def findIDs(d, idis, exdpression):
    f = []
    wordlist = nltk.word_tokenize(exdpression)
    words = []
    for word in wordlist:
        if word in d:
            words.append(word)
            f.append(list(d[word]))
    outs = findOutcomes(f, words,[])
    res = []
    for path in outs:
        new = []
        for i in range(len(path)):
            if path[i] in idis:
                new.append(idis[path[i]])
            else:
                new.append('')
        res.append(new)
    return res
def findWords(annotation,exdpression,t_orig):
    d, idis = createMappDictionary1(annotation, t_orig)
    return findIDs(d, idis, exdpression)

def findHeader(WordIds,g_dir): #given WordIDs, returns head
    finalw=[]
    if len(WordIds)==1:
        return WordIds[0]
    for ent in WordIds:
        if ent!='':
            finalw.append(ent)
    lens=[]
    if len(finalw)==1:
        return finalw[0]
    blocked_ind=[]
    for i in range(len(finalw)):
        if str(finalw[i]).find('NN')>-1 or str(finalw[i]).find('PRP')>-1:
            lens.append(nx.shortest_path_length(g_dir, source="ROOT-NNP-0", target=finalw[i]))
        else:
            lens.append(-1)
            blocked_ind.append(i)
    m_val=max(lens)
    for j in blocked_ind:
        lens[j]=m_val+1
    if len(lens)==0:
        return WordIds[0]
    ind=lens.index(min(lens))
    # while lens[ind] in lens[ind+1:]:
    #     break
    #     if finalw[ind].split('-')[1].find('N')>-1:
    #         break
    #     ind=lens[ind+1:].index(min(lens[ind+1:]))+ind+1
    #     if ind==len(lens)-1:
    #         break
    min_val=min(lens)
    for i in range(len(lens)):
        if lens[i]==min_val:
            if str(finalw[i]).find('NN')>-1 or str(finalw[i]).find('PRP')>-1 :
                return finalw[i]
    #######TODO: check here if you are willing to change anythin
    # for word in finalw:
    #     if word!=finalw[ind] and finalw[ind] and word!='':
    #         try:
    #             nx.shortest_path_length(g_dir, source=finalw[ind], target=word)
    #         except:
    #             # print "SRL error but"
    #             return finalw[ind]
    return finalw[ind]

def create_dep_graph(annotation):
    #print "\n\nIn create_dep_graph function and this is the annotation: ", annotation
    dep_parse = annotation['dep_parse']
    if dep_parse == '':
        return None
    dp_list = dep_parse.split('\n')
    #print dp_list
    pattern = re.compile(r'.+?\((.+?), (.+?)\)')    
    #g = nx.Graph()
    g_dir = nx.DiGraph()
    for dep in dp_list:
        m = pattern.search(dep)
        n1 = dp_str_to_node_id(m.group(1),annotation['pos'])
        n2 = dp_str_to_node_id(m.group(2),annotation['pos'])
        n1_att = create_node_attributes(n1, annotation)
        n2_att = create_node_attributes(n2, annotation)
        if n1_att is None or n2_att is None:
            return None
        
        g_dir.add_node(n1, **n1_att)
        g_dir.add_node(n2, **n2_att)
        e_rel = dep[:dep.find("(")]
        #edges.append(e)
        g_dir.add_edge(n1, n2, rel=e_rel, label=e_rel)
    return g_dir

def get_simp_rel(rel, option = "SVO", dataset='mothering'):
    # add options later
    '''
    Lower case, Strip
    '''
    arg1 = word_id_get_word(rel['arg1']).lower().strip()
    arg2 = word_id_get_word(rel['arg2']).lower().strip()
    r = word_id_get_word(rel['rel']).lower().strip()

    '''
    Mapping:
    (I,You,We -> Parents)
    '''    
    if dataset == "mothering":
        parent_list = ["i","we","us"]#,"you"]
        if arg1 in parent_list:
            arg1 = "parent"
        if arg2 in parent_list:
            arg2 = "parent"

        child_list = ["child","children","kid","kids","son", "sons","daughter","daughters","toddler","toddlres","boy"]
        if arg1 in child_list:
            arg1 = "child"
        if arg2 in child_list:
            arg2 = "child"    
    '''
    Stemming
    '''
    stemmer = SnowballStemmer("english")
    arg1 = stemmer.stem(arg1) 
    arg2 = stemmer.stem(arg2)
    r = stemmer.stem(r)
    
    rel_simp = rel.copy()
    rel_simp['arg1'] = arg1
    rel_simp['arg2'] = arg2
    rel_simp['rel'] = r
    return rel_simp

def get_conj_and_rels(rel, rel_expanded, g_dir, annotation):
    rel_updated = defaultdict(list)
    rel_updated = rel.copy()
    list_conj_and_rels = []
    list_conj_and_rels_simp = []
    field_names = ["arg1", "arg2", "rel"]
    arg_expansion_list = get_expansion_list(rel["arg1"], "arg", "SVO")
    for field in field_names:
        main_word_id = rel[field]  
        g_dir_v = None
        try:
            g_dir_v = g_dir[main_word_id]
        except:
            ##print "Faild to get adjacency network of ", main_word_id, " while expanding it."
            pass
        if g_dir_v is not None:
            for word_id, e in g_dir_v.iteritems():
                if "conj_and" in e["rel"]:
                    has_own_obj = False
                    has_own_subj = False
                    obj_id = -1
                    single_nested_rel = defaultdict(list)
                    single_nested_rel = rel_expanded.copy()
                    #print "conj_and relation"
                    #print single_nested_rel                 
                    single_nested_rel_ids = []
                    single_nested_rel_ids.append(word_id)
                    #reason: type of preposition
                    single_nested_rel["type"] = rel_expanded["type"]+"_conj_and_"+field
                    nested_rel_head_id = word_id
                    nested_rel_head_ind = word_id_get_index(word_id)

                    g_dir_v_nested = None
                    res_expanded_conj_and_obj = []
                    #rel_updated[field] = nested_rel_head_ind
                    rel_updated[field] = nested_rel_head_id
                    try:
                        g_dir_v_nested = g_dir[nested_rel_head_id]
                    except:
                        pass
                        ##print "Faild to get adjacency network of ", nested_rel_head_id, " while extracting nested relations."
                    if "arg" in field:
                        expansion_list = get_expansion_list(nested_rel_head_id, "arg", "SVO")
                    if "rel" == field:
                        t_verbs = get_verbs(annotation)#annotation['verbs']
                        v_ids = []
                        for v_ind_tmp, v in enumerate(t_verbs):
                            v_id = word_to_node_id(t_verbs, v_ind_tmp,annotation)
                            v_ids.append(v_id)
                        if nested_rel_head_id not in v_ids:
                            continue
                        expansion_list = get_expansion_list(nested_rel_head_id, "rel", "SVO")
                    for word_id_nested, e_nested in g_dir_v_nested.iteritems():
                        if "rel" == field:
                            if e_nested["rel"] == "nsubj" or e_nested["rel"] == "xsubj" or e_nested["rel"] == "nsubjpass" or e_nested["rel"] == "xsubjpass":
                                has_own_subj = True
                            if e_nested["rel"] == "dobj":
                                has_own_obj = True
                                obj_id = word_id_nested 
                                rel_updated["arg2"] = word_id_nested
                        if e_nested["rel"] in expansion_list:
                            single_nested_rel_ids.append(word_id_nested)
                    nested_rel_final_word_str, nested_rel_final_with_pos_str = sort_word_ids(single_nested_rel_ids, nested_rel_head_ind)
                    #print "nested rel : ", nested_rel_final_word_str
                    #single_nested_rel["text"] = nested_rel_final_word_str
                    
                    single_nested_rel[field] = nested_rel_final_word_str
                    
                    if has_own_obj and has_own_subj:
                        continue
                    
                    if has_own_obj :
                        res_expanded_conj_and_obj = expand_node_id(obj_id, g_dir, node_type="arg",
                                                                   rel_type="SVO",EXTRACT_NESTED_PREPOSITIONS_RELS=False)

                        single_nested_rel["type"] += "_dobj"
                        single_nested_rel["arg2"] = res_expanded_conj_and_obj["final_word_str"]
                        single_nested_rel["arg2_with_pos"] = res_expanded_conj_and_obj["final_word_with_pos_str"] 
                        single_nested_rel["rel_prepositions"] = res_expanded_conj_and_obj["rel_prepositions"]
                        expansion_list = get_expansion_list(obj_id, "arg", "SVO")
                        single_nested_rel["arg2_prepositions"] = get_nested_preposition_rels(g_dir, 
                                                                                             obj_id, 
                                                                                             expansion_list)
                        
                        
                    single_nested_rel['rel_prepositions'] = get_nested_preposition_rels(g_dir, rel_updated["rel"], arg_expansion_list)
                    single_nested_rel['arg1_prepositions'] = get_nested_preposition_rels(g_dir, rel_updated["arg1"], arg_expansion_list)
                    single_nested_rel['arg2_prepositions'] = get_nested_preposition_rels(g_dir, rel_updated["arg2"], arg_expansion_list)
                    
                    list_conj_and_rels.append(single_nested_rel)
                    
                    single_nested_rel_simp = get_simp_rel(single_nested_rel.copy(),option="SVO")
                    list_conj_and_rels_simp.append(single_nested_rel_simp)
                    #get svp relations
                    list_svop_rels, list_svop_rels_simp = get_svop_rels(single_nested_rel.copy())
                    list_conj_and_rels += list_svop_rels
                    list_conj_and_rels_simp += list_svop_rels_simp                            
           
    #nested_rels_str = ""
    #for item in nested_rels:
    #    nested_rels_str += "REASON: " + item["reason"] + " TEXT: " + item["text"] + " -- "
    #nested_rels_str.strip()    
    return list_conj_and_rels, list_conj_and_rels_simp

## TODO: Instead of parsing the output string, we should extract these rels from the original dep tree.
def parse_rel_prepositions(rel_prep):
    res = []
    res_entry = defaultdict(list)
    prepositions = rel_prep.split("--")
    for p in prepositions:
        p = p.strip()
        if p:
            res_entry = {}
            p = p.strip()
            m = re.search('REASON:(.+?)TEXT:', p)
            if m:
                found = m.group(1).strip()
                found = found.split("_")[1:]
                prep_str = ""
                for i in found:
                    prep_str += i + " "

                prep_str = prep_str.strip()
                res_entry["prep"] = prep_str
                
            m = re.search('TEXT:(.*)$', p)
            if m:
                found = m.group(1).strip()
                res_entry["text"] = found              
            res.append(res_entry)
            
    return res

def get_svop_rels(rel_expanded):
    list_rels_with_svp = []
    list_rels_with_svp_simp = []    
    if rel_expanded["rel_prepositions"]:
        prepositions = parse_rel_prepositions(rel_expanded["rel_prepositions"])
        for p in prepositions:
            new_row = rel_expanded.copy()
            new_row["type"] = "SV(O)P"
            new_row["pattern"] = "(nsubj, verb, (O)prep)"
            new_row["rel"] = rel_expanded["rel"] + " <<" + rel_expanded["arg2"] + ">> " + p["prep"]
            new_row["arg2"] = p["text"]
            new_row["arg2_with_pos"] = ''
            new_row["rel_prepositions"] = ''
            list_rels_with_svp.append(new_row)
            new_row_simp = get_simp_rel(new_row.copy(),option="SVO")
            list_rels_with_svp_simp.append(new_row_simp)
            
    return list_rels_with_svp, list_rels_with_svp_simp


def get_appos_rels(g_dir, annotation, EXTRACT_NESTED_PREPOSITIONS_RELS):
    list_appos_rels = []
    list_appos_rels_simp = []
    rel = {}
    for e_tuple in g_dir.edges(data=True):
        arg1_id, arg2_id, e = e_tuple[0],e_tuple[1],e_tuple[2]
        if e["rel"] == "appos":
            rel["arg1"] = arg1_id
            rel["arg2"] = arg2_id
            rel["rel"] = "{is}"
            rel["type"] = "appos"
            rel["pattern"] = "(word, appos, word)"
            #rel_expanded = expand_rel(rel.copy(), g_dir, EXTRACT_NESTED_PREPOSITIONS_RELS)
            rel_expanded = rel.copy()

            field_names = [("arg1", arg1_id), ("arg2",arg2_id)]
            for arg_name,arg_id in field_names:
                res_expanded_arg = expand_node_id(arg_id, g_dir, node_type="arg",
                                                           rel_type="SVO",EXTRACT_NESTED_PREPOSITIONS_RELS=True)

                rel_expanded[arg_name] = res_expanded_arg["final_word_str"]
                rel_expanded[arg_name + "_with_pos"] = res_expanded_arg["final_word_with_pos_str"] 
                rel_expanded["rel_prepositions"] = res_expanded_arg["rel_prepositions"]
                expansion_list = get_expansion_list(arg_id, "arg", "SVO")
                rel_expanded[arg_name + "_prepositions"] = get_nested_preposition_rels(g_dir, 
                                                                                     arg_id, 
                                                                                     expansion_list)
            
            list_appos_rels.append(rel_expanded.copy())
            rel_simp = get_simp_rel(rel_expanded.copy(),option="SVO")
            list_appos_rels_simp.append(rel_simp)            
            
    return list_appos_rels, list_appos_rels_simp

def get_sv_rels(g_dir, annotation, EXTRACT_NESTED_PREPOSITIONS_RELS):
    non_extended_copular_verbs = ["die", "died", "dies", "walk", "walks", "walked"]
    list_sv_rels = []
    list_sv_rels_simp = []
    rel = {}
    t_verbs = get_verbs(annotation)#annotation['verbs']
    for v_ind_tmp, v in enumerate(t_verbs):
        v_id = word_to_node_id(t_verbs, v_ind_tmp, annotation)
        try:
            g_dir_v = g_dir[v_id] #adjacency of v_id
        except:
            ##print v_id, " does not appeared as a separate node in parsing tree."
            continue
        subj_list = []
        subj_list_type = []
        has_subj_edge_only = True
        for word_id, e in g_dir_v.iteritems(): 
            if e["rel"] == "nsubj" or e["rel"] == "nsubjpass" or e["rel"] == "xsubj" or e["rel"] == "xsubjpass":
                subj_list.append(word_id)
                subj_list_type.append(e["rel"])
            else:
                has_subj_edge_only = False
        # for now we only take those 
        if len(subj_list) > 0:
            if has_subj_edge_only or v in non_extended_copular_verbs:
                for s_ind, s in enumerate(subj_list):
                    rel = {}
                    rel["rel"] = v_id
                    rel["arg1"] = s#s.split("-")[0]
                    rel["arg2"] = ""#o.split("-")[0]
                    rel["type"] = "SV"
                    rel["pattern"] = "(" + subj_list_type[s_ind] + ", verb)"
                    rel_expanded = expand_rel(rel.copy(), g_dir, EXTRACT_NESTED_PREPOSITIONS_RELS)
                    #print rel_expanded
                    list_sv_rels.append(rel_expanded.copy())
                    rel_simp = get_simp_rel(rel_expanded.copy(),option="SVO")
                    list_sv_rels_simp.append(rel_simp)                    
                    
    return list_sv_rels, list_sv_rels_simp
def findPssibleRels(header1,header2,header3,g_dir,isSRL=False):
    res = []
    if not header1 or not header3:
        return
    if len(header1)==1 and len(header3)==1:
        res.append([header1[0],header2[0],header3[0]])
        return res
    for root in header2:
        for leaf1 in header1:
            try:
                if not isSRL:
                    nx.shortest_path_length(g_dir, source=root, target=leaf1)
                for leaf2 in header3:
                    try:
                        if not isSRL:
                            nx.shortest_path_length(g_dir, source=root, target=leaf2)
                        res.append([leaf1,root,leaf2])
                        # return res
                    except:
                        pass
            except:
                pass
    if not res:
        res.append([header1[0], header2[0], header3[0]])
    return res


def markHeader(arg, header):
    ind=arg.find(header)
    tmp=""
    if not header:
        return "{"+arg+"}"
    repi = 0
    while ind > 0 and arg[ind] != " ":
        ind = ind + arg[ind + 1:].find(header)
        tmp = arg[0:ind + 1] + "{" + header + "}" + arg[ind + 1 + len(header):]
        repi += 1
        if repi > 10:
            return tmp
    if tmp:
        return tmp
    if ind>-1:
        return arg[0:ind]+"{"+header+"}"+arg[ind+len(header):]

    return "{"+header+"}"

# ind=arg.find(' '+header)
#     if ind>-1:
#         return arg[0:ind+1]+"{"+header+"}"+arg[ind+1+len(header):]
#     return ""

def verify_headers(leafs):
    if len(leafs)==1:
        return leafs

    for h in leafs:
        if h.split("-")[1].find('N')>-1:
            return [h]
    return leafs
def fixSRL(t_annotated, arg0, arg1, root, t_orig,g_dir):
    def findargs(h,arg_temp):
        for aaa in arg_temp:
            if h in aaa:
                res_ar=""
                for item in aaa:
                    res_ar+=item+" "
                return res_ar[0:len(res_ar)-1]
    leaf1 = findWords(t_annotated, arg0, t_orig)
    leaf2 = findWords(t_annotated, arg1, t_orig)
    root0 = findWords(t_annotated, root, t_orig)
    leaf1_h = []
    leaf2_h = []
    root0_h=[]
    for i1,ans in enumerate(leaf1):
        h = findHeader(ans, g_dir)
        leaf1_h.append(h)
    leaf1_h=verify_headers(leaf1_h)
    for i2,ans in enumerate(leaf2):
        h = findHeader(ans, g_dir)
        leaf2_h.append(h)
    leaf2_h = verify_headers(leaf2_h)
    for i3,ans in enumerate(root0) :
        h = findHeader(ans, g_dir)
        root0_h.append(h)

    res=findPssibleRels(leaf1_h,root0_h,leaf2_h,g_dir,isSRL=True)
    srl_ans=[]
    for ans in res:
        r={}
        h1,h2,h3=ans
        r['arg1']=markHeader(arg0,h1.split("-")[0])
        r['arg1_with_pos'] = findargs(h1,leaf1)
        r['arg2']=markHeader(arg1,h3.split("-")[0])
        r['arg2_with_pos'] =findargs(h3,leaf2)
        r['rel_with_id']=h2
        srl_ans.append(r)
    return srl_ans
def findIDwithIndex(t_annotated,t_orig,ind):
    _, idis=createMappDictionary1(t_annotated,t_orig)
    return idis[ind]
def createCorefMap(t_annotated,t_orig,g_dir,output):
    coref_map = {}
    for i in output['corefs']:
        main = ""
        mentions = set()
        for j in output['corefs'][i]:
            WordIds = []
            for k in range(j['startIndex'], j['endIndex']):
                WordIds.append(findIDwithIndex(t_annotated,t_orig,k))
            if j['isRepresentativeMention']:
                main = findHeader(WordIds, g_dir)
            else:
                mentions.add(findHeader(WordIds, g_dir))
        for each in mentions:
            coref_map[each] = main
    return coref_map
# Base function to extract the relationships
def get_relations(g_dir, annotation, EXTRACT_NESTED_PREPOSITIONS_RELS, option="SVO",setHeaderSRL=True,t_orig=""):
    '''
    inputs:
    g_dir : type:networkx -> directed dependency graph
    annotation : type:dictionary -> it keeps the annotations such as dep_tree, SRL, list of words and so on
    EXTRACT_NESTED_PREPOSITIONS_RELS : type:boolean -> If you want to extract nested prepositions. Example: Behnam presents Strands in the KDD conference -> nested rel : (Behnam, presents Strands in, the KDD conference)
    option : type:string -> type of relationships we want to extract (for now it is always SVO even if we extract other type of relationships)
    '''
    relations = []
    '''
    Simplified relations:
    meaning that we only keep head words, do stemming, map words to their actual actor ( I,we,you -> parents)
    '''
    relations_simp = []
    t_verbs = get_verbs(annotation)#annotation['verbs']
    for v_ind_tmp, v in enumerate(t_verbs):
        v_id = word_to_node_id(t_verbs, v_ind_tmp,annotation)
        try:
            g_dir_v = g_dir[v_id] #adjacency of v_id
        except:
            ##print v_id, " does not appeared as a separate node in parsing tree."
            continue
        nsubj_list = []
        dobj_list = []
        prep_list = []

        subj_list_type = [] #nsubj, nsubjpass, xsubjpass
        prep_list_type = [] #prep_ , prepc
        for word_id, e in g_dir_v.iteritems():
            if e["rel"] == "nsubj" or e["rel"] == "xsubj" or e["rel"] == "nsubjpass" or e["rel"] == "xsubjpass":
                nsubj_list.append(word_id)
                subj_list_type.append(e["rel"])
            if e["rel"] == "dobj":
                dobj_list.append(word_id)
            if "prep_" in e["rel"] or "prepc_" in e["rel"]:
                prep_list.append((word_id, e["rel"]))
                prep_list_type.append(e["rel"])
        if len(nsubj_list) > 0 and len(dobj_list) > 0:
            for s_ind, s in enumerate(nsubj_list):
                for o in dobj_list:
                    rel = {}
                    rel["rel"] = v_id
                    rel["arg1"] = s#s.split("-")[0]
                    rel["arg2"] = o#o.split("-")[0]
                    rel["type"] = option
                    rel["pattern"] = "(" + subj_list_type[s_ind] + ", verb, dobj)"
                    rel_expanded = expand_rel(rel.copy(), g_dir, EXTRACT_NESTED_PREPOSITIONS_RELS)
                    #print rel_expanded
                    relations.append(rel_expanded.copy())
                    rel_simp = get_simp_rel(rel_expanded.copy(),option)
                    relations_simp.append(rel_simp)

                    list_conj_and_rels, list_conj_and_rels_simp = get_conj_and_rels(rel.copy(),
                                                                                    rel_expanded.copy(), g_dir, annotation)

                    relations += list_conj_and_rels
                    relations_simp += list_conj_and_rels_simp

                    list_svop_rels, list_svop_rels_simp = get_svop_rels(rel_expanded.copy())

                    relations += list_svop_rels
                    relations_simp += list_svop_rels_simp

        # To extract SVP relationships with no object.
        #if len(nsubj_list) > 0  and len(prep_list) > 0:
        if len(nsubj_list) > 0 and len(dobj_list) == 0: #and len(prep_list) > 0:
            for s_ind, s in enumerate(nsubj_list):
                for p_ind, p in enumerate(prep_list):
                    rel = {}
                    rel["rel"] = v_id
                    rel["arg1"] = s
                    rel["arg2"] = p[0]
                    prep_type = prep_list_type[p_ind].split("_")[0]

                    if len(dobj_list) == 0:
                        rel["type"] = "SVP"
                    else:
                        rel["type"] = "SVP-O"
                        
                    if prep_type == "prepc":
                        rel["type"] += "c" #SVPc type
                    rel["pattern"] = "(" + subj_list_type[s_ind] + ", verb (no obj)," + prep_type #taking prep or prepc
                    rel_expanded = expand_rel(rel.copy(), g_dir, EXTRACT_NESTED_PREPOSITIONS_RELS,True)
                    prep_text = " ".join(p[1].split("_")[1:])
                    prep_text += " "
                    rel_expanded["arg2"] = prep_text + rel_expanded["arg2"]

                    #check if we get to the wierd scenario in which preposition is on the edge and also on the node.
                    #ex. we have edge prepc_out_of which is connected to "of" node!
                    #we skip these.
                    if rel["arg2"].split("-")[0] == p[1].split("_")[-1]:
                        #print "prepc weird syntax has been skipped!"
                        #print "sentence: ", annotation
                        #print rel_expanded["arg1"], rel_expanded["rel"], rel_expanded["arg2"]
                        continue

                    relations.append(rel_expanded.copy())
                    rel_simp = get_simp_rel(rel_expanded.copy(),option)
                    relations_simp.append(rel_simp)

    #SRL extractions:
    list_srl_dicts = annotation["srl"]
    for item in list_srl_dicts:
        rel = {}
        rel = item.copy()

        if "A0" in rel and "A1" in rel and "V" in rel:
            temp = rel.pop("V")
            rel["rel"] = "{" + temp + "}"
            if "AM-NEG" in rel and rel["AM-NEG"]:
                rel["rel"] = rel["AM-NEG"] + " " + rel["rel"]
            if not setHeaderSRL:
                rel["arg1"] = "{" + rel.pop("A0") + "}"
                rel["arg2"] = "{" + rel.pop("A1") + "}" #TODO: need to add A2 appended to A1 in case we need
                rel["type"] = "SRL"
                rel["pattern"] = "(srl-A0, srl-v, srl-A1)"
                relations.append(rel.copy())
                rel_simp = get_simp_rel(rel.copy(), option)
                relations_simp.append(rel_simp)
            else:
                try:
                    ans_srl=fixSRL(annotation,rel.pop("A0"),rel.pop("A1"),temp,t_orig,g_dir)
                    for rr in ans_srl:
                        rel["arg1"]=rr['arg1']
                        rel["arg2"]=rr['arg2']
                        rel["type"] = "SRL"
                        rel["arg1_with_pos"]=rr['arg1_with_pos']
                        rel["arg2_with_pos"] = rr['arg2_with_pos']
                        rel["pattern"] = "(srl-A0, srl-v, srl-A1)"
                        rel["rel_with_id"]=rr['rel_with_id']
                        relations.append(rel.copy())
                        rel_simp = get_simp_rel(rel.copy(),option)
                        relations_simp.append(rel_simp)
                        break
                except:
                    #print t_orig
                    #print "error happend"
                    pass
        elif "A1" in rel and "A2" in rel and "V" in rel:
            temp = rel.pop("V")
            rel["rel"] = "{" + temp + "}"
            if "AM-NEG" in rel and rel["AM-NEG"]:
                rel["rel"] = rel["AM-NEG"] + " " + rel["rel"]
            if not setHeaderSRL:
                rel["arg1"] = "{" + rel.pop("A1") + "}"
                rel["arg2"] = "{" + rel.pop("A2") + "}"
                rel["type"] = "SRL"
                rel["pattern"] = "(srl-A1, srl-v, srl-A2)"
                relations.append(rel.copy())
                rel_simp = get_simp_rel(rel.copy(), option)
                relations_simp.append(rel_simp)
            else:
                try:
                    ans_srl=fixSRL(annotation,rel.pop("A1"),rel.pop("A2"),temp,t_orig,g_dir)
                    for rr in ans_srl:
                        rel["arg1"]=rr['arg1']
                        rel["arg2"]=rr['arg2']
                        rel["type"] = "SRL"
                        rel["arg1_with_pos"]=rr['arg1_with_pos']
                        rel["arg2_with_pos"] = rr['arg2_with_pos']
                        rel["pattern"] = "(srl-A1, srl-v, srl-A2)"
                        rel["rel_with_id"] = rr['rel_with_id']
                        relations.append(rel.copy())
                        rel_simp = get_simp_rel(rel.copy(),option)
                        relations_simp.append(rel_simp)
                        break
                except:
                    #print t_orig
                    #print "error happend"
                    pass
        elif "A0" in rel and "A2" in rel and "V" in rel:
            temp = rel.pop("V")
            rel["rel"] = "{" + temp + "}"
            if "AM-NEG" in rel and rel["AM-NEG"]:
                rel["rel"] = rel["AM-NEG"] + " " + rel["rel"]
            if not setHeaderSRL:
                rel["arg1"] = "{" + rel.pop("A0") + "}"
                rel["arg2"] = "{" + rel.pop("A2") + "}"
                rel["type"] = "SRL"
                rel["pattern"] = "(srl-A0, srl-v, srl-A2)"
                relations.append(rel.copy())
                rel_simp = get_simp_rel(rel.copy(), option)
                relations_simp.append(rel_simp)
            else:
                try:
                    ans_srl=fixSRL(annotation,rel.pop("A0"),rel.pop("A2"),temp,t_orig,g_dir)
                    for rr in ans_srl:
                        rel["arg1"]=rr['arg1']
                        rel["arg2"]=rr['arg2']
                        rel["type"] = "SRL"
                        rel["arg1_with_pos"]=rr['arg1_with_pos']
                        rel["arg2_with_pos"] = rr['arg2_with_pos']
                        rel["pattern"] = "(srl-A0, srl-v, srl-A2)"
                        rel["rel_with_id"] = rr['rel_with_id']
                        relations.append(rel.copy())
                        rel_simp = get_simp_rel(rel.copy(),option)
                        relations_simp.append(rel_simp)
                        break
                except:
                    #print t_orig
                    #print "error happend"
                    pass




    # Subject, Verb, Cop extractions: It was an accident. Behnam is a great student..
    t_nouns = get_nouns(annotation)
    t_adjs = get_adjectives(annotation)

    t_nouns_adjs = t_nouns + t_adjs
    for n_ind, n in enumerate(t_nouns_adjs):
        n_id = word_to_node_id(t_nouns_adjs, n_ind, annotation)
        try:
            g_dir_n = g_dir[n_id]  # adjacency of n_id
        except:
            continue

        cop_list = []
        nsubj_list = []
        subj_list_type = [] #nsubj, nsubjpass, xsubjpass
        for word_id, e in g_dir_n.iteritems():
            if e["rel"] == "nsubj" or e["rel"] == "xsubj" or e["rel"] == "nsubjpass" or e["rel"] == "xsubjpass":
                nsubj_list.append(word_id)
                subj_list_type.append(e["rel"])
            if e["rel"] == "cop":
                cop_list.append(word_id)

        if len(nsubj_list) > 0 and len(cop_list) > 0:
            for s_ind, s in enumerate(nsubj_list):
                for c in cop_list:
                    rel = {}
                    rel["rel"] = c
                    rel["arg1"] = s  # s.split("-")[0]
                    rel["arg2"] = n_id  # o.split("-")[0]
                    rel["type"] = "SVCop"
                    rel["pattern"] = "(" + subj_list_type[s_ind] + ", verb, noun-cop)"
                    rel_expanded = expand_rel(rel.copy(), g_dir, EXTRACT_NESTED_PREPOSITIONS_RELS)
                    # print rel_expanded
                    relations.append(rel_expanded.copy())
                    rel_simp = get_simp_rel(rel_expanded.copy(), option)
                    relations_simp.append(rel_simp)

                    list_conj_and_rels, list_conj_and_rels_simp = get_conj_and_rels(rel.copy(),
                                                                                    rel_expanded.copy(), g_dir,
                                                                                    annotation)

                    relations += list_conj_and_rels
                    relations_simp += list_conj_and_rels_simp

                    list_svp_rels, list_svp_rels_simp = get_svop_rels(rel_expanded.copy())

                    relations += list_svp_rels
                    relations_simp += list_svp_rels_simp

    # Appos relationships (my brother, Ali)
    list_appos_rels, list_appos_rels_simp = get_appos_rels(g_dir, annotation, EXTRACT_NESTED_PREPOSITIONS_RELS)
    relations += list_appos_rels
    relations_simp += list_appos_rels_simp


    # SV relationships (She walks)
    list_sv_rels, list_sv_rels_simp = get_sv_rels(g_dir, annotation, EXTRACT_NESTED_PREPOSITIONS_RELS)
    relations += list_sv_rels
    relations_simp += list_sv_rels_simp
            
        
    return relations, relations_simp

def create_argument_graph(df, source, target, edge_attr=None, graph_type="directed"):
    ''' Return a graph from Pandas DataFrame.
    Modified version of "from_pandas_dataframe" function.
    '''
    if graph_type == "undirected":
        g = nx.Graph()
    elif graph_type == "directed":
        g = nx.DiGraph()
    else:
        g = nx.MultiGraph()
    
    src_i = df.columns.get_loc(source)
    tar_i = df.columns.get_loc(target)
    label_i = df.columns.get_loc(edge_attr)
    if edge_attr:
        # If all additional columns requested, build up a list  tuples
        # [(name, index),...]
        if edge_attr is True:
            # Create a list of all columns indices, ignore nodes
            edge_i = []
            for i, col in enumerate(df.columns):
                if col is not source and col is not target:
                    edge_i.append((col, i))
        # If a list or tuple of name is requested
        elif isinstance(edge_attr, (list, tuple)):
            edge_i = [(i, df.columns.get_loc(i)) for i in edge_attr]
        # If a string or int is passed
        else:
            edge_i = [(edge_attr, df.columns.get_loc(edge_attr)),]

        # Iteration on values returns the rows as Numpy arrays
        for row in df.values:
            g.add_edge(row[src_i], row[tar_i], label = row[label_i])#{i:row[j] for i, j in edge_i},label=row[label_i])
    
    # If no column names are given, then just return the edges.
    else:
        for row in df.values:
            g.add_edge(row[src_i], row[tar_i])

    return g

def create_argument_multiGraph(df, source, target,edge_attr):

    src_i = df.columns.get_loc(source)
    tar_i = df.columns.get_loc(target)
    label_i = df.columns.get_loc(edge_attr)
    
    g = nx.MultiDiGraph()
    nodes = set()
    nodes = list(nodes.union(df[source],df[target]))
    for n in nodes:
        g.add_node(n)
        ''' Get dataframe in which n is the source'''
        df_n = df[df[source] == n]
        cnt = Counter()
        for row in df_n.values:
            cnt[(row[label_i],row[tar_i])] += 1
        for k,v in cnt.most_common():
            #print n,k,v
            label_rel_freq = str(k[0])+"-"+str(v)
            g.add_edge(n,str(k[1]),label=label_rel_freq)
    return g

def filter_nodes(df,source,target, selected_nodes):
    df_filtered = df[np.logical_and(df[source].isin(selected_nodes), df[target].isin(selected_nodes))]
    return df_filtered

def filter_nodes_OR(df,source,target, selected_nodes):
    df_filtered = df[np.logical_or(df[source].isin(selected_nodes), df[target].isin(selected_nodes))]
    return df_filtered

'''
def glob_version(entity, entity_versions):

    #Extraction part -> arg or rel
    #Take an argument or relation entry, with a list of the main actors and different versions of the main actors.
    #Return the global name for the main actor.
    
    entity_new = ""
    entity_new = entity
    entity_new = entity_new.lower()
    entity_head = re.search(r'\{(.*)\}', entity_new).group(1)
    for ent_glob_name, ent_version_list in entity_versions.iteritems():
        if entity_head in ent_version_list:
            entity_new = ent_glob_name
            break
    return entity_new
'''    
    
def glob_version(entity, entity_versions):
    '''
    Extraction part -> arg or rel
    Take an argument (or relation entry), with a list of the main actors and different versions of the main actors.
    Return the global name for the main actor.
    1. We compare entity versions (and even each of their words if they have multiple parts) with the head word of an argument. 
    We map the representation of the longest entity version that part of it matches with the head word.
    2. If non of the entity versions matches with the head word, we remove {}s, 
    and map the longest entity version that is in the argument.

    Ex...
    #ent = "{apple} pay"
    #entv = {"apple":["apple"],"apple pay":["apple pay"]}
    res -> apple pay
    #ent = "second {creature}"
    #entv = {"creature":['creature'], "2nd creature":["second creature"]}
    res -> 2nd creature
    '''
    #temporary workaround - avoid crashing when there is no "{" or "}"
    if "{" not in entity and "}" not in entity:
        entity = "{" + entity.strip() + "}"
    
    if not entity:
        return None
    entity_new = ""
    entity_new = entity
    entity_new = entity_new.lower()
    entity_head = re.search(r'\{(.*)\}', entity_new).group(1)
    entv_max_len = 0
    flag_head_matches = False
    entity_no_bracket = entity_new.replace("{","").replace("}","")
    #iterate through the dictionary
    for ent_glob_name, ent_version_list in entity_versions.iteritems():
        #iterate through each of the entity version lists
        for entv_item in ent_version_list:
            #separate each entity version into separate words
            entv_item_words = entv_item.split(" ")
            #if head word matches any of the words and the complete entity version is in the argument, then we have a match
            if entity_head in entv_item_words:
                if entv_item in entity_no_bracket:
                    flag_head_matches = True
                    #we find all the matches and return the representation of the longest entity version.
                    if len(entv_item) > entv_max_len:
                        entv_max_len = len(entv_item)
                        entity_new = ent_glob_name
                    
    # if head noun does not matche, then find the longest entity version in the 
    if not flag_head_matches:
        entv_max_len = 0
        for ent_glob_name, ent_version_list in entity_versions.iteritems():
            for entv_item in ent_version_list:
                if entv_item in entity_no_bracket:
                    entv_item_as_separate_words = False
                    # if it appears as separate words inside the text
                    if (" " + entv_item + " ") in entity_no_bracket:
                        entv_item_as_separate_words = True
                    # if it appears as separate words at first
                    if not entv_item_as_separate_words: 
                        ind_first_match = entity_no_bracket.find(entv_item+" ")
                        if ind_first_match == 0:
                            entv_item_as_separate_words = True
                    # if it appears as separate words at the end
                    if not entv_item_as_separate_words:
                        ind_last_match = entity_no_bracket.rfind(" "+entv_item)
                        if ind_last_match + len(entv_item) + 1 == len(entity_no_bracket):
                            entv_item_as_separate_words = True
                    if entv_item_as_separate_words and len(entv_item) > entv_max_len:
                        entv_max_len = len(entv_item)
                        entity_new = ent_glob_name
    return entity_new    
    
def get_simp_df(df,entity_versions):
    for index, row in df.iterrows():
        # lower case the letters
        if row['arg1']:
            arg1_new = glob_version(row['arg1'],entity_versions)
        if row['arg2']:
            #print row['arg2']
            arg2_new = glob_version(row['arg2'],entity_versions)
        #row['arg1'] = arg1_new
        #row['arg2'] = arg2_new
        df.loc[index,'arg1'] = arg1_new
        df.loc[index,'arg2'] = arg2_new    
    return df

def getCorefMAp(output_1,annots,g_dirs,sents):
    coref_map_rep={}
    core_map_mens={}
    if not output_1:
        return {},{}
    for i in output_1['corefs']:
        main=[]
        for j in output_1['corefs'][i]:
            WordIds=[]
            try:
                if j['isRepresentativeMention']:
                    main=findWordIDs(sents[j['sentNum']],annots[j['sentNum']],j['startIndex'],j['text'],j['sentNum'],False) #TODO: add all words and mark header
                    coref_map_rep[i] = main

                else:
                    mention=findWordIDs(sents[j['sentNum']],annots[j['sentNum']],j['startIndex'],j['text'],j['sentNum'],True)
                    for word in mention:
                        core_map_mens[word]=i
            except:
                pass
    return coref_map_rep,core_map_mens




def findWordIDs(t_orig,t_annotated,startIndex,text,sentNum,IsRep):
    res=[]
    res_t=[]
    inds, idis=createMappDictionary1(t_annotated,t_orig)
    tokenized_text = nltk.word_tokenize(text)


    for j in range(len(tokenized_text)):
        if startIndex+j in inds[tokenized_text[j]]:
            try:
                res.append(str(idis[startIndex+j]))
                #+'*'+str(sentNum))
                res_t.append(str(idis[startIndex+j])+'*'+str(sentNum))
            except:
                #print "Some thing happend",text
                continue
        else:
            #print "The id don't match",text
            continue
    if IsRep:
        return res_t
    res_2=[]
    g_dir = create_dep_graph(t_annotated)
    hh=findHeader(res,g_dir)
    for i_1 in range(len(res)):
        if res[i_1]==hh:
            t_mp=hh.split('-')
            arg_h='{'+t_mp[0]+'}'
            for i_3 in range(1,len(t_mp)):
                arg_h+='-'+str(t_mp[i_3])
            res_2.append(arg_h+'*'+str(sentNum))
        else:
            res_2.append(res[i_1]+'*'+str(sentNum))
    return res_2

def getheadWord(s):
    res=str(s).split('{')
    if len(res)==1:
        return res[0].split('}')[0]
    else:
        return res[1].split('}')[0]

def addPronouns(coref_map_rep, core_map_mens, rels_pure, t_ind, ind,PRONOUN_RESOLUTION):
    def findHindex(args_only):
        for i, w in enumerate(args_only):
            if '{' in w:
                return i
        return -1
    def getArgSimplified(inp):
        args1=inp.split(' ')
        res_4=""
        for a in args1:
            res_4+=' '+a.split('-')[0]
        return res_4[1:]

    def findCoref(arg_with_pos, arg1):
        args = arg_with_pos.split(' ')
        args_only = arg1.split(' ')
        index_h = findHindex(args_only)
        ans = ""
        rep = ""
        for index, arg in enumerate(args):
            if arg + '*' + str(t_ind+1) in core_map_mens and not rep:
                if core_map_mens[arg + '*' + str(t_ind+1)] in coref_map_rep:
                    rep = coref_map_rep[core_map_mens[arg + '*' + str(t_ind+1)]]
                    for w in rep:
                        if index == index_h:
                            ans += ' ' + w
                            ans += '*H'
                        else:
                            ans += ' ' + w.replace('{','').replace('}','')
                        # else:
                        #     ans += ' '+w.replace('{','').replace('}','')
            elif arg + '*' + str(t_ind+1) in core_map_mens and rep:
                pass
                # if core_map_mens[arg + '*' + str(t_ind+1)] in coref_map_rep:
                #     print "This happended when ",rep
                #     rep += coref_map_rep[core_map_mens[arg + '*' + str(t_ind+1)]]
                #     for w in rep:
                #         if index == index_h:
                #             ans += ' ' + w
                #             ans += '*H'
                        # else:
                        #     ans += ' '+w.replace('{','').replace('}','')
            elif arg + '*' + str(t_ind+1) not in core_map_mens:
                # ans += ' ' + arg
                if index == index_h:
                    tmp_l=arg.split('-')
                    ans+=' {'+tmp_l[0]+'}'
                    for ind_t in range(1,len(tmp_l)):
                        ans+='-'+tmp_l[ind_t]
                else:
                    ans += ' ' + arg
        if  rep:
            return ans
        else:
            return ""

    for rel in rels_pure:
        try:
            if 'arg1_with_pos' in rel:
                arg1_with_pos = rel['arg1_with_pos']
                fixed_arg1 = findCoref(arg1_with_pos, rel['arg1'])
                rel['arg1_with_pos_pronoun'] = fixed_arg1[1:]
                rel['arg1_pronoun']=getArgSimplified(fixed_arg1)
                if PRONOUN_RESOLUTION:
                    rel['arg1_orig']=rel['arg1']
                    if rel['arg1_pronoun']:
                        tmp1=rel['arg1_pronoun']
                        while tmp1[0]==' ':
                            tmp1=tmp1[1:]
                        if '{' not in tmp1 and isPronoun(getheadWord(str(rel['arg1']))):
                            rel['arg1']=tmp1
                        elif '{' in tmp1:
                            rel['arg1'] = tmp1


            if 'arg2_with_pos' in rel:
                arg2_with_pos = rel['arg2_with_pos']
                fixed_arg2 = findCoref(arg2_with_pos, rel['arg2'])
                rel['arg2_with_pos_pronoun'] = fixed_arg2[1:]
                rel['arg2_pronoun'] = getArgSimplified(fixed_arg2)
                if PRONOUN_RESOLUTION:
                    rel['arg2_orig'] = rel['arg2']
                    if rel['arg2_pronoun']:
                        tmp1 = rel['arg2_pronoun']
                        while tmp1[0] == ' ':
                            tmp1 = tmp1[1:]
                        if '{' not in tmp1 and isPronoun(getheadWord(str(rel['arg2']))):
                            rel['arg2']=tmp1
                        elif '{' in tmp1:
                            rel['arg2'] = tmp1

        except:
            continue
    return rels_pure


def getheadWord(s):
    res = str(s).split('{')
    if len(res) == 1:
        return res[0].split('}')[0]
    else:
        return res[1].split('}')[0]


def resolveDuplicates(rels_pure):
    s_dict=defaultdict(set)
    d_dict=defaultdict(set)

    for rel in rels_pure:
        rel['isDup'] = False
        if rel['rel'] not in s_dict:
            s_dict[rel['rel']].add(getheadWord(rel['arg1']))
            d_dict[rel['rel']].add(getheadWord(rel['arg2']))
        elif getheadWord(rel['arg1']) not in s_dict[rel['rel']]:
            s_dict[rel['rel']].add(getheadWord(rel['arg1']))
            d_dict[rel['rel']].add(getheadWord(rel['arg2']))
        elif getheadWord(rel['arg2']) not in d_dict[rel['rel']]:
            s_dict[rel['rel']].add(getheadWord(rel['arg1']))
            d_dict[rel['rel']].add(getheadWord(rel['arg2']))
        else:
            rel['isDup']=True

    return rels_pure


def text_corpus_to_rels(file_input_arg,
                        DATA_SET,
                        INPUT_DELIMITER,
                        input_fname,
                        output_dir_arg,
                        MAX_ITERATION,
                        CLEAN_SENTENCES,
                        SEPARATE_SENT,
                        SHOW_DP_PLOTS,
                        SHOW_REL_EXTRACTIONS,
                        SAVE_ALL_RELS,
                        EXTRACT_NESTED_PREPOSITIONS_RELS,
                        SAVE_ANNOTATIONS_TO_FILE,
                        LOAD_ANNOTATIONS,
                        KEEP_ORDER_OF_EXTRACTIONS,
                        PRINT_EXCEPTION_ERRORS,
                        SAVE_ALL_SENTENCES_AND_ANNOTATIONS,
                        PRONOUN_RESOLUTION=False
                       ):
    
    df = read_data(file_input_arg, DATA_SET, INPUT_DELIMITER, LOAD_ANNOTATIONS)
    
    text_col_name = 'text'
    if 'text' not in df.columns:
        text_col_name = 'sentence'
    try:
        texts = df[text_col_name].tolist()
    except:
        print "Invalid input! You should have either 'text' or 'sentence' as one of the input headers."
        return
    
    output_prefix = output_dir_arg + input_fname
    f_rel = open(output_prefix +"_"+"relations_" + str(MAX_ITERATION) +".csv", "w")
    
    #f_input_plus_annotations = open(output_prefix +"_with_annotations" +".csv", "w")
    srl_headers =     ['A2', 'A3','A4', 'AM-ADV', 'AM-CAU','AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC','AM-MNR', \
     'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD','AM-TMP','C-A0', 'C-A1', 'C-A2', 'C-V', 'R-A0','R-A1', 'R-A2']
    # A0 -> has been changed to arg1, A1->arg2, V->rel 

    header = ['sentence','arg1','rel','arg2','type','pattern','arg1_with_pos','rel_with_pos','arg2_with_pos','arg1_prepositions', 'rel_prepositions', 'arg2_prepositions','arg2_with_pos_pronoun','arg1_with_pos_pronoun','arg1_pronoun','arg2_pronoun','rel_with_id','arg1_orig','arg2_orig','isDup'] + srl_headers

    #
    missing_headers = set()
    if SAVE_ANNOTATIONS_TO_FILE:
        header = header + ['annotation']
    if KEEP_ORDER_OF_EXTRACTIONS:
        header = ["post_num", "sentence_num"] + header
    if DATA_SET=="twitter":
        header = df.columns.values.tolist() + header
    if DATA_SET=="bridgegate_with_dates":
        header = df.columns.values.tolist() + header
        header.remove("text")
    if DATA_SET == "deathreports":
        header = df.columns.values.tolist() + header#[h for h in header if h not in ["sentence"]]
    dict_writer = csv.DictWriter(f_rel, header)
    dict_writer.writeheader()#writerow(header)    
    
    annotator = Annotator()
    all_rels_str = []
    all_rels = []
    output = []
    all_sents_and_annots = []
    for ind, t_orig in enumerate(tqdm(texts, ascii=True, desc="Relation Extraction From Each Post")):
        if DATA_SET=="bridgegate_with_dates" and ind == 242: #post 242 causes memory issues "kill 9" error.
            continue
        #print ind, " "
        #print t_orig
        try:
            #t_orig = unicodedata.normalize('NFKD', t_orig).encode('ascii','ignore')
            t_orig = str(t_orig)
        #t_orig = unicode(t_orig, errors='replace')
        except:
            #print t_orig
            #print ind, " "
            continue
        if MAX_ITERATION >= 0:
            if ind > MAX_ITERATION:
                break
        t_sentences = []
        try:
            if CLEAN_SENTENCES and not LOAD_ANNOTATIONS:
                t_orig = clean_sent(t_orig)
            if SEPARATE_SENT and not LOAD_ANNOTATIONS:
                t_sentences = sent_tokenize(t_orig)
                for tmp_ind in range(len(t_sentences)):
                    t_sentences[tmp_ind] = t_sentences[tmp_ind]
            else:
                t_sentences = [t_orig]
        except:
            if PRINT_EXCEPTION_ERRORS:
                print "Error in sentence tokenizer! - ", t_orig
        output_1={}
        try:
            output_1 = nlp.annotate(str(t_orig), properties={'annotators': 'coref', 'outputFormat': 'json'})
        except:
            # print t_orig
            #print "Error with pronoun"
            #:output_1={}
            continue
        annots = {}
        g_dirs = {}
        sents = {}
        coref_map_rep={}
        core_map_mens={}
        try:
            for i in range(len(t_sentences)):
                annots[i + 1] = annotator.getAnnotations(t_sentences[i], dep_parse=True)
                g_dirs[i + 1] = create_dep_graph(annots[i + 1])
                sents[i + 1] = t_sentences[i]
            coref_map_rep, core_map_mens = getCorefMAp(output_1, annots, g_dirs, sents)
        except:
            pass
        for t_ind, t in enumerate(t_sentences):
            try:
                if LOAD_ANNOTATIONS:
                    t_annotated = df.iloc[ind]["annotation"]
                    t_annotated = ast.literal_eval(t_annotated) 
                else:
                    t_annotated = annots[t_ind]#annotator.getAnnotations(t, dep_parse=True)
                if SAVE_ALL_SENTENCES_AND_ANNOTATIONS:
                    if "post_num" in df.columns:
                        post_num_tmp = df.iloc[ind]["post_num"]
                    else:
                        post_num_tmp = ind
                    if "sentence_num" in df.columns:
                        sentence_num_tmp = df.iloc[ind]["sentence_num"]
                    else:
                        sentence_num_tmp = t_ind
                    all_sents_and_annots.append([post_num_tmp, sentence_num_tmp, t, t_annotated])
            except:
                if PRINT_EXCEPTION_ERRORS:
                    print "Error in sentence annotation"
                continue
            try:
                g_dir = create_dep_graph(t_annotated)
                if g_dir is None:
                    if PRINT_EXCEPTION_ERRORS:
                        print "No extraction found"
                    continue
                if SHOW_DP_PLOTS:
                    plot_dep(g_dir,t)
                g_undir = g_dir.to_undirected()
            except:
                if PRINT_EXCEPTION_ERRORS:
                    print "Unexpected error while extracting relations:", sys.exc_info()[0]
                continue
            rels_pure, rels_simp = get_relations(g_dir, t_annotated, EXTRACT_NESTED_PREPOSITIONS_RELS, option="SVO",setHeaderSRL=True,t_orig=t)
            rels_pure = addPronouns(coref_map_rep, core_map_mens, rels_pure, t_ind, ind,PRONOUN_RESOLUTION)
            rels_pure = resolveDuplicates(rels_pure)
            rels = rels_pure
            if SHOW_REL_EXTRACTIONS:
                print ind, t, "\n"
                print "Simplifided Version:"
                print_relations(rels)
                print "More detailed Version:"
                print_relations(rels_pure)
            #else:
            #    if ind % 1000 == 0:
            #        print ind,
            all_rels_str = all_rels_str + get_rels_str(rels) #For simply counting the exact strings
            all_rels = all_rels + rels # to later create a dataframe
            for r in rels:
                output_row = defaultdict(list)
                output_row = r.copy()
                #output_row["original_text"] = t_orig
                output_row["sentence"] = t
                if SAVE_ANNOTATIONS_TO_FILE:
                    output_row["annotation"] = t_annotated
                if KEEP_ORDER_OF_EXTRACTIONS:
                    if "post_num" in df.columns:
                        post_num_tmp = df.iloc[ind]["post_num"]
                    else:
                        post_num_tmp = ind
                    if "sentence_num" in df.columns:
                        sentence_num_tmp = df.iloc[ind]["sentence_num"]
                    else:
                        sentence_num_tmp = t_ind                    
                    output_row["post_num"] = post_num_tmp
                    output_row["sentence_num"] = sentence_num_tmp
                output.append(output_row)
                #print " output is : ", output
                #output_subset = dict((k,output[k]) for k in header)
                if DATA_SET == "twitter" or DATA_SET == "deathreports":
                    #print "-------------------------df.iloc[[ind]]-------------------------"
                    #print df.iloc[[ind]]
                    d_orig = df.iloc[[ind]].to_dict(orient='records')#dict(df.iloc[[ind]].)
                    #print "after dict()"
                    #print type(d_orig)
                    #d_final = dict(d_orig)
                    #d_final.update(output_row)
                    output_row.update(d_orig[0])
                    #print output_row
                if DATA_SET == "bridgegate_with_dates":
                    d_orig = df.iloc[[ind]].to_dict(orient='records')
                    output_row.update(d_orig[0])
                    output_row.pop('text', None)
                #if DATA_SET == "deathreports":
                try:
                    dict_writer.writerow(output_row)
                except:
                    for key_tmp in output_row:
                        if key_tmp not in header:
                            #print "Extra key needed to be added to the header: ", key_tmp
                            missing_headers.add(key_tmp)
                
    
    if SAVE_ALL_SENTENCES_AND_ANNOTATIONS:
        columns = ["post_num", "sentence_num", "sentence", "annotation"]
        
        
        f_sent_annot = open(output_prefix +"_" + "sentences_and_annotations_" + str(MAX_ITERATION) +".csv", "w")
        dict_writer_sent_annots = csv.writer(f_sent_annot, columns)
        dict_writer_sent_annots.writerow(columns)
        
        for row_tmp in all_sents_and_annots:
            dict_writer_sent_annots.writerow(row_tmp)
        
        f_sent_annot.close()
        #df_sent_annot = pd.DataFrame(all_sents_and_annots, columns = columns)
        #df_sent_annot.to_csv(output_dir_arg + input_fname + "_" + "sentences_and_annotations.csv",sep=',', encoding='utf-8',header=True, columns=columns, index=False, line_terminator=None)  
        
    #'''            
    #if SAVE_ALL_RELS:
    if len(missing_headers) > 0:
        print "\n%%%%%% NOTE -> headers needed to be added: ", missing_headers , " %%%%%%\n"
        columns = ['sentence','arg1','rel','arg2','type','pattern','arg1_with_pos','rel_with_pos','arg2_with_pos']
        columns = columns + srl_headers + list(missing_headers)
        if KEEP_ORDER_OF_EXTRACTIONS:
            columns = ["post_num", "sentence_num"] + columns
        df_output = pd.DataFrame(output)
        df_output.to_csv(output_dir_arg + input_fname + "_" + "output_relations.csv",sep=',', encoding='utf-8',header=True, columns=columns)       
    #'''
    
        
    f_rel.close()
    return all_rels_str, all_rels, output


def rels_to_network(df_rels,
                    input_fname,
                    output_dir_arg,
                    MAX_ITERATION,
                    NODE_SELECTION,
                    DATA_SET,
                    SAVE_GEFX,
                    SAVE_PAIRWISE_RELS,
                    SHOW_ARGUMENT_GRAPH,
                    SAVE_G_JSON,
                    SAVE_DF_SELECTED,
                    PATH_TO_SAVED_ARG_GRAPH=None):

    if NODE_SELECTION:
        # get the list of different versions of an entity. Example : parents,parent,i,we -> parents
        entity_versions = get_entity_versions(DATA_SET)
        df_simp = get_simp_df(df_rels.copy(),entity_versions)  
        selected_nodes = entity_versions.keys()
        df_rels_selected = filter_nodes(df_simp.copy(),source='arg1',target='arg2',selected_nodes = selected_nodes)
        if SAVE_DF_SELECTED:
            df_rels_selected.to_csv(output_dir_arg + input_fname + "_" + "selected_relations.csv",sep=',', encoding='utf-8',header=True, columns=df_rels_selected.columns.tolist())
        g_arg = create_argument_multiGraph(df_rels_selected.copy(),source='arg1',target='arg2',edge_attr = 'rel')
        if SAVE_GEFX:
            nx.write_gexf(g_arg, output_dir_arg + input_fname + "_" + "g_arg_selected_"+str(MAX_ITERATION)+"_"+str(time.time())+".gexf")
        if SAVE_G_JSON:
            with open(output_dir_arg + input_fname + "_" + "g_arg_selected"+str(MAX_ITERATION)+"_"+str(time.time())+".json", 'w') as outfile:

                json.dump(node_link_data(g_arg), outfile)
        if SHOW_ARGUMENT_GRAPH:
            plot_argument_graph(g_arg)
        if SAVE_PAIRWISE_RELS:
            file_loc = output_dir_arg + input_fname + "_" + "pairwise_rels_selected_"+str(MAX_ITERATION)+"_"+DATA_SET+".txt"
            save_pairwise_rels(file_loc,g_arg,print_option=True)      
    else:
        g_arg = create_argument_multiGraph(df_rels.copy(),source='arg1',target='arg2',edge_attr = 'rel')
        if SAVE_GEFX:
            nx.write_gexf(g_arg, output_dir_arg + input_fname + "_" + "g_arg_"+str(MAX_ITERATION)+"_"+str(time.time())+".gexf")
        if SAVE_G_JSON:
            with open(output_dir_arg + input_fname + "_" + "g_arg_"+str(MAX_ITERATION)+"_"+str(time.time())+".json", 'w') as outfile:

                json.dump(node_link_data(g_arg), outfile)

        if SHOW_ARGUMENT_GRAPH:
            if PATH_TO_SAVED_ARG_GRAPH:
                plot_argument_graph(g_arg, PATH_TO_SAVED_ARG_GRAPH)
            else:
                plot_argument_graph(g_arg)
        if SAVE_PAIRWISE_RELS:
            file_loc = output_dir_arg + input_fname + "_"  + "pairwise_rels_"+str(MAX_ITERATION)+"_"+DATA_SET+".txt"
            save_pairwise_rels(file_loc,g_arg,print_option=False)  

