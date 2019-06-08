from RE_init import *
import string
from collections import Counter
#from collections import Counter
#import pandas as pd

def read_data(file_input,dataset="twitter",delim=",", LOAD_ANNOTATIONS=False):
    print(file_input)
    if dataset == "twitter-v0":      
        ff = open(file_input)
        h = ff.readline()
        header_orig = h.split(delim)
        df = pd.read_csv(file_input,delimiter=delim, header=0)#names=header_orig)
        #print df.tweet_id[0:10]
        df['tweet_posted_time'] = df['tweet_posted_time'].apply(lambda x: datetime.strptime(x.split('.')[0],                                                                                             '%Y-%m-%dT%H:%M:%S'))
        selected_columns = ['tweet_posted_time', 'tweet_text', 'main_tweet', 'ollie_conf',                 'ollie_arg1', 'ollie_rel', 'ollie_arg2', 'clean_tweet_polarity','clean_tweet_subjectivity']
        df_selected = df[[i for i in df.columns if i in selected_columns]]
        #print "df_selected values - 1"
        #df_selected.values_counts()
        #print len(df_selected.index)
        df_selected = df_selected.dropna(how = 'any')
        print("Number of instances: ")    
        print(len(df_selected.index))
        #print " selected dataframe - index 0 : ", df_selected.iloc[0]
        return df_selected
    
    if dataset == "twitter":
        #ff = open(file_input)
        #h = ff.readline()
        #header_orig = h.split(delim)
        df = pd.read_csv(file_input,delimiter=delim, header=0, error_bad_lines=False)
        df.rename(columns={'Replaced Version of Main Tweet': 'text'}, inplace=True)
        '''
        if LOAD_ANNOTATIONS:
            df_selected = df[['sentence', 'annotation']]
            df_selected.columns = ['text', 'annotation']
        else:
            df.rename(columns={'Replaced Version of Main Tweet': 'text'}, inplace=True)
            #df_selected = df[['Replaced Version of Main Tweet']]
            #df_selected.columns = ['text'] 
        '''
        return df
    
    #if dataset == "mothering" or dataset == "sentence_only" or dataset == "goodreads" or "goodreads" in dataset or dataset == "deathreports" or dataset == "ssdb" or dataset == "fakenews": 
    ff = open(file_input)
    df = pd.read_csv(file_input,delimiter=delim,header=0,error_bad_lines=False)
    return df


def get_file_input(DATA_SET):
    if DATA_SET == "twitter":
        based_dir = data_dir+ 'Tweets/'
        file_input_name = 'tweets_textOnly_sample.txt'#'sample.csv'
        file_input = based_dir + file_input_name      
        df = read_data(file_input,"twitter",",")#read the input sentences
        texts = df['text'].tolist()

    if DATA_SET == "sentence_only":
        based_dir = data_dir+ 'Tweets/'
        file_input_name = 'tweets_textOnly.txt'#'sample.csv'
        file_input = based_dir + file_input_name      
        df = read_data(file_input,"sentence_only","\n")#read the input sentences
        texts = df['text'].tolist()  

    elif DATA_SET == "mothering":
        based_dir = data_dir + 'Vaccination/'
        file_input_name = 'sents.txt'
        #file_input_name = 'sent_cdb_child_exemption.txt'
        file_input = based_dir + file_input_name
        # file_input is extra - should be removed later

    return file_input
    
def save_pairwise_rels(file_loc,g,print_option=True):
    f = open(file_loc,'w')
    nodes = g.nodes()
    for n1 in nodes:
        for n2 in nodes:
            if n1 is not n2:
                l = g.get_edge_data(n1,n2)
                if l:
                    line = str(n1) + "\t" + str(n2) + "\t" + str(l) + "\n"
                    f.write(line)
                    if print_option:
                        print(n1,n2,l)    
    f.close()
def plot_argument_graph(g, path_to_file=None):
    A = nx.nx_agraph.to_agraph(g)
    A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
    d = draw(g, show='ipynb')
    #display(d)
    if path_to_file:
        with open(path_to_file, "wb") as png:
            A.draw(path_to_file)
        #d.savefig(path_to_file)
    
def plot_dep(g,title):
    '''
    This function takes a DIRECTED graph as input, and plot it inline in Ipython.
    '''
    #set figure size
    '''
    plt.figure(figsize=(14,8))
    #set style of the graph
    pos = graphviz_layout(g,prog='dot')

    # 
    node_labels = nx.get_node_attributes(g, "id")
    nx.draw_networkx_labels(g, pos, labels = node_labels, font_size=11)
    edge_labels_tupels = nx.get_edge_attributes(g, "rel")
    #print edge_labels_tupels
    #edge_labels = [(e[0][0],e[0][1],e[1].value()) for e in edge_labels_tupels]
    edge_labels = edge_labels_tupels
    #edge_labels = edge_labels_dict.values()
    #print edge_labels
    nx.draw_networkx_edge_labels(g, pos, labels = edge_labels)
    nx.draw_networkx(g,pos=pos,  arrows=True, with_labels=False, node_size=1500, alpha=0.3, node_shape = 's') 
    
    #nx.nx_pylab.  
    plt.title(title)
    plt.savefig('Dep_tree.png')
    plt.show()
    '''
    A = nx.nx_agraph.to_agraph(g)
    
    A.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
    A.draw('test.png')
    d = draw(g, show='ipynb')
    display(d)
    
def print_relations(rels):
    if len(rels) < 1:
        print("No extraction.")
        return
    for ind,r in enumerate(rels):
        print(">Extraction Number: ",ind+1, " - ", "Pattern: ", r["type"]," - relation : (", r["arg1"], ", ", r["rel"], ", ", r["arg2"] ,")")
        if "arg1_prepositions" in r and "rel_prepositions" in r and "arg2_prepositions" in r:
            if r["arg1_prepositions"]:
                print(" arg1_prep: ", r["arg1_prepositions"], end=' ')
            if r["rel_prepositions"]:
                print(" rel_prep: ", r["rel_prepositions"], end=' ')
            if r["arg2_prepositions"]:
                print(" arg2_prep: ", r["arg2_prepositions"])   
            #print "----- Extra: arg1_prep: ", r["arg1_prepositions"], " rel_prep: ", r["rel_prepositions"], " arg2_prep: ", r["arg2_prepositions"] ,"\n\n"

        print("\n\n")

def get_rels_str(rels):
    if len(rels) < 1:
        return []
    rels_str = []
    for r in rels:
        r_str = "( " + r["arg1"] + ", " + r["rel"] + ", " + r["arg2"] + " )"
        rels_str.append(r_str)
    return rels_str
    
def saveToFile_rows(outputLoc, inputList, delim):
    with open(outputLoc,"wb") as f:
        writer = csv.writer(f,delimiter=delim)
        writer.writerows(inputList)   
        
def print_top_relations(all_rels,output_file, top_num=-1):
    f = open(output_file, 'w')
    cnt = Counter()
    for r in all_rels:
        cnt[r] += 1
    if top_num == -1: # means print all
        print("Frequent relations:", file=f)
        for letter,count in cnt.most_common():
            print(letter, ": ", count, file=f)
    else:
        print("top ", top_num, " frequent relations:", file=f)
        for letter,count in cnt.most_common(top_num):
            print(letter, ": ", count, file=f)                 

def save_pairwise_relations_with_node_selection(df_rels,entity_versions,output_file):
    f = open(output_file, 'w')
    cnt = Counter()
    for entity in entity_versions:
        print("-------------------------", file=f)
        print("       ", entity, file=f)
        print("-------------------------", file=f)    
        for ent_one_version in entity_versions[entity]:
            print("\n\n**** ", ent_one_version, " ****", file=f)
            df_all_versions = defaultdict(list)
            df_one_version = df_rels[np.logical_or(df_rels['arg1'].str.contains(ent_one_version),df_rels['arg2'].str.contains(ent_one_version))]
            list_one_version = df_one_version['rel'].tolist()
            for r in list_one_version:
                cnt[r] += 1
            print("Frequent relations:", file=f)
            for letter,count in cnt.most_common():
                print(letter, ": ", count, file=f)             

def rel_to_stemRel(r):
    stemmer = SnowballStemmer("english")
    r_new = ""
    # let's remove the {} inside the << >> -> to not mistakenly take it as the head noun
    r_has_less_than_equal = re.search(r'\<<(.*)\>>', r)
    if r_has_less_than_equal:
        r = r.split("<<")[0] + re.search(r'\<<(.*)\>>', r).group(0).replace("{","").replace("}","") + r.split(">>")[1]
    if is_entity_present(r.replace("{","").replace("}",""), "not") or "cannot" in r:        
        rel_head = re.search(r'\{(.*)\}', r).group(1).replace("{","").replace("}","")
        rel_head = stemmer.stem(rel_head) 
        rel_head = get_relation_representative(rel_head, dataset="dream")
        r_new = r.split("{")[0] + "{" + rel_head + "}" + ''.join([x for i,x in enumerate(r.split("}")) if i > 0 ])
    else:
        rel_head = re.search(r'\{(.*)\}', r).group(1).replace("{","").replace("}","")
        rel_head = stemmer.stem(rel_head) 
        rel_head = get_relation_representative(rel_head, dataset="dream")
        r_new = "{" + rel_head + "}"    
    return r_new


def get_top_extractions(df_rels, output_file=None, top_num=-1, save_to_file=False, stem_rels=False, just_head_arg=False):
    cnt = Counter()
    for ind, r in df_rels.iterrows():
        # get arg1, and arg2 headwords
        if just_head_arg:
            arg1_simp = "{" + re.search(r'\{(.*)\}', r["arg1"].strip()).group(1) + "}"
            arg2_simp = "{" + re.search(r'\{(.*)\}', r["arg2"].strip()).group(1) + "}"
        else:
            arg1_simp = r["arg1"].strip()
            arg2_simp = r["arg2"].strip()
        # get stem version of rel
        if stem_rels:
            rel_simp = rel_to_stemRel(r["rel"].strip())
        else:
            rel_simp = r["rel"].strip()
        key_str = arg1_simp + ";" + rel_simp + ";" + arg2_simp
        cnt[key_str] += 1

    list_aggregated_rels = []
    header_aggregated_rels = ["relation tuple", "counts"]
    if top_num == -1:    
        for k,v in cnt.most_common():
            list_aggregated_rels.append([k,v])  

    else:
        for k,v in cnt.most_common(top_num):
            list_aggregated_rels.append([k,v])         
    
    df = pd.DataFrame(list_aggregated_rels, columns = header_aggregated_rels)
    
    if output_file is not None and save_to_file:
        #df.to_csv(based_dir + input_name_prefix + '_rels_aggregated.csv', index=False) 
        df.to_csv(output_file + '_rels_aggregated.csv', index=False) 
    
    return df
         
        
    
def get_top_entities(df_rels, output_file=None, top_num=-1, save_to_file=False, just_head_arg=True):
    entities = []
    if just_head_arg:
        for ind, item in df_rels.iterrows():
            if "{" not in item["arg1"]:
                entities.append("{" + item["arg1"].strip() + "}")
            else:
                entities.append("{" + re.search(r'\{(.*)\}', item["arg1"].strip()).group(1) + "}")
            if "{" not in item["arg2"]:
                entities.append("{" + item["arg2"].strip() + "}")
            else:
                entities.append("{" + re.search(r'\{(.*)\}', item["arg2"].strip()).group(1) + "}")
    else:
        entities = list(df_rels['arg1']) + list(df_rels['arg2'])
    cols = ['entity', 'pos' ,'frequency']
    df_entity_rankings = pd.DataFrame(columns = cols)
    cnt = Counter()
    for e in entities:
        cnt[e] += 1
    #'''
    if top_num == -1: # means print all
        #print "Frequent relations:"
        for letter,count in cnt.most_common():
            #print letter, ": ", count
            letter_no_bracket = letter.replace("{","").replace("}","")
            if letter_no_bracket:
                letter_pos = nltk.tag.pos_tag([letter_no_bracket])
                df_entity_rankings.loc[len(df_entity_rankings)] = [letter, letter_pos[0][1], count]
    else:
        #print "top ", top_num, " frequent relations:"
        for letter,count in cnt.most_common(top_num):
            #print letter, ": ", count 
            letter_no_bracket = letter.replace("{","").replace("}","")
            if letter_no_bracket:
                letter_pos = nltk.tag.pos_tag([letter_no_bracket])
                df_entity_rankings.loc[len(df_entity_rankings)] = [letter, letter_pos[0][1], count]
    #'''
    
    
    
    if output_file is not None and save_to_file:
        f = open(output_file, 'w')
        df_entity_rankings.to_csv(output_file,sep=',', encoding='utf-8',header=True, columns=cols)      
        
    return df_entity_rankings





def get_top_relations(df_rels, output_file=None, top_num=-1, save_to_file=False, stem_rels=True, dataset="dream"):
    stemmer = SnowballStemmer("english")
    
    relations = []#list(df_rels['rel'])    
    
    for ind, item in df_rels.iterrows():
        r = item["rel"]
        r_new = ""
        
        # if there is no head noun specified in the relation phrase, then take the whole phrase as the main content.
        if "{" not in r:
            r_new = stemmer.stem(r)
            r_new = get_relation_representative(r_new, dataset)
            r_new = "{" + r_new + "}"
            continue
        
        # if we have the head words ({...})
        # let's remove the {} inside the << >> -> to not mistakenly take it as the head noun
        r_has_less_than_equal = re.search(r'\<<(.*)\>>', r)
        if r_has_less_than_equal:
            r = r.split("<<")[0] + re.search(r'\<<(.*)\>>', r).group(0).replace("{","").replace("}","") + r.split(">>")[1]
        if is_entity_present(r.replace("{","").replace("}",""), "not") or "cannot" in r:
            
            rel_head = re.search(r'\{(.*)\}', r).group(1).replace("{","").replace("}","")
            rel_head = stemmer.stem(rel_head) 
            rel_head = get_relation_representative(rel_head, dataset)
            r_new = r.split("{")[0] + "{" + rel_head + "}" + ''.join([x for i,x in enumerate(r.split("}")) if i > 0 ])
        else:
            rel_head = re.search(r'\{(.*)\}', r).group(1).replace("{","").replace("}","")
            rel_head = stemmer.stem(rel_head) 
            rel_head = get_relation_representative(rel_head, dataset)
            r_new = "{" + rel_head + "}"
        
        
        relations.append(r_new)
    
    cols = ['relation', 'frequency']
    df_relation_rankings = pd.DataFrame(columns = cols)
    cnt = Counter()
    for r in relations:
        cnt[r] += 1
    if top_num == -1: # means print all
        #print "Frequent relations:"
        for letter,count in cnt.most_common():
            #print letter, ": ", count
            df_relation_rankings.loc[len(df_relation_rankings)] = [letter, count]
    else:
        #print "top ", top_num, " frequent relations:"
        for letter,count in cnt.most_common(top_num):
            #print letter, ": ", count 
            df_relation_rankings.loc[len(df_relation_rankings)] = [letter, count]

    if output_file is not None and save_to_file:
        f = open(output_file, 'w')
        df_relation_rankings.to_csv(output_file,sep=',', encoding='utf-8',header=True, columns=cols)      
        
    return df_relation_rankings

def get_relation_representative(r, dataset="dream"):
    rel_to_representative_mapping = get_relation_versions_reverse_mapping(dataset)
    res = r
    if len(rel_to_representative_mapping[r]) > 0:
        res = rel_to_representative_mapping[r]
    return res

def print_full(df):
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def error_msg(error_type):
    if error_type == "tokenizer":
        return "Tokenizer failed during parsing, Ex. there might be a dash in the sentence!"
    

def get_relation_versions_reverse_mapping(dataset="dream"):
    relation_versions = get_relation_versions(dataset)
    relation_versions_reverse_mapping = defaultdict(list)
    
    for rel_glob_name, rel_version_list in relation_versions.items():
        for ind in range(len(rel_version_list)):
            relation_versions_reverse_mapping[rel_version_list[ind]] = rel_glob_name 
    
    return relation_versions_reverse_mapping
    
def get_relation_versions(dataset="dream"):
    relation_versions = defaultdict(list)
    if dataset == "dream":
        relation_versions['have'] = ['have', 'hav', 'has', 'had']
        relation_versions['see'] = ['see','saw']
        relation_versions['take'] = ['take', 'took']
        relation_versions['tell'] = ['tell', 'told','say', 'said', 'ask', 'asked']
        
        
    # make everything lower case
    for rel_glob_name, rel_version_list in relation_versions.items():
        for ind in range(len(rel_version_list)):
            rel_version_list[ind] = rel_version_list[ind].lower()
            
    return relation_versions
    
def get_entity_versions(dataset="mothering"):
    entity_versions = defaultdict(list)
    if dataset=="mothering":
        entity_versions['parents'] = ['parents', 'parent', 'i', 'we' , 'us']#, 'you']
        entity_versions['children'] = ['child', 'kid', 'kids', 'children', 'daughter', 'daughters',
                                       'son', 'sons', 'toddler',
                                       'toddlers', 'kiddo', 'boy','dd','ds']
        entity_versions['medical prof'] = ['doctor', 'doctors', 'pediatrician', 
                                           'pediatricians', 'nurse', 'nurses', 'ped', 'md', 'dr']
        entity_versions['government'] = ['government', 'cdc', 'federal', 'feds',
                                         'center for disease control', 'officials',
                                         'politician', 'official', 'law']
        entity_versions['religous inst'] = ['faith', 'religion', 'pastor', 'pastors',
                                            'parish', 'parishes', 'church', 'churches',
                                            'congregation', 'congregations', 'clergy']
        entity_versions['schools'] = ['teacher', 'teachers', 'preschools', 'preschool', 
                                      'school', 'schools', 'class', 
                                      'daycare', 'daycares', 'classes']
        entity_versions['vaccines'] = ['vaccines', 'vax', 'vaccine', 'vaccination', 
                                       'vaccinations', 'shots', 'shot', 'vaxed',
                                       'unvax', 'unvaxed', 'nonvaxed', 'vaccinate',
                                       'vaccinated', 'vaxes', 'vaxing', 'vaccinating',
                                       'substances', 'ingredients']
        entity_versions['exemptions'] = ['exemption', 'exempt']
        entity_versions['VPDs'] = ['varicella', 'chickenpox', 
                                   'flu', 'whooping cough', 'tetanus', 'pertussis', 
                                   'hepatitis', 'polio', 'mumps', 'measles', 'diphtheria']
        entity_versions['adverse effects'] = ['autism', 'autistic', 'fever', 'fevers',
                                              'reaction', 'reactions', 'infection', 'infections', 'inflammation', 'inflammations',
                                              'pain', 'pains', 'bleeding', 'bruising', 'diarrhea', 'diarrhoea']
        
    if dataset=="twitter":
        entity_versions['applepay'] = ['applepay']
        entity_versions['samsungpay'] = ['samsungpay']
        entity_versions['samsung'] = ['samsung']
        entity_versions['apple'] = ['apple']
        entity_versions['verizon'] = ['verizon']
        entity_versions['googlewallet'] = ['googlewallet']
        entity_versions['mastercard'] = ['mastercard']
        entity_versions['barclaycard'] = ['barclaycard']
        entity_versions['mcdonalds'] = ['mcdonalds']
        entity_versions['riteaid'] = ['riteaid']
        entity_versions['barclays'] = ['barclays']
        entity_versions['hsbc'] = ['hsbc', 'hsbc direct']
        entity_versions['paypal'] = ['paypal']
        entity_versions['youtube'] = ['youtube']
        entity_versions['amex'] = ['american express', 'express', 'amex']
        entity_versions['starbucks'] = ['starbucks']
        entity_versions['looppay'] = ['looppay']
        entity_versions['chinese hackers'] = ['hackers', 'hackers', 'chinese hackers']
        entity_versions['the uk'] = ['the uk', 'uk']
        entity_versions['the us'] = ['the us', 'us']
        entity_versions['chilis'] = ['chilis']
        entity_versions['banks'] = ['banks', 'bank']
        entity_versions['lloyds'] = ['lloyds']
        entity_versions['tesco'] = ['tesco']
        entity_versions['chevron'] = ['chevron']

        
    if dataset=="goodreads-Hobbit":
        #entity_versions['hobbit'] = ['hobbit', 'hobbits', 'hobit']
        entity_versions['bilbo'] = ['bilbo', 'baggins', 'the hobbit', 'burglar','bilbo baggins','hobbit', 'hobbits', 'hobit']
        entity_versions['dwarves'] = ['dwarf', 'dwarves', 'dwarvs', 'dwarfs']
        #entity_versions['tolkien'] = ['tolkien', 'tolkein']
        entity_versions['the ring'] = ['the ring', 'ring']
        entity_versions['gandalf'] = ['gandalf', 'wizard', 'gandolf']
        entity_versions['dragon'] = ['dragon', 'smaug', 'dragons']
        entity_versions['goblins'] = ['goblin', 'goblins']
        entity_versions['elves'] = ['elf', 'elves', 'elvs', 'elfs']
        entity_versions['wood-elves'] = ['wood-elves', 'wood elves', 'woodelves', 'woodelvs', 'woodelfs', 'woodelf']
        entity_versions['treasure'] = ['treasure', 'treasures', 'arkenstone', 'gem']
        entity_versions['gollum'] = ['gollum', 'smeagol', 'smagol', 'smegol', 'golum', 'gollem', 'golem', 'gollumn', 'golumn']
        entity_versions['bard'] = ['bard', 'archer']
        entity_versions['human'] = ['human', 'humans', 'man', 'men', 'lake-men']
        entity_versions['laketown'] = ['laketown', 'lake-town', 'lake town', 'village']
        entity_versions['warg'] = ['warg', 'wolves', 'wolf', 'wolverine', 'wolverines', 'wargs']
        entity_versions['beorn'] = ['beorn', 'bear', 'woodsman', 'man-bear', 'skin-changer']
        entity_versions['mountain'] = ['mountain', 'mountains']
        entity_versions['elrond'] = ['elrond']
        entity_versions['sauron'] = ['sauron', 'lord sauron', 'dark lord sauron']
        entity_versions['mirkwood'] = ['mirkwood', 'forest']
        entity_versions['spiders'] = ['spiders', 'spider', 'giant spiders','giant spider']
        entity_versions['rivendell'] = ['rivendell']
        entity_versions['eagles'] = ['eagles']
        entity_versions['hobbitown'] = ['hobbitown','hobbit hole','shire']
        entity_versions['bombur'] = ['bombur']
        entity_versions['bofur'] = ['bofur']
        entity_versions['bifur'] = ['bifur']
        entity_versions['nori'] = ['nori']
        entity_versions['dori'] = ['dori']
        entity_versions['ori'] = ['ori']
        entity_versions['gloin'] = ['gloin']
        entity_versions['oin'] = ['oin']
        entity_versions['balin'] = ['balin']
        entity_versions['dwalin'] = ['dwalin']
        entity_versions['kili'] = ['kili']
        entity_versions['fili'] = ['fili']
        entity_versions['thror'] = ['thror']
        entity_versions['thorin'] = ['thorin', 'oakenshield', 'thorin oakenshield']
        entity_versions['trolls']=['troll','trolls','trol','trols']        
        

    if dataset=="goodreads-Frankenstein": 
        entity_versions['monster'] = ['monster', 'monsters', 'monsterous', 'creature','creatures', 'creation','creations','the monster']
        entity_versions['frankenstein'] = ['frankenstein','frankensteins','victor','victors', 'victor_frankenstein', 'victor_frankensteins', 'victor frankenstein', 'creator', 'creators', 'doctor', 'dr', 'doctor frankenstein', 'dr. frankenestein']
        #entity_versions['god'] = ['creator','creators', 'god']
        #entity_versions['mary shelley'] = ['mary_shellei', 'mary_shelley','mary','shellei','shelley','mary shellei', 'mary shelley', 'author']
        #entity_versions['man'] = ['man','mans']
        entity_versions['female monster'] = ['new creature', 'second creature', '2nd creature', 'another creature','female',' female companion','counterpart']
        #entity_versions['novel'] = ['novel','novelization','stori','stories','tale', 'story', 'book']
        #entity_versions['human'] = ['human','humane','humanity','humans','humanness']
        #entity_versions['life'] = ['life','lifes','life.']
        #entity_versions['death'] = ['death','deaths'] 
        #entity_versions['revenge'] = ['revenge','reveng']
        #entity_versions['letter'] = ['letter','letters']
        entity_versions['elizabeth'] = ['elizabeth', 'wife', 'elizabeth lavenza', 'lavenza']
        entity_versions['walton'] = ['walton', 'robert', 'robert walton']
        entity_versions['henry'] = ['henry', 'clerval', 'henry clerval']
        #entity_versions['doctor'] = ['doctor']
        entity_versions['dracula'] = ['dracula']
        entity_versions['justine moritz'] = ['justine','justine moritz', 'nanny']
        entity_versions['crowd'] = ['crowd']
        entity_versions['alphonse frankenstein']=['alphonse frankenstein', 'alphonse', 'father']
        entity_versions['william']=['william','william frankenstein','borther william','borther']
        entity_versions['krempe']=['krempe']
        entity_versions['kiwin']=['kirwin']
        entity_versions['beaufort']=['beauford','beaufort']
        entity_versions['caroline']=['caroline', 'caroline beaufort', 'mother'] 
        entity_versions['margaret saville']=['margaret saville','margaret','saville']
        entity_versions['waldman']=['waldman', 'm. waldman', 'mother'] 
 
 
    if dataset=="goodreads-Mice-and-men":
        entity_versions['Lennie'] = ['lennie', 'lennie small', 'lenny', 'small']
        entity_versions['George'] = ['george', 'george milton', 'milton']
        entity_versions['Candy'] = ['candy']
        entity_versions["Curley's wife"] = ["curley's wife", "curley wife", "tramp", "tart", "looloo", 'wife', "curleys wife", "curley s wife"]
        entity_versions['Crooks'] = ['crooks', 'crooke']
        entity_versions['Curley'] = ['curley'] 
        entity_versions['Slim'] = ['slim']
        entity_versions['Carlson'] = ['carlson']
        entity_versions['The Boss'] = ['the boss', "curley's father", "boss"]
        entity_versions['Aunt Clara'] = ['aunt clara', "lennie's aunt", 'aunt', 'clara']
        entity_versions['Whit'] = ['whit']
        entity_versions["Candy's dog"] = ["Candy's dog", "dog","old dog", "Candy s dog", "Candys dog"]
        #entity_versions['Steinbeck'] = ['steinbeck', 'john steinbeck', 'john', 'author', 'steinback']
        entity_versions['Puppy'] = ['puppy', 'the puppy']
        entity_versions['Mice'] = ['mice', 'mouse']
        #entity_versions['Neck'] = ['neck']
        #entity_versions['Rabbits'] = ['rabbits', 'rabbit']
        #entity_versions['Body'] = ['body']
        #entity_versions['They'] = ['they']
        entity_versions['Ranchhands'] = ['ranchhands', 'workers', 'ranch hands']
        entity_versions['Ranch'] = ['ranch', 'farm']
        entity_versions['Dream'] = ['dream', 'dreams']
        entity_versions['Soft Things'] = ['soft things', 'soft']
        entity_versions['Mental Disability'] = ['mental disability', 'disability', 'mental']

 
 
    if dataset=="goodreads-Mockingbird":
        entity_versions['scout'] = ['scout', 'scout finch', 'sister']
        entity_versions['atticus'] = ['atticus', 'atticus finch', 'father']
        entity_versions['jem'] = ['jem', 'jem finch', 'brother']
        entity_versions['lee'] = ['lee', 'harper lee', 'author']
        entity_versions['tom'] = ['tom', 'tom robinson', 'robinson', 'black man', 'negro', 'black negro','mockingbird']
        entity_versions['bob ewell'] = ['bob ewell', 'bob', 'mr. ewell', 'mr ewell','bob ewel', "Mayella's father", "Mayella father"]
        entity_versions['boo'] = ['boo', 'arthur', 'arthur radley', 'boo radley','arthur boo radley']
        entity_versions['dill'] = ['dill', 'charles baker', 'dill harris', 'charles baker dill harris']
        entity_versions['mayella'] = ['mayella', 'mayella ewell', 'mayella ewel', 'white girl','ewell daughter', "ewell's daughter", "bob ewell's daughter",'white woman']
        entity_versions['nathan radley'] = ['nathan radley', 'nathan']
        entity_versions['radley place'] = ['radley place', "radley's place", 'radley property', "radley's property", 'radley house']
        entity_versions['judge'] = ['judge', 'judge taylor']
        entity_versions['jury'] = ['jury']
        entity_versions['alexandra'] = ['alexandra', 'aunt alexandra', 'aunt', 'his aunt', 'her aunt', "jem's aunt"]
        entity_versions['maycomb'] = ['maycomb']
        #entity_versions['racism'] = ['racism']
        entity_versions['bluejays'] = ['bluejays']
        entity_versions['miss maudie'] = ['miss maudie', 'maudie', 'maudie atkinson']
        entity_versions['trial'] = ['trial', 'court']
        #entity_versions['fact'] = ['fact']
        #entity_versions['case'] = ['case']
        entity_versions['mrs. dubose'] = ['mrs. dubose', 'dubose', 'mrs dubose']
        entity_versions['gregory peck'] = ['gregory peck', 'peck', 'gregory']
        #entity_versions['justice'] = ['justice']
        #entity_versions['prejudice'] = ['prejudice']
        entity_versions['jean louise finch'] = ['jean louise finch']
        entity_versions['school'] = ['school']
        entity_versions['gift'] = ['gift', 'gifts']
        entity_versions['heck tate'] = ['heck tate', 'heck', 'tate']
        entity_versions['children'] = ['children']
        entity_versions['calpurnia'] = ['calpurnia']
        entity_versions['people'] = ['people', 'townspeople', 'town']
        entity_versions['mr walter cunningham'] = ['mr walter cunningham', 'cunningham', 'mr cunningham', 'walter', 'walter cunningham']
        entity_versions['mr dolphus raymond'] = ['mr dolphus raymond', 'raymond', 'mr raymond']
        entity_versions["tom's widow"]=["tom's widow", "tom robinson's widow", 'widow', "tom's wife"]
    
    
    if dataset=="sddb-car":
        entity_versions['car'] = ['car', 'cars']
        entity_versions['I'] = ['i', 'me']
        entity_versions['person'] = ['he', 'she', 'him', 'her']
        entity_versions['dream'] = ['dream']
        entity_versions['mom'] = ['mom', 'mother']
        entity_versions['dad'] = ['dad', 'father']
        entity_versions['people'] = ['people']
        entity_versions['home'] = ['home']
        entity_versions['door'] = ['door']
        entity_versions['house'] = ['house']
        entity_versions['lila'] = ['lila']
        entity_versions['hand'] = ['hand']
        entity_versions['man'] = ['man', 'guy']        
        entity_versions['woman'] = ['woman']
        entity_versions['baby'] = ['baby']
        entity_versions['card'] = ['card']
        entity_versions['police'] = ['police', 'policeman', 'cop']
        entity_versions['aids'] = ['aids']
        entity_versions['sex'] = ['sex']        
        entity_versions['room'] = ['room', 'place']
        entity_versions['gun'] = ['gun']
        entity_versions['knife'] = ['knife']
        entity_versions['feeling'] = ['feeling']
        entity_versions['janie'] = ['janie']
        entity_versions['wilgespruit'] = ['wilgespruit']
        entity_versions['love'] = ['love']
        entity_versions['suicide'] = ['suicide']        
        entity_versions['bus/train'] = ['bus', 'train']
        entity_versions['mouse'] = ['mouse']
        entity_versions['hilary'] = ['hilary']  
        
        
    if dataset=="pizzagate":
        entity_versions['Oliver Willis'] = ['oliver willis', 'oliver', 'willis']
        entity_versions['Alefantis'] = ['alefantis', 'james', 'james alefantis', 'james alefantis comet ping pong owner', 'mr. alefantis', 'the owner alefantis']
        entity_versions['Hillary Clinton'] = ['clinton', 'hillary', 'hillary clinton']  
        entity_versions['Trump'] = ['trump', 'donald', 'donald trump']
        entity_versions['pedophiles'] = ['pedophiles', 'pedos', 'pedo']  
        entity_versions['David Brock'] = ['david brock', 'david', 'brock']
        entity_versions['Alig'] = ['alig', 'michael']        
        entity_versions['Katie Reilly'] = ['katie reilly', 'katie', 'reilly']        
        entity_versions['Ring'] = ['ring']        
        entity_versions['Comet/Pizza'] = ['comet', 'ping','pong', 'restaurant', 'pizza']
        entity_versions['Pizzagate'] = [ 'pizzagate']
        #entity_versions['Podesta'] = ['podesta', 'podestas']        
        entity_versions['Obama'] = ['obama', 'barack', 'barack obama']    
        entity_versions['(sub)reddit'] = ['subreddit', 'reddit', 'this subreddit']
        entity_versions['Children'] = ['children','child', 'kids', 'kid', 'girl', 'boy', 'girls', 'boys']
        entity_versions['John Podesta'] = ['john', 'john podesta', 'j. podesta', 'podesta']
        entity_versions['Tony Podesta'] = ['tony', 'tony podesta', 't. podesta']  
        entity_versions['Democrats'] = ['democrat', 'democrats', 'democratic', 'democratics']
        entity_versions['WikiLeaks'] = ['wikileaks' , 'wikileak', 'by wikileaks']
        entity_versions['Social Media Account/Content'] = ['instagram' , 'about instagram photos', 'instagram account', 'account', 'an account', 'posts', 'meme', 'photos', 'pictures', 'photo', 'picture', 'image', 'twitter']
        entity_versions['Authorities/Police/FBI'] = ['police', 'cop', 'cops', 'polices', 'fbi', 'authorities', 'the fbi'] 
        entity_versions['Voat'] = ['voat']
        entity_versions['Abuse'] = ['rape', 'abuse']#, 'pedophilia', 'cannibalism', 'victims']
        entity_versions['cannibalism'] = ['cannibalism']
        entity_versions['pedophilia'] = ['pedophilia']
        entity_versions['sex'] = ['sex']
        entity_versions['Hollywood'] = ['hollywood', 'actors', 'actor', 'woods', 'james woods', 'wood']
        entity_versions['Admins'] = ['admins']        
        entity_versions['Money'] = ['money']
        entity_versions['Vatican'] = ['vatican', 'church']
        entity_versions['Haiti'] = ['haiti'] 
        entity_versions['jews'] = ['jews']        
        entity_versions['Facebook'] = ['facebook']        
        entity_versions['Government'] = ['government']                
        entity_versions['Email'] = ['email', 'emails']            
        entity_versions['Williams'] = ['williams', 'katt', 'katt williams']        
        entity_versions['Soros'] = ['soros']        
        entity_versions['Israel'] = ['israel']
        entity_versions['journalists'] = ['journalists', 'journalist', 'carpenter']        
        entity_versions['Petersen'] = ['petersen', 'monica', 'monica petersen']
        entity_versions['mods'] = ['mods', 'mod']        
        entity_versions['porn(ography)'] = ['porn', 'pornography']
        entity_versions['Rosenberg'] = ['rosenberg', 'eli']
        entity_versions['Welch'] = ['welch', 'edgar', 'maddison'] 
        entity_versions['Mascot'] = ['mascot'] 
        entity_versions['Breitbart'] = ['breitbart'] 
        entity_versions['Washington'] = ['washington', 'dc', 'washington dc']         
        entity_versions['Trafficking'] = ['trafficking', 'traffickers', 'traffic']
        
    if dataset == "pizzagate-podcast":
        #entity_versions['I/We'] = ['i','we']
        entity_versions['story'] = ['story', 'stories', 'the story', 'this story']
        entity_versions['Amanda'] = ['amanda']
        entity_versions['Pizzagate'] = ['pizzagate']
        entity_versions['bots'] = ['bot','bots']
        #entity_versions['guy/person'] = ['guy', 'person']
        entity_versions['welch'] = ['welch']
        entity_versions['accounts'] = ['accounts','account']
        #entity_versions['facebook'] = ['facebook']
        entity_versions['Breitbart'] = ['breitbart']
        entity_versions['Laura'] = ['laura']
        #entity_versions['twitter/tweets'] = ['tweets','twitter']
        entity_versions['election'] = ['election']
        entity_versions['hilary'] = ['hilary','clinton']
        entity_versions['Commet Ping Pong'] = ['pong', 'commet', 'ping', 'pingpong', 'pizzeria', 'pizza place']
        #entity_versions['Missouri'] = ['missouri']
        entity_versions['Hagmann'] = ['hagmann']
        entity_versions['ring'] = ['ring']
        entity_versions['Trump'] = ['trump','donald']
        entity_versions['Prince'] = ['prince']
        entity_versions['Carmen'] = ['carmen', 'katz', 'cat', 'cats']
        entity_versions['Alefantis'] = ['james', 'alefantis']
        entity_versions['Alex Jones'] = ['alex', 'jones']
        entity_versions['Wooley'] = ['wooley']
        entity_versions['November'] = ['november']
        entity_versions['October'] = ['october']
        entity_versions['December'] = ['december']
        entity_versions['Podesta'] = ['podesta']
        entity_versions['Campbell'] = ['campbell']
        entity_versions['fake news'] = ['fake','fake news']
        entity_versions['Amanda and Laura'] = ['amanda and laura', 'laura and amanda']
    
    if dataset=="bridgegate-old":
        entity_versions['Christie'] = ['christie', 'chris', 'chris christie','Governor Christie']
        entity_versions['Wildstein'] = ['wildstein', 'david', 'david wildstein']
        entity_versions['Kelly'] = ['kelly', 'anne', 'bridget anne kelly', 'bridget']
        entity_versions['Baroni'] = ['baroni']
        entity_versions['Samson'] = ['samson']
        entity_versions['Stepien'] = ['stepien']
        entity_versions['Sokolich'] = ['sokolich', 'fort lee mayor mark sokolich'] # mayor        
        entity_versions['closures'] = ['closures', 'closure', 'scandal', 'the lane closures', 'the lane closings']
        
        entity_versions['Wisniewski'] = ['wisniewski']
        entity_versions['Mastro'] = ['mastro']        
        entity_versions['attorney'] = ['attorney']
        entity_versions['Critchley'] = ['critchley']
        entity_versions['Drewniak'] = ['drewniak']
        entity_versions['Democrat'] = ['democrat', 'democrats']
        entity_versions['Republic'] = ['republic', 'republicans', 'republican']
        entity_versions['Fort Lee'] = ['fort', 'lee', 'fort lee']
        
        '''
        entity_versions['governor'] = ['governor', 'the governor']
        entity_versions['authority'] = ['authority']
        entity_versions['documents'] = ['documents', 'reports', 'report']
        entity_versions['Email'] = ['email', 'emails']
        entity_versions['media'] = ['media', 'by the record']
        entity_versions['Kelly and Stepien'] = ['kelly and stepien']
        entity_versions['Bridge'] = ['bridge', 'ther bridge', 'the george washington bridge']
        '''
        
        
    if dataset=="bridgegate" or dataset=="bridgegate_with_dates":
        entity_versions['Christie'] = ['christie', 'chris christie', 'chris']
        entity_versions['Wildstein'] = ['wildstein', 'david wildstein']
        entity_versions['Kelly'] = ['kelly', 'anne', 'bridget anne kelly', 'bridget', 'bridget kelly']
        entity_versions['Baroni'] = ['baroni', 'bill baroni']
        entity_versions['Samson'] = ['samson', 'david samson']
        entity_versions['Stepien'] = ['stepien', 'bill stepien']
        entity_versions['Sokolich'] = ['sokolich', 'fort lee mayor mark sokolich', 'mark sokolich'] # mayor        
        entity_versions['closures'] = ['closures', 'closure', 'the lane closures', 'the lane closings']
        
        entity_versions['Wisniewski'] = ['wisniewski', 'john wisniewski']
        entity_versions['Mastro'] = ['mastro', 'randy mastro']        
        entity_versions['attorney'] = ['attorney']
        entity_versions['Critchley'] = ['critchley', 'michael critchley']
        entity_versions['Drewniak'] = ['drewniak', 'michael drewniak']
        entity_versions['Democrat'] = ['democrat', 'democrats','democratic', 'democrat party']
        entity_versions['Republic'] = ['republic', 'republicans', 'republican', 'republican party']
        entity_versions['Fort Lee'] = ['fort', 'lee', 'fort lee']        
        
        entity_versions['Zimmer'] = ['zimmer', 'dawn zimmer']
        entity_versions['Port Authority'] = ['port authority', 'port', 'authority']  
        entity_versions['New Jersey'] = ['new jersey', 'n j', 'nj']  
        entity_versions['New York'] = ['new york', 'n y', 'ny']
        entity_versions['George Washington Bridge'] = ['george washington bridge', 'gwb']
        entity_versions['Bridgegate'] = ['bridgegate', 'scandal']
        entity_versions['Record'] = ['record']
        
        entity_versions['Trump'] = ['trump', 'donald trump']
        entity_versions['Hillary'] = ['hillary', 'hillary clinton', 'clinton']
        entity_versions['Obama'] = ['obama', 'barack obama']
        entity_versions['North Jersey Media Group']= ['north jersey media group']
        entity_versions['Hoboken'] = ['hoboken']#, 'record hoboken']
        
    
    # make everything lower case
    for ent_glob_name, ent_version_list in entity_versions.items():
        for ind in range(len(ent_version_list)):
            ent_version_list[ind] = ent_version_list[ind].lower()
    
    return entity_versions
        
    
def change_nt_to_not(sent):
    sent = sent.replace(" can't ", " cannot ").replace(" won't ", " will not ")
    res_sent = ""
    ind = 0
    while ind < len(sent):
        # current character
        c = sent[ind]
        # avoid out of range access.
        if ind > len(sent)-3:
            res_sent += c
            ind += 1
            continue
        # n't at the end of the sentence.
        if ind == len(sent)-3 and c == "n" and sent[ind+1] == "'" and sent[ind+2] == "t":
            res_sent += " not"
            break
        if ind == len(sent)-4 and c == "n" and sent[ind+1] == "'" and sent[ind+2] == "t":
            res_sent += " not" + sent[ind+3]
            break            
        if c == "n" and sent[ind+1] == "'" and sent[ind+2] == "t" and sent[ind+3] == " ":
            res_sent += " not "
            ind += 4
            continue
        if c == "n" and sent[ind+1] == "'" and sent[ind+2] == "t" and sent[ind+3] == ".":
            res_sent += " not."
            ind += 4
            continue            
        res_sent += c
        ind += 1
    return res_sent

def change_multi_dots_to_single_dot(sent):
    ind = 0
    res_sent = ""
    while ind < len(sent):
        c = sent[ind]
        if c == ".":
            res_sent += c
            ind2 = ind
            while ind2 < len(sent) and sent[ind2] == ".":
                ind2 += 1
            if ind2 < len(sent) and sent[ind2] != " ":
                res_sent += " "
            ind = ind2
            
        else:
            ind += 1
            res_sent += c
    return res_sent
            
def strip_non_ascii(sent):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in sent if 0 < ord(c) < 127)
    return ''.join(stripped)

def clean_sent(sent):
    '''
    This function 
            1. Remove non-accii characters (encode to utf-8).
            2. Replace - with .
            3. Remove punctuations - except ".",",",";", "!", "?", "'", # '"' -> we remove " for some reasons (dep makes mistake?)
            4. Change n't to not
            5. Change multidots to a single dot - Example: ... -> .
    '''
    
    #sent = sent.encode('utf-8')
    #printable = set(string.printable)
    #''.join(filter(lambda x: x in printable, sent))
    
    sent = strip_non_ascii(sent)
    
    
    #sent = sent.replace("-",".")#.replace("(","").replace(")","")    
    
    exclude = set(string.punctuation) - {".",",",";", "!", "?", "'"}#, '"'}
    sent = ''.join(ch for ch in sent if ch not in exclude)

    sent = change_nt_to_not(sent)
    sent = change_multi_dots_to_single_dot(sent)
    return sent


def is_entity_present(sent, entity):
    # in case we would like to match car's to cars
    sent = sent.translate(None, string.punctuation.replace("'","")).lower()
    entity = entity.lower()
    ent_item_as_separate_word = False
    
    if sent == entity:
        return True
    # if it appears as separate words inside the text
    if (" " + entity + " ") in sent:
        return True
    # if it appears as separate words at first
    ind_first_match = sent.find(entity+" ")
    if ind_first_match == 0:
        return True
    # if it appears as separate words at the end
    ind_last_match = sent.rfind(" "+entity)
    if ind_last_match == -1:
        return False
    if ind_last_match + len(entity) + 1 == len(sent):
        return True
    return False


def aggregate_who_are_entities(df_rels, output_file=None, top_num=-1, save_to_file=False, stem_rels=False, just_head_arg=False):
    
    df_is_rels = df_rels[np.logical_or(df_rels["rel"] == "is", df_rels["rel"] == "{is}")]
    df = get_top_extractions(df_is_rels, output_file, top_num, save_to_file, stem_rels, just_head_arg)
    
    return df


def create_equivalent_dict(df_is_rels, main_ent_name, ent_version_list):
    '''
    input: main_entity name like {sokolich} (including the {})
    output: dictionary of dictionaries.
            - we keep the equivalent head words and their count as the first layer of dictionary.
            - then for each of them we store their different versions (or their full appearances) together with their counts.
            - for example for sokolich we have: 
            {
            'Democrat': {'count': 23, 'versions': {'a {Democrat}': 23}}, 
            'target': {'count': 2, 'versions': {'the {target} of jams': 1, 'the alleged {target} of closures': 1}}, 
            'mayor': {'count': 16, 'versions': {'the Hoboken {mayor}': 1, 'borough Democratic {mayor}': 1, 'town Democratic {mayor}': 1, 'Lee {mayor}': 4, 'borough {mayor}': 2, 'Lee Democratic {mayor}': 3, 'town {mayor}': 2, 'the {mayor}': 1, 'The {mayor}': 1}}
            ...
            }
    '''
    dict_main_ent_equiv = {}
    #main_ent_name = "{Sokolich}"
    for ind, item in df_is_rels.iterrows():
        equiv_ent = ""
        # check if any of the entity versions are present in any of the arguments, then take the other argument as their equivalent word (descriptive word).
        for ent_v in ent_version_list: 
            if "{" + ent_v + "}" in item["arg1"]:
                equiv_ent = item["arg2"]
                break
            if "{" + ent_v + "}" in item["arg2"]:
                equiv_ent = item["arg1"]
                break
        # if this entity is not head word of any of the arguments, then skip this row.
        if not equiv_ent:
            continue
        equiv_ent_head = re.search(r'\{(.*)\}', equiv_ent).group(1).replace("{","").replace("}","")
        if equiv_ent_head not in dict_main_ent_equiv:
            dict_main_ent_equiv[equiv_ent_head] = {}
            dict_main_ent_equiv[equiv_ent_head]["count"] = 1
            dict_main_ent_equiv[equiv_ent_head]["versions"] = {}
            dict_main_ent_equiv[equiv_ent_head]["versions"][equiv_ent] = 1
        else:
            dict_main_ent_equiv[equiv_ent_head]["count"] += 1
            if equiv_ent not in dict_main_ent_equiv[equiv_ent_head]["versions"]:
                dict_main_ent_equiv[equiv_ent_head]["versions"][equiv_ent] = 1
            else:
                dict_main_ent_equiv[equiv_ent_head]["versions"][equiv_ent] += 1


    return dict_main_ent_equiv        
        
    
def save_entity_sorted_equivalents(dict_main_ent_equiv, main_ent_name, f):    
    print("*" * 60, file=f)
    print("*" * ((59-len(main_ent_name))/2), main_ent_name, "*" * ((59-len(main_ent_name))/2), file=f)
    #print "**************************   ", main_ent_name, "   **************************"
    print("*" * 60, file=f)
    print("", file=f)
    for s in sorted(iter(dict_main_ent_equiv.items()), key=lambda x_y1: x_y1[1]['count'], reverse=True):
        print(s[0], "->", s[1]["count"], file=f)
        print("-" * 60, file=f)
        #print s[1]
        #print s[1]["versions"]
        for s_versions in sorted(iter(s[1]["versions"].items()), key=lambda x_y: x_y[1], reverse=True):
            print(s_versions, file=f)
        print("", file=f)    
        
def write_df_to_csv(path_with_file_name, df_input, header=None):
    if header is None:
        header = df_input.columns
    
    with open(path_with_file_name, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')#, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for ind, item in df_input.iterrows():
            #if ind >5 : 
            #    break
            csv_writer.writerow(item)         