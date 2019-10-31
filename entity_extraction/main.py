import torch
from entity_extractor import *
from visualizer import *
import ast
import pandas as pd
import pickle
import math

''' 
# In order to explore the results in Ipython shell, run the following commands: 
    1. (highlight them -> press alt+shift+e)
%run entity_extractor.py
%reload_ext autoreload
%autoreload 2
    2. highlight the peice of code you would like to examine (say inside your main func) and press (alt+shift+e)
'''
#%run entity_extractor.py
#%reload_ext autoreload
#%autoreload 2

#base directory where you load input files, and save output files
BASE_DIR = "/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/Fenestra/FENESTRA-Fake-News-Structure-and-Threat-Assessment/data/"
# dataset name
DATASET = "bridgegate_with_dates"
# relationship file name
RELATIONSHIPS_RAW_NAME = "bridgegate_minimal_orig_text_with_dates_with_supervised_cleaning_with_sorted_datetime_relations_1.csv"

## If this is the first time running this code, then keep the following names as it is, and these files will get generated
# processed relationship file name which contains NER taggs
DF_EXTRACTION_NAME = "df_extractions_with_ner_1.csv"
# name of the dataframe (and csv file) which contains ranking of the named entities
DF_NER_RANKING_NAME = "df_ner_ranking_1.csv"
# name of the dataframe (and csv file) which contains ranking of the argument headwords
DF_ARG_RANKING_NAME = "df_arg_ranking_1.csv"

# Turn this to file, if you have already generated the ner tags, and have ``flair sentences" -- to save some computations
REGENERATE_DF_EXTRACTIONS_WITH_NER_FLAIR_SENTENCES_AND_TAGS = True

# "TRUE" -> to overwrite the NER ranking.
OVERWRITE_NER_RANKING = True

# NUMBER OF ENTITIES FOR CLUSTERING
NUMBER_OF_ENTITIES_TO_CLUSTER = 50 #130 -> for the paper


def generate_avg_embeddings_for_entities_tensorboard_format(base_dir, file_postfix):
    file_prefix = "entity_embedding_dict_"
    file_name = file_prefix + file_postfix + ".pkl"
    path_to_file = base_dir +  file_name

    with open(path_to_file, "rb") as f:
        ent_embeddings = pickle.load(f)

    print(ent_embeddings)

    embs = []
    ent_names = []
    ent_types = []
    for k, v in ent_embeddings.items():
        if not isinstance(v["embedding"], float):# and not math.isnan(v):
            embs.append(v["embedding"])
            ent_names.append(k)
            ent_types.append(v["type"])
    df_embs = pd.DataFrame(embs)
    df_embs.to_csv(base_dir + file_postfix + "_ent_avg_embeddings.tsv", sep='\t', index=False, header=False)
    #ent_names = [k for k, v in ent_embeddings.items()]
    df_ent_names = pd.DataFrame(ent_names, columns=["Text"])
    df_ent_names["Type"] = ent_types
    df_ent_names.to_csv(base_dir + file_postfix + "_ent_meta_data_names_and_types.tsv", sep='\t', index=False, header=True)
    print(embs)

def get_entity_versions(dataset="bridgegate_entity_trends"):
    entity_versions = defaultdict(list)

    if dataset == "bridgegate_entity_trends":
        entity_versions['Chris Christie'] = ['christie', 'chris christie', 'chris']
        entity_versions['David Wildstein'] = ['wildstein', 'david wildstein']
        entity_versions['Bridget Anne Kelly'] = ['kelly', 'anne', 'bridget anne kelly', 'bridget', 'bridget kelly']
        entity_versions['Bill Baroni'] = ['baroni', 'bill baroni']
        entity_versions['David Samson'] = ['samson', 'david samson']
        entity_versions['Bill Stepien'] = ['stepien', 'bill stepien']
        entity_versions['Mark Sokolich'] = ['sokolich', 'fort lee mayor mark sokolich', 'mark sokolich']  # mayor
        #entity_versions['closures'] = ['closures', 'closure', 'the lane closures', 'the lane closings']

        entity_versions['John Wisniewski'] = ['wisniewski', 'john wisniewski']
        entity_versions['Randy Mastro'] = ['mastro', 'randy mastro']
        entity_versions['Michael Critchley'] = ['critchley', 'michael critchley']
        entity_versions['Michael Drewniak'] = ['drewniak', 'michael drewniak']
        entity_versions['Dawn Zimmer'] = ['zimmer', 'dawn zimmer']
        entity_versions['Donald Trump'] = ['trump', 'donald trump']
        entity_versions['Hillary Clinton'] = ['hillary', 'hillary clinton', 'clinton']
        entity_versions['Barack Obama'] = ['obama', 'barack obama']
        entity_versions['Paul Fishman'] = ['fishman', 'paul fishman']
        entity_versions['Pat Foye'] = ['foye', 'pat foye']
        entity_versions['Gibson Dunn'] = ['gibson dunn', 'gibson', 'dunn']
        entity_versions['Rudy Giuliani'] = ['giuliani', 'rudy giuliani']

    if dataset == "bridgegate_entity_trends_full_names":
        entity_versions['Chris Christie'] = ['chris christie']
        entity_versions['David Wildstein'] = ['david wildstein']
        entity_versions['Bridget Anne Kelly'] = ['bridget anne kelly']
        entity_versions['Bill Baroni'] = ['bill baroni']
        entity_versions['David Samson'] = ['david samson']
        entity_versions['Bill Stepien'] = ['bill stepien']
        entity_versions['Mark Sokolich'] = ['mark sokolich']  # mayor
        #entity_versions['closures'] = ['closures', 'closure', 'the lane closures', 'the lane closings']

        entity_versions['John Wisniewski'] = ['john wisniewski']
        entity_versions['Randy Mastro'] = ['randy mastro']
        entity_versions['Michael Critchley'] = ['michael critchley']
        entity_versions['Michael Drewniak'] = ['michael drewniak']
        entity_versions['Dawn Zimmer'] = ['dawn zimmer']
        entity_versions['Donald Trump'] = ['donald trump']
        entity_versions['Hillary Clinton'] = ['hillary clinton']
        entity_versions['Barack Obama'] = ['barack obama']
        entity_versions['Paul Fishman'] = ['paul fishman']
        entity_versions['Pat Foye'] = ['pat foye']
        entity_versions['Gibson Dunn'] = ['gibson dunn']
        entity_versions['Rudy Giuliani'] = ['rudy giuliani']

    # make everything lower case
    for ent_glob_name, ent_version_list in entity_versions.items():#iteritems():
        for ind in range(len(ent_version_list)):
            ent_version_list[ind] = ent_version_list[ind].lower()

    return entity_versions


def generate_entity_versions_automatically(base_dir, entity_version_pickle_name="entity_versions_auto_generated.pkl",min_freq=-1, overwrite=False):
    entity_versions = defaultdict(list)
    #base_dir = "/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/data/FakeNews/bridgegate/small_accurate_set/results/data_with_dates/"
    path_to_file = base_dir + entity_version_pickle_name.split(".")[0] + "_" + str(min_freq) + ".pkl"#"entity_versions_auto_generated.pkl"
    if os.path.isfile(path_to_file) and not overwrite:
        with open(path_to_file, "rb") as f:
            entity_versions_auto_generated = pickle.load(f)
        return entity_versions_auto_generated
    else:
        df_ent_final_ranking = pd.read_csv(base_dir + "df_ent_final_ranking.csv")
        df_persons = df_ent_final_ranking[df_ent_final_ranking["type"] == "PERSON"]
        #Droping some entities which are either article writers, or noises from ner pipeline
        entities_to_drop = ["George Washington", "Mark Sokou00adlich"]
        #df_persons = df_persons[df_persons["entity"] != "George Washington"] #dropping the NER mistake
        df_persons = df_persons[~df_persons["entity"].isin(entities_to_drop)]  # dropping the NER mistake
        if min_freq != -1:
            df_persons = df_persons[df_persons["frequency_score_sum_NER_arg"] >= min_freq]

        def get_cap_fullnames_only(ent):
            ent_cap = ""
            if len(ent.split(" ")) == 2:
                ent_cap = ent_capitalized(ent)
            return ent_cap
        df_persons["fullnames"] = df_persons.apply(lambda x: get_cap_fullnames_only(x["entity"]), axis=1)
        df_persons = df_persons[df_persons["fullnames"] != ""]
        df_persons["entity"] = df_persons["fullnames"]
        df_persons.drop(columns=["fullnames"], inplace=True)
        df_persons.to_csv(base_dir + "df_persons_"+str(min_freq) + ".csv")
        for ind, row in df_persons.iterrows():
            ent = row["entity"]
            if len(ent.split(" ")) == 2:
                ent_cap = ent_capitalized(ent)
                if ent_cap not in entity_versions:
                    entity_versions[ent_cap] = [ent]
        with open(path_to_file, "wb") as f:
            pickle.dump(entity_versions, f, protocol=pickle.HIGHEST_PROTOCOL)
    # make everything lower case
    for ent_glob_name, ent_version_list in entity_versions.items():#iteritems():
        for ind in range(len(ent_version_list)):
            ent_version_list[ind] = ent_version_list[ind].lower()

    print("Number of total entities: ", len(entity_versions.keys()))
    return entity_versions

def ent_capitalized(ent):
    ent_splitted = ent.split(" ")
    res = ""
    for w in ent_splitted:
        res += w[0].upper() + w[1:] + " "
    return res.strip()


def get_entity_versions_reverse_mapping(dataset="bridgegate_entity_trends"):
    entity_versions = get_entity_versions(dataset)
    entity_versions_reverse_mapping = defaultdict(list)

    for ent_glob_name, ent_version_list in entity_versions.iteritems():
        for ind in range(len(ent_version_list)):
            entity_versions_reverse_mapping[ent_version_list[ind]] = ent_glob_name

    return entity_versions_reverse_mapping

def _get_main_persons(base_dir, entity_min_freq):
    '''
    To overlap our entity list with an entity list provided by a news page -- just for experimentations -- didn't use the results in the paper.
    :param base_dir:
    :param entity_min_freq:
    :return:
    '''

    path_to_nytimes = base_dir + "timeline_online_txt_only.rtf"
    import codecs
    f = codecs.open(path_to_nytimes, "r", "utf-8")
    #with open(, 'r', encoding='utf-8') as f:
    #df = pd.read_csv(path_to_nytimes, encoding='utf-8')
    #print(df.head())
    txt_nytimes_list = f.readlines()
    txt_nytimes = " ".join(txt_nytimes_list)
    txt_nytimes = txt_nytimes.lower()
    df_persons = pd.read_csv(base_dir + "df_persons_" + str(entity_min_freq) + ".csv")

    df_persons["exist_in_ny_times"] = df_persons.apply(lambda x: x["entity"].lower() in txt_nytimes, axis=1)
    df_persons.to_csv(base_dir + "df_persons_join_nytimes_" + str(entity_min_freq) + ".csv")


def experiment_visualize_first_mention_of_entities(base_dir, input_file_name, create_new_ents_dict=False):
    #base_dir = "/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/data/FakeNews/bridgegate/small_accurate_set/results/data_with_dates/"
    vis = visualizer(base_dir)
    entity_min_freq = 10
    new_ent_dict_name = "dict_new_ents_per_date_persons_only_minFreq" + str(entity_min_freq) + ".pkl"
    path_to_new_ent_dict = base_dir + new_ent_dict_name
    '''
    path_to_entity_emb_dict = base_dir + "entity_embedding_dict_top_130.pkl"
    embs, ent_names, ent_types, ent_embeddings = vis.load_ent_embedding_dict(path_to_entity_emb_dict)
    selected_ent_names = []
    num_selected_ents = 30
    for ind in range(len(ent_names)):
        if num_selected_ents <= 0:
            break
        if ent_types[ind] == "PERSON":
            selected_ent_names.append(ent_names[ind])
            num_selected_ents -= 1
    '''

    '''
    print(" get main persons ...")
    _get_main_persons(base_dir, entity_min_freq)
    print("done.")
    '''

    entity_versions = generate_entity_versions_automatically(base_dir, entity_version_pickle_name="entity_versions_auto_generated.pkl",min_freq = entity_min_freq, overwrite=True)
    '''
    entity_versions_reverse_mapping = get_entity_versions_reverse_mapping(dataset="bridgegate_entity_trends")
    df_ent_final_ranking = pd.read_csv(base_dir + "df_ent_final_ranking.csv")
    df_persons = df_ent_final_ranking[np.logical_and(df_ent_final_ranking["type"] == "PERSON", df_ent_final_ranking["frequency_score_sum_NER_arg"] > 5)]
    for ind, row in df_persons.iterrows():
        if row["entity"] in entity_versions_reverse_mapping:
            
    '''

    if create_new_ents_dict:
        vis.create_first_mention_of_entities_dict(input_file_name, entity_versions=entity_versions, output_name=new_ent_dict_name, generate_df_with_dates=True)

    vis.visualize_new_ents_dict(path_to_new_ent_dict, output_post_fix_name="minFreq_"+str(entity_min_freq))


def experiment_main_generate_ent_rankings(base_dir,
                                          df_extraction_raw_name="bridgate_minimal_clean_text_relations_-1.csv",
                                          df_extraction_name="df_extractions_with_ner.csv",
                                          df_ner_ranking_name="df_ner_ranking.csv",
                                          df_arg_ranking_name="df_arg_ranking.csv",
                                          dataset_name="bridgegate",
                                          regenerate_df_extractions_with_ner_flair_sentences_and_tags=False,
                                          overwrite_ner_ranking=False
                                          ):
    ee = EntityExtractor(base_dir,
                            df_extraction_raw_name,
                            df_extraction_name,
                            df_ner_ranking_name,
                            df_arg_ranking_name,
                            dataset_name,
                            regenerate_df_extractions_with_ner_flair_sentences_and_tags,
                            overwrite_ner_ranking
                            )

    df_ent_final_ranking = ee.generate_or_load_final_ent_ranking(path_to_file= base_dir + "df_ent_final_ranking.csv", overwrite=overwrite_ner_ranking)
    print(df_ent_final_ranking.head())

    start_time = time.time()
    ent_emb_lists = ee.get_ent_emb_dict(df_ent_final_ranking, only_top_N_entitis=NUMBER_OF_ENTITIES_TO_CLUSTER)
    print("entity embedding generation done - execution time: ", (time.time()-start_time)/60.0)
    print("entity lists:", ent_emb_lists.keys())
    ent_single_emb_lists = {}
    for ent_name, ent_cnt_and_emb in ent_emb_lists.items():
        ent_single_emb_lists[ent_name] = {}
        ent_single_emb_lists[ent_name]["type"] = ent_cnt_and_emb["type"]
        ent_single_emb_lists[ent_name]["count"] = ent_cnt_and_emb["count"]
        ent_single_emb_lists[ent_name]["embedding"] = np.mean(ent_cnt_and_emb["embeddings"], axis=0)
        #ent_single_emb_lists[ent_name] = (ent_cnt_and_emb["type"],ent_cnt_and_emb["count"], np.mean(ent_cnt_and_emb["embeddings"], axis=0))

    print(ent_single_emb_lists)
    PIK = base_dir + "entity_embedding_dict_top_" + str(NUMBER_OF_ENTITIES_TO_CLUSTER) + ".pkl"
    print("saving pickle object at:  ", PIK)
    with open(PIK, "wb") as f:
        pickle.dump(ent_single_emb_lists, f, protocol=pickle.HIGHEST_PROTOCOL)
    #'''

if __name__ == '__main__':

    base_dir = BASE_DIR

    '''
    experiment_main_generate_ent_rankings(base_dir,
                                          df_extraction_raw_name=RELATIONSHIPS_RAW_NAME,
                                          df_extraction_name=DF_EXTRACTION_NAME,
                                          df_ner_ranking_name=DF_NER_RANKING_NAME,
                                          df_arg_ranking_name=DF_ARG_RANKING_NAME,
                                          dataset_name=DATASET,
                                          regenerate_df_extractions_with_ner_flair_sentences_and_tags=REGENERATE_DF_EXTRACTIONS_WITH_NER_FLAIR_SENTENCES_AND_TAGS,
                                          overwrite_ner_ranking=OVERWRITE_NER_RANKING
                                          )

    experiment_visualize_first_mention_of_entities(base_dir, input_file_name="bridgegate_minimal_orig_text_with_dates_with_supervised_cleaning_with_sorted_datetime.csv", create_new_ents_dict=True)



    # Generate embeddings for visualizing the entities in TensorBoard
    file_postfix = "top_" + str(NUMBER_OF_ENTITIES_TO_CLUSTER)
    generate_avg_embeddings_for_entities_tensorboard_format(base_dir=base_dir, file_postfix=file_postfix)

    '''
    # Visualize entities into 2D plot using PCA projection
    vis = visualizer(base_dir)
    path_to_entity_emb_dict = base_dir + "entity_embedding_dict_top_" + str(NUMBER_OF_ENTITIES_TO_CLUSTER) + ".pkl"
    vis.visualize_clusters(path_to_entity_emb_dict, output_file_name="top_130_entity_clusters.png")