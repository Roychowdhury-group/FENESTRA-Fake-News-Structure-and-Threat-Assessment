import os
import pickle
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
from nltk.tokenize import sent_tokenize
import collections
import string
from datetime import datetime
import csv
from nltk.tokenize import sent_tokenize
import re
from collections import defaultdict


class visualizer:
    def __init__(self,
                 base_dir = ""):
        self.base_dir = base_dir

    def load_ent_embedding_dict(self, path_to_file=""):
        if not path_to_file:
            file_prefix = "entity_embedding_dict_"
            file_postfix = "top_130"
            file_name = file_prefix + file_postfix + ".pkl"
            path_to_file = self.base_dir + file_name
        if not os.path.isfile(path_to_file):
            #raise Exception("Unable to find the file at: ", path_to_file)
            print("Unable to find the file at: ", path_to_file)
            return None

        with open(path_to_file, "rb") as f:
            ent_embeddings = pickle.load(f)

        embs = []
        ent_names = []
        ent_types = []
        for k, v in ent_embeddings.items():
            if not isinstance(v["embedding"], float):  # and not math.isnan(v):
                ent_names.append(k)
                embs.append(v["embedding"])
                ent_types.append(v["type"])

        return embs, ent_names, ent_types, ent_embeddings


    def load_pickle_dict(self, path_to_file=""):
        if not path_to_file:
            file_name = "dict_new_ents_per_date.pkl"
            path_to_file = self.base_dir + file_name
        if not os.path.isfile(path_to_file):
            print("Unable to find the file at: ", path_to_file)
            return None

        with open(path_to_file, "rb") as f:
            dict_pickle = pickle.load(f)

        return dict_pickle

    def get_color(self, type):
        if type == "PERSON":
            return 'r'
        if type == "ORG":
            return 'g'
        if type == "LOC":
            return 'b'
        if type == "EVENT":
            return 'c'
        if type == "FAC":
            return 'm'
        if type == "GPE":
            return 'y'
        if type == "LAW":
            return 'k'
        if type == "NORP":
            return (0.2,0.4,0.7)
        if type == "OTHER(ARG)":
            return (0,0.2,0.8)
        return '(0,0,0.3)'

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)



    def visualize_clusters(self, path_to_file="", pca_n_components=2, output_file_name="entity_clusters.png"):
        embs, ent_names, ent_types, ent_embeddings = self.load_ent_embedding_dict(path_to_file)


        pca = PCA(n_components=pca_n_components)
        principalComponents = pca.fit_transform(embs)
        df_pca = pd.DataFrame(data=principalComponents
                                   , columns=['x', 'y'])
        #print(df_pca.head())


        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=9)
        ax.set_ylabel('Principal Component 2', fontsize=9)
        ax.set_title('Bert Embeddings of Entities Projected in 2-D using PCA', fontsize=20)
        #'''
        types = list(set(ent_types))
        cmap = ['r','g','b','c','m','y', 'k','w']#self.get_cmap(len(types))
        type_to_color_mapping = {}
        for ind, t in enumerate(types):
            type_to_color_mapping[t] = cmap[ind]

        '''
        for ind, row in df_pca.iterrows():
            x, y = row['x'], row['y']
            c = self.get_color(ent_types[ind])

        for target, color in zip(types, colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=cmap(i)
                       , s=50)

        '''
        for ind, t in enumerate(ent_types):
            ax.scatter(df_pca.iloc[ind]['x'], df_pca.iloc[ind]['y'], c=type_to_color_mapping[t])# c=

        '''
        for i, txt in enumerate(ent_names):
            ax.annotate(txt, (df_pca['x'].iloc[i], df_pca['y'].iloc[i]))
        '''
        texts = []
        for x, y, s in zip(df_pca['x'], df_pca['y'], ent_names):
            texts.append(plt.text(x, y, s))
        adjust_text(texts, only_move={'text':'xy'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        print("done")
        plt.legend(types)
        plt.show()
        plt.savefig(self.base_dir + output_file_name)


    def create_first_mention_of_entities_dict(self, input_file_name, entity_versions, output_name = "dict_new_ents_per_date.pkl", generate_df_with_dates=False):
        if generate_df_with_dates:
            #df_with_dates = pd.read_csv("/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/data/FakeNews/bridgegate/small_accurate_set/results/data_with_dates/relationship_results/bridgegate_minimal_orig_text_with_dates_relations_-1.csv")
            df_with_dates = pd.read_csv(self.base_dir + input_file_name, encoding = "ISO-8859-1")
            df_with_dates.dropna(subset=['date'], inplace=True)
            #experiment: only take the ones with closure in them. 'closures', 'closure', 'the lane closures', 'the lane closings'
            print("number of posts: ", len(df_with_dates))
            df_with_dates = df_with_dates[df_with_dates["text"].str.contains("lane closure|closure|closures|lane closures|lane closings|closing|closings")]
            print("number of posts with closure/lane closures/etc keywords: ", len(df_with_dates))
            if "sentence" not in set(df_with_dates.columns):
                df_with_dates.rename(columns={"file_num":"post_num"}, inplace=True)
                df_with_dates["date"] = df_with_dates["date"].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))
            else:
                df_with_dates["date"] = df_with_dates["date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

            df_with_dates.sort_values("date", inplace=True)
            ## Drop non-related posts before sept 2013.
            df_with_dates = df_with_dates[df_with_dates["date"] >= datetime(2013, 9, 9)]
            df_with_dates.to_csv(self.base_dir + input_file_name.split(".")[0] + "_with_sorted_datetime.csv")
            #df_ent_final_ranking = pd.read_csv("/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/data/FakeNews/bridgegate/small_accurate_set/results/rels_v2_with_pronoun/df_ent_final_ranking.csv")
        else:
            df_with_dates = pd.read_csv(self.base_dir + input_file_name.split(".")[0] +"_with_sorted_datetime.csv")

        def is_entity_present(sent, entity):
            # in case we would like to match car's to cars
            sent = sent.translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower()
            #sent = sent.translate(None, string.punctuation.replace("'", "")).lower()
            entity = entity.lower()
            ent_item_as_separate_word = False

            if sent == entity:
                return True
            # if it appears as separate words inside the text
            if (" " + entity + " ") in sent:
                return True
            # if it appears as separate words at first
            ind_first_match = sent.find(entity + " ")
            if ind_first_match == 0:
                return True
            # if it appears as separate words at the end
            ind_last_match = sent.rfind(" " + entity)
            if ind_last_match == -1:
                return False
            if ind_last_match + len(entity) + 1 == len(sent):
                return True
            return False


        dict_new_ents_per_date = {}
        remained_entities = set(entity_versions.keys())

        regex = re.compile(r"Follow * on Twitter", re.IGNORECASE)
        for ind, row in df_with_dates.iterrows():
            t_sentences = sent_tokenize(row["text"])
            for sentence in t_sentences:

                if re.match(regex, sentence) is not None:
                    continue
                #sentence = row["sentence"]
                post = str(row["post_num"])
                date = row["date"]
                found_entities = set()
                sentences_with_mentions = set()
                posts_with_mentions = set()
                if len(remained_entities) < 1:
                    break

                for ent in remained_entities:
                    for entv_item in entity_versions[ent]:
                        if is_entity_present(sentence, entv_item):
                            found_entities.add(ent)
                            sentences_with_mentions.add(sentence)
                            posts_with_mentions.add(post)
                for ent in found_entities:
                    remained_entities.remove(ent)

                if len(found_entities) > 0:
                    if date not in dict_new_ents_per_date:
                        dict_new_ents_per_date[date] = {}
                        dict_new_ents_per_date[date]["new_entities"] = []
                        dict_new_ents_per_date[date]["num_new_entities"] = 0
                        dict_new_ents_per_date[date]["sentences_with_new_mentions"] = []
                        dict_new_ents_per_date[date]["posts_with_new_mentions"] = []
                    dict_new_ents_per_date[date]["new_entities"] += list(found_entities)
                    dict_new_ents_per_date[date]["sentences_with_new_mentions"] += list(sentences_with_mentions)
                    dict_new_ents_per_date[date]["posts_with_new_mentions"] += list(posts_with_mentions)
                    dict_new_ents_per_date[date]["num_new_entities"] += 1

        output_path = self.base_dir + output_name
        with open(output_path, "wb") as f:
            pickle.dump(dict_new_ents_per_date, f, protocol=pickle.HIGHEST_PROTOCOL)

        return dict_new_ents_per_date


    def visualize_new_ents_dict(self, path_to_file, output_post_fix_name="minFreq_-1"):
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        dict_new_ents_per_date = self.load_pickle_dict(path_to_file)
        res = []
        for date,ent_list_and_ent_freq in dict_new_ents_per_date.items():
            if isinstance(date, float):
                continue
            try:
                datetime_object = datetime.strptime(date, '%Y-%m-%d')
            except:
                datetime_object = date
            num_new_ents = ent_list_and_ent_freq["num_new_entities"]
            all_entities = ",".join(list(ent_list_and_ent_freq["new_entities"]))
            all_sentences = " ---- ".join(list(ent_list_and_ent_freq["sentences_with_new_mentions"]))
            posts_num = " ---- ".join(list(ent_list_and_ent_freq["posts_with_new_mentions"]))
            example_entities = ",".join(list(ent_list_and_ent_freq["new_entities"])[:3])
            res.append([datetime_object, num_new_ents, example_entities, all_entities, all_sentences, posts_num])
        df = pd.DataFrame(res, columns=["date", "number of new entities", "example_entities", "all_entities", "all_sentences", "posts_num"])
        df.sort_values("date", inplace=True)

        df.to_csv(self.base_dir+"df_new_entities_"+ str(output_post_fix_name) + ".csv", quoting=csv.QUOTE_ALL)

        fig, ax = plt.subplots(figsize=(15.5, 6), facecolor='white', edgecolor='white')
        # fig = plt.figure()
        time_span = df["date"].values
        freq_trend = df["number of new entities"].values
        example_entities_list = df["example_entities"].values
        plt.bar(time_span, freq_trend, edgecolor='k', alpha = 0.5, color= 'b')
        plt.gcf().autofmt_xdate()
        plt.title("New mentions of entities")
        plt.ylabel('Number of new entities introduced daily')
        plt.xlabel('Time')

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1, y2 + 2))

        #date = datetime(2015, 10, 1)
        #date1 = datetime.strptime('2015-06-15', '%Y-%m-%d')

        for label, x, y in zip(example_entities_list, time_span, freq_trend):
            thr = 150
            if y > thr:# or x > date1:
                ax.annotate(
                    label,
                    xy=(x, y), xytext=(100, 20),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        #plt.bar(df["date"], df["number of new entities"])

        plt.savefig(self.base_dir + "first_mention_of_entities_over_time_" + str(output_post_fix_name) + ".eps", format='eps', dpi=1000)
        #plt.show()