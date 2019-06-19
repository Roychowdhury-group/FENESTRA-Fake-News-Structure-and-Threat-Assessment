import pandas as pd
import os
from collections import Counter
import nltk
import re
import math
import time
import ast
from flair.embeddings import FlairEmbeddings, BertEmbeddings, CharacterEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import StackedEmbeddings
import pickle
from nltk.corpus import stopwords
# You will have to download the set of stop words the first time
import nltk
#nltk.download('stopwords')
import string
import numpy as np



class DataLoader:
    def __init__(self,
                 base_dir="",
                 df_extractions_raw_name="",
                 df_extraction_name="",
                 df_ner_ranking_name="",
                 df_arg_ranking_name="",
                 dataset_name=""):
        # self.input_data = None
        self.base_dir = base_dir
        self.df_extractions_raw_name = df_extractions_raw_name
        self.df_extractions_name = df_extraction_name
        self.df_ner_ranking_name = df_ner_ranking_name
        self.df_arg_ranking_name = df_arg_ranking_name
        self.dataset_name = dataset_name
        self.df_extractions = None
        self.df_ner_ranking = None
        self.df_arg_ranking = None


    # def load_input_txt(self):
    #    self.

    # @classmethod
    @staticmethod
    def load_csv(path_to_file=""):
        #input_file_path = self.base_dir + input_file_name
        print("Loading ... ", path_to_file)
        if not os.path.isfile(path_to_file):
            #raise Exception("Unable to find the file at: ", path_to_file)
            print("Unable to find the file at: ", path_to_file)
            return None
        try:
            df = pd.read_csv(path_to_file)
            print("Number of records: ", len(df))
            return df
        except IOError as e:
            # print(e.message)
            print("Unable to open the file at", path_to_file)

    # @classmethod
    def load_extractions(self, path_to_file="", pickle_name="flair_res.pkl"):
        if path_to_file and os.path.isfile(path_to_file):
            print("Loading -> df_extractions")
            self.df_extractions = self.load_csv(path_to_file)
        else:
            path_to_extractions = self.base_dir + self.df_extractions_name
            if os.path.isfile(path_to_extractions):
                self.df_extractions = self.load_csv(path_to_extractions)
            else:
                path_to_extractions = self.base_dir + self.df_extractions_raw_name
                self.df_extractions = self.load_csv(path_to_extractions)
        PIK = self.base_dir + pickle_name
        if os.path.isfile(PIK):
            self.df_extractions["flair_res"] = self.load_from_pickle_object_list(PIK)
        return self.df_extractions

    def load_ner_ranking(self, path_to_file=""):
        if path_to_file:
            print("Loading -> ner ranking")
            self.df_ner_ranking = self.load_csv(path_to_file)
        else:
            self.df_ner_ranking = self.load_csv(self.base_dir + self.df_ner_ranking_name)
        return self.df_ner_ranking

    def load_arg_ranking(self, path_to_file=""):
        if path_to_file:
            print("Loading -> arg ranking")
            self.df_arg_ranking = self.load_csv(path_to_file)
        else:
            self.df_arg_ranking = self.load_csv(self.base_dir + self.df_arg_ranking_name)
        return self.df_arg_ranking

    def save_to_pickle_object_list(self, data, pickle_name="flair_res.pkl"):
        PIK = self.base_dir + pickle_name
        print("saving pickle object at:  ", PIK)
        with open(PIK, "wb") as f:
            pickle.dump(len(data), f)
            for value in data:
                pickle.dump(value, f)

    def load_from_pickle_object_list(self, PIK):
        data2 = []
        with open(PIK, "rb") as f:
            for _ in range(pickle.load(f)):
                data2.append(pickle.load(f))
        return data2


class EntityExtractor:
    def __init__(self,
                 base_dir="../data/FakeNews/bridgegate/small_accurate_set/results/rels_v2_with_pronoun/",
                 df_extractions_raw_name="df_rels_with_ner.csv",
                 df_extraction_name="df_extractions_with_ner.csv",
                 df_ner_ranking_name="df_ner_ranking.csv",
                 df_arg_ranking_name="df_arg_ranking.csv",
                 dataset_name="bridgegate",
                 regenerate_df_extractions_with_ner_flair_sentences_and_tags=False,
                 overwrite_ner_ranking=False):

        self.data_loader = DataLoader(base_dir,
                                     df_extractions_raw_name,
                                     df_extraction_name,
                                     df_ner_ranking_name,
                                     df_arg_ranking_name,
                                     dataset_name)
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.df_extractions_raw_name = df_extractions_raw_name
        self.df_extraction_name = df_extraction_name
        self.df_ner_ranking_name = df_ner_ranking_name
        self.df_arg_ranking_name = df_arg_ranking_name
        #if not load_all_data:
        #    return
        self.df_extractions = self.data_loader.load_extractions(self.base_dir + df_extraction_name)
        self.generate_or_load_flair_tags(regenerate_df_extractions_with_ner_flair_sentences_and_tags=regenerate_df_extractions_with_ner_flair_sentences_and_tags)  # -> uncomment this if df_extractions does not have flair results
        self.generate_or_load_ner_ranking(self.base_dir + df_ner_ranking_name, overwrite=overwrite_ner_ranking)
        self.generate_or_load_arg_ranking(self.base_dir + df_arg_ranking_name, just_head_arg=True)


    @staticmethod
    def get_top_arguments(df_rels, output_file=None, top_num=-1, save_to_file=False, just_head_arg=True):
        entities = []
        if just_head_arg:
            for ind, item in df_rels.iterrows():
                # print(item["arg2"])
                if isinstance(item["arg2"], float) and math.isnan(item["arg2"]):
                    continue
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
        cols = ['entity', 'pos', 'frequency']
        df_entity_rankings = pd.DataFrame(columns=cols)
        cnt = Counter()
        for e in entities:
            cnt[e] += 1
        # '''
        if top_num == -1:  # means print all
            # print "Frequent relations:"
            for letter, count in cnt.most_common():
                # print letter, ": ", count
                letter_no_bracket = letter.replace("{", "").replace("}", "")
                if letter_no_bracket:
                    letter_pos = nltk.tag.pos_tag([letter_no_bracket])
                    df_entity_rankings.loc[len(df_entity_rankings)] = [letter, letter_pos[0][1], count]
        else:
            # print "top ", top_num, " frequent relations:"
            for letter, count in cnt.most_common(top_num):
                # print letter, ": ", count
                letter_no_bracket = letter.replace("{", "").replace("}", "")
                if letter_no_bracket:
                    letter_pos = nltk.tag.pos_tag([letter_no_bracket])
                    df_entity_rankings.loc[len(df_entity_rankings)] = [letter, letter_pos[0][1], count]
        # '''

        if output_file is not None and save_to_file:
            df_entity_rankings.to_csv(output_file, sep=',', encoding='utf-8', header=True, columns=cols)

        return df_entity_rankings

    @staticmethod
    def clean_ent(ent):
        ent = ent.lower()
        ent = ent.replace("{", "").replace("}", "").replace("<<", "").replace(">>", "")
        # remove punctuations
        ent = ent.translate(str.maketrans('','',string.punctuation))
        #remove stop words
        stop_words = stopwords.words('english')
        ent = " ".join([w for w in ent.split(" ") if w not in stop_words])
        return ent.strip()

    #@staticmethod
    def get_top_entities(self,
                         df_rels,
                         output_file=None,
                         top_num=-1,
                         save_to_file=False,
                         just_head_arg=False,
                         nouns_only=True):
        entities = []
        if just_head_arg:
            for ind, item in df_rels.iterrows():
                if isinstance(item["arg1"], float) and math.isnan(item["arg1"]):
                    continue
                if isinstance(item["arg2"], float) and math.isnan(item["arg2"]):
                    continue
                if "{" not in item["arg1"]:
                    ent1 = "{" + item["arg1"].strip() + "}"
                else:
                    ent1 = "{" + re.search(r'\{(.*)\}', item["arg1"].strip()).group(1) + "}"
                if "{" not in item["arg2"]:
                    ent2 = "{" + item["arg2"].strip() + "}"
                else:
                    ent2 = "{" + re.search(r'\{(.*)\}', item["arg2"].strip()).group(1) + "}"

                ent1 = self.clean_ent(ent1)
                if ent1:
                    entities.append(ent1)
                ent2 = self.clean_ent(ent2)
                if ent2:
                    entities.append(ent2)
        else:
            print("arg1,arg2 gen")
            arg1_list = list(df_rels['arg1'].apply(lambda x: self.clean_ent(x)))
            arg2_list = list(df_rels['arg2'].apply(lambda x: self.clean_ent(x)))
            #entities = list(df_rels['arg1']) + list(df_rels['arg2'])
            entities = arg1_list + arg2_list
        cols = ['entity', 'pos', 'frequency']
        df_entity_rankings = pd.DataFrame(columns=cols)
        cnt = Counter()
        # print(entities[0:10])
        for e in entities:
            cnt[e] += 1
        # '''
        if top_num == -1:  # means print all
            # print "Frequent relations:"
            for letter, count in cnt.most_common():
                # print letter, ": ", count
                letter_no_bracket = letter.replace("{", "").replace("}", "")
                if letter_no_bracket:
                    letter_pos = nltk.tag.pos_tag([letter_no_bracket])
                    df_entity_rankings.loc[len(df_entity_rankings)] = [letter, letter_pos[0][1], count]
        else:
            # print "top ", top_num, " frequent relations:"
            for letter, count in cnt.most_common(top_num):
                # print letter, ": ", count
                letter_no_bracket = letter.replace("{", "").replace("}", "")
                if letter_no_bracket:
                    letter_pos = nltk.tag.pos_tag([letter_no_bracket])
                    df_entity_rankings.loc[len(df_entity_rankings)] = [letter, letter_pos[0][1], count]
        # '''
        print(df_entity_rankings.head())
        if nouns_only:
            df_entity_rankings = df_entity_rankings[df_entity_rankings["pos"].str.contains("NN")]
        if output_file is not None and save_to_file:
            df_entity_rankings.to_csv(output_file, sep=',', encoding='utf-8', header=True, columns=cols)

        return df_entity_rankings

    def generate_or_load_arg_ranking(self, path_to_file="", overwrite=False, just_head_arg=True):
        if not path_to_file:
            path_to_file = self.base_dir + self.df_arg_ranking_name
        if os.path.exists(path_to_file) and not overwrite:
            self.df_arg_ranking = self.data_loader.load_arg_ranking(path_to_file)
        else:
            print("Generating df_arg_ranking ..")
            start_time = time.time()
            print("Generating df_arg_ranking -- ranking of the subject and objects along with their POS tags")
            df_entity_rankings = self.get_top_entities(self.df_extractions,
                                                  output_file=self.base_dir + self.df_arg_ranking_name,
                                                  top_num=-1,
                                                  save_to_file=True,
                                                  just_head_arg=just_head_arg,
                                                  nouns_only=True)
            self.df_arg_ranking = df_entity_rankings
            end_time = time.time()
            print("df_arg_ranking generation is done. Execution Time: ", (end_time-start_time)/60.0, " minutes.")

    def _get_sentence_space_delimited(self, x):
        annotation = ast.literal_eval(x["annotation"])
        return " ".join([str(w) for w in annotation["words"]])

    def generate_or_load_flair_tags(self, path_to_file="", regenerate_df_extractions_with_ner_flair_sentences_and_tags=False):
        if not path_to_file:
            path_to_file = self.base_dir + self.df_extraction_name #file_name #+ "_with_flair_res.csv"
        if os.path.exists(path_to_file) and not regenerate_df_extractions_with_ner_flair_sentences_and_tags:
            self.df_extractions = self.data_loader.load_extractions(path_to_file)
            return self.df_extractions
        else:
            df_rels = self.df_extractions.copy()
            print("Generating Flair Sentence Objects")
            tagger = SequenceTagger.load('ner-ontonotes-fast')
            #df_rels = self.data_loader.load_extractions(path_to_file)
            '''
            def get_sentence_space_delimited(x):
                annotation = ast.literal_eval(x["annotation"])
                return " ".join([str(w) for w in annotation["words"]])
            '''
            df_rels["flair_sentence"] = df_rels.apply(lambda x: Sentence(self._get_sentence_space_delimited(x)), axis=1)

            print("Predicting Named Entities and Saving them to ", path_to_file)
            start_time = time.time()

            res = tagger.predict(df_rels["flair_sentence"])

            df_rels["flair_res"] = res
            df_rels.to_csv(self.base_dir + "df_extractions_with_ner.csv")

            self.data_loader.save_to_pickle_object_list(res, pickle_name="flair_res.pkl")
            self.df_extractions = df_rels.copy()
            end_time = time.time()
            print("Flair Sentences + Tagger Prediction Done: ", end_time - start_time, "(seconds) - ", (end_time - start_time) / 60,
                  "(min)")

            '''
            if regenerate_flair_sentences_and_tags:
                blah blah
            else:
                res = self.data_loader.load_from_pickle_object_list(self.base_dir + "flair_res.pkl")
            '''

            # If dataset is small and you would like to keep the embeddings into a pkl file, uncomment the following block.
            '''
            print("Setting up the stacked embedding --BERT + CharacterEmbeddings")  # Flair (backward/forward) +
            # Generate embeddings
            # init Flair embeddings
            # flair_forward_embedding = FlairEmbeddings('multi-forward')
            # flair_backward_embedding = FlairEmbeddings('multi-backward')
            # init multilingual BERT
            start_time = time.time()
            bert_embedding = BertEmbeddings('bert-base-cased')
            character_embeddings = CharacterEmbeddings()
            # now create the StackedEmbedding object that combines all embeddings
            # we take bert + character embeddings -> so that we get embeddings in 3k space (character level only adds 50 dimensions. each flair embeddings adds about 2k dimensions)
            print(len(res))
            stacked_embeddings = StackedEmbeddings(
                embeddings=[bert_embedding])#, character_embeddings]) #, flair_forward_embedding, flair_backward_embedding])
            #stacked_embeddings.embed(res) # this function returns nothing and just updates the "res" with embeddings.
            bert_embedding.embed(res) ### Crashes if the data is large
            print("stacked_embedding done, minutes: ", (time.time()-start_time)/60.0)
            print(len(res))
            df_rels["flair_res"] = res
            df_rels.to_csv(self.base_dir + "df_extractions.csv")

            self.data_loader.save_to_pickle_object_list(res, pickle_name="flair_res_characterEmb_Bert.pkl")
            self.df_extractions = df_rels.copy()
            end_time = time.time()
            print("Embed Flair Sentences Done: ", end_time - start_time, "(seconds) - ", (end_time - start_time) / 60,
                  "(min)")
            '''
    def generate_or_load_ner_ranking(self, path_to_file="", overwrite=False):
        if not path_to_file:
            path_to_file = self.base_dir + self.df_ner_ranking_name
        if os.path.exists(path_to_file) and not overwrite:
            self.df_ner_ranking = self.data_loader.load_ner_ranking(path_to_file)
            return self.df_ner_ranking
        else:
            #if "flair_res" not in set(self.df_extractions.columns):
            #    raise "Make sure you run named entity prediction task before aggregating the named entities. call generate_or_load_flair_tags method predict flair tags."
            print("Generating ", path_to_file)
            self.df_extractions["ner"] = self.df_extractions["flair_res"].apply(lambda x: x.get_spans('ner'))
            print(self.df_extractions.iloc[0]["ner"])
            print(self.df_extractions["ner"].head())
            def get_list_of_span_dicts(row):
                res = []
                for span in row:
                    res.append(span.to_dict())
                return res

            res = []
            for ind, row in self.df_extractions["ner"].iteritems():
                res += get_list_of_span_dicts(row)
            print(len(res))
            print(res[0])
            df_ner_formatted = pd.DataFrame(res)
            # clean entities:
            df_ner_formatted["text"] = df_ner_formatted["text"].apply(lambda x: self.clean_ent(x))
            print(df_ner_formatted.head())
            df_ner_sorted = df_ner_formatted.groupby(by=["text", "type"]).agg({'start_pos':'size', 'confidence':'mean'}) \
                                                        .rename(columns={'start_pos':'count','confidence':'mean_confidence'}) \
                                                         .sort_values(['count'], ascending=False).reset_index()
            df_ner_sorted.to_csv(path_to_file)
            self.df_ner_ranking = df_ner_sorted.copy()
            return self.df_ner_ranking

    #@staticmethod
    def generate_or_load_final_ent_ranking(self, path_to_file="", overwrite=False):
        if not path_to_file:
            path_to_file = self.base_dir + "df_ent_final_ranking.csv"
        if os.path.exists(path_to_file) and not overwrite:
            df_final_ent_ranking = self.data_loader.load_csv(path_to_file)
            return df_final_ent_ranking
        else:
            df_ent_final_ranking = self.df_ner_ranking.copy()
            df_ent_final_ranking = df_ent_final_ranking[df_ent_final_ranking["type"].isin(["PERSON",
                                                                                           "ORG",
                                                                                           "LOC",
                                                                                           "EVENT",
                                                                                           "FAC",
                                                                                           "GPE",
                                                                                           "LAW",
                                                                                           "NORP"])]
            df_ent_final_ranking = df_ent_final_ranking[["text", "type", "count"]]
            df_ent_final_ranking.rename(columns={'text': 'entity', 'count': 'frequency'}, inplace=True)
            df_arg_ranking_with_type = self.df_arg_ranking.copy()
            df_arg_ranking_with_type["type"] = "OTHER(ARG)"
            df_ent_final_ranking = df_ent_final_ranking.append(df_arg_ranking_with_type[["entity", "type", "frequency"]], sort=True)
            print(df_ent_final_ranking.head(200))
            # u = df_ent_final_ranking["entity"].unique()
            # pd.Series(u).to_csv(ee.base_dir+"df_ent_ranking_unigue.csv")
            df_ent_final_ranking = df_ent_final_ranking.groupby(by=["entity"], as_index=False).agg({'type': 'first', 'frequency': 'sum'}) \
                .sort_values(['frequency'], ascending=False)
            df_ent_final_ranking.rename(columns={'frequency': 'frequency_score_sum_NER_arg'}, inplace=True)
            # df_ent_final_ranking["avgFreq"] = (df_ent_final_ranking["frequency"] + df_ent_final_ranking["count"])/2
            # df_ent_final_ranking.to_csv(ee.base_dir + "df_ent_final_ranking.csv")
            # df_ent_final_ranking.sort_values("avgFreq", axis=0, ascending=False, inplace=True, na_position ='last')
            df_ent_final_ranking.to_csv(self.base_dir + "df_ent_final_ranking.csv", index=False)
            return df_ent_final_ranking

    def get_ent_emb_dict(self, df_ent_final_ranking, only_top_N_entitis = 10, dump_flair_res_to_pickle=False):
        print("In function: get_ent_emb_dict")
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' #To get rid of this error: OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
        bert_embedding = BertEmbeddings('bert-base-cased')#do_lower_case=False
        if dump_flair_res_to_pickle:
            PIK = self.base_dir + "flair_res_embeddings.pkl"
            print("saving pickle object at:  ", PIK)
            f = open(PIK, "wb")
            pickle.dump(len(self.df_extractions), f)

        entities = df_ent_final_ranking["entity"][:only_top_N_entitis]
        ent_emb_lists = {}
        ent_has_enough_embs = {}
        for ind, ent in enumerate(entities):
            ent_emb_lists[ent] = {"count": 0, "type": df_ent_final_ranking.iloc[ind]["type"], "embeddings": []}
            ent_has_enough_embs[ent] = False

        cnt_found_entities = 0
        for ind_row, row in self.df_extractions.iterrows():
            if ind_row % 500 == 0:
                print(ind_row)
            #if ind_row > 1000:
            #    break
            has_entity = False
            for ent in entities:
                if ent_has_enough_embs[ent]:
                    continue
                if ent in row["sentence"].lower():
                    has_entity = True
                    break
            if not has_entity:
                continue
            sent_words = [t.text.lower() for t in row["flair_res"]]
            row_with_embeddings = row["flair_res"]
            bert_embedding.embed(row_with_embeddings)
            if dump_flair_res_to_pickle:
                pickle.dump(row_with_embeddings, f)
            #sent = self._get_sentence_space_delimited(row)
            #sent = sent.lower()
            #sent_words = sent.split(" ")
            '''
            Algo:
            for every word (w) in sentence, find the embedding for entities that start from w.
            '''
            for ind_w, w in enumerate(sent_words):
                for ent in entities:
                    if ent_has_enough_embs[ent]:
                        continue
                    ent_words = ent.split(" ")
                    ent_embs = []
                    ent_words_len = len(ent_words)
                    cnt = 0
                    while(cnt < ent_words_len):
                        if ind_w + cnt >= len(sent_words) or sent_words[ind_w+cnt] != ent_words[cnt]:
                            ent_embs = []
                            break
                        else:
                            #print(ind_w+1)
                            #print(row_with_embeddings.get_token(ind_w+1))
                            #print(row_with_embeddings.get_token(ind_w+1).embedding)
                            # flair get_token function is 1-based -> ind_w + 1 is needed
                            ent_embs.append(np.array(row_with_embeddings.get_token(ind_w + 1).embedding))
                        cnt += 1
                    if len(ent_embs) > 0:
                        ent_emb_lists[ent]["embeddings"].append(np.mean(ent_embs, axis=0))
                        ent_emb_lists[ent]["count"] += 1
                        # let's only take average of some mentions of them (for speed-up purposes) -- remove the followin if condition to average over all the entity mentions
                        if ent_emb_lists[ent]["count"] > 0:
                            ent_has_enough_embs[ent] = True
                            print(ent , " --- Embedding found.")
                            cnt_found_entities += 1
                            print("Number of found entities: ", cnt_found_entities)

        return ent_emb_lists
