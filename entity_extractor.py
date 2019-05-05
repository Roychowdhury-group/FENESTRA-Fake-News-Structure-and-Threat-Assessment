import pandas as pd
import os
from collections import Counter
import nltk
import re
import math
import time

class DataLoader:
    def __init__(self,
                 base_dir="",
                 df_extraction_name="",
                 df_ner_ranking_name="",
                 df_arg_ranking_name="",
                 dataset_name=""):
        # self.input_data = None
        self.base_dir = base_dir
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
            raise Exception("Unable to find the file at: ", path_to_file)
        try:
            df = pd.read_csv(path_to_file)
            print("Done. Number of records: ", len(df))
            return df
        except IOError as e:
            # print(e.message)
            print("Unable to open the file at", path_to_file)

    # @classmethod
    def load_extractions(self, path_to_file=""):
        if path_to_file:
            self.df_extractions =  self.load_csv(path_to_file)
        else:
            self.df_extractions = self.load_csv(self.base_dir + self.df_extractions_name)
        return self.df_extractions

    def load_ner_ranking(self, path_to_file=""):
        if path_to_file:
            self.df_ner_ranking = self.load_csv(path_to_file)
        else:
            self.df_ner_ranking = self.load_csv(self.base_dir + self.df_ner_ranking_name)
        return self.df_ner_ranking

    def load_arg_ranking(self, path_to_file=""):
        if path_to_file:
            self.df_arg_ranking = self.load_csv(path_to_file)
        else:
            self.df_arg_ranking = self.load_csv(self.base_dir + self.df_arg_ranking_name)
        return self.df_arg_ranking



class EntityExtractor:
    def __init__(self,
                 base_dir="../data/FakeNews/bridgegate/small_accurate_set/results/rels_v2_with_pronoun/",
                 df_extraction_name="df_rels_with_ner.csv",
                 df_ner_ranking_name="Entities_NER_Flair_Ranking_From_Sentences_with_conf.csv",
                 df_arg_ranking_name="df_arg_ranking.csv",
                 dataset_name="bridgegate"):

        self.data_loader = DataLoader(base_dir,
                                     df_extraction_name,
                                     df_ner_ranking_name,
                                     df_arg_ranking_name,
                                     dataset_name)

        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.df_extraction_name = df_extraction_name
        self.df_ner_ranking_name = df_ner_ranking_name
        self.df_arg_ranking_name = df_arg_ranking_name
        self.df_extractions = self.data_loader.load_extractions()
        self.df_ner_ranking = self.data_loader.load_ner_ranking()
        if not os.path.exists(self.base_dir + self.df_arg_ranking_name):
            self.generate_arg_ranking(just_head_arg=True) # also sets self.df_arg_ranking variable
        else:
            self.df_arg_ranking = self.data_loader.load_arg_ranking()

    @staticmethod
    def get_top_entities(df_rels, output_file=None, top_num=-1, save_to_file=False, just_head_arg=True):
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

    def generate_arg_ranking(self, just_head_arg):
        start_time = time.time()
        print("Generating df_arg_ranking -- ranking of the subject and objects along with their POS tags")
        df_entity_rankings = self.get_top_entities(self.df_extractions,
                                              output_file=self.base_dir + self.df_arg_ranking_name,
                                              top_num=-1,
                                              save_to_file=True,
                                              just_head_arg=just_head_arg)
        self.df_arg_ranking = df_entity_rankings
        end_time = time.time()
        print("df_arg_ranking generation is done. Execution Time: ", (end_time-start_time)/60.0, " minutes.")
