import torch
from entity_extractor import *
import ast
import pandas as pd


def test(ee):
    a = ee.df_extractions.head()
    a = a.iloc[0]
    ann = a["annotation"]
    ann = ast.literal_eval(ann)
    print(ann)
    print(ann.keys())
    print(a["sentence"], len(a["sentence"].split(" ")))
    words = " ".join(ann["words"])
    print(words, len(ann["words"]))

def main():
    base_dir = "/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/data/FakeNews/bridgegate/small_accurate_set/results/rels_v2_with_pronoun/"
    ee = EntityExtractor(base_dir=base_dir,
                            df_extraction_name="df_rels_with_ner.csv",
                            df_ner_ranking_name="df_ner_ranking.csv",#"Entities_NER_Flair_Ranking_From_Sentences_with_conf.csv",
                            df_arg_ranking_name="df_arg_ranking.csv",
                            dataset_name="bridgegate",
                            load_all_data=False)



    df_ent_final_ranking = ee.generate_or_load_final_ent_ranking(path_to_file= base_dir + "df_ent_final_ranking.csv", overwrite=False)
    print(df_ent_final_ranking.head())


if __name__ == '__main__':
    main()