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
    ee = EntityExtractor(base_dir="/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/data/FakeNews/bridgegate/small_accurate_set/results/rels_v2_with_pronoun/",
                            df_extraction_name="df_rels_with_ner.csv",
                            df_ner_ranking_name="df_ner_ranking.csv",#"Entities_NER_Flair_Ranking_From_Sentences_with_conf.csv",
                            df_arg_ranking_name="df_arg_ranking.csv",
                            dataset_name="bridgegate")

    '''
    data_loader = ent_extractor.dataLoader

    df_extractions = data_loader.load_extractions()
    df_ner_ranking = data_loader.load_ner_ranking()
    ent_extractor.generate_arg_ranking()
    df_arg_ranking = data_loader.generate_arg_ranking("df_argument_ranking_name")
    '''
    # print(ee.df_ner_ranking.head())
    # print(ee.df_arg_ranking.head())
    #test(ee)
    ee.generate_or_load_flair_tags(overwrite=True)
    ee.generate_or_load_ner_ranking(overwrite=True)




if __name__ == '__main__':
    main()