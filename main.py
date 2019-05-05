from entity_extractor import *


def main():
    ee = EntityExtractor(base_dir="/Users/behnam/Desktop/Behnam_Files/vwani_text_mining/RE_Behnam/data/FakeNews/bridgegate/small_accurate_set/results/rels_v2_with_pronoun/",
                            df_extraction_name="df_rels_with_ner.csv",
                            df_ner_ranking_name="Entities_NER_Flair_Ranking_From_Sentences_with_conf.csv",
                            df_arg_ranking_name="df_arg_ranking.csv",
                            dataset_name="bridgegate")

    '''
    data_loader = ent_extractor.dataLoader

    df_extractions = data_loader.load_extractions()
    df_ner_ranking = data_loader.load_ner_ranking()
    ent_extractor.generate_arg_ranking()
    df_arg_ranking = data_loader.generate_arg_ranking("df_argument_ranking_name")
    '''
    print(ee.df_arg_ranking.head())


if __name__ == '__main__':
    main()