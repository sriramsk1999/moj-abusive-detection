"""
Contains some helper functions
"""
import pandas as pd
from tqdm.auto import tqdm


def cache_csv():
    """Cache and store csv files"""
    for data_dir in [
        "eam2021-test-set-public/eam2021-test-set-public.csv",
        "eam2021-train-set/bq-results-20210825-203004-swh711l21gv2.csv",
    ]:
        cached_csv = "%s.cached" % data_dir
        df = pd.read_csv(data_dir)
        print("Transforming metadata")
        metadata_cols = [
            "report_count_comment",
            "report_count_post",
            "like_count_comment",
            "like_count_post",
        ]
        for post_index in tqdm(df.post_index.unique()):
            for col in metadata_cols:
                df.loc[df.post_index == post_index, col] = df[
                    df.post_index == post_index
                ][col].max()
        for col in metadata_cols:
            df[col] = df[col] / df[col].max()
        df.to_csv(cached_csv, index=False)

        print("%s complete!" % cached_csv)


def ensemble_by_voting():
    """Load predictions and ensemble by voting"""
    muril = pd.read_csv("submission_muril-base-cased.csv")
    indic_bert = pd.read_csv("submission_indic-bert.csv")
    mbert = pd.read_csv("submission_bert-base-multilingual-cased.csv")
    df = pd.concat([muril.Expected, indic_bert.Expected, mbert.Expected], axis=1)
    df["ensemble"] = df.mode(axis=1)
    df = df.drop("Expected", axis=1)
    df = df.rename(columns={"ensemble": "Expected"})
    df.insert(0, "Id", muril.Id)
    df.to_csv("submission.csv", index=False)
