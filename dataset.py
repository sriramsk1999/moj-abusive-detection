"""Dataset and processing pipeline"""

from typing import Optional
import pandas as pd
from transformers import AutoTokenizer
import demoji
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch


class MojAbusiveDataModule(pl.LightningDataModule):
    """LightningDataModule for preparing dataset"""

    def __init__(
        self,
        model_name: str,
        data_dir: str = "eam-train-set/bq-results-20210825-203004-swh711l21gv2.csv",
        test_data_dir: str = "eam-test-set-public/eam2021-test-set-public.csv",
        batch_size: int = 64,
        seed: int = 61,
        test_size: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.test_size = test_size
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_and_prep_data(self, data_dir):
        cached_csv = "%s.cached" % data_dir
        try:
            df = pd.read_csv(cached_csv)
        except:
            df = pd.read_csv(data_dir)
            # df = df.head(2048) # Mini dataset for experimenting
            print("Transforming metadata")
            # Resolve metatdata inconsistency due to time variation
            metadata_cols = [
                "report_count_comment",
                "report_count_post",
                "like_count_comment",
                "like_count_post",
            ]
            for post_index in df.post_index.unique():
                for col in metadata_cols:
                    df.loc[df.post_index == post_index, col] = df[
                        df.post_index == post_index
                    ][col].max()
            # normalize
            for col in metadata_cols:
                df[col] = df[col] / df[col].max()
            df.to_csv(cached_csv, index=False)
        print("Tokenizing data")
        token_df = pd.DataFrame.from_dict(
            self.fast_encode(list(df["commentText"].values))
        )
        df = pd.concat([df, token_df], axis=1)
        dataset = MojAbusiveDataset(df)
        return dataset

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = self.load_and_prep_data(self.data_dir)
            self.moj_abusive_train, self.moj_abusive_val = train_test_split(
                dataset,
                shuffle=True,
                random_state=self.seed,
                test_size=self.test_size,
                stratify=dataset.dataframe["language"].values,
            )
        if stage == "test":
            self.moj_abusive_test = self.load_and_prep_data(self.test_data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.moj_abusive_train, batch_size=self.batch_size, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.moj_abusive_val, batch_size=self.batch_size, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.moj_abusive_test, batch_size=self.batch_size, num_workers=4
        )

    def fast_encode(self, texts, chunk_size=512, maxlen=128):

        input_ids = []
        tt_ids = []
        at_ids = []

        for i in tqdm(range(0, len(texts), chunk_size)):
            text_chunk = texts[i : i + chunk_size]
            text_chunk = [demoji.replace_with_desc(sent, " ") for sent in text_chunk]

            encs = self.tokenizer(
                text_chunk, max_length=128, padding="max_length", truncation=True
            )

            input_ids.extend(encs["input_ids"])
            tt_ids.extend(encs["token_type_ids"])
            at_ids.extend(encs["attention_mask"])

        return {
            "input_ids": input_ids,
            "token_type_ids": tt_ids,
            "attention_mask": at_ids,
        }


class MojAbusiveDataset(Dataset):
    """Implementation of a Dataset which is then loaded into torch's DataLoader"""

    def __init__(self, dataframe):
        self.dataframe = dataframe.drop(
            "label", axis=1, errors="ignore"
        )  # Ignore error to handle test scenario
        self.target = dataframe.get(
            "label", pd.Series(index=dataframe.index, name="label")
        )  # Empty column to handle test scenario

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        txt = self.dataframe.iloc[index]
        return {
            "input_ids": torch.tensor(txt["input_ids"]),
            "token_type_ids": torch.tensor(txt["token_type_ids"]),
            "attention_mask": torch.tensor(txt["attention_mask"]),
            "like_count_comment": torch.tensor(txt["like_count_comment"]),
            "like_count_post": torch.tensor(txt["like_count_post"]),
            "report_count_comment": torch.tensor(txt["report_count_comment"]),
            "report_count_post": torch.tensor(txt["report_count_post"]),
            "label": torch.tensor(self.target[index], dtype=torch.float32),
        }
