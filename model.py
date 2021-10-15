"""Model"""

from transformers import AutoModelForSequenceClassification
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torchmetrics.functional import f1


class MojAbusiveModel(pl.LightningModule):
    """Model based on MUrIL"""

    def __init__(self, model_name, dataset_size=600000, batch_size=64):
        super().__init__()
        self.muril = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        self.lin = nn.Linear(5, 1)
        self.total_steps = dataset_size // batch_size
        self.num_samples = 30
        self.batch_size = batch_size
        self.model_name = model_name

    def shared_forward(self, batch, prefix):
        label = batch["label"]
        pred = self(batch)
        loss = F.binary_cross_entropy(pred, label)
        self.log("%s_loss" % prefix, loss)
        pred_label = torch.round(pred).long()
        f1_score = f1(pred_label, label.long())
        self.log("%s_f1" % prefix, f1_score)
        return loss

    def forward(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]

        x = self.muril(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        ).logits

        like_count_comment = torch.unsqueeze(batch["like_count_comment"], dim=1)
        like_count_post = torch.unsqueeze(batch["like_count_post"], dim=1)
        report_count_comment = torch.unsqueeze(batch["report_count_comment"], dim=1)
        report_count_post = torch.unsqueeze(batch["report_count_post"], dim=1)

        x = torch.cat(
            [
                x,
                like_count_comment,
                like_count_post,
                report_count_comment,
                report_count_post,
            ],
            dim=1,
        )
        x = torch.sigmoid(self.lin(x.float()))
        return x.view(-1)

    def training_step(self, batch, batch_idx):
        loss = self.shared_forward(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_forward(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        self.enable_dropout()

        pred_list = torch.empty((self.num_samples, len(batch["label"]), 1))
        for i in range(self.num_samples):
            pred_list[i] = torch.unsqueeze(self(batch), dim=1)
        pred = torch.mean(pred_list, dim=0)
        pred_label = torch.round(pred).long()
        return pred_label

    def test_epoch_end(self, outputs):
        outputs = pd.DataFrame(torch.cat(outputs).tolist(), columns=["Expected"])
        outputs.index += 2
        outputs.to_csv(
            "submission_%s.csv" % self.model_name.split("/")[-1], index_label="Id"
        )

    def enable_dropout(self):
        """Function to enable the dropout layers during test-time"""
        for m in self.muril.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=5e-6)
        return optimizer
