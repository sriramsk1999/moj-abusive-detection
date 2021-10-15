"""
Main
"""

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import MojAbusiveDataModule
from model import MojAbusiveModel


def main():
    """Main function"""
    # Different seeds for different data splits
    ensemble = [
        ("google/muril-base-cased", 61),
        ("ai4bharat/indic-bert", 76),
        ("bert-base-multilingual-cased", 42),
    ]

    # Choosing which model to train
    index = 0
    model_name, seed = ensemble[index]

    seed_everything(seed, workers=True)
    data_module = MojAbusiveDataModule(model_name)
    model = MojAbusiveModel(model_name)
    wandb_logger = WandbLogger(project="moj-abuse-detection")

    trainer = Trainer(
        gpus=1,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="ckpts",
                filename="moj_abusive-%s-{epoch:02d}-{val_loss:.2f}"
                % model_name.split("/")[-1],
                save_top_k=1,
                mode="min",
            ),
        ],
    )

    trainer.fit(model, data_module)
    print(ModelCheckpoint(dirpath="ckpts/").best_model_path)
    model = MojAbusiveModel.load_from_checkpoint(
        checkpoint_path=ModelCheckpoint(dirpath="ckpts/").best_model_path
    )

    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
