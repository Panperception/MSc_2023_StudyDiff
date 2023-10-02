import lightning.pytorch as pl
import torch
from forward_process import p_losses, timesteps
from net import Unet
from torch import optim


class DiffusionModel(pl.LightningModule):
    def __init__(self, image_size, channels):
        super().__init__()
        self.net = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(
                1,
                2,
                4,
            ),
        )

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"]

        t = torch.randint(0, timesteps, (batch_size,)).long().to("cuda")

        loss = p_losses(self.net, batch, t, loss_type="huber")

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        return optimizer
