import matplotlib.pyplot as plt
import torch
import typer
from sklearn.metrics import RocCurveDisplay

import wandb
from template.data import corrupt_mnist
from template.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")



def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_accuracy = 0, 0
        preds, targets = [], []

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            batch_accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": batch_accuracy})

            epoch_loss += loss.item()
            epoch_accuracy += batch_accuracy

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
                
                # Log input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # Log gradient histogram
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.detach().cpu())})

        # Compute average epoch loss and accuracy
        epoch_loss /= len(train_dataloader)
        epoch_accuracy /= len(train_dataloader)
        wandb.log({"epoch_loss": epoch_loss, "epoch_accuracy": epoch_accuracy})

        # Generate and log ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)
        for class_id in range(10):
            one_hot = (targets == class_id).float()
            roc_display = RocCurveDisplay.from_predictions(
                one_hot.numpy(),
                preds[:, class_id].numpy(),
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
            )
            fig, ax = plt.subplots()
            roc_display.plot(ax=ax)
            wandb.log({f"ROC_curve_class_{class_id}": wandb.Image(fig)})
            plt.close()

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={},
    )
    artifact.add_file("model.pth")
    run.log_artifact(artifact)

if __name__ == "__main__":
    typer.run(train)


