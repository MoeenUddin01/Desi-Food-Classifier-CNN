import torch
from src.data.loader import train_loader, test_loader
from src.model.cnn import CNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator
import wandb
import os
from datetime import datetime
from dotenv import load_dotenv  # ✅ for reading .env files

# Load environment variables from .env
load_dotenv()  # looks for .env in the current directory

def main():
    try:
        # Training Config
        EPOCHS = 10
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        config = {
            "Epochs": EPOCHS,
            "Batch Size": BATCH_SIZE,
            "Learning Rate": LEARNING_RATE,
            "Device": DEVICE,
            "Model": CNN
        }

        # Initialize W&B
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key is None:
            raise ValueError("WANDB_API_KEY not found in environment variables")
        wandb.login(key=wandb_key)

        wandb.init(
            project="Desi-Food-Classifier-CNN",
            config=config,
            name=f'Experiment-{datetime.now().strftime("%d_%m_%Y_%H_%M")}'
        )

        # Model
        my_model = CNN()
        print("Using device:", DEVICE)
        torch.set_default_device(DEVICE)

        # Trainer & Evaluator
        model_trainer = Trainer(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            data=train_loader,
            model=my_model,
            model_path="my_cnn",
            device=DEVICE
        )

        model_evaluator = Evaluator(
            batch_size=BATCH_SIZE,
            data=test_loader,
            model=my_model,
            device=DEVICE
        )

        BEST_ACCURACY = 0

        # Epoch loop
        for epoch in range(EPOCHS):
            # Training
            avg_train_loss, _, train_acc = model_trainer.start_training_loop(epoch)

            # Validation
            avg_val_loss, _, val_acc = model_evaluator.start_evaluation_loop(epoch)

            # Log to WandB
            wandb.log({
                "Training Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Epoch": epoch,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })

            # Save best model
            if val_acc > BEST_ACCURACY:
                BEST_ACCURACY = val_acc
                final_model_path = model_trainer.save_model()
                if final_model_path is not None:
                    print(f"Model with Accuracy {val_acc} Saved Successfully")
                    wandb.log_model(final_model_path, "desi_food_classifier_cnn",
                                    aliases=[f"epoch-{epoch+1}"])

    except Exception as e:
        print(f"Error in Training Script: {e}")
        raise

if __name__ == "__main__":
    main()
    print("Training Script Finished Successfully")
