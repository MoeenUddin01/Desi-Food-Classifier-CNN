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
        EPOCHS = 100
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        DEVICE = "cuda" if torch.cuda.is_available()else "cpu"
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
        my_model = CNN().to(DEVICE)
        print("Using device:", DEVICE)
        

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

        # epoch loop

        for epoch in range(EPOCHS):
            # run training loop
            average_epoch_training_loss, _, epoch_training_acc = model_trainer.start_training_loop(epoch)

            # run validation loop
            average_epoch_validation_loss, _, epoch_validation_acc = model_evaluator.start_evaluation_loop(epoch)
            wandb.log(
                {"Training Loss": average_epoch_training_loss, 
                 "Validation Loss": average_epoch_validation_loss,
                 "Epoch": epoch,
                 "Training Accuracy": epoch_training_acc,
                 "Validation Accuracy": epoch_validation_acc
                 })
            
            if epoch_validation_acc > BEST_ACCURACY:
                BEST_ACCURACY = epoch_validation_acc
                
                final_model_path = model_trainer.save_model()
                if final_model_path !=None:
                    print(f"Model with Accuracy {epoch_validation_acc} Saved Successfully")
                    wandb.log_model(final_model_path, "desi_food_classifier_cnn",aliases=[f"epoch-{epoch+1}"] )

    except Exception as e:
        print(f"Error in Training Script {e}")
        raise Exception


if __name__=="__main__":
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))
    main()