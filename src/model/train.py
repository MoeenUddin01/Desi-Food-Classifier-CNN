import torch
from torch.utils.data import DataLoader
import os
class Trainer:
    # training loop
    def __init__(self, batch_size:int,learning_rate:float, data: DataLoader, model, model_path, device):
        self.batch_size = batch_size
        self.data = data
        self.model = model
        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model_directory = "models"
        os.makedirs(self.model_directory, exist_ok= True)
        self.model_path = model_path
        self.device = device
    
    def start_training_loop(self, epoch):
        try:
            self.model.train()
            training_losses = []
            correct = 0
            total = 0
            for batch, (x, y) in enumerate(self.data):
                # forward pass
                x = x.to(torch.device(self.device))
                y = y.to(torch.device(self.device))
                prediction = self.model(x)
                training_loss = self.loss(prediction, y)
                training_losses.append(training_loss.item())

                # backward pass
                self.optimizer.zero_grad()
                training_loss.backward()
                self.optimizer.step()

                # predicted class
                _, predicted = torch.max(prediction.data, 1)
                # ground class
                total += y.size(0)
                correct += (predicted == y).sum().item()

                if batch % 5 == 0:
                    print(f"Training-> Epoch {epoch} -> Batch No{batch}: {training_loss.item():.4f}")
            epoch_acc = 100.0 * correct / total if total > 0 else 0.0
            average_epoch_training_loss = (sum(training_losses) / len(training_losses)) if training_losses else 0.0

            print(f"Average Epoch Training Loss {epoch} -> {average_epoch_training_loss}")
            print(f"Training-> Epoch{epoch}: {epoch_acc}")
            return average_epoch_training_loss, training_losses, epoch_acc
        
        except Exception as e:
            print(f"Error in Training Loop Epoch {epoch} and Batch No {batch} due to {e}")
            return None
        
    def save_model(self):
        try:
            final_path = os.path.join(self.model_directory, f"{self.model_path}.pth")
            torch.save(
                {  "model_state_dict": self.model.state_dict(),},
                final_path
            )
            return final_path
        except Exception as e:
            print(f"Error in Saving Model: {e}")
            return None

