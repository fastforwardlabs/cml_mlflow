import argparse

import torch
import mlflow
import mlflow.pytorch


# Make this script parametrizable

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden', type=int, default=100)
args = parser.parse_args()


# Generate some fake data

n_data = 64
input_dim = 1000
output_dim = 10


x = torch.randn(n_data, input_dim)
y = torch.randn(n_data, output_dim)


# Define a model

hidden_dim = args.hidden

model = torch.nn.Sequential(
  torch.nn.Linear(input_dim, hidden_dim),
  torch.nn.ReLU(),
  torch.nn.Linear(hidden_dim, output_dim)
)



# Define some loss functions

mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()


# Define an opitimizer

lr = args.lr

opt = torch.optim.SGD(model.parameters(), lr = lr)


# Run a training loop

# Experiments group related runs in MLFlow.
mlflow.set_experiment('demo')

with mlflow.start_run():
  
  # Log some parameters valid for the whole run
  mlflow.log_params({
    'lr': lr,
    'hidden_dim': hidden_dim
  })
  
  for epoch in range(100):
    predictions = model(x)
    mse_loss = mse(predictions, y)
    l1_loss = l1(predictions, y)
    
    # Log metrics for each epoch of the run
    mlflow.log_metrics({
      'mse_loss': mse_loss.item(),
      'l1_loss': l1_loss.item()
    }, step=epoch)
    
    opt.zero_grad()
    mse_loss.backward() # choose mse as optimization target
    opt.step()
    
    
  # save the model artefact
  mlflow.pytorch.log_model(model, 'model')

