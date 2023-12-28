import torch
import torch.nn as nn
from model import BLSTM


sequence_length = 1440
feature_per_time = 7


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BLSTM(input_size=feature_per_time, hidden_size=64, num_layers=4, output_size=1440, device=device).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

dummy_input = torch.randn(1, sequence_length, feature_per_time).to(device)
print(dummy_input)
print(dummy_input.shape)

with torch.no_grad():
    model.eval()
    output = model(dummy_input)

print(output)
print(output.shape)


# num_samples = 100  # You can adjust the number of samples as needed
# sequence_length = 1440
# feature_per_time = 7
# output_size = 1440

# # Generate random data for training
# train_data = np.random.rand(num_samples, sequence_length, feature_per_time)
# train_labels = np.random.rand(num_samples, output_size)

# # Convert your training data and labels to PyTorch tensors
# train_data = torch.Tensor(train_data).to(device)
# train_labels = torch.Tensor(train_labels).to(device)

# # Set the number of epochs and batch size
# num_epochs = 10
# batch_size = 4

# # Create a DataLoader for batching your training data
# from torch.utils.data import DataLoader, TensorDataset
# train_dataset = TensorDataset(train_data, train_labels)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Define the number of batches
# num_batches = len(train_loader)

# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode
#     total_loss = 0.0
    
#     for batch_data, batch_labels in train_loader:
#         optimizer.zero_grad()  # Zero the gradients
#         output = model(batch_data)  # Forward pass
        
#         # Compute the loss
#         loss = criterion(output, batch_labels)
        
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     # Print the average loss for this epoch
#     average_loss = total_loss / num_batches
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
