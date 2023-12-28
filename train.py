import torch
import torch.nn as nn
from model import BLSTM, SeqModel
from daily_data_loader import create_one_dataloader
from sklearn.metrics import accuracy_score


num_epochs = 40
learning_rate = 0.000005

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

# model = BLSTM(input_size=7, hidden_size=1440, num_layers=2, output_size=7, device=device).to(device)
model = SeqModel(input_size=7, hidden_size=1440, num_layers=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


daily_data_root = "daily_data/test_samples/train"
train_loader  = create_one_dataloader(daily_data_root=daily_data_root)

num_batches = len(train_loader)

for epoch in range(num_epochs):
    
    model.train()
    total_losses = 0.0
    accuracies = 0.0
    for batch_data, batch_labels in train_loader:

        batch_labels = batch_labels.long()

        optimizer.zero_grad()
        output = model(batch_data.to(device)).float()

        # print(batch_labels)
        # print(output.shape)
        # print(output[0])
        
        _, predicted_classes = torch.max(output, dim=2)
        predicted_classes_list = predicted_classes.squeeze().tolist()
        print(predicted_classes_list)

        unlabeleds = int(float(list(batch_labels[0]).count(7.0)))
        accuracy = accuracy_score(batch_labels[0], predicted_classes_list, normalize=False)/float(1440-unlabeleds)
        accuracies += accuracy


        # Compute the loss
        # total_loss = 0.0
        # for t, t_softmax_output in enumerate(output[0]):
        #     if int(batch_labels[0][t]) == 7:
        #         continue

        #     loss = criterion(t_softmax_output, batch_labels[0][t])

        #     # loss.backward(retain_graph=True) # if update every time point
        #     # print(f"t: {t}, loss: {loss}")
        #     total_loss += loss

        total_loss = model.neg_log_likelihood_loss(output, batch_labels, device='cpu')
        total_losses += total_loss
        
        print(f'loss: {total_loss}, acc: {accuracy:.3f}')
        
        total_loss.backward()
        optimizer.step()
        

    # for batch_data, batch_labels in train_loader:
    #     model.eval()
    #     output = model(batch_data)


    average_loss = total_losses / num_batches
    average_accuracy = accuracies / num_batches

    print(f'------------------------\nEpoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.2f}, Accuracy: {average_accuracy:.3f}\n------------------------')
