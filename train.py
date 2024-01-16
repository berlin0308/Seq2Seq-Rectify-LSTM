import torch
import torch.nn as nn
from model import BLSTM, SeqModel
from daily_data_loader import create_one_dataloader
from sklearn.metrics import accuracy_score

num_epochs = 40
learning_rate = 0.003
clip_value = 1.0  

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

model = SeqModel(input_size=7, hidden_size=32, num_layers=4, window_size=32, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,15,20,25,30],gamma=0.5, verbose=True)

daily_data_root = "daily_data/raw/V9_lr5e-4_ep80_best"
# daily_data_root = "daily_data/test_samples/train"
train_loader  = create_one_dataloader(daily_data_root=daily_data_root)

num_batches = len(train_loader)


log_file = 'result_train/train.log'
with open(log_file, 'w') as f:
    f.write('Epoch, Loss, Acc\n')
    f.flush()

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
            # print(predicted_classes_list)

            unlabeleds = int(float(list(batch_labels[0]).count(7.0)))
            accuracy = accuracy_score(batch_labels[0], predicted_classes_list, normalize=False)/float(1440-unlabeleds)
            accuracies += accuracy

            total_loss = model.neg_log_likelihood_loss(output, batch_labels)
            total_losses += total_loss
            
            print(f'loss: {total_loss:.4f}, acc: {accuracy:.3f}')
            
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            scheduler.step()
            
        average_loss = total_losses / num_batches
        average_accuracy = accuracies / num_batches

        f.write(f'{epoch + 1}, {average_loss:.4f}, {average_accuracy:.3f}\n')
        f.flush()
        
        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(), f"result_train/seqmodel_ep{epoch+1}.pth")

        print(f'------------------------\nEpoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.2f}, Accuracy: {average_accuracy:.3f}\n------------------------')
