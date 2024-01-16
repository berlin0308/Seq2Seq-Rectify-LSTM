import torch
from model import SeqModel
from daily_data_loader import load_predict_probs, load_labels_predicts

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = SeqModel(input_size=7, hidden_size=32, num_layers=4, window_size=32, device=device).to(device)
model.load_state_dict(torch.load("result_train/seqmodel_ep40.pth"))
model.eval()

""" inference with any input
inputs = torch.randn(1, 1440, 7).to(device)
with torch.no_grad():
    tag_seq = model(inputs)

print(tag_seq)
print(tag_seq.shape)

"""

data_path = "daily_data/raw/V9_lr5e-4_ep80_best/1112.csv"
assert data_path[-4:]==".csv" 

print(f"Loading: {data_path}")
predict_probs = load_predict_probs(data_path)
labels, predicts = load_labels_predicts(data_path)

X = torch.tensor([predict_probs], dtype=torch.float32)
Y = torch.tensor([labels], dtype=torch.float32)

print(X.shape)
print(Y.shape)

with torch.no_grad():
    output = model(X.to(device))
    print(output)
    print(output.shape)

_, predicted_classes = torch.max(output, dim=2)
predicted_classes_list = predicted_classes.squeeze().tolist()
print(predicted_classes_list)

