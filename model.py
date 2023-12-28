import torch
import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=2)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        out = self.fc(out)
        out += x

        # out, (h_n, c_n) = self.lstm(out, (h0, c0))

        # out = self.fc(out)
        # out += x

        # out = self.softmax(out)
        
        return out

import torch.nn.functional as F
from crf import CRF

class SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SeqModel, self).__init__()

        self.use_crf = False
        self.gpu = False
        label_size = 7

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, label_size)

        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

    def neg_log_likelihood_loss(self, inputs, batch_label, mask=torch.ones(1, 1440).long(), device='cpu'):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        outputs = self.forward(inputs)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outputs, mask, batch_label)
        else:
            outputs = outputs.view(batch_size * seq_len, -1)
            batch_label = batch_label.view(batch_size * seq_len)
            loss_function = nn.CrossEntropyLoss(ignore_index=7, reduction='sum', weight=torch.tensor([1.0/211, 1.0/216, 1.0/26, 1.0/20, 1.0/718, 1.0/6, 1.0/200]))
            total_loss = loss_function(outputs.to(device), batch_label.to(device))

        total_loss = total_loss / batch_size
        return total_loss

 

    def forward(self, inputs, mask=torch.ones(1, 1440).long()):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        lstm_out, _ = self.lstm(inputs)
        """
        lstm_out = lstm_out.view(batch_size * seq_len, -1)
        output = self.fc(lstm_out)

        softmax_out = F.softmax(output, dim=1)
        softmax_out = softmax_out.view(batch_size, seq_len, -1)
        """


        lstm_out = lstm_out.view(batch_size * seq_len, -1)
        output = self.fc(lstm_out)

        output = output.view(batch_size, seq_len, -1)

        output += inputs
        # out = F.softmax(output, dim=1)

        return output



if __name__ == '__main__':

    seq_length = 1440
    feature_per_time = 7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = BLSTM(input_size=feature_per_time, hidden_size=64, num_layers=4, output_size=7, device=device).to(device)

    # criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters())

    # dummy_input = torch.randn(1, seq_length, feature_per_time).to(device)
    # print(dummy_input)
    # print(dummy_input.shape)

    # with torch.no_grad():
    #     model.eval()
    #     output = model(dummy_input)

    # print(output)
    # print(output.shape)



    model = SeqModel(input_size=7, hidden_size=1440, num_layers=2, device=device).to(device)

    inputs = torch.randn(1, seq_length, 7)
    with torch.no_grad():
        tag_seq = model(inputs)

    print(tag_seq)
    print(tag_seq.shape)


    loss = model.neg_log_likelihood_loss(inputs, batch_label=7)
    loss.backward()
    
    print(loss.item())

