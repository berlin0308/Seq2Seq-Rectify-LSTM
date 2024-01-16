import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


class Attention(nn.Module):
    def __init__(self, hidden_size, window_size=10):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.attention = nn.Linear(hidden_size * 4, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]

        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):

        energy = torch.zeros_like(encoder_outputs[:, :, 0])
        center = self.window_size // 2

        for i in range(encoder_outputs.size(1)):

            start = max(0, i - center)
            end = min(encoder_outputs.size(1), i + center + 1)

            local_hidden = hidden[:, start:end, :]
            local_encoder_outputs = encoder_outputs[:, start:end, :]

            local_combined = torch.cat([local_hidden, local_encoder_outputs], dim=2)
            local_energy = torch.tanh(self.attention(local_combined))
            local_energy = local_energy.transpose(1, 2)
            local_v = self.v.repeat(local_encoder_outputs.size(0), 1).unsqueeze(1)
            local_energy = torch.bmm(local_v, local_energy).squeeze(1)

            energy[:, start:end] = local_energy

        return energy


class SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, window_size=10, device='cpu'):
        super(SeqModel, self).__init__()

        self.gpu = False
        label_size = 7

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        # self.lstm_bn = nn.BatchNorm1d(self.hidden_size * 2)
        # self.lstm_relu = nn.ReLU()

        self.attention = Attention(hidden_size, window_size=window_size)
        
        self.fc1 = nn.Linear(self.hidden_size * 2, label_size*8)
        self.bn1 = nn.BatchNorm1d(label_size*8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(label_size*8, label_size)

        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=7, reduction='sum'
                                           ,weight=torch.tensor([1.0/211, 1.0/216, 1.0/26, 1.0/20, 1.0/718, 1.0/6, 1.0/200])).to(self.device)



    def neg_log_likelihood_loss(self, inputs, batch_label, mask=torch.ones(1, 1440).long()):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        outputs = self.forward(inputs)

        outputs = outputs.view(batch_size * seq_len, -1)
        batch_label = batch_label.view(batch_size * seq_len)

        total_loss = self.loss_fn(outputs.to(self.device), batch_label.to(self.device))

        return total_loss


    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        lstm_out, (hidden, cell) = self.lstm(inputs)
        # lstm_out = self.lstm_bn(lstm_out.transpose(1, 2)).transpose(1, 2)
        # lstm_out = self.lstm_relu(lstm_out)

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = hidden.repeat(1, lstm_out.size(1), 1)  # [1, 1440, H*2]

        attn_weights = self.attention(hidden, lstm_out)  # [B, 1, T]

        lstm_out = torch.bmm(attn_weights, lstm_out.transpose(0, 1))  # [B, 1, H*2]
        lstm_out = lstm_out.squeeze(1)

        output = self.fc1(lstm_out)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = output.view(batch_size, seq_len, -1)

        output += inputs
        # output = F.softmax(output, dim=1)

        return output



if __name__ == '__main__':

    device = torch.device('cpu')
    
    model = SeqModel(input_size=7, hidden_size=1440, num_layers=2).to(device)

    inputs = torch.randn(1, 1440, 7).to(device)
    with torch.no_grad():
        tag_seq = model(inputs)

    print(tag_seq)
    print(tag_seq.shape)


    # loss = model.neg_log_likelihood_loss(inputs, batch_label=7)
    # loss.backward()
    
    # print(loss.item())

