# coding: utf-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F


def get_mask(x):
    x, lens = pad_packed_sequence(x)
    mask = [[0] * _ + [-10000] * (max(lens) - _) for _ in lens]
    return x.to('cuda'), torch.FloatTensor(mask).to('cuda')


class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, num_head=6, dropout=0):
        super(BaseModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2 * num_head, output_dim)
        self.dropout = nn.Dropout(dropout)
        # self.batch_norm = nn.BatchNorm1d(output_dim)
        # self.heads = [nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim * 4), nn.Dropout(0.2), nn.ReLU(), nn.Linear(hidden_dim * 4, 1)).to('cuda') for _ in range(num_head)]
        self.heads = [
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 4),
                          nn.ReLU(), nn.Linear(hidden_dim * 4, 1)).to('cuda') for _ in range(num_head)]

    def forward(self, x):
        x, mask = get_mask(x)
        x, (hidden, cell) = self.lstm(x)
        x = x.transpose(0, 1)
        # x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        att_li = []
        for head in self.heads:
            weight = torch.squeeze(head(x), -1) + mask
            weight = F.softmax(weight, dim=-1)
            att_li.append((x * torch.unsqueeze(weight, -1)).sum(axis=1))
        x = torch.cat(att_li, -1)
        # x = self.dropout(x)
        x = self.fc(x)
        # x = self.batch_norm(x)
        return x


if __name__ == '__main__':
    from StockDataSet import StockDataSet, padding
    from torch.utils.data import DataLoader, random_split

    book_dir = './data/book_train.parquet/'
    trade_dir = './data/trade_train.parquet/'
    target_file = './data/train.csv'
    stocks = StockDataSet(book_dir, trade_dir, target_file)
    len_data = len(stocks)
    train_len = int(len_data * 0.9)
    valid_len = len_data - train_len
    train_set, valid_set = random_split(stocks, [train_len, valid_len], generator=torch.Generator().manual_seed(2021))
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=padding)
    valid_loader = DataLoader(valid_set, batch_size=2, shuffle=True, collate_fn=padding)

    net = BaseModel(11, 64, 64, 2)
    for step, data in enumerate(train_loader):
        net.train()
        books, trades, targets = data
        books = books.float()
        trades = trades.float()
        outputs = net(books)
        print(outputs.shape)
        break
