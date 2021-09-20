# coding: utf-8
import torch
import torch.nn as nn
from BaseModel import BaseModel


class BookTradeNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, num_head=6, dropout=0):
        super(BookTradeNet, self).__init__()
        self.books = BaseModel(input_dim, output_dim, hidden_dim, num_layers, num_head, dropout)
        self.trades = BaseModel(input_dim, output_dim, hidden_dim, num_layers, num_head, dropout)
        self.context_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim // 4),
            nn.ReLU(),
            # nn.BatchNorm1d(output_dim // 4),
            nn.Linear(output_dim // 4, output_dim * 2),
            nn.Sigmoid())
        self.fc0 = nn.Linear(output_dim * 2, output_dim * 2)
        self.fc = nn.Linear(output_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, books, trades):
        books = self.books(books)
        trades = self.trades(trades)
        x = torch.cat([books, trades], dim=1)
        x = self.fc0(x)
        x = self.dropout(x)
        gate = self.context_gate(x)
        x = x * gate
        x = self.fc(x)
        return x
