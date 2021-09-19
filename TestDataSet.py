# coding: utf-8
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import numpy as np


def padding(batch):
    books = [_[0] for _ in batch]
    books = pack_sequence(books, enforce_sorted=False)
    # books = pad_sequence(books, batch_first=True)
    trades = [_[1] for _ in batch]
    trades = pack_sequence(trades, enforce_sorted=False)
    # trades = pad_sequence(trades, batch_first=True)
    targets = [_[2] for _ in batch]
    return books, trades, targets


class TestDataSet(Dataset):

    def __init__(self, _book_dir, _trade_dir):
        super(TestDataSet, self).__init__()
        self.book_dir = _book_dir
        self.trade_dir = _trade_dir
        self.data = {}
        self.books = {}
        self.trades = {}

        index = 0
        stock_list = sorted([int(_.split('=')[1]) for _ in os.listdir(self.book_dir)])
        for stock_id in stock_list:
            book_train = pd.read_parquet(self.book_dir + f'stock_id={stock_id}')
            trade_train = pd.read_parquet(self.trade_dir + f'stock_id={stock_id}')
            self.books[stock_id] = book_train
            self.trades[stock_id] = trade_train
            for tick in trade_train.time_id.unique():
                self.data[index] = (stock_id, tick)
                index += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stock_id, tick = self.data[idx]
        book_train = self.books[stock_id]
        trade_train = self.trades[stock_id]

        book_data = ((book_train[book_train.time_id == tick].iloc[:, 2:] - [9.997256e-01, 1.000272e+00, 9.995367e-01,
                                                                            1.000460e+00, 9.264611e+02, 9.215677e+02,
                                                                            1.179289e+03, 1.143937e+03]) / [
                         3.811612e-03, 3.810952e-03, 3.822038e-03, 3.820868e-03, 5.769687e+03, 5.252489e+03,
                         7.155213e+03, 6.095076e+03]).values
        trade_data = ((trade_train[trade_train.time_id == tick].iloc[:, 2:] - [9.999719e-01, 3.560419e+02,
                                                                               4.173204e+00]) / [4.607074e-03,
                                                                                                 1.245112e+03,
                                                                                                 7.799558e+00]).values
        books_mean = book_data.mean(axis=0)
        trades_mean = trade_data.mean(axis=0)
        repeat_book_mean = np.vstack([books_mean] * len(trade_data))
        trade_data = np.hstack([trade_data, repeat_book_mean])
        repeat_trade_mean = np.vstack([trades_mean] * len(book_data))
        book_data = np.hstack([book_data, repeat_trade_mean])

        return torch.FloatTensor(book_data), torch.FloatTensor(trade_data)


if __name__ == '__main__':
    book_dir = './data/book_test.parquet/'
    trade_dir = './data/trade_test.parquet/'
    stocks = TestDataSet(book_dir, trade_dir)
    print(stocks[0])
