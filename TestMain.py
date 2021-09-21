# coding: utf-8
import pandas as pd
import torch
from TestDataSet import TestDataSet, padding
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    book_dir = '../input/optiver-realized-volatility-prediction/book_test.parquet/'
    trade_dir = '../input/optiver-realized-volatility-prediction/trade_test.parquet/'
    target_file = '../input/optiver-realized-volatility-prediction/test.csv'
    model = torch.load('../input/pretrain-models/models/2021-09-20_12-06-54_0.232564.pkl').to('cuda')
    # book_dir = './data/book_test.parquet/'
    # trade_dir = './data/trade_test.parquet/'
    # target_file = './data/test.csv'
    # model = torch.load('./models/2021-09-20_12-06-54_0.232564.pkl').to('cuda')
    model.eval()
    test_set = TestDataSet(book_dir, trade_dir, target_file)
    result = []
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=padding)
    with torch.no_grad():
        for step, data in enumerate(tqdm(test_loader, position=0, leave=True)):
            books, trades = data
            books = books.float().cuda()
            trades = trades.float().cuda()
            output = model(books, trades).item() / 100.0
            keys = test_set.data[step]
            row_id = test_set.target[
                (test_set.target.stock_id == keys[0]) & (test_set.target.time_id == keys[1])].row_id.values[0]
            result.append(pd.DataFrame({'row_id': row_id, 'target': output}, index=[0]))
    result = pd.concat(result)
    print(result)
    result.to_csv('submission.csv', index=False)
