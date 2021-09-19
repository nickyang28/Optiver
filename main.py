# coding: utf-8
from torch.utils.data import DataLoader, random_split
import torch
from StockDataSet import StockDataSet, padding
from BookTradeNet import BookTradeNet
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime


def RMSPELoss(y_hat, y):
    return torch.sqrt(torch.mean(((y_hat - y) / y) ** 2))


if __name__ == '__main__':
    book_dir = './data/book_train.parquet/'
    trade_dir = './data/trade_train.parquet/'
    target_file = './data/train.csv'
    stocks = StockDataSet(book_dir, trade_dir, target_file)
    len_data = len(stocks)
    train_len = int(len_data * 0.9)
    valid_len = len_data - train_len
    train_set, valid_set = random_split(stocks, [train_len, valid_len], generator=torch.Generator().manual_seed(2021))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=padding)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True, collate_fn=padding)

    # book_net = BookNet(8, 64, 64, 4).to('cuda')
    # trade_net = TradeNet(3, 64, 64, 4).to('cuda')
    # net = CombinedNet(book_net, trade_net).to('cuda')
    # net = trade_net
    net = BookTradeNet(11, 64, 64, 4, 8, 0.1).to('cuda')

    """for name, param in net.named_parameters():
        if name.startswith("weight"):
            nn.init.xavier_normal_(param)
        else:
            nn.init.zeros_(param)"""
    # criterion = nn.MSELoss()
    criterion = RMSPELoss
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):
        running_loss = 0.0
        for step, data in enumerate(tqdm(train_loader, position=0, leave=True)):
            net.train()
            books, trades, targets = data
            books = books.float().cuda()
            trades = trades.float().cuda()
            optimizer.zero_grad()
            outputs = net(books, trades)
            # outputs = net(trades)
            targets = torch.tensor(targets).reshape(outputs.shape).cuda()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 1000 == 999:
                print('[%d, %5d] Training Loss: %.6f' % (epoch + 1, step + 1, (running_loss / 1000)))
                running_loss = 0.0

            """if step % 1000 == 999:  # print every 100 steps
                net.eval()
                eval_loss = 0.0
                batches = 0
                for e_step, e_data in enumerate(tqdm(valid_loader, position=0, leave=True)):
                    batches += 1
                    books, trades, targets = e_data
                    books = books.float().cuda()
                    trades = trades.float().cuda()
                    outputs = net(books, trades)
                    targets = torch.tensor(targets).reshape(outputs.shape).cuda()
                    loss = criterion(outputs, targets)
                    eval_loss += loss.item()
                print('[%d, %5d] Eval_Loss: %.6f' % (epoch + 1, step + 1, (eval_loss / batches)))"""
        net.eval()
        eval_loss = 0.0
        batches = 0
        for e_step, e_data in enumerate(tqdm(valid_loader, position=0, leave=True)):
            batches += 1
            books, trades, targets = e_data
            books = books.float().cuda()
            trades = trades.float().cuda()
            outputs = net(books, trades)
            targets = torch.tensor(targets).reshape(outputs.shape).cuda()
            loss = criterion(outputs, targets)
            eval_loss += loss.item()
        print('[Epoch %d] Eval_Loss: %.6f' % (epoch + 1, (eval_loss / batches)))
        now = datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(net, './models/{}_{}.pkl'.format(now, round(eval_loss / batches, 6)))
        # model = torch.load('model.pth')
