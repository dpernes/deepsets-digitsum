import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def accuracy(output, target):
    return (target.long() == output.round().long()).float().mean().item()

def train(model, loss_fn, optimizer, epochs, train_loader,
          valid_loader=None, device='cuda:0', visdom=None, model_path='./mnist_adder.pth'):
    reduce_lr = ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=0.000001)

    for epoch in range(epochs):
        print('Epoch {}/{}:'.format(epoch, epochs-1))

        model.train()
        loss_hist = []
        train_acc = 0.
        for i, (X, labels, set_sizes) in enumerate(train_loader):
            # copy mini-batch to device
            X = X.to(device)
            labels = labels.to(device)
            set_sizes = set_sizes.to(device)

            # compute the ground-truth (sum of digits)
            Y = torch.sum(labels, dim=1).float().unsqueeze(1)

            # forward pass through the model and get the loss
            Ypred = model(X, set_sizes)
            loss = loss_fn(Ypred, Y)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute the accuracy for evaluation purposes
            with torch.no_grad():
                train_acc += accuracy(Ypred, Y)

            print('{}/{} mini-batch loss: {:.3f}'
                  .format(i, len(train_loader)-1, loss.item()),
                  flush=True, end='\r')
            loss_hist.append(loss.item())

        train_loss = sum(loss_hist)/len(loss_hist)
        train_acc /= len(loss_hist)
        print()
        print('Train loss: {:.3f}'.format(train_loss))
        print('Train accuracy: {:.3f}'.format(train_acc))
        if visdom:
            visdom.plot(loss_fn.__name__, 'train', 'global loss', epoch, train_loss)
            visdom.plot('accuracy', 'train', 'accuracy', epoch, train_acc)

        if valid_loader is not None:
            with torch.no_grad():
                model.eval()
                loss_hist = []
                valid_acc = 0.
                for X, labels, set_sizes in valid_loader:
                    # copy mini-batch to device
                    X = X.to(device)
                    labels = labels.to(device)
                    set_sizes = set_sizes.to(device)

                    # compute the ground-truth (sum of digits)
                    Y = torch.sum(labels, dim=1).float().unsqueeze(1)

                    # forward pass through the model and get the loss and accuracy
                    Ypred = model(X, set_sizes)
                    loss = loss_fn(Ypred, Y)
                    loss_hist.append(loss.item())
                    valid_acc += accuracy(Ypred, Y)

                valid_loss = sum(loss_hist)/len(loss_hist)
                valid_acc /= len(loss_hist)
                reduce_lr.step(valid_loss)
                print('Valid loss: {:.3f}'.format(valid_loss))
                print('Valid accuracy: {:.3f}'.format(valid_acc))
                print()
                if visdom:
                    visdom.plot(loss_fn.__name__, 'valid', 'global loss', epoch, valid_loss)
                    visdom.plot('accuracy', 'valid', 'accuracy', epoch, valid_acc)

    torch.save(model.state_dict(), model_path)

def test(model, loss_fn, test_loader,
         size_range=[5, 51], device='cuda:0', visdom=None):
    with torch.no_grad():
        model.eval()
        loss_hg = {size: 0. for size in range(*size_range)}
        acc_hg = {size: 0. for size in range(*size_range)}

        for size in range(*size_range):
            print('Testing at size {}'.format(size))

            test_loader.dataset.set_max_seq_len(size)

            loss_hist = []
            test_acc = 0.
            for i, (X, labels, _) in enumerate(test_loader):
                # copy mini-batch to device
                X = X.to(device)
                labels = labels.to(device)

                # compute the ground-truth (sum of digits)
                Y = torch.sum(labels, dim=1).float().unsqueeze(1)

                # forward pass through the model and get the loss
                Ypred = model(X)
                loss = loss_fn(Ypred, Y)
                loss_hist.append(loss.item())

                test_acc += accuracy(Ypred, Y)

                print('...{:1.0f}% done '.format(100.*(i + 1)/len(test_loader)),
                      flush=True, end='\r')

            test_loss = sum(loss_hist)/len(loss_hist)
            test_acc /= len(loss_hist)
            print()
            print('Loss: {:.3f}'.format(test_loss))
            print('Accuracy: {:.3f}'.format(test_acc))
            print()

            loss_hg[size] = test_loss
            acc_hg[size] = test_acc

            if visdom:
                visdom.plot(loss_fn.__name__, 'cardinality', 'test loss', loss_hg)
                visdom.plot('accuracy', 'cardinality', 'test accuracy', acc_hg)

        return loss_hg, acc_hg
