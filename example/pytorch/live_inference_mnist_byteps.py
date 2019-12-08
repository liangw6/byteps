from __future__ import print_function
import argparse
import os
import copy
import threading
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import byteps.torch as bps

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during pushpull')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# BytePS: initialize library.
bps.init()
torch.manual_seed(args.seed)

if args.cuda:
    # BytePS: pin GPU to local rank.
    torch.cuda.set_device(bps.local_rank())
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.MNIST('data-%d' % bps.rank(), train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
# BytePS: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bps.size(), rank=bps.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

test_dataset = \
    datasets.MNIST('data-%d' % bps.rank(), train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
# BytePS: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bps.size(), rank=bps.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# BytePS: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.lr * bps.size(),
                      momentum=args.momentum)

# BytePS: (optional) compression algorithm.
compression = bps.Compression.fp16 if args.fp16_pushpull else bps.Compression.none

# BytePS: wrap optimizer with DistributedOptimizer.
optimizer = bps.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)


# BytePS: broadcast parameters.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
# bps.broadcast_optimizer_state(optimizer, root_rank=0)

def train(epoch):
    model.train()
    # BytePS: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # BytePS: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = bps.push_pull(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # BytePS: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # BytePS: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # BytePS: print output only on first rank.
    if bps.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))

    return test_loss, test_accuracy

best_test_accuracy = None
best_test_loss = None
best_model = None
model_update_lock = threading.Lock()

def byteps_model_train():
    global best_test_accuracy
    global best_test_loss
    global best_model
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss, test_accuracy = test()

        if best_test_accuracy == None or test_accuracy < best_test_accuracy:
            model_update_lock.acquire()
            best_test_accuracy = test_accuracy
            best_test_loss = test_loss

            best_model = copy.deepcopy(model)
            model_update_lock.release()

train_thread = threading.Thread(target=byteps_model_train)
train_thread.start()

continue_inference = True

while continue_inference:
    image_file_path = input("Live Inference (Path to Local Image File or q to quit): ")

    if os.path.isfile(image_file_path):

        if image_file_path == 'q':
            continue_inference = False
            continue

        image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE) # Default CV2 image format is BGR

        # Potentially need to convert image here to be right shape (or assume correct)
        model_update_lock.acquire()

        output = model(np.array([image]))
        pred = output.data.max(1, keepdim=True)[1]

        print("Prediction: ", pred)
        model_update_lock.release()
    else:
        print("Local file not found: ", image_file_path)