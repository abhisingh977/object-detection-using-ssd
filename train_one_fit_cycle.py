import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
#checkpoint=None
batch_size = 16  # batch size

#iterations = 125  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
#print_freq = 80  # print training status every __ batches
#lr = 1e-3  # learning rate
#decay_lr_at = [30,60,75]  # decay learning rate after these many iterations
#decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
#momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():


    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here


    test_dataset = PascalVOCDataset(data_folder,
                                    split='test',
                                    keep_difficult=keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)





    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        lr=0.01
        optimizer = torch.optim.SGD([
        {'params': model.base.parameters(), 'lr': lr/100},
        {'params': model.aux_convs.parameters(), 'lr ':lr/10 },
        ], lr=0.01, momentum=0.8)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[0.0001,0.001], max_lr=[0.001,0.005],step_size_up=31,step_size_down=31)
        #print(model)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=[0.000001, 0.00001, 0.00001],
                                                      max_lr=[0.000005, 0.00009, 0.00005], step_size_up=31,
                                                      step_size_down=31)


    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    print(model)
    #print(next(iter(test_loader)))
#    print(train_loader)
#    a=next(iter(train_loader))
#    print(a)
    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    #epochs = iterations // (len(train_dataset) // 8)
    #print(epochs)
    #decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]
    epochs = 125
    # Epochs
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,div_factor=100.0, final_div_factor=100000.0, max_lr=0.001, total_steps=66)
    for epoch in range(start_epoch, epochs):


        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              epoch=epoch)
        #print(scheduler.get_lr())

        test(test_loader=test_loader,
              model=model,
              criterion=criterion,
              #optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer,scheduler, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

   # batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        #print(i)
        # scheduler.step()
        # print(scheduler.get_lr())
#        print(boxes)
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model

        optimizer.step()

        #print(loss.item())
        losses.update(loss.item(), images.size(0))
        #batch_time.update(time.time() - start)

        start = time.time()

        scheduler.step()
        #print(scheduler.get_lr())
        # Print status
        if i  ==  len(train_loader)-1:
            print('Epoch: [{0}]\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
                                                                 # batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored



def test(test_loader, model, criterion, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.eval()   # training mode enables dropout

    with torch.no_grad():
        #batch_time = AverageMeter()  # forward prop. time
        data_time = AverageMeter()  # data loading time
        losses_test = AverageMeter()  # loss

        start = time.time()

        # Batches
        for i, (images, boxes, labels, _) in enumerate(test_loader):
    #        print(i)
    #        print(boxes)
            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            # Backward prop.
            # optimizer.zero_grad()
            # loss.backward()

            # Clip gradients, if necessary
            # if grad_clip is not None:
            #     clip_gradient(optimizer, grad_clip)
            #
            # # Update model
            # optimizer.step()
            #print(loss.item())
            losses_test.update(loss.item(), images.size(0))
            #batch_time.update(time.time() - start)

            start = time.time()


            # Print status
            if i  == len(test_loader)-1:
                print('Epoch: [{0}]\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
                                                                      #batch_time=batch_time,
                                                                      data_time=data_time, loss=losses_test))
        del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored





















if __name__ == '__main__':
    main()
