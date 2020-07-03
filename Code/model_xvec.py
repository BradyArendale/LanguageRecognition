import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Load training set
X_train = torch.load('tensors/filterbanks_train.pt')
X_train.detach_()
y_train = torch.load('tensors/labels_train.pt')
print("Train shape:", X_train.shape)
# Load validation set
X_valid = torch.load('tensors/filterbanks_valid.pt')
X_valid.detach_()
y_valid = torch.load('tensors/labels_valid.pt')
print("Validation shape:", X_valid.shape)

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, pin_memory=True)

# Pooling function for x-vector network
def mean_std_pooling(x, eps=1e-9):
    m = torch.mean(x, dim=2)
    s = torch.sqrt(torch.mean((x - m.unsqueeze(2))**2, dim=2) + eps)
    x = torch.cat([m, s], dim=1)
    return x

# courtesy of Daniel Garcia-Romero
class xvec(nn.Module):

    def __init__(self, input_dim, layer_dim, mid_dim, embedding_dim, expansion_rate=3, drop_p=0.0, bn_affine=False):

        super(xvec, self).__init__()

        layers = []
        # conv blocks                                                                                                                                                                                                                                                                                         
        layers.extend(self.conv_block(1, input_dim, layer_dim, mid_dim, 5, 1, 0, bn_affine))
        layers.extend(self.conv_block(2, mid_dim, layer_dim, mid_dim, 3, 2, 0, bn_affine))
        layers.extend(self.conv_block(3, mid_dim, layer_dim, mid_dim, 3, 3, 0, bn_affine))
        layers.extend(self.conv_block(4, mid_dim, layer_dim, layer_dim, 3, 4, 0, bn_affine))

        # expansion layer                                                                                                                                                                                                                                                                                     
        layers.extend([('expand_linear', nn.Conv1d(layer_dim, layer_dim*expansion_rate, kernel_size=1)),
                       ('expand_relu', nn.LeakyReLU(inplace=True)),
                       ('expand_bn', nn.BatchNorm1d(layer_dim*expansion_rate, affine=False))])

        # Dropout pre-pooling                                                                                                                                                                                                                                                                                 
        if drop_p > 0.0:
            layers.extend([('drop_pre_pool', nn.Dropout2d(p=drop_p, inplace=True))])

        self.prepooling_layers = nn.Sequential(OrderedDict(layers))

        # pooling defined below                                                                                                                                                                                                                                                                               

        # embedding                                                                                                                                                                                                                                                                                           
        self.embedding = nn.Linear(layer_dim*expansion_rate*2, embedding_dim)

        self.init_weight()

    def conv_block(self, index, in_channels, mid_channels, out_channels, kernel_size, dilation, padding, bn_affine=False):
         return [('conv%d' % index, nn.Conv1d(in_channels, mid_channels, kernel_size, dilation=dilation, padding=padding)),
                 ('relu%d' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%d' % index, nn.BatchNorm1d(mid_channels, affine=bn_affine)),
                 ('linear%d' % index, nn.Conv1d(mid_channels, out_channels, kernel_size=1)),
                 ('relu%da' % index, nn.LeakyReLU(inplace=True)),
                 ('bn%da' % index, nn.BatchNorm1d(out_channels, affine=bn_affine))]

    def init_weight(self):
        """                                                                                                                                                                                                                                                                                                   
        Initialize weight with sensible defaults for the various layer types                                                                                                                                                                                                                                  
        :return:                                                                                                                                                                                                                                                                                              
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                print("Initializing %s with kaiming_normal" % str(m))                                                                                                                                                                                                                                                      
                nn.init.kaiming_normal_(m.weight, a=0.01)
            
            if isinstance(m, nn.Linear):
                print("Initializing %s with kaiming_normal" % str(m))                                                                                                                                                                                                                                                          
                nn.init.kaiming_normal_(m.weight, a=0.01)


    def extract_pre_pooling(self, x):
        x = self.prepooling_layers(x)
        return x

    def extract_post_pooling(self, x):
        x = self.extract_pre_pooling(x)
        x = mean_std_pooling(x)
        return x

    def extract_embedding(self, x):        
        x = self.extract_post_pooling(x)
        x = self.embedding(x)
        return x

    def forward(self, x):
        # Compute embeddings                                                                                                                                                                                   #                                                                                               
        x = self.extract_embedding(x)
        return x

# E-TDNN architecture
model = nn.Sequential(
    xvec(input_dim=64, layer_dim=512, mid_dim=198, embedding_dim=512),
    nn.LeakyReLU(inplace=True),
    nn.Linear(512, 512),
    nn.LeakyReLU(inplace=True),
    nn.Linear(512,8),
    nn.LogSoftmax(dim=1)
)
device = torch.device("cuda:0")
model.to(device)

opt = torch.optim.Adam(model.parameters())
loss_func = torch.nn.NLLLoss()

log_interval = 1000
metrics_df = pd.DataFrame()

def train(model, epoch, data_loader):
    print("\n\n#### Starting training epoch %s ####\n\n" % epoch)
    model.train()
    for batch_idx, (x, y) in enumerate(data_loader):
        opt.zero_grad()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_func(pred, y) #the loss functions expects a batchSizex10 input
        loss.backward()
        opt.step()
        if batch_idx % log_interval == 999: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
            
def test(model, epoch, data_loader, label):
    print("\n\n******* Evaluate %s *******\n" % label)
    model.eval()
    y_pred = []
    y_true = []
    loss = []
    correct = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        pred_prob = model(x)
        batch_loss = loss_func(pred_prob, y)
        loss.append(batch_loss.item())
        preds = torch.argmax(pred_prob, dim=1)
        y_pred.extend(preds.tolist())
        y_true.extend(y.tolist())
        correct += (preds == y).float().sum()
        accuracy = 100. * correct / len(data_loader.dataset)
    mean_loss = torch.tensor(loss).mean().item()
    metrics[label+'_loss'].append(mean_loss)
    metrics[label+'_accuracy'].append(accuracy.item())
    print('\nLoss: {:.4f}'.format(mean_loss))
    print('Accuracy: {}/{} ({:.4f}%)'.format(
        correct, len(data_loader.dataset), accuracy))
    print("\nConfusion metrics: \n%s" % confusion_matrix(y_true, y_pred))
    
for epoch in range(100):
    # Initialize training curves
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': []
    }
    # Train and evaluate model
    train(model, epoch, train_loader)
    test(model, epoch, train_loader, 'train')
    test(model, epoch, val_loader, 'validation')
    torch.save(model.state_dict(), 'checkpoints/xvec_model/xvec_model_checkpoint_'+str(epoch)+'.pt')
    # Save training curves
    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics)]).reset_index(drop=True)
    metrics_df.to_csv('model_xvec_metrics.csv')
    # Early stopping (no validation accuracy improvement in last 10)
    if epoch >= 10:
        acc = metrics_df['validation_accuracy']
        # Count number of validation accuracies less than current in last 10
        es_criterion = sum(acc[epoch] > i for i in acc[epoch-10:epoch])
        if es_criterion == 0:
            print('Early stopping criterion by validation accuracy reached')
            break

# Save model
torch.save(model.state_dict(), 'model_xvec_weights.pt')

# Test results
X_test = torch.load('tensors/filterbanks_test.pt')
y_test = torch.load('tensors/labels_test.pt')

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    torch.save(y_pred, 'model_xvec_predictions.pt')
    # Get predicted class
    y_pred = y_pred.argmax(dim=1)
    print('Test accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion matrix:', confusion_matrix(y_test, y_pred))