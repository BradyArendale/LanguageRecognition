import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Load training set
X_train = torch.load('tensors/waveforms_train.pt')
y_train = torch.load('tensors/labels_train.pt')
# Load validation set
X_valid = torch.load('tensors/waveforms_valid.pt')
y_valid = torch.load('tensors/labels_valid.pt')

train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True, num_workers=10)
val_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, pin_memory=True, num_workers=10)

# SincNet module
CNN_arch = {'input_dim': 16000,
          'fs': 8000,
          'cnn_N_filt': [80,60,60],
          'cnn_len_filt': [251,5,5],
          'cnn_max_pool_len':[3,3,3],
          'cnn_use_laynorm_inp': True,
          'cnn_use_batchnorm_inp': False,
          'cnn_use_laynorm':[True,True,True],
          'cnn_use_batchnorm':[False,False,False],
          'cnn_act': ["leaky_relu","leaky_relu","leaky_relu"],
          'cnn_drop':[0.0,0.0,0.0],          
          }
CNN_net = CNN(CNN_arch)

DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': [2048,2048,2048],
          'fc_drop': [0.2,0.2,0.2], 
          'fc_use_batchnorm': [True,True,True],
          'fc_use_laynorm': [False,False,False],
          'fc_use_laynorm_inp': True,
          'fc_use_batchnorm_inp': False,
          'fc_act': ["leaky_relu","leaky_relu","leaky_relu"],
          }
DNN1_net = MLP(DNN1_arch)

DNN2_arch = {'input_dim': 2048,
          'fc_lay': [8],
          'fc_drop': [0.2], 
          'fc_use_batchnorm': [False],
          'fc_use_laynorm': [False],
          'fc_use_laynorm_inp': False,
          'fc_use_batchnorm_inp': False,
          'fc_act': ["softmax"],
          }
DNN2_net = MLP(DNN2_arch)

model = nn.Sequential(
    CNN_net,
    DNN1_net,
    DNN2_net
)

device = torch.device("cuda:0")
model.to(device)

opt = torch.optim.RMSprop(model.parameters(), lr=0.001,alpha=0.95, eps=1e-8) 
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
    torch.save(model.state_dict(), 'checkpoints/sincnet/sincnet_checkpoint_'+str(epoch)+'.pt')
    # Save training curves
    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics)]).reset_index(drop=True)
    metrics_df.to_csv('model_sincnet_metrics.csv')
    # Early stopping (no validation loss improvement in last 10)
    if epoch >= 10:
        val_losses = metrics_df['validation_loss']
        # Count number of validation losses less than 10 epochs ago
        es_criterion = sum(val_losses[epoch-10] > i for i in val_losses[epoch-9:epoch+1])
        if es_criterion == 0:
            print('Early stopping criterion by validation loss reached')
            break

# Save model
torch.save(model.state_dict(), 'model_sincnet_weights.pt')

# Test results
# Load Common Voice/Bengali test data
wav_test = torch.load('tensors/waveforms_test.pt')
wav_test.detach_()
labels_test = torch.load('tensors/labels_test.pt')

wav_ds = TensorDataset(wav_test, labels_test)
wav_loader = DataLoader(wav_ds, batch_size=128, shuffle=False, 
                          pin_memory=True, num_workers=10)

# Load Audio Lingua test data
wav_al = torch.load('tensors/audio_lingua_waveforms.pt')
labels_al = torch.load('tensors/audio_lingua_labels.pt')

al_wav_ds = TensorDataset(wav_al, labels_al)
al_wav_loader = DataLoader(al_wav_ds, batch_size=128, shuffle=False, 
                          pin_memory=True, num_workers=10)

cpu = torch.device("cpu")
# Find minimum validation loss and load weights
best_epoch = metrics_df['validation_loss'].argmin()
model.load_state_dict(
    torch.load('checkpoints/sincnet/sincnet_checkpoint_'+str(best_epoch)+'.pt')
    )

# Define evaluation function
def test_eval(model, data_loader, y_true, label):
    print("\n\n******* Evaluate %s *******\n" % label)
    model.eval()
    y_pred = torch.tensor([])
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            pred_prob = model(x)
            y_pred = torch.cat([y_pred, pred_prob.to(cpu)])
    print(''.join(["Test accuracy for ", label, ": ", 
                   str(accuracy_score(y_true, y_pred.argmax(dim=1)))]))
    print(''.join(["Test confusion matrix for ", label, ":\n", 
                   str(confusion_matrix(y_true, y_pred.argmax(dim=1)))]))
    torch.save(y_pred, 'preds/test_preds_'+label+'.pt')
    torch.save(y_pred.argmax(dim=1), 'preds/test_preds_'+label+'_labels.pt')

# Evaluate performance on test sets
test_eval(model, wav_loader, labels_test, 'sincnet')
test_eval(model, al_wav_loader, labels_al, 'sincnet_al')