import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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

# thanks Samira
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.d1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.d2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.d3 = nn.Dropout(0)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.d4 = nn.Dropout(0)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(14) #input should be 512x14 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 100)
        self.d5 = nn.Dropout(0)
        self.fc2 = nn.Linear(100, 8)
    def forward(self, x):
        x = x.unsqueeze(1)
        print(x.shape)
        x = self.conv1(x)
        x = F.relu(self.d1(self.bn1(x)))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.d2(self.bn2(x)))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.d3(self.bn3(x)))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.d4(self.bn4(x)))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = F.relu(self.d5(self.fc1(x)))
        x = self.fc2(x)
        x = x.reshape(x.shape[0],-1)
        x = x.log_softmax(dim=1)
        return x
model = Net()
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
    torch.save(model.state_dict(), 'checkpoints/m5/m5_checkpoint_'+str(epoch)+'.pt')
    # Save training curves
    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics)]).reset_index(drop=True)
    metrics_df.to_csv('model_m5_metrics.csv')
    # Early stopping (no validation loss improvement in last 10)
    if epoch >= 10:
        val_losses = metrics_df['validation_loss']
        # Count number of validation losses less than 10 epochs ago
        es_criterion = sum(val_losses[epoch-10] > i for i in val_losses[epoch-9:epoch+1])
        if es_criterion == 0:
            print('Early stopping criterion by validation loss reached')
            break

# Save model
torch.save(model.state_dict(), 'model_m5_weights.pt')

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
    torch.load('checkpoints/m5/m5_checkpoint_'+str(best_epoch)+'.pt')
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
test_eval(model, wav_loader, labels_test, 'm5')
test_eval(model, al_wav_loader, labels_al, 'm5_al')