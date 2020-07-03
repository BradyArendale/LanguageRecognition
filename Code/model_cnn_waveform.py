import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Concatenate vectors from parts
X_train = torch.load('tensors/waveforms_train.pt')
X_train.detach_()
y_train = torch.load('tensors/labels_train.pt')
# Subset training data
# X_train, _, y_train, _ = train_test_split(X_train, y_train, stratify=y_train, train_size=0.2, random_state=9001)
print("Train shape:", X_train.shape)
# Concatenate vectors from parts
X_valid = torch.load('tensors/waveforms_valid.pt')
X_valid.detach_()
y_valid = torch.load('tensors/labels_valid.pt')
# Subset validation data
# X_valid, _, y_valid, _ = train_test_split(X_valid, y_valid, stratify=y_valid, train_size=0.2, random_state=9001)
print("Validation shape:", X_valid.shape)

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
        self.conv3 = nn.Conv1d(128, 256, 2)
        self.bn3 = nn.BatchNorm1d(256)
        self.d3 = nn.Dropout(0)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.d4 = nn.Dropout(0)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(15) #input should be 512x15 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 100)
        self.d5 = nn.Dropout(0)
        self.fc2 = nn.Linear(100, 8)
    def forward(self, x):
        x = x.unsqueeze(1)
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
        loss.backward(retain_graph=True)
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
    torch.save(model.state_dict(), 'checkpoints/cnn_waveform/cnn_waveform_checkpoint_'+str(epoch)+'.pt')
    # Save training curves
    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics)]).reset_index(drop=True)
    metrics_df.to_csv('model_cnn_metrics.csv')
    # Early stopping (no validation accuracy improvement in last 10)
    if epoch >= 10:
        acc = metrics_df['validation_accuracy']
        # Count number of validation accuracies less than current in last 10
        es_criterion = sum(acc[epoch] > i for i in acc[epoch-10:epoch])
        if es_criterion == 0:
            print('Early stopping criterion by validation accuracy reached')
            break


# Save model
torch.save(model.state_dict(), 'model_cnn_weights.pt')

# Test results
X_test = torch.load('tensors/waveforms_test.pt')
y_test = torch.load('tensors/labels_test.pt')

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    torch.save(y_pred, 'model_cnn_predictions.pt')
    # Get predicted class
    y_pred = y_pred.argmax(dim=1)
    print('Test accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion matrix:', confusion_matrix(y_test, y_pred))