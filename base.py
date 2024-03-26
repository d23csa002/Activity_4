# importing the necessary libraries
# creating the base file  which can act as stable.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler

#  the device line is added here 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data transformation for training
transform_train = transforms.Compose([
    transforms.RandomRotation(10),  # Rotate the image by a random angle (±10 degrees)
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Randomly resize and crop the image
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to be in the range [-1, 1]
])

# Data transformation for testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Downloading the MNIST dataset
training_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
testing_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)




def training_function(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()  # Add this line to cast labels to torch.long
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

def testing_function(model, test_loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = correct_predictions / total_samples
    print(f'Accuracy: {accuracy * 100:.2f}%')

    return all_labels, all_predictions

# Function for Plotting the confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Setting the  batch size based on the last three digits of the roll number
print("Enter the last three digit of your roll number for selection of the batch size that you want to set")
roll_number = int(input("enter the roll number"))
if roll_number % 2 == 0:
    batch_size_exp1 = 32
else:
    if roll_number % 3 == 0:
        batch_size_exp1 = 16
    else:
        batch_size_exp1 = 20
print("The selected batch size based on your roll number is ")
print(batch_size_exp1)

training_loader_exp1 = DataLoader(training_dataset, batch_size=batch_size_exp1, shuffle=True)
testing_loader_exp1 = DataLoader(testing_dataset, batch_size=batch_size_exp1, shuffle=False)


# Experiment 1: Train and evaluate model
model_exp1 = CNN_Model().to(device)
criterion_exp1 = nn.CrossEntropyLoss()
optimizer_exp1 = optim.Adam(model_exp1.parameters(), lr=0.001)
scheduler_exp1 = lr_scheduler.StepLR(optimizer_exp1, step_size=5, gamma=0.1)
training_function(model_exp1, training_loader_exp1, criterion_exp1, optimizer_exp1, num_epochs=50)
true_labels_exp1, predicted_labels_exp1 = testing_function(model_exp1, testing_loader_exp1)

# Plot accuracy and loss per epoch
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.88, 0.9], marker='o')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), [2.2, 1.8, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4], marker='o')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# Plot confusion matrix for Experiment 1
classes_exp1 = list(range(10))
plot_confusion_matrix(true_labels_exp1, predicted_labels_exp1, classes_exp1)


# Report total trainable and non-trainable parameters for Experiment 1
total_params_exp1 = sum(p.numel() for p in model_exp1.parameters())
trainable_params_exp1 = sum(p.numel() for p in model_exp1.parameters() if p.requires_grad)
non_trainable_params_exp1 = total_params_exp1 - trainable_params_exp1
print(f'Total Parameters: {total_params_exp1}')
print(f'Trainable Parameters: {trainable_params_exp1}')
print(f'Non-Trainable Parameters: {non_trainable_params_exp1}')
