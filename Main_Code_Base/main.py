import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from CNNmodel import CNNmodell

def main():
#Define image transformation (Resizing, augmenting, converting to tensors)
    transform = transforms.Compose([
        transforms.Resize((150,150)), #Resize Image
        transforms.Grayscale(num_output_channels=1), #convert to greyscale
        transforms.RandomHorizontalFlip(), #Random flip
        transforms.RandomRotation(10), #Random rotate
        transforms.ToTensor(), #convert to Pytorch sensor
        transforms.Normalize(mean=[0.5], std=[0.5]) #Normalize grayscale image for efficiently
    ])

    # Load DataSet
    dataset = datasets.ImageFolder(root='C:/Users/Jacks/Python/Cell_Recognition/Cell_Dataset', transform=transform)

    # Split dataset into training and validation sets (80% train, 20% validation)goal
    train_size = int(0.8* len(dataset))
    val_size = len(dataset) - train_size
    train_dataset,val_dataset = torch.utils.data.random_split(dataset,[train_size, val_size])

    # Create DataLoader for efficient loading
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Initalize the Model
    model = CNNmodell()

    #Training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Loss function amd optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    num_epochs = 10 
    for epoch in range(num_epochs):
        model.train() # Set model to training
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            #Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), 'image_classification_model.pth')
    # Validation loop
    model.eval() #evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Validation Accuracy: {100 * correct / total}%')

    # Load trained model
    model.load_state_dict(torch.load('image_classification_model.pth', weights_only=True))
    model.eval()

    #Function to predict a single image
    def predict_image(image_path):
        img = Image.open(image_path).convert('L') # Convert to greyscale
        img = transform(img).unsqueeze(0) #Add batch Dimension

        with torch.no_grad():
            outputs = model(img.to(device))
            _, predicted = torch.max(outputs, 1)

        return dataset.classes[predicted.item()]

    #Example
    image_path = 'C:/Users/Jacks/Python/Cell_Recognition/Main_Code_Base/Sample_1206.jpg'
    predicted_class = predict_image(image_path)
    print(f'Predicted class: {predicted_class}')
if __name__ == '__main__':
    main()