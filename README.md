# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Image denoising is a key task in computer vision where the objective is to remove unwanted noise from images while preserving important visual details.
Traditional filtering techniques like Gaussian or median filters often blur the image, whereas Convolutional Autoencoders (CAEs) can learn efficient representations to restore clean images automatically.

In this experiment, a convolutional autoencoder is developed to denoise images by learning compressed representations of noisy inputs.
The dataset used is the MNIST dataset, consisting of 28×28 grayscale handwritten digit images.
Random noise is added to the images to train the model, which learns to reconstruct the original clean images from their noisy counterparts.

## DESIGN STEPS

### Step 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, and Matplotlib.

### Step 2:
Load the MNIST dataset and add random noise to create noisy input images.

### Step 3:
Normalize the image data and prepare dataloaders for training and testing.

### Step 4:
Define the Convolutional Autoencoder model with encoder and decoder layers.

### Step 5:
Specify the loss function (MSELoss) and optimizer (Adam).

### Step 6:
Train the model using noisy images as input and clean images as target output.

### Step 7:
Monitor training loss to ensure the model learns effective noise removal.

### Step 8:
Test the trained model on noisy test images to evaluate performance.

### Step 9:
Visualize original, noisy, and denoised images for comparison.
## PROGRAM
### Name:SHARAN G
### Register Number:212223230203

```py
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
      super(DenoisingAutoencoder,self).__init__()
      self.encoder=nn.Sequential(
          nn.Conv2d(1, 16, 3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(16, 32, 3, stride=2, padding=1),
          nn.ReLU()
      )
      self.decoder=nn.Sequential(
          nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
          nn.Sigmoid()
      )
    def forward(self,x):
      x=self.encoder(x)
      x=self.decoder(x)
      return x

## Model Training
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)

            # Add noise
            noisy_inputs = add_noise(inputs)
            noisy_inputs = noisy_inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

    print('Finished Training')
```

## OUTPUT

### Model Summary
<img width="717" height="477" alt="image" src="https://github.com/user-attachments/assets/672e715a-7c67-462e-a7fc-2ae9553f1b17" />


### Original vs Noisy Vs Reconstructed Image
<img width="1539" height="612" alt="image" src="https://github.com/user-attachments/assets/b74972bb-8ce3-4fe0-9b4e-036a7a4cf2fd" />



## RESULT

The convolutional autoencoder model was successfully implemented and effectively removed noise from images, restoring them close to their original quality.
