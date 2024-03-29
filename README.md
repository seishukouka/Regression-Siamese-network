# Regression-Based Siamese Network for Eye Opening Rate Prediction

This project implements a regression-based Siamese network that predicts the rate of eye opening from images. The network uses pairs of eye images to compare "normal state images" and "current state images" taken from various angles to estimate on a scale of 0 to 1 how open the eyes are. It is particularly useful in applications where understanding the state of the eyes is important, such as driver drowsiness detection systems.

## Project Structure

- `augmentation_images/`: Directory containing the training and validation image sets.
- `validation_data/`: Directory containing additional images for validation.
- `best_model.pth`: The saved model weights after training.
- `siamese_network.py`: Main script for defining and training the Siamese Network model.
- `dataset.py`: Script for data loading and preprocessing.
- `loss.py`: Definition of the ModifiedContrastiveLoss used for training the network.
- `utils.py`: Miscellaneous utility functions used across the project.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8 or higher
- PyTorch 1.7 or higher
- torchvision 0.8 or higher
- PIL (Pillow)
- NumPy
- Matplotlib
- OpenCV

You can install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```
## Usage

To use this project, follow these steps:

1. Prepare your dataset by placing your normal and diagonal images in the `augmentation_images/` directory.
2. Run the `siamese_network.py` script to train the model:

    ```bash
    python siamese_network.py
    ```

   After training, the best model will be saved as `best_model.pth`.

3. For evaluating the model on the validation set, you can modify and use the evaluation code section in `siamese_network.py`.

## Customization

You can customize the following aspects of the project:

- **Data Path**: Change `data_dir` and `normal_image_path` in the `siamese_network.py` script to point to your dataset directories.
- **Hyperparameters**: Modify learning rate, batch size, and number of epochs as needed in the training section of the script.
- **Network Architecture**: Adjust the Siamese Network architecture in `siamese_network.py` according to your requirements.

## Contributing

Contributions to this project are welcome! Please fork the repository and create a pull request with your improvements.

## License

This project is open-source and available under the MIT License. See the LICENSE file for more information.
