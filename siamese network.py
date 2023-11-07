
# 必要なライブラリをインポート
import os
from matplotlib.lines import Line2D
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# データをロード
def load_data(data_dir, normal_image_path):
    diagonal_dir = os.path.join(data_dir, "augmentation_diagonal")  # 斜めから撮った現在の状態の画像のディレクトリ

    diagonal_images = []  # 画像リスト
    file_names = []  # ファイル名リスト

    # 眼の開眼率を読み込む
    eye_opening_rates = {}
    with open(os.path.join(data_dir, "output.txt"), "r") as f:
        for line in f.readlines():
            file_name, rate = line.strip().split()
            eye_opening_rates[file_name] = float(rate)

    # 画像を読み込む
    for file in sorted(os.listdir(diagonal_dir)):
        if file.endswith(".jpg"):  # 全ての画像を読む
            img_path = os.path.join(diagonal_dir, file)
            img = Image.open(img_path).convert("RGB")  # 画像をRGBモードで読む
            diagonal_images.append(img)
            file_names.append(file)

    # 正常画像を読む
    normal_image = Image.open(normal_image_path).convert("RGB")  # 画像をRGBモードで読む

    return diagonal_images, normal_image, eye_opening_rates, file_names

# データセットクラス
class EyeDataset(Dataset):
    def __init__(self, diagonal_images, normal_image, eye_opening_rates, file_names):
        self.diagonal_images = []
        self.normal_image = normal_image
        self.eye_opening_rates = {}
        self.file_names = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 画像をリサイズ
            transforms.ToTensor()  # テンソルに変換
        ])

        # 眼の開き度が1.0を超える画像を無視（1.0以上の取り扱いを考え中）
        for img, file_name in zip(diagonal_images, file_names):
            if eye_opening_rates[file_name] <= 1.0:
                self.diagonal_images.append(img)
                self.eye_opening_rates[file_name] = eye_opening_rates[file_name]
                self.file_names.append(file_name)

    def __len__(self):
        return len(self.diagonal_images)

    def __getitem__(self, idx):
        # 画像をトランスフォーム
        diagonal_image = self.transform(self.diagonal_images[idx])
        normal_image = self.transform(self.normal_image)

        file_name = self.file_names[idx]
        eye_opening_rate = self.eye_opening_rates[file_name]

        # 2つの画像、眼の開眼率、ファイル名のタプルを返す
        return (diagonal_image, normal_image), eye_opening_rate, file_name

# シャムネットワークのクラス
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 事前訓練済みのResNet18を使用
        self.resnet.fc = nn.Identity()  # 全結合層をIdentityに置き換え
        self.fc = nn.Linear(512, 1)  # 眼の開き度を予測するための回帰層を追加

    def forward(self, img1, img2):
        feat1 = self.resnet(img1)
        feat2 = self.resnet(img2)
        diff = torch.sqrt(torch.sum(torch.square(feat1 - feat2), dim=1, keepdim=True))  # ユークリッド距離を使用
        eye_opening_rate = self.fc(diff)  # 特徴量の差を用いて眼の開眼率を予測
        return eye_opening_rate, diff  # 予測された眼の開眼率と特徴量の差を返す




class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        self.linear = nn.Linear(512, 1)

    def forward(self, diff):
        return self.linear(diff)

class ModifiedContrastiveLoss(nn.Module):
    def __init__(self, base_margin=1.0):
        super(ModifiedContrastiveLoss, self).__init__()
        self.base_margin = base_margin

    def forward(self, eye_opening_rate, D):
        # y_trueが正しい形状かつタイプになるように調整します
        #print("eye_opening_rate",eye_opening_rate)
        y_true = eye_opening_rate.view(-1, 1).float()
        # Compute the contrastive loss as before
        weight = torch.abs(1.0 - y_true)
        weight_repeated = weight.repeat_interleave(repeats=D.shape[1], dim=1)
        #print(weight_repeated)
        term1 = D**2
        margin = self.base_margin * weight_repeated
        term2 = torch.max(torch.zeros_like(D), margin - D)**2
        contrastive_loss = 0.5 * (term1 + term2)
        #print("D",D)
        #print("margin",margin)
        #print("y_true",y_true)
        #print("contrastive_loss",contrastive_loss)
        return torch.mean(contrastive_loss)





def plot_feature_difference_per_dimension(siamese_network, data_loader, epoch, data_dir):
    # Switch siamese_network to evaluation mode
    siamese_network.eval()

    # Create a dictionary to store feature differences for each augmentation type
    feature_diffs = {"normal": [], "flipped": [], "hue_changed": [], "sharpness_changed": [], "cropped": [], "brightness_adjusted": []}
    true_eye_opening_rates = {"normal": [], "flipped": [], "hue_changed": [], "sharpness_changed": [], "cropped": [], "brightness_adjusted": []}

    with torch.no_grad():
        for ((diagonal_image, normal_image), eye_opening_rate, file_name) in data_loader:
            diagonal_image = diagonal_image.to(device)  # Move the images to the device
            normal_image = normal_image.to(device)  # Move the images to the device

            # Take images from the dataset and pass them through the network
            predicted_eye_opening_rate, D = siamese_network(diagonal_image, normal_image)
            feat1, feat2 = siamese_network.resnet(diagonal_image), siamese_network.resnet(normal_image)


            # Move tensors back to CPU before converting to numpy
            diff_np = D.cpu().detach().numpy()

            # Add difference and eye opening rate to the list
            for i in range(len(diff_np)):
                # Calculate mean of diffs_np[i]
                mean_diff = np.mean(diff_np[i])
                # Use the filename to determine the augmentation type
                for key in feature_diffs.keys():
                    if key in file_name[i]:
                        feature_diffs[key].append(mean_diff)
                        true_eye_opening_rates[key].append(eye_opening_rate[i].item())
                        break
                else:
                    feature_diffs["normal"].append(mean_diff)
                    true_eye_opening_rates["normal"].append(eye_opening_rate[i].item())

    # Plotting for each augmentation type
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']

    for i, key in enumerate(feature_diffs.keys()):
        normal_eye_opening_rates = [rate for rate in true_eye_opening_rates[key] if rate < 1.0]
        high_eye_opening_rates = [rate for rate in true_eye_opening_rates[key] if rate >= 1.0]
        normal_feature_diffs = [feature_diffs[key][j] for j, rate in enumerate(true_eye_opening_rates[key]) if rate < 1.0]
        high_feature_diffs = [feature_diffs[key][j] for j, rate in enumerate(true_eye_opening_rates[key]) if rate >= 1.0]

        ax.scatter(normal_eye_opening_rates, normal_feature_diffs, color=colors[i], marker=markers[0], label=f"{key} < 1.0")
        ax.scatter(high_eye_opening_rates, high_feature_diffs, color=colors[i], marker=markers[1], label=f"{key} >= 1.0")

    ax.set_xlabel("True Eye Opening Rate")
    ax.set_ylabel("Mean Feature Difference")
    ax.legend()
    plt.title('True Eye Opening Rate vs Mean Feature Difference for Different Augmentations')
    plt.savefig(f"{data_dir}/plot/plot_epoch_{epoch}.png")
    plt.close(fig)



# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the data
data_dir = "augmentation_images"
normal_image_path = "augmentation_images/augmentation_normal/93_diagonal.jpg"
diagonal_images, normal_image, eye_opening_rates, file_names = load_data(data_dir, normal_image_path)

# Create the dataset
dataset = EyeDataset(diagonal_images, normal_image, eye_opening_rates, file_names)

# Create the data loader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the siamese network
siamese_network = SiameseNetwork().to(device)

# Specify the loss function and optimizer
criterion = ModifiedContrastiveLoss(base_margin=1.0)
optimizer = torch.optim.Adam(siamese_network.parameters(), lr=0.001)

# Load the validation data
val_data_dir = "validation_data"  # validationデータのディレクトリを設定してください
val_normal_image_path = "validation_data/augmentation_normal/93_diagonal.jpg"  # validationデータの正常画像のパスを設定してください
val_diagonal_images, val_normal_image, val_eye_opening_rates, val_file_names = load_data(val_data_dir, val_normal_image_path)

val_dataset = EyeDataset(val_diagonal_images, val_normal_image, val_eye_opening_rates, val_file_names)

# Create the validation data loader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import matplotlib.pyplot as plt

# Training
num_epochs = 20
best_loss = float('inf')
best_epoch = -1

for epoch in range(num_epochs):
    for i, ((diagonal_image, normal_image), eye_opening_rate, file_name) in enumerate(train_loader):
        diagonal_image = diagonal_image.to(device)
        normal_image = normal_image.to(device)
        eye_opening_rate = eye_opening_rate.float().to(device)

        predicted_eye_opening_rate, D = siamese_network(diagonal_image, normal_image)
        loss = criterion(eye_opening_rate, D)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot some images during the training
        # if i % 50 == 0:  # Change this condition according to your needs
        #     diagonal_image_cpu = diagonal_image.cpu().numpy().transpose((0, 2, 3, 1))
        #     normal_image_cpu = normal_image.cpu().numpy().transpose((0, 2, 3, 1))

        #     fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

        #     for j, ax in enumerate(axes.flat):
        #         if j < 4:
        #             ax.imshow(diagonal_image_cpu[j])
        #             ax.set_title("Diagonal Image")
        #         else:
        #             ax.imshow(normal_image_cpu[j-4])
        #             ax.set_title("Normal Image")
        #         ax.axis("off")

        #     plt.tight_layout()
        #     plt.show()

    # Save the model at the end of each epoch
    torch.save({
        'siamese': siamese_network.state_dict(),
    },'best_model.pth')

    # Plot
    # plot_feature_difference_per_dimension(siamese_network, train_loader, epoch + 1, data_dir)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

