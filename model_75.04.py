import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torchvision.transforms.functional import to_pil_image


class Net(nn.Module):
    def __init__(
        self,
        num_classes: int = 500,
        pretrained_model_name: str = "facebook/dinov2-large-imagenet1k-1-layer",
        freeze_backbone: bool = True,
    ):
        super(Net, self).__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model_name)
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.feature_dim = self.base_model.config.hidden_size

        if freeze_backbone:
            # Freeze all layers in the backbone
            self.base_model.requires_grad_(False)

        # Add a convolutional head for additional learning
        self.conv_head = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Bottleneck for feature compression
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        pil_images = [to_pil_image(img) for img in x]
        pixel_values = self.processor(images=pil_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(next(self.base_model.parameters()).device)

        # Pass through base model
        features = self.base_model(pixel_values).last_hidden_state.mean(dim=1)
        features = self.bottleneck(features)
        return self.classifier(features)

    def unfreeze_last_layers(self, num_layers: int):
        """
        Unfreeze the last `num_layers` layers in the base model.
        """
        for param in list(self.base_model.parameters())[-num_layers:]:
            param.requires_grad = True




"""2024-12-04 21:42:48.047176: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-12-04 21:42:48.067850: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-12-04 21:42:48.073910: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-12-04 21:42:48.088984: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-12-04 21:42:49.548683: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
config.json: 100% 549/549 [00:00<00:00, 4.55MB/s]
model.safetensors: 100% 1.22G/1.22G [00:08<00:00, 149MB/s]
preprocessor_config.json: 100% 436/436 [00:00<00:00, 2.96MB/s]
/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
/content/recvis24_a3/main.py:98: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler()
/content/recvis24_a3/main.py:36: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
Train Epoch: 1 [0/20000 (0%)]	Loss: 6.243225
Train Epoch: 1 [1280/20000 (6%)]	Loss: 6.194855
Train Epoch: 1 [2560/20000 (13%)]	Loss: 6.154709
Train Epoch: 1 [3840/20000 (19%)]	Loss: 6.146210
Train Epoch: 1 [5120/20000 (25%)]	Loss: 6.007462
Train Epoch: 1 [6400/20000 (32%)]	Loss: 5.770798
Train Epoch: 1 [7680/20000 (38%)]	Loss: 5.503616
Train Epoch: 1 [8960/20000 (45%)]	Loss: 5.268448
Train Epoch: 1 [10240/20000 (51%)]	Loss: 5.226624
Train Epoch: 1 [11520/20000 (57%)]	Loss: 5.144554
Train Epoch: 1 [12800/20000 (64%)]	Loss: 4.597930
Train Epoch: 1 [14080/20000 (70%)]	Loss: 4.675539
Train Epoch: 1 [15360/20000 (76%)]	Loss: 4.455877
Train Epoch: 1 [16640/20000 (83%)]	Loss: 4.132351
Train Epoch: 1 [17920/20000 (89%)]	Loss: 4.301312
Train Epoch: 1 [19200/20000 (96%)]	Loss: 4.013618

Train set: Accuracy: 7.09%


Validation set: Loss: 71.8018, Accuracy: 23.08%

Epoch 1/50
Train Loss: 5.1761, Train Accuracy: 7.09%
Validation Loss: 3.5901, Validation Accuracy: 23.08%
Saved new best model with accuracy: 23.08%
Train Epoch: 2 [0/20000 (0%)]	Loss: 3.864696
Train Epoch: 2 [1280/20000 (6%)]	Loss: 3.923305
Train Epoch: 2 [2560/20000 (13%)]	Loss: 3.658268
Train Epoch: 2 [3840/20000 (19%)]	Loss: 3.612826
Train Epoch: 2 [5120/20000 (25%)]	Loss: 3.707743
Train Epoch: 2 [6400/20000 (32%)]	Loss: 3.478880
Train Epoch: 2 [7680/20000 (38%)]	Loss: 3.351091
Train Epoch: 2 [8960/20000 (45%)]	Loss: 3.460060
Train Epoch: 2 [10240/20000 (51%)]	Loss: 3.250678
Train Epoch: 2 [11520/20000 (57%)]	Loss: 3.449864
Train Epoch: 2 [12800/20000 (64%)]	Loss: 3.485150
Train Epoch: 2 [14080/20000 (70%)]	Loss: 3.104901
Train Epoch: 2 [15360/20000 (76%)]	Loss: 3.271820
Train Epoch: 2 [16640/20000 (83%)]	Loss: 3.023738
Train Epoch: 2 [17920/20000 (89%)]	Loss: 3.132603
Train Epoch: 2 [19200/20000 (96%)]	Loss: 3.281939

Train set: Accuracy: 25.06%


Validation set: Loss: 56.1749, Accuracy: 36.72%

Epoch 2/50
Train Loss: 3.4216, Train Accuracy: 25.06%
Validation Loss: 2.8087, Validation Accuracy: 36.72%
Saved new best model with accuracy: 36.72%
Train Epoch: 3 [0/20000 (0%)]	Loss: 3.053749
Train Epoch: 3 [1280/20000 (6%)]	Loss: 2.940909
Train Epoch: 3 [2560/20000 (13%)]	Loss: 2.689650
Train Epoch: 3 [3840/20000 (19%)]	Loss: 2.739312
Train Epoch: 3 [5120/20000 (25%)]	Loss: 2.758316
Train Epoch: 3 [6400/20000 (32%)]	Loss: 2.762547
Train Epoch: 3 [7680/20000 (38%)]	Loss: 2.677431
Train Epoch: 3 [8960/20000 (45%)]	Loss: 3.124665
Train Epoch: 3 [10240/20000 (51%)]	Loss: 2.884636
Train Epoch: 3 [11520/20000 (57%)]	Loss: 2.873888
Train Epoch: 3 [12800/20000 (64%)]	Loss: 2.651575
Train Epoch: 3 [14080/20000 (70%)]	Loss: 2.894416
Train Epoch: 3 [15360/20000 (76%)]	Loss: 2.607211
Train Epoch: 3 [16640/20000 (83%)]	Loss: 2.851775
Train Epoch: 3 [17920/20000 (89%)]	Loss: 2.838894
Train Epoch: 3 [19200/20000 (96%)]	Loss: 2.935453

Train set: Accuracy: 35.10%


Validation set: Loss: 49.6797, Accuracy: 43.48%

Epoch 3/50
Train Loss: 2.8571, Train Accuracy: 35.10%
Validation Loss: 2.4840, Validation Accuracy: 43.48%
Saved new best model with accuracy: 43.48%
Train Epoch: 4 [0/20000 (0%)]	Loss: 2.880927
Train Epoch: 4 [1280/20000 (6%)]	Loss: 2.524331
Train Epoch: 4 [2560/20000 (13%)]	Loss: 2.607768
Train Epoch: 4 [3840/20000 (19%)]	Loss: 2.365879
Train Epoch: 4 [5120/20000 (25%)]	Loss: 2.347708
Train Epoch: 4 [6400/20000 (32%)]	Loss: 2.429231
Train Epoch: 4 [7680/20000 (38%)]	Loss: 2.540229
Train Epoch: 4 [8960/20000 (45%)]	Loss: 2.296226
Train Epoch: 4 [10240/20000 (51%)]	Loss: 2.455867
Train Epoch: 4 [11520/20000 (57%)]	Loss: 2.553393
Train Epoch: 4 [12800/20000 (64%)]	Loss: 2.953862
Train Epoch: 4 [14080/20000 (70%)]	Loss: 2.459110
Train Epoch: 4 [15360/20000 (76%)]	Loss: 2.476786
Train Epoch: 4 [16640/20000 (83%)]	Loss: 2.402116
Train Epoch: 4 [17920/20000 (89%)]	Loss: 2.257056
Train Epoch: 4 [19200/20000 (96%)]	Loss: 2.580213

Train set: Accuracy: 41.27%


Validation set: Loss: 45.6712, Accuracy: 47.00%

Epoch 4/50
Train Loss: 2.5320, Train Accuracy: 41.27%
Validation Loss: 2.2836, Validation Accuracy: 47.00%
Saved new best model with accuracy: 47.00%
Train Epoch: 5 [0/20000 (0%)]	Loss: 2.236325
Train Epoch: 5 [1280/20000 (6%)]	Loss: 2.428977
Train Epoch: 5 [2560/20000 (13%)]	Loss: 2.318335
Train Epoch: 5 [3840/20000 (19%)]	Loss: 2.398433
Train Epoch: 5 [5120/20000 (25%)]	Loss: 2.566135
Train Epoch: 5 [6400/20000 (32%)]	Loss: 1.918605
Train Epoch: 5 [7680/20000 (38%)]	Loss: 2.574050
Train Epoch: 5 [8960/20000 (45%)]	Loss: 2.402810
Train Epoch: 5 [10240/20000 (51%)]	Loss: 2.182798
Train Epoch: 5 [11520/20000 (57%)]	Loss: 2.363168
Train Epoch: 5 [12800/20000 (64%)]	Loss: 2.484988
Train Epoch: 5 [14080/20000 (70%)]	Loss: 2.540169
Train Epoch: 5 [15360/20000 (76%)]	Loss: 2.240129
Train Epoch: 5 [16640/20000 (83%)]	Loss: 2.549039
Train Epoch: 5 [17920/20000 (89%)]	Loss: 2.258169
Train Epoch: 5 [19200/20000 (96%)]	Loss: 2.344391

Train set: Accuracy: 45.65%


Validation set: Loss: 42.6596, Accuracy: 51.08%

Epoch 5/50
Train Loss: 2.3016, Train Accuracy: 45.65%
Validation Loss: 2.1330, Validation Accuracy: 51.08%
Saved new best model with accuracy: 51.08%
Train Epoch: 6 [0/20000 (0%)]	Loss: 2.222276
Train Epoch: 6 [1280/20000 (6%)]	Loss: 2.357488
Train Epoch: 6 [2560/20000 (13%)]	Loss: 1.925530
Train Epoch: 6 [3840/20000 (19%)]	Loss: 1.947985
Train Epoch: 6 [5120/20000 (25%)]	Loss: 2.027557
Train Epoch: 6 [6400/20000 (32%)]	Loss: 1.982605
Train Epoch: 6 [7680/20000 (38%)]	Loss: 2.122714
Train Epoch: 6 [8960/20000 (45%)]	Loss: 2.205601
Train Epoch: 6 [10240/20000 (51%)]	Loss: 2.071353
Train Epoch: 6 [11520/20000 (57%)]	Loss: 1.963651
Train Epoch: 6 [12800/20000 (64%)]	Loss: 2.174948
Train Epoch: 6 [14080/20000 (70%)]	Loss: 2.214070
Train Epoch: 6 [15360/20000 (76%)]	Loss: 2.078383
Train Epoch: 6 [16640/20000 (83%)]	Loss: 2.309936
Train Epoch: 6 [17920/20000 (89%)]	Loss: 2.029630
Train Epoch: 6 [19200/20000 (96%)]	Loss: 2.265667

Train set: Accuracy: 48.05%


Validation set: Loss: 40.3542, Accuracy: 52.04%

Epoch 6/50
Train Loss: 2.1557, Train Accuracy: 48.05%
Validation Loss: 2.0177, Validation Accuracy: 52.04%
Saved new best model with accuracy: 52.04%
Train Epoch: 7 [0/20000 (0%)]	Loss: 1.977690
Train Epoch: 7 [1280/20000 (6%)]	Loss: 1.940620
Train Epoch: 7 [2560/20000 (13%)]	Loss: 2.177101
Train Epoch: 7 [3840/20000 (19%)]	Loss: 1.887102
Train Epoch: 7 [5120/20000 (25%)]	Loss: 1.990625
Train Epoch: 7 [6400/20000 (32%)]	Loss: 1.894410
Train Epoch: 7 [7680/20000 (38%)]	Loss: 2.095971
Train Epoch: 7 [8960/20000 (45%)]	Loss: 2.285549
Train Epoch: 7 [10240/20000 (51%)]	Loss: 2.013550
Train Epoch: 7 [11520/20000 (57%)]	Loss: 1.943586
Train Epoch: 7 [12800/20000 (64%)]	Loss: 2.289412
Train Epoch: 7 [14080/20000 (70%)]	Loss: 2.395793
Train Epoch: 7 [15360/20000 (76%)]	Loss: 1.785366
Train Epoch: 7 [16640/20000 (83%)]	Loss: 1.910154
Train Epoch: 7 [17920/20000 (89%)]	Loss: 1.944835
Train Epoch: 7 [19200/20000 (96%)]	Loss: 1.863765

Train set: Accuracy: 51.58%


Validation set: Loss: 38.7738, Accuracy: 54.12%

Epoch 7/50
Train Loss: 1.9995, Train Accuracy: 51.58%
Validation Loss: 1.9387, Validation Accuracy: 54.12%
Saved new best model with accuracy: 54.12%
Train Epoch: 8 [0/20000 (0%)]	Loss: 2.223836
Train Epoch: 8 [1280/20000 (6%)]	Loss: 1.596057
Train Epoch: 8 [2560/20000 (13%)]	Loss: 2.197597
Train Epoch: 8 [3840/20000 (19%)]	Loss: 1.720091
Train Epoch: 8 [5120/20000 (25%)]	Loss: 1.764351
Train Epoch: 8 [6400/20000 (32%)]	Loss: 1.952002
Train Epoch: 8 [7680/20000 (38%)]	Loss: 1.929415
Train Epoch: 8 [8960/20000 (45%)]	Loss: 1.714742
Train Epoch: 8 [10240/20000 (51%)]	Loss: 1.942881
Train Epoch: 8 [11520/20000 (57%)]	Loss: 1.736067
Train Epoch: 8 [12800/20000 (64%)]	Loss: 1.662598
Train Epoch: 8 [14080/20000 (70%)]	Loss: 1.760387
Train Epoch: 8 [15360/20000 (76%)]	Loss: 1.816612
Train Epoch: 8 [16640/20000 (83%)]	Loss: 1.792652
Train Epoch: 8 [17920/20000 (89%)]	Loss: 2.029210
Train Epoch: 8 [19200/20000 (96%)]	Loss: 1.668720

Train set: Accuracy: 54.22%


Validation set: Loss: 37.7352, Accuracy: 54.96%

Epoch 8/50
Train Loss: 1.8936, Train Accuracy: 54.22%
Validation Loss: 1.8868, Validation Accuracy: 54.96%
Saved new best model with accuracy: 54.96%
Train Epoch: 9 [0/20000 (0%)]	Loss: 1.807729
Train Epoch: 9 [1280/20000 (6%)]	Loss: 1.658109
Train Epoch: 9 [2560/20000 (13%)]	Loss: 1.779597
Train Epoch: 9 [3840/20000 (19%)]	Loss: 2.111397
Train Epoch: 9 [5120/20000 (25%)]	Loss: 1.563000
Train Epoch: 9 [6400/20000 (32%)]	Loss: 1.635603
Train Epoch: 9 [7680/20000 (38%)]	Loss: 2.277086
Train Epoch: 9 [8960/20000 (45%)]	Loss: 1.830872
Train Epoch: 9 [10240/20000 (51%)]	Loss: 2.019650
Train Epoch: 9 [11520/20000 (57%)]	Loss: 1.786906
Train Epoch: 9 [12800/20000 (64%)]	Loss: 1.844442
Train Epoch: 9 [14080/20000 (70%)]	Loss: 1.696721
Train Epoch: 9 [15360/20000 (76%)]	Loss: 1.867225
Train Epoch: 9 [16640/20000 (83%)]	Loss: 1.893130
Train Epoch: 9 [17920/20000 (89%)]	Loss: 1.907324
Train Epoch: 9 [19200/20000 (96%)]	Loss: 1.892855

Train set: Accuracy: 55.34%


Validation set: Loss: 37.0182, Accuracy: 56.04%

Epoch 9/50
Train Loss: 1.8281, Train Accuracy: 55.34%
Validation Loss: 1.8509, Validation Accuracy: 56.04%
Saved new best model with accuracy: 56.04%
Train Epoch: 10 [0/20000 (0%)]	Loss: 1.492394
Train Epoch: 10 [1280/20000 (6%)]	Loss: 2.054547
Train Epoch: 10 [2560/20000 (13%)]	Loss: 1.616238
Train Epoch: 10 [3840/20000 (19%)]	Loss: 2.158579
Train Epoch: 10 [5120/20000 (25%)]	Loss: 1.616591
Train Epoch: 10 [6400/20000 (32%)]	Loss: 2.095662
Train Epoch: 10 [7680/20000 (38%)]	Loss: 2.026660
Train Epoch: 10 [8960/20000 (45%)]	Loss: 1.248604
Train Epoch: 10 [10240/20000 (51%)]	Loss: 1.914369
Train Epoch: 10 [11520/20000 (57%)]	Loss: 1.552148
Train Epoch: 10 [12800/20000 (64%)]	Loss: 1.754579
Train Epoch: 10 [14080/20000 (70%)]	Loss: 1.865479
Train Epoch: 10 [15360/20000 (76%)]	Loss: 1.937565
Train Epoch: 10 [16640/20000 (83%)]	Loss: 1.969214
Train Epoch: 10 [17920/20000 (89%)]	Loss: 1.801228
Train Epoch: 10 [19200/20000 (96%)]	Loss: 1.794165

Train set: Accuracy: 55.87%


Validation set: Loss: 36.8270, Accuracy: 55.96%

Epoch 10/50
Train Loss: 1.8109, Train Accuracy: 55.87%
Validation Loss: 1.8414, Validation Accuracy: 55.96%
Train Epoch: 11 [0/20000 (0%)]	Loss: 1.782272
Train Epoch: 11 [1280/20000 (6%)]	Loss: 2.119839
Train Epoch: 11 [2560/20000 (13%)]	Loss: 2.017746
Train Epoch: 11 [3840/20000 (19%)]	Loss: 1.968539
Train Epoch: 11 [5120/20000 (25%)]	Loss: 2.085973
Train Epoch: 11 [6400/20000 (32%)]	Loss: 2.265835
Train Epoch: 11 [7680/20000 (38%)]	Loss: 2.198032
Train Epoch: 11 [8960/20000 (45%)]	Loss: 2.118560
Train Epoch: 11 [10240/20000 (51%)]	Loss: 1.971774
Train Epoch: 11 [11520/20000 (57%)]	Loss: 2.067227
Train Epoch: 11 [12800/20000 (64%)]	Loss: 2.331684
Train Epoch: 11 [14080/20000 (70%)]	Loss: 1.972342
Train Epoch: 11 [15360/20000 (76%)]	Loss: 2.514864
Train Epoch: 11 [16640/20000 (83%)]	Loss: 2.000781
Train Epoch: 11 [17920/20000 (89%)]	Loss: 2.435524
Train Epoch: 11 [19200/20000 (96%)]	Loss: 2.008354

Train set: Accuracy: 50.19%


Validation set: Loss: 39.4872, Accuracy: 53.36%

Epoch 11/50
Train Loss: 2.0421, Train Accuracy: 50.19%
Validation Loss: 1.9744, Validation Accuracy: 53.36%
Train Epoch: 12 [0/20000 (0%)]	Loss: 1.893888
Train Epoch: 12 [1280/20000 (6%)]	Loss: 2.297589
Train Epoch: 12 [2560/20000 (13%)]	Loss: 2.042397
Train Epoch: 12 [3840/20000 (19%)]	Loss: 2.092692
Train Epoch: 12 [5120/20000 (25%)]	Loss: 2.239656
Train Epoch: 12 [6400/20000 (32%)]	Loss: 1.827992
Train Epoch: 12 [7680/20000 (38%)]	Loss: 1.929027
Train Epoch: 12 [8960/20000 (45%)]	Loss: 1.825140
Train Epoch: 12 [10240/20000 (51%)]	Loss: 1.920640
Train Epoch: 12 [11520/20000 (57%)]	Loss: 1.845791
Train Epoch: 12 [12800/20000 (64%)]	Loss: 1.894294
Train Epoch: 12 [14080/20000 (70%)]	Loss: 2.168600
Train Epoch: 12 [15360/20000 (76%)]	Loss: 2.154316
Train Epoch: 12 [16640/20000 (83%)]	Loss: 2.052050
Train Epoch: 12 [17920/20000 (89%)]	Loss: 2.219502
Train Epoch: 12 [19200/20000 (96%)]	Loss: 1.948979

Train set: Accuracy: 52.00%


Validation set: Loss: 38.9516, Accuracy: 53.16%

Epoch 12/50
Train Loss: 1.9849, Train Accuracy: 52.00%
Validation Loss: 1.9476, Validation Accuracy: 53.16%
Train Epoch: 13 [0/20000 (0%)]	Loss: 1.847250
Train Epoch: 13 [1280/20000 (6%)]	Loss: 2.119093
Train Epoch: 13 [2560/20000 (13%)]	Loss: 1.975345
Train Epoch: 13 [3840/20000 (19%)]	Loss: 2.129539
Train Epoch: 13 [5120/20000 (25%)]	Loss: 1.905967
Train Epoch: 13 [6400/20000 (32%)]	Loss: 1.558351
Train Epoch: 13 [7680/20000 (38%)]	Loss: 1.887854
Train Epoch: 13 [8960/20000 (45%)]	Loss: 1.808656
Train Epoch: 13 [10240/20000 (51%)]	Loss: 1.874394
Train Epoch: 13 [11520/20000 (57%)]	Loss: 1.973505
Train Epoch: 13 [12800/20000 (64%)]	Loss: 1.719415
Train Epoch: 13 [14080/20000 (70%)]	Loss: 1.819938
Train Epoch: 13 [15360/20000 (76%)]	Loss: 1.927760
Train Epoch: 13 [16640/20000 (83%)]	Loss: 2.471841
Train Epoch: 13 [17920/20000 (89%)]	Loss: 1.897143
Train Epoch: 13 [19200/20000 (96%)]	Loss: 2.038897

Train set: Accuracy: 52.94%


Validation set: Loss: 37.3983, Accuracy: 55.20%

Epoch 13/50
Train Loss: 1.9215, Train Accuracy: 52.94%
Validation Loss: 1.8699, Validation Accuracy: 55.20%
Train Epoch: 14 [0/20000 (0%)]	Loss: 1.665704"""