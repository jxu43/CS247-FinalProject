# Image Captioning (Let Images Speak For Themselves)

Image Captioning is a task that given an image, it generates an sentense to describe it. 

Our implementation is based on the is inspired from "Show and Tell" [1] by Vinyals et al.

## Dataset
We use Flickr8k [2] as dataset. 

## Model 
<div align="center">
  <img src="Model.png"><br><br>
</div>

## Train
We trained 70 epoches.

## Result
Average BLEU-4 score is 0.22
Average BLEU-1 score is 0.53

## Usage

### From Scratch
1. `python encode_image.py`
2. `python train.py`
3. `python test.py`

### Load Pretrained Weights
Download pre-trained weights from [releases](https://github.com/jxu43/CS247-FinalProject/releases)
and Move it into Output/ folder

## References
#### NIC Model
[1] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
#### Data
[2] https://illinois.edu/fb/sec/1713398
#### VGG16 Model
https://github.com/fchollet/deep-learning-models
#### Saved Model
https://drive.google.com/drive/folders/1aukgi_3xtuRkcQGoyAaya5pP4aoDzl7r
#### Code reference
https://github.com/anuragmishracse/caption_generator

You can find a detailed report in the Report folder.
