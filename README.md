# RetinaDetection Object Detector

## Introduction

RetinaDetector是基于RetinaFace修改的检测方法，原论文is a practical single-stage [SOTA](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) face detector which is initially described in [arXiv technical report](https://arxiv.org/abs/1905.00641)

## Data

0. Organise the dataset directory as follows:

```Shell
  data/retinaface/
    train/
      images/
      label.txt
    val/
      images/
      label.txt
    test/
      images/
      label.txt
```

## Install

1. Install MXNet with GPU support.
2. Install Deformable Convolution V2 operator from [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets) if you use the DCN based backbone.
3. Type ``make`` to build cxx tools.

## Training

Please check ``train.py`` for training.

1. Copy ``rcnn/sample_config.py`` to ``rcnn/config.py``

为了获得更好的训练效果，可针对性的修改一些参数，如下：

```Shell
config.TRAIN.MIN_BOX_SIZE = 10 #最小bbox
config.FACE_LANDMARK = False #使用landmark
config.USE_BLUR = False
config.BBOX_MASK_THRESH = 0
config.COLOR_MODE = 2 #增强
config.COLOR_JITTERING = 0.125
```

无效人脸的过滤，如下：
```Shell
if (x2 - x1) < config.TRAIN.MIN_BOX_SIZE or (y2 - y1) < config.TRAIN.MIN_BOX_SIZE:
   continue
if self._split.startswith('train'):
   blur[ix] = values[19]
   if blur[ix] < 0.25:
      continue
if config.BBOX_MASK_THRESH > 0:
   if (x2 - x1) < config.BBOX_MASK_THRESH or (y2 - y1) < config.BBOX_MASK_THRESH:
      boxes_mask.append(np.array([x1, y1, x2, y2], np.float))
      continue
   if self._split.startswith('train'):
      if blur[ix] < 0.35:
         boxes_mask.append(np.array([x1, y1, x2, y2], np.float))
         continue
```

2. Download pretrained models and put them into ``model/``. 

    ImageNet ResNet50 ([baidu cloud](https://pan.baidu.com/s/1WAkU9ZA_j-OmzO-sdk9whA) and [dropbox](https://www.dropbox.com/s/48b850vmnaaasfl/imagenet-resnet-50.zip?dl=0)). 

    ImageNet ResNet152 ([baidu cloud](https://pan.baidu.com/s/1nzQ6CzmdKFzg8bM8ChZFQg) and [dropbox](https://www.dropbox.com/s/8ypcra4nqvm32v6/imagenet-resnet-152.zip?dl=0)).

3. Start training with ``sh train_model.sh``.  
Before training, you can check the ``resnet`` network configuration (e.g. pretrained model path, anchor setting and learning rate policy etc..) in ``rcnn/config.py``.

## Testing

Please check ``test.py`` for testing.

## Result

1. 缺陷检测

![MASK1](https://github.com/bleakie/RetinaDetection/blob/master/images/00001673.jpg)

2. 人脸检测+人脸对齐

![MASK1](https://github.com/bleakie/RetinaDetection/blob/master/images/0000.png)

## Models

人脸检测模型，比原版误检更低，角度较大和模糊超过0.6的face会自动忽略，更适合人脸识别的应用：click [here](http://www.multcloud.com/share/5079e926-283b-4833-a216-b3de42eea0fe).

## ToDo

由于缺陷检测数据涉及私密性，缺陷检测的模型暂时不会释放

## References

```
@inproceedings{yangsai1991@163.com,
year={2019}
}
```


