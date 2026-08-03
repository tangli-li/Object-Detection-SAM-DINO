## Project Description

This is a demo of my object detection project. I referred to the [DINO] (https://arxiv.org/abs/2203.03605) and [SAM-DETR] (https://arxiv.org/abs/2203.06883) models. The code implementation is also modified based on the official [DINO code] (https://github.com/IDEA-Research/DINO).

In brief, I insert a semantic alignment module into DINO to evaluate how much the detection accuracy can be improved. In addition, I designed a temporal module to experiment with processing multi-frame information using DINO.

The training and testing data come from [HA-VID] (https://iai-hrc.github.io/ha-vid), an assembly dataset for industrial scenarios.
