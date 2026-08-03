## Project Description

This is a demo of my object detection project. I mainly referred to the [DINO](https://arxiv.org/abs/2203.03605) and [SAM-DETR](https://arxiv.org/abs/2203.06883) paper. The code implementation is also modified based on the official [DINO code](https://github.com/IDEA-Research/DINO).

In brief, I insert a semantic alignment module into DINO to evaluate how much the detection accuracy can be improved. DINO with semantic alignment module is named as SAM-DINO in this project. In addition, I designed a temporal module to experiment with processing multi-frame information using DINO.

The training and testing data come from [HA-VID](https://iai-hrc.github.io/ha-vid), an assembly dataset for industrial scenarios.

## Environment

It only needs to support normal inference for DINO and SAM-DETR.

## Command

### Training

(1) Train SAM-DINO.
```
sh DINO_train_swin.sh
```

(2) Train temporal module. 
```
python small_model.py
```

Before training, ensure that lines 553 to 555 in the small_model.py script are consistent with the following code.
```
if __name__=="__main__":
    #evaluate()
    main()
```

Cause the temporal module is designed based on SAM-DINO. Only after the SAM-DINO has been trained can you train the temporal module.

### Testing

(1) Test SAM-DINO.
```
sh DINO_eval.sh
```

(2) Test temporal module. 
```
python small_model.py
```

Before testing, ensure that lines 553 to 555 in the small_model.py script are consistent with the following code.
```
if __name__=="__main__":
    evaluate()
    #main()
```

### Inferencing image

(1) SAM-DINO.
```
python showing-sam-dino.py -c config/DINO/DINO_4scale_swin.py
```

(2) SAM-DINO with temporal module.
```
python showing-sam-dino-time.py -c config/DINO/DINO_4scale_swin.py
```

## Demo of SAM-DINO

| ... | scene 1 | scene 2 |
|---------|---------|---------|
| without temporal module | 文字2<br><video src="./video_sam_dino_0.mp4" ...></video> | ... |
| with temporal module | ... | ... |

