## Project Description

This is a demo of my object detection project. I mainly referred to the [DINO](https://arxiv.org/abs/2203.03605) and [SAM-DETR](https://arxiv.org/abs/2203.06883) paper. The code implementation is also modified based on the official [DINO code](https://github.com/IDEA-Research/DINO).

In brief, I insert a semantic alignment module into DINO to evaluate how much the detection accuracy can be improved. DINO with semantic alignment module is named as SAM-DINO in this project. In addition, I designed a temporal module to experiment with processing multi-frame information using SAM-DINO.

The training and testing data come from [HA-VID](https://iai-hrc.github.io/ha-vid), an assembly dataset for industrial scenarios.

The [checkpoints](https://huggingface.co/tl121212/sam-dino) of SAM-DINO and temporal module have been uploaded to Hugging Face. Checkpoint_best_regular.pth is SAM-DINO's checkpoint, while temporal_module.pth is the checkpoint of temporal module.

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

Only after the SAM-DINO has been trained can you train the temporal module.  Don't forget to change the SAM-DINO's checkpoint at line 373 in small_model.py.

### Testing

(1) Test SAM-DINO. Change the SAM-DINO's checkpoint in DINO_eval.sh.
```
sh DINO_eval.sh
```

(2) Test temporal module. Change the checkpoints of SAM-DINO and temporal module from line 438 to 439 in small_model.py.
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

(1) SAM-DINO. Change the SAM-DINO's checkpoint at line 41 in showing-sam-dino.py.
```
python showing-sam-dino.py -c config/DINO/DINO_4scale_swin.py
```

(2) SAM-DINO with temporal module. Change the checkpoints of SAM-DINO and temporal module from line 154 to 155 in showing-sam-dino-time.py.
```
python showing-sam-dino-time.py -c config/DINO/DINO_4scale_swin.py
```

## Demo of SAM-DINO

The following scenes are from the HA‑VID validation set. Click the image to view the video.

|  | scene 1 | scene 2 |
|---------|---------|---------|
| without temporal module | [![](frame_000000.PNG)](./video_sam_dino_0.mp4) | [![](frame_017800.PNG)](./video_sam_dino_17800.mp4) |
| with temporal module | [![](frame_000000_time.PNG)](./video_sam_dino_time_0.mp4) | [![](frame_017800_time.PNG)](./video_sam_dino_time_17800.mp4) |

