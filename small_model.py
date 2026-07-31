import torch 
import torch.nn as nn 
import argparse
from main import build_model_main, get_args_parser_formini
from models.dino.utils import MLP
from models.dino.dino import DINO, SetCriterion
from models.dino.deformable_transformer import DeformableTransformer
from models.dino.backbone import Joiner
from util.slconfig import SLConfig
from util.misc import (NestedTensor, inverse_sigmoid,get_world_size)
from torch.utils.data import DataLoader
from datasets import build_dataset,get_coco_api_from_dataset
import util.misc as utils
from tqdm import tqdm
from pathlib import Path
import os
import copy
from engine import evaluate_withmini
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class minimodel(nn.Module):
    def __init__(self,num_classes,hidden_dim,dec_num,d_model=256,dropout=0.1,head=8,curr_rate=0.7,post_frames=3,batch=1,num_queries=900):
        super().__init__()
        self.dec_num=dec_num
        self.curr_rate=curr_rate
        self.batch=batch
        self.post_frames=post_frames
        self.d_model=d_model
        self.num_queries=num_queries
        _class_embed = nn.Linear(hidden_dim, num_classes)
        self.init_weights(_class_embed)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.init_weights(_bbox_embed)
        _hs_atten =  nn.MultiheadAttention(d_model,head,dropout=dropout,batch_first=True)
        self.init_weights(_hs_atten)
        _ref_atten =nn.MultiheadAttention(4,2,dropout=dropout,batch_first=True)
        self.init_weights(_ref_atten)
        _hs_K = nn.Linear(d_model, d_model)#转化K
        self.init_weights(_hs_K)
        _ref_K = nn.Linear(4,4)#K的特征映射一般不用激活函数
        self.init_weights(_ref_K)
        #_hs_down = nn.Conv1d(num_queries*post_frames,num_queries,kernel_size=3,padding=1)#应该在词条L上做，不该在batch上做
        _hs_down = nn.Sequential(nn.Linear(num_queries*post_frames,num_queries),nn.SiLU(),nn.LayerNorm(num_queries))
        self.init_weights(_hs_down)
        #_ref_down = nn.Conv1d(num_queries*post_frames,num_queries,kernel_size=1)#使用卷积降低维度，不用线性层（线性层相当于对每个特征进行相同处理）
        _ref_down = nn.Sequential(nn.Linear(num_queries*post_frames,num_queries),nn.SiLU(),nn.LayerNorm(num_queries))
        self.init_weights(_ref_down)
        _hs_ffn = nn.Sequential(nn.Linear(num_queries,num_queries),nn.SiLU(),nn.LayerNorm(num_queries))
        self.init_weights(_hs_ffn[1])
        _ref_ffn = nn.Sequential(nn.Linear(num_queries,num_queries),nn.SiLU(),nn.LayerNorm(num_queries))
        self.init_weights(_ref_ffn[1])
        norm1 = nn.LayerNorm(d_model)
        dropout = nn.Dropout(dropout)

        self.hs_atten=self.copy_layers(_hs_atten,num=self.dec_num)
        self.ref_atten=self.copy_layers(_ref_atten,num=self.dec_num)
        self.hs_K=self.copy_layers(_hs_K,num=self.dec_num)
        self.ref_K=self.copy_layers(_ref_K,num=self.dec_num)
        self.hs_down=self.copy_layers(_hs_down,num=self.dec_num)
        self.ref_down=self.copy_layers(_ref_down,num=self.dec_num)
        self.hs_ffn=self.copy_layers(_hs_ffn,num=self.dec_num)
        self.ref_ffn=self.copy_layers(_ref_ffn,num=self.dec_num)
        self.bbox_embed = self.copy_layers(_bbox_embed,num=self.dec_num)
        self.class_embed = self.copy_layers(_class_embed,num=self.dec_num)
        self.norm1 = self.copy_layers(norm1,num=self.dec_num)
        self.dropout = self.copy_layers(dropout,num=self.dec_num)

        #self.Linear_1 = nn.Sequential(nn.Linear(4,self.d_model),nn.ReLU())#一般后面有norm就可以不激活
        #self.Linear_2 = nn.Sequential(nn.Linear(self.d_model,4),nn.ReLU())

    def init_weights(self,module):
        #print(type(module))
        if isinstance(module, nn.MultiheadAttention):
        # 初始化 in_proj_weight 和 out_proj
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias,0)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.constant_(module.out_proj.bias,0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias,0)
        elif isinstance(module,MLP):
            for layer1 in module.layers:
                nn.init.xavier_uniform_(layer1.weight)
                nn.init.constant_(layer1.bias,0)
            
    def copy_layers(self,module,num):
        list0=[module for i in range(num)]
        return nn.ModuleList(list0)

    def process(self,reference,hs):
        # print(reference[0].shape)
        # print(hs[0].shape)
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)        

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        # print("outputs_class")
        # print(outputs_class.shape)
        # print("outputs_coord_list")
        # print(outputs_coord_list.shape)
        return outputs_coord_list, outputs_class
    
    def forward_atten(self,reference,hs,reference_post,hs_post,copy_time):
        ref_new=[]
        hs_new=[]
        for i in range(self.dec_num):
            #print(reference[i].shape)
            curr_input=inverse_sigmoid(reference[i])
            post_input=inverse_sigmoid(reference_post[i])
            reference_curr=curr_input.repeat(1,copy_time,1)
            hs_curr=hs[i].repeat(1,copy_time,1)
            ref_out=self.ref_atten[i](reference_curr,self.ref_K[i](post_input),post_input)[0]
            hs_out=self.hs_atten[i](hs_curr,self.hs_K[i](hs_post[i]),hs_post[i])[0]
            # print(hs_out.permute(0,2,1).shape)
            ref_before_add=self.ref_ffn[i](self.ref_down[i](ref_out.permute(0,2,1)))
            hs_before_add=self.hs_ffn[i](self.hs_down[i](hs_out.permute(0,2,1)))
            out_ref=curr_input+self.dropout[i](ref_before_add.permute(0,2,1))
            out_hs=hs[i]+self.dropout[i](hs_before_add.permute(0,2,1))

            ref_new.append(out_ref.sigmoid())
            hs_new.append(self.norm1[i](out_hs))
        return ref_new, hs_new

    def forward(self,reference0,hs0):#ref:4,hs:256
        # print(reference0[0][0].shape)
        # print(hs0[0][0].shape)
        reference_pos_frames=[]
        hs_pos_frames=[]
        #reference0中最后一帧为当前帧
        for idx, (reference,hs) in enumerate(zip(reference0,hs0)):
            #print(reference[0].shape)#[1, 900, 256]
            assert self.batch==hs[0].shape[0]
            if idx!=len(reference0)-1:
                #用框辅助物体识别，使得hs获得框信息，更专注物体
                # hs_pos_frames.append([self.Linear_1(reference[i].detach())+hs[i].detach() for i in range(len(hs))])
                hs_pos_frames.append([hs[i].detach() for i in range(len(hs))])
                #框之间直接分析注意力，希望使得前后帧的框能更加统一
                reference_pos_frames.append([reference[i].detach() for i in range(len(reference))])
                
        hs_post=[]
        reference_post=[]
        for i in range(self.dec_num):#过去的帧stack到一起
            hs_post_dec=torch.cat([hs_pos_frames[j][i] for j in range(self.post_frames)],dim=1)
            reference_post_dec=torch.cat([reference_pos_frames[j][i] for j in range(self.post_frames)],dim=1)
            hs_post.append(hs_post_dec)
            reference_post.append(reference_post_dec)
        
        reference_after_atten,hs_after_atten=self.forward_atten(reference,hs,reference_post,hs_post,copy_time=self.post_frames)

        outputs_coord_list,outputs_class =self.process(reference_after_atten,hs_after_atten)
        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        return out

class minimodel_LSTM(nn.Module):
    def __init__(self,num_classes,hidden_dim,dec_num,d_model=256,dropout=0.1,head=8,curr_rate=0.7,post_frames=3,batch=1,num_queries=900):
        super().__init__()
        self.dec_num=dec_num
        self.curr_rate=curr_rate
        self.batch=batch
        self.post_frames=post_frames
        self.d_model=d_model
        self.num_queries=num_queries
        _class_embed = nn.Linear(hidden_dim, num_classes)
        self.init_weights(_class_embed)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.init_weights(_bbox_embed)
        _hs_atten = nn.MultiheadAttention(d_model,head,dropout=dropout,batch_first=True)
        self.init_weights(_hs_atten)
        _ref_atten = nn.MultiheadAttention(4,2,dropout=dropout,batch_first=True)
        self.init_weights(_ref_atten)
        _hs_K = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU())#转化K
        self.init_weights(_hs_K[0])
        _ref_K = nn.Sequential(nn.Linear(4,4),nn.ReLU())
        self.init_weights(_ref_K[0])
        _hs_down = nn.Linear(num_queries*post_frames,num_queries)#应该在词条L上做，不该在batch上做
        self.init_weights(_hs_down)
        _ref_down = nn.Linear(num_queries*post_frames,num_queries)
        self.init_weights(_ref_down)
        _hs_lstm= nn.LSTM(input_size=num_queries*d_model,hidden_size=num_queries*d_model)

        _ref_lstm= nn.LSTM(input_size=num_queries*4,hidden_size=num_queries*4)
        norm1 = nn.LayerNorm(d_model)
        dropout = nn.Dropout(dropout)

        self.hs_atten=self.copy_layers(_hs_atten,num=self.dec_num)
        self.ref_atten=self.copy_layers(_ref_atten,num=self.dec_num)
        self.hs_K=self.copy_layers(_hs_K,num=self.dec_num)
        self.ref_K=self.copy_layers(_ref_K,num=self.dec_num)
        self.hs_down=self.copy_layers(_hs_down,num=self.dec_num)
        self.ref_down=self.copy_layers(_ref_down,num=self.dec_num)
        self.bbox_embed = self.copy_layers(_bbox_embed,num=self.dec_num)
        self.class_embed = self.copy_layers(_class_embed,num=self.dec_num)
        self.norm1 = self.copy_layers(norm1,num=self.dec_num)
        self.dropout = self.copy_layers(dropout,num=self.dec_num)
        self.hs_lstm=self.copy_layers(_hs_lstm,num=self.dec_num)
        self.ref_lstm=self.copy_layers(_ref_lstm,num=self.dec_num)

        self.Linear_1 = nn.Sequential(nn.Linear(4,self.d_model),nn.ReLU())#一般后面有norm就可以不激活
        #self.Linear_2 = nn.Sequential(nn.Linear(self.d_model,4),nn.ReLU())

    def init_weights(minimodel,module):
        #print(type(module))
        if isinstance(module, nn.MultiheadAttention):
        # 初始化 in_proj_weight 和 out_proj
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.constant_(module.in_proj_bias,0)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.constant_(module.out_proj.bias,0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias,0)
        elif isinstance(module,MLP):
            for layer1 in module.layers:
                nn.init.xavier_uniform_(layer1.weight)
                nn.init.constant_(layer1.bias,0)
            
    def copy_layers(self,module,num):
        list0=[module for i in range(num)]
        return nn.ModuleList(list0)

    def process(self,reference,hs):
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)        

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        # print("outputs_class")
        # print(outputs_class.shape)
        # print("outputs_coord_list")
        # print(outputs_coord_list.shape)
        return outputs_coord_list, outputs_class
    
    def forward_atten(self,reference,hs,reference_post,hs_post,copy_time):
        ref_new=[]
        hs_new=[]
        for i in range(self.dec_num):
            reference_post_input=self.ref_lstm[i](reference_post[i])[-1,:,:]
            hs_post_input=self.hs_lstm[i](hs_post[i])[-1,:,:]
            reference0_input=torch.unflatten(reference_post_input,dim=1,size=(self.num_queries,4))
            hs0_input=torch.unflatten(hs_post_input,dim=1,size=(self.num_queries,self.d_model))

            ref_out=self.ref_atten[i](reference_curr,self.ref_K[i](reference0_input),reference0_input)[0]
            hs_out=self.hs_atten[i](hs_curr,self.hs_K[i](hs0_input),hs0_input)[0]
            #print(ref_out.permute(0,2,1).shape)
            ref_before_add=self.ref_down[i](ref_out.permute(0,2,1)).permute(0,2,1)
            hs_before_add=self.hs_down[i](hs_out.permute(0,2,1)).permute(0,2,1)

            out_ref=reference[i]+self.dropout[i](ref_before_add)
            out_hs=hs[i]+self.dropout[i](hs_before_add)

            ref_new.append(out_ref.sigmoid())
            hs_new.append(self.norm1[i](out_hs))
        return ref_new, hs_new

    def forward(self,reference0,hs0):#reference的embedding为4，hs0的embedding为256
        reference_pos_frames=[]
        hs_pos_frames=[]
        #reference0中最后一帧为当前帧
        for idx, (reference,hs) in enumerate(zip(reference0,hs0)):
            #print(reference[0].shape)#[1, 900, 256]
            assert self.batch==hs[0].shape[0]
            if idx!=len(reference0)-1:
                #用框辅助物体识别，使得hs获得框信息，更专注物体
                hs_pos_frames.append([self.Linear_1(reference[i].detach())+hs[i].detach() for i in range(len(hs))])
                #框之间直接分析注意力，希望使得前后帧的框能更加统一
                reference_pos_frames.append([reference[i].detach() for i in range(len(hs))])
                
        hs_post=[]
        reference_post=[]
        for i in range(self.dec_num):#过去的帧stack到一起
            #post_frames, batch, 900*4
            hs_post_dec=torch.stack([torch.flatten(hs_pos_frames[j][i],start_dim=1) for j in range(self.post_frames)],dim=0)
            reference_post_dec=torch.stack([torch.flatten(reference_pos_frames[j][i],start_dim=1) for j in range(self.post_frames)],dim=1)
            hs_post.append(hs_post_dec)
            reference_post.append(reference_post_dec)
        
        reference_after_atten,hs_after_atten=self.forward_atten(reference,hs,reference_post,hs_post,copy_time=self.post_frames)

        outputs_coord_list,outputs_class =self.process(reference_after_atten,hs_after_atten)
        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        return out

def calculate_loss(outputs, target,criterion,device="cuda"):
    for item in target:#转移设备
        for k,v in item.items():
            if isinstance(item[k],torch.Tensor):
                item[k]=item[k].to(device)
    losses={}
    losses['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
    losses['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
    losses['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
    losses['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
    losses['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
    losses['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')

    losses_map=['labels','cardinality','boxes']
    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
    # print(outputs_without_aux.keys())
    # print(target)
    indices = criterion.matcher(outputs_without_aux, target)
    num_boxes = sum(len(t["labels"]) for t in target)
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
    num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
    for loss in losses_map:
        losses.update(criterion.get_loss_minimodel(loss, outputs, target, indices, num_boxes))

    loss_dict=losses
    weight_dict = criterion.weight_dict
    return sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

def train_one_epoch_mini(model_DINO,model_post,criterion,optimizer,data_loader,device):
    for param in model_DINO.parameters():#只会传播model_post，model_DINO不会受影响
        param.requires_grad = False 
    for sample_list, target in tqdm(data_loader):
        #target.to(device)
        reference_frames=[]
        hs_frames=[]
        for sample0 in sample_list:
            with torch.no_grad():
                sample0=sample0.to(device)
                hs,reference=model_DINO(sample0,targets=None,return_ref=True)
            reference_frames.append(reference)
            hs_frames.append(hs)
        #print(reference_frames[0][0].shape)
        outputs=model_post(reference_frames,hs_frames)#hs_frames,reference_frames与函数默认的参数名设置反了，这里需要倒过来
        loss=calculate_loss(outputs, target,criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

def generate_args(config):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser_formini()])
    args = parser.parse_args()
    args.config_file=config

    cfg = SLConfig.fromfile(config)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    return args

def main(device="cuda",epoch=10,post_frames=3):
    pretrain_dino_model="/media/mldadmin/home/s125mdg35_05/DINO/DINO/logs/DINO/R50-MS4/checkpoint_best_regular.pth"
    pretrain_mini_model=None
    config_file="/media/mldadmin/home/s125mdg35_05/DINO/DINO/config/DINO/DINO_4scale_swin.py"
    coco_path="/media/mldadmin/home/s125mdg35_05/dataset/data"
    backbone_dir="/media/mldadmin/home/s125mdg35_05/DINO/DINO/backbone"
    output_dir='/media/mldadmin/home/s125mdg35_05/DINO/DINO/output_mini_702'
    #更改coco数据地址在main.py中更改
    args=generate_args(config_file)
    args.backbone_dir=backbone_dir
    args.coco_path=coco_path
    args.output_dir=output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.dn_scalar=100
    args.embed_init_tgt=True
    args.dn_label_coef=1.0
    args.dn_bbox_coef=1.0
    args.use_ema=False
    args.dn_box_noise_scale=1.0
    args.dn_number=0

    dataset_train = build_dataset(image_set='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    print(f"训练batchsize:{args.batch_size}")
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=utils.collate_fn_mini)

    model_DINO, criterion, postprocessors = build_model_main(args)
    model_post=minimodel(dec_num=model_DINO.transformer.num_decoder_layers,post_frames=post_frames,
                         num_classes=model_DINO.num_classes,hidden_dim=model_DINO.hidden_dim,batch=args.batch_size)
    
    model_DINO.to(device)
    model_post.to(device)

    state_dict_dino_origin=torch.load(pretrain_dino_model)
    state_dict_dino=state_dict_dino_origin["model"]
    model_DINO.load_state_dict(state_dict_dino)
    if pretrain_mini_model:
        state_dict_mini=torch.load(pretrain_mini_model)
        model_post.load_state_dict(state_dict_mini["model"])

    new_dict={}
    model_dict=model_post.state_dict()
    #print(model_dict.keys())
    for key,value in state_dict_dino.items():
        if key in model_dict and model_dict[key].shape==value.shape:
            new_dict[key]=value
    model_dict.update(new_dict)
    model_post.load_state_dict(model_dict,strict=False)

    optimizer = torch.optim.AdamW(model_post.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    output_dir=args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    for i in range(epoch):
        print(f"start epoch:{i}")
        loss=train_one_epoch_mini(model_DINO,model_post,criterion,optimizer,data_loader_train,device)
        print(f"epoch:{i}--loss:{loss}")

        checkpoint={"epoch":i,"model":model_post.state_dict(),'optimizer': optimizer.state_dict()}
        checkpoint_paths = f"{output_dir}/checkpoint_epoch_{i}.pth"
        torch.save(checkpoint, checkpoint_paths)
    print("END")

def evaluate(device="cuda",post_frames=3):
    pretrain_dino_model="/media/mldadmin/home/s125mdg35_05/DINO/DINO/logs/DINO/R50-MS4/checkpoint_best_regular.pth"
    pretrain_mini_model="/media/mldadmin/home/s125mdg35_05/DINO/DINO/output_mini_702/checkpoint_epoch_9.pth"
    config_file="/media/mldadmin/home/s125mdg35_05/DINO/DINO/config/DINO/DINO_4scale_swin.py"
    coco_path="/media/mldadmin/home/s125mdg35_05/dataset/data"
    backbone_dir="/media/mldadmin/home/s125mdg35_05/DINO/DINO/backbone"
    output_dir='/media/mldadmin/home/s125mdg35_05/DINO/DINO/output_mini'
    #更改coco数据地址在main.py中更改
    args=generate_args(config_file)
    args.backbone_dir=backbone_dir
    args.coco_path=coco_path
    args.output_dir=output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.dn_scalar=100
    args.embed_init_tgt=True
    args.dn_label_coef=1.0
    args.dn_bbox_coef=1.0
    args.use_ema=False
    args.dn_box_noise_scale=1.0
    args.dn_number=0
    #frame_start_list=[0,1260,2520,3780,4980,6180,7380,8175,8970,9765,10725,11685,12645,13590,14535,15480,16455,17430]#验证集视频分割节点

    dataset_test = build_dataset(image_set='val', args=args)#单帧单帧输入
    sampler_val = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_val,drop_last=False, collate_fn=utils.collate_fn)

    model_DINO, criterion, postprocessors = build_model_main(args)
    model_post=minimodel(dec_num=model_DINO.transformer.num_decoder_layers,post_frames=post_frames,
                         num_classes=model_DINO.num_classes,hidden_dim=model_DINO.hidden_dim,batch=1)
    model_DINO.to(device)
    model_post.to(device)

    state_dict_dino_origin=torch.load(pretrain_dino_model)
    state_dict_dino=state_dict_dino_origin["model"]
    model_DINO.load_state_dict(state_dict_dino)
    print("DINO load finished")
    state_dict_mini=torch.load(pretrain_mini_model)
    model_post.load_state_dict(state_dict_mini["model"])
    print("minimodel load finished")

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_test)
    os.environ['EVAL_FLAG'] = 'TRUE'
    test_stats, coco_evaluator = evaluate_withmini(model_DINO, model_post,criterion, postprocessors,
                                        data_loader_test, base_ds, device, output_dir, frame_start_list=[],wo_class_error=False, args=args)
    if args.output_dir:
        utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

def main_lstm(device="cuda",epoch=10,post_frames=3):
    pretrain_dino_model="/media/mldadmin/home/s125mdg35_05/DINO/DINO/logs/DINO/R50-MS4/checkpoint_best_regular.pth"
    config_file="/media/mldadmin/home/s125mdg35_05/DINO/DINO/config/DINO/DINO_4scale_swin.py"
    coco_path="/media/mldadmin/home/s125mdg35_05/dataset/data"
    backbone_dir="/media/mldadmin/home/s125mdg35_05/DINO/DINO/backbone"
    output_dir='/media/mldadmin/home/s125mdg35_05/DINO/DINO/output_mini_lstm'
    #更改coco数据地址在main.py中更改
    args=generate_args(config_file)
    args.backbone_dir=backbone_dir
    args.coco_path=coco_path
    args.output_dir=output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.dn_scalar=100
    args.embed_init_tgt=True
    args.dn_label_coef=1.0
    args.dn_bbox_coef=1.0
    args.use_ema=False
    args.dn_box_noise_scale=1.0
    args.dn_number=0

    dataset_train = build_dataset(image_set='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    print(f"训练batchsize:{args.batch_size}")
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=utils.collate_fn_mini)

    model_DINO, criterion, postprocessors = build_model_main(args)
    model_post=minimodel_LSTM(dec_num=model_DINO.transformer.num_decoder_layers,post_frames=post_frames,
                         num_classes=model_DINO.num_classes,hidden_dim=model_DINO.hidden_dim,batch=args.batch_size)
    model_DINO.to(device)
    model_post.to(device)

    state_dict_dino_origin=torch.load(pretrain_dino_model)
    state_dict_dino=state_dict_dino_origin["model"]
    model_DINO.load_state_dict(state_dict_dino)

    new_dict={}
    model_dict=model_post.state_dict()
    #print(model_dict.keys())
    for key,value in state_dict_dino.items():
        if key in model_dict and model_dict[key].shape==value.shape:
            new_dict[key]=value
    model_dict.update(new_dict)
    model_post.load_state_dict(model_dict,strict=False)

    optimizer = torch.optim.AdamW(model_post.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    output_dir=args.output_dir

    for i in range(epoch):
        print(f"start epoch:{i}")
        loss=train_one_epoch_mini(model_DINO,model_post,criterion,optimizer,data_loader_train,device)
        print(f"epoch:{i}--loss:{loss}")

        checkpoint={"epoch":i,"model":model_post.state_dict(),'optimizer': optimizer.state_dict()}
        checkpoint_paths = f"{output_dir} / checkpoint_epoch{i}.pth"
        torch.save(checkpoint, checkpoint_paths)
    print("END")
    #模型训练
    #模型保存
if __name__=="__main__":
    evaluate()
    #main()
