import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image

from torch.utils import data
from network.DBSNet import DBSNet
from datasets import ext_transforms as et
from datasets.LC8BASTestset import LC8BASDataset
import argparse
from tqdm import tqdm
import collections

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir,size=(512,512)):
    resolution=list(size)
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
   
    imo = im.resize((resolution[0], resolution[1]),resample=Image.BILINEAR)

    imo.save(d_dir+image_name+'.png')

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_dir", type=str, 
        default='./BAS512_preds/',
        help="path to Dataset")
    
    parser.add_argument("--testset_path", type=str, 
    default='D:/2023_Files/BAS_Dataset/LC8BAS512_Aug/Test',
                        help="path to a single image or image directory")
    
    parser.add_argument("--dataset", type=str, default='LC8BAS512_Aug', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
    parser.add_argument("--phi", type=str, default='s',
                       help='PVT version: t,s,m,l')
  
    parser.add_argument("--model", type=str, default='DBSNet',
                    help='model name:[DBSNet]')
 
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--n_cpu", type=int, default=1,
                        help="download datasets")

    parser.add_argument("--ckpt",type=str,
            default=None, 
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def get_dataset(opts):
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_dst = LC8BASDataset(is_train=False,voc_dir=opts.testset_path,
                            transform=val_transform)
    return val_dst

def main():
    torch.cuda.empty_cache()

    opts = get_argparser().parse_args()

    test_dst = get_dataset(opts)
    
    opts.prediction_dir += opts.model+'/'
    if not os.path.exists(opts.prediction_dir):
        os.makedirs(opts.prediction_dir, exist_ok=True)
    print('opts:',opts)

    test_loader = data.DataLoader(
        test_dst, batch_size=opts.batch_size, shuffle=False, num_workers=opts.n_cpu)
    print("Dataset: %s, Test set: %d" %
            (opts.dataset, len(test_dst)))
   
   
    net = DBSNet(n_channels=3, phi=opts.phi)
  
   
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        try:
            net.load_state_dict(checkpoint["model_state"])
            print('try: load pth from:', opts.ckpt)
        except:
            dic = collections.OrderedDict()
            for k, v in checkpoint["model_state"].items():
                mlen=len('module')+1
                newk=k[mlen:]
                dic[newk]=v
            net.load_state_dict(dic)
            print('except: load pth from:', opts.ckpt)
        del checkpoint

    net=net.cuda()
    net.eval()

    for data_test in tqdm(test_loader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        inputs_test = Variable(inputs_test).cuda()
        
        SOUTS= net(inputs_test)
        
        pred = SOUTS[0]
        pred = pred[:,0,:,:]
        pred = normPRED(pred)
    
        if not os.path.exists(opts.prediction_dir):
            os.makedirs(opts.prediction_dir, exist_ok=True)

        save_output(data_test['img_name'][0], pred, opts.prediction_dir, 
                    size=(opts.input_size, opts.input_size))
        
        del SOUTS
       

if __name__ == "__main__":
    main()
