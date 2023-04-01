import torch
from collections import OrderedDict
import sys


def moco_convert(src, dst):
    """Convert keys in pycls pretrained moco models to mmdet style."""
    # load caffe model
    moco_model = torch.load(src, map_location='cpu')
    blobs = moco_model['state_dict']
    # convert to pytorch style
    state_dict = OrderedDict()
    for k, v in blobs.items():
        if not k.startswith('module.encoder_q.'):
            continue
        if 'fpn' in k:
            continue
        old_k = k
        k = k.replace('module.encoder_q.', '')
        state_dict[k] = v
        print(old_k, '->', k)
    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


src = sys.argv[1]
dst = sys.argv[2]
moco_convert(src, dst)