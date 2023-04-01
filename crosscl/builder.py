import torch
import torch.nn as nn
from lib.prroi_pool.functional import prroi_pool2d


class CrossCL(nn.Module):
    """
    Build a CrossCL model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, fpn_bn=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: CrossCL momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CrossCL, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, use_fpn_norm=fpn_bn)
        self.encoder_k = base_encoder(num_classes=dim, use_fpn_norm=fpn_bn)

        '''
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        '''

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def logits(self, q, k, neg):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, neg])
        return torch.cat([l_pos, l_neg], dim=1) / self.T

    def forward(self, im_q=None, im_k=None, ac_q=None, ac_k=None, encode_only=False):
        if encode_only:
            return self.encoder_q(im_q, return_middle=False)  # queries: NxC

        img_size = im_q.size()[-1]
        batch_im_idx = torch.arange(im_q.size(0)).type_as(ac_q)
        batch_idx = batch_im_idx.unsqueeze(1).repeat(1, ac_q.size(1)).unsqueeze(2)

        # compute query features
        q, p4_q, p3_q = self.encoder_q(im_q, return_middle=True)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, p4_k, p3_k = self.encoder_k(im_k, return_middle=True)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            p4_k = self._batch_unshuffle_ddp(p4_k, idx_unshuffle)
            p3_k = self._batch_unshuffle_ddp(p3_k, idx_unshuffle)

            # p4 feats
            if img_size == 224:
                neg_p4 = nn.functional.avg_pool2d(p4_k, 3, stride=1, padding=0)
            elif img_size == 384:
                neg_p4 = nn.functional.avg_pool2d(p4_k, 4, stride=2, padding=0)
            else:  # 480
                neg_p4 = nn.functional.avg_pool2d(p4_k, 5, stride=2, padding=0)
            neg_p4 = nn.functional.normalize(neg_p4, dim=1)
            neg_p4 = concat_all_gather(neg_p4)
            neg_p4 = neg_p4.transpose(1, 0).contiguous().view(256, -1)

            # p3 feats
            if img_size == 224:
                neg_p3 = nn.functional.avg_pool2d(p3_k, 3, stride=1, padding=0)
            elif img_size == 384:
                neg_p3 = nn.functional.avg_pool2d(p3_k, 4, stride=2, padding=0)
            else:  # 480
                neg_p3 = nn.functional.avg_pool2d(p3_k, 5, stride=2, padding=0)
            neg_p3 = nn.functional.normalize(neg_p3, dim=1)
            neg_p3 = neg_p3.transpose(1, 0).contiguous().view(256, -1)  # (128, Nx12x12)

        # local vector
        ac_q = torch.cat([batch_idx, ac_q], dim=2)
        ac_k = torch.cat([batch_idx, ac_k], dim=2)
        ac_q = ac_q.view(-1, 5)
        ac_k = ac_k.view(-1, 5)

        p4_q = prroi_pool2d(p4_q, ac_q.detach(), 1, 1, 1/16.)
        p3_q = prroi_pool2d(p3_q, ac_q.detach(), 1, 1, 1/8.)
        with torch.no_grad():
            pos_p4 = prroi_pool2d(p4_k, ac_k, 1, 1, 1/16.)
            pos_p3 = prroi_pool2d(p3_k, ac_k, 1, 1, 1/8.)

        p4_q = p4_q.squeeze(3).squeeze(2)
        pos_p4 = pos_p4.squeeze(3).squeeze(2)
        p4_q = nn.functional.normalize(p4_q, dim=1)
        pos_p4 = nn.functional.normalize(pos_p4, dim=1)

        p3_q = p3_q.squeeze(3).squeeze(2)
        pos_p3 = pos_p3.squeeze(3).squeeze(2)
        p3_q = nn.functional.normalize(p3_q, dim=1)
        pos_p3 = nn.functional.normalize(pos_p3, dim=1)

        # global
        logits_global = self.logits(q, k, self.queue.clone().detach())
        labels = torch.zeros(logits_global.shape[0], dtype=torch.long).cuda()

        # local
        logits_p4 = self.logits(p4_q, pos_p4, neg_p4)
        logits_p3 = self.logits(p3_q, pos_p3, neg_p3)

        # cross-scale
        logits_p4_p3 = self.logits(p4_q, pos_p3, neg_p3)
        logits_p3_p4 = self.logits(p3_q, pos_p4, neg_p4)

        self._dequeue_and_enqueue(k)

        return (logits_global, labels), logits_p4, logits_p3, logits_p4_p3, logits_p3_p4


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
