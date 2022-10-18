
import math
import os
from turtle import shape
import torch
from tqdm import tqdm
from config import config
from functools import partial
from torch.utils.data import DataLoader

def epoch_train(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, cur_epoch, all_epoch, local_rank=0):

    #---------------------------------------#
    #   在主程序加载进度条
    #---------------------------------------#
    if local_rank ==0:
        pmgressbar_train = tqdm(train_loader, desc=f"Train epoch {cur_epoch + 1}/{all_epoch}", postfix=dict, mininterval=0.5)
    
    # 确保网络模式为可训练模式
    model.train()
    train_losses = []

    #---------------------------------------#
    #   迭代抽取数据
    #---------------------------------------#
    for iteration, (images, targets) in enumerate(train_loader):
        # 将数据放到 gpu 上面
        with torch.no_grad():
            if config.GPU:
                images = images.cuda(local_rank)
                targets_cuda = [target.cuda(local_rank) for target in targets]

        # 清除梯度优化器中的值
        optimizer.zero_grad()

        if config.FP16:
            # 使用混合双精进行训练
            with torch.cuda.amp.autocast():
                out     = model(images)
                loss    = (
                        loss_fn(out[0], targets_cuda[0], scaled_anchors[0])
                        + loss_fn(out[1], targets_cuda[1], scaled_anchors[1])
                        + loss_fn(out[2], targets_cuda[2], scaled_anchors[2])
                    )
            
            train_losses.append(loss.item())   

            # 反向传播
            # 先将 loss 放大
            scaler.scale(loss).backward()
            # 对权重求导
            scaler.step(optimizer)
            # 将梯度更新，缩小梯度信息
            scaler.update()
        else:
            # 否则正常训练
            out         = model(images)
            loss        = (
                        loss_fn(out[0], targets_cuda[0], scaled_anchors[0])
                        + loss_fn(out[1], targets_cuda[1], scaled_anchors[1])
                        + loss_fn(out[2], targets_cuda[2], scaled_anchors[2])
                    )
            loss.backward()
            optimizer.step()
        
        # 计算平均 loss，并更新进度条
        train_mean_loss = sum(train_losses) / len(train_losses)
        if local_rank == 0:
            pmgressbar_train.set_postfix(**{'train_loss' : train_mean_loss,
                                        'lr'         : get_lr(optimizer)})
            pmgressbar_train.update()

    if local_rank == 0:
        pmgressbar_train.close()
        print("一个epoch的训练集的训练结束. ")
        return train_mean_loss


#---------------------------------------------------#
#   加载数据集相应的 txt 
#---------------------------------------------------#
def get_anno_lines(train_annotation_path, val_annotation_path):
    with open(os.path.join(train_annotation_path, "train.txt")) as f:
        train_lines = f.readlines()
    with open(os.path.join(val_annotation_path, "val.txt")) as f:
        val_lines   = f.readlines()
    
    return train_lines, val_lines

#---------------------------------------------------#
#   从优化器中获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   对优化器中设置新的学习率
#---------------------------------------------------#
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#---------------------------------------------------#
#   选择不同的学习率下降公式
#---------------------------------------------------#
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    
    # 余弦退火算法
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr
    # step 下降算法
    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    # 返回有关装载了相关参数的函数
    return func

#---------------------------------------------------#
#   设置模型和额优化器保存到地方
#---------------------------------------------------#
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

#---------------------------------------------------#
#   从保存的文件中重新加载相应的参数
#---------------------------------------------------#
def load_checkpoint(checkpoint_file, model, optimizer, device , lr):
    print("=> Loading checkpoint")

    # 记录当前 网络 和 优化器 的静态参数
    # model_dict      = model.state_dict()
    # optimize_dict   = optimizer.state_dict()
    # 加载 预训练 权重
    checkpoint = torch.load(checkpoint_file, map_location=device)
    # load_key, no_load_key, temp_dict = [], [], {}
    # for k, v in checkpoint.items():
    #     if k in model_dict.key() and np.shape(model_dict[k]) = np.shape(v):
    #         temp_dict[k] = v
    #         load_key.append(k)
    #     else:
    #         no_load_key.append(k)
    # model_dict.update(temp_dict)
    model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr

    # ---------------------------------------#
