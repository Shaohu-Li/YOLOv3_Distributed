
import os
import torch
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from model.loss import YoloLoss
from model.model import YOLOv3
from model.dataloader import Dataset_loader, yolo_dataset_collate, train_transforms, val_transforms
from config import config
from utils.utils_train import load_checkpoint, epoch_train,get_lr_scheduler, set_optimizer_lr,get_anno_lines, save_checkpoint
from utils.utils_map import epoch_eval_loss_map
from utils.utils_box import get_evaluation_bboxes
from utils.callbacks import Loss_Map_History, LossHistory


if __name__ == "__main__":
    
    #---------------------------------------#
    #   设置分布式训练使用的后端和 gpu
    #---------------------------------------#
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    ngpus_per_node  = torch.cuda.device_count()
    # print(ngpus_per_node)
    if config.DISTRIBUTED:
        # 使用的通信后端 ncc1->NVIDIA、gloo->Facebook、mpi->OpenMPI
        if dist.is_nccl_available():
            dist.init_process_group(backend="nccl")
        elif dist.is_gloo_available():
            dist.init_process_group(backend="mpi")
        elif dist.is_mpi_available():
            dist.init_process_group(backend="mpi")
        else:
            raise TypeError("==> 显卡没有支持的通信后端，请重新选择训练模式. ")

        # 获取当前进程的 rank(类似 pid)
        local_rank  = int(os.environ["LOCAL_RANK"])
        # print("current local_rank is : ", local_rank)
        # 获取当前主机的 rank (多主机的情况，主机的 id 号码)
        rank        = int(os.environ["RANK"])
        # print("current rank is :", rank) 
        # 设置当前进程所使用的 gpu, 这样设置为一个 进程 对应一个 gpu，也可以不这样设置
        device      = torch.device("cuda", local_rank) 
        # 只在最开始的进程中打印下面的信息
        if local_rank == 0:
            print(f"==> [{os.getpid()}]-> rank = {rank}, local_rank = {local_rank} training...")
            print("==> GPU Device Count : {}".format(ngpus_per_node))
    else:
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank  = 0  

    #---------------------------------------#
    #   加载自己的网络模型
    #---------------------------------------#
    model           = YOLOv3(config.model_config, num_classes=config.NUM_CLASSES)
    if local_rank == 0:
        print("==> 模型加载完毕... ")

    #---------------------------------------#
    #   加载优化器
    #---------------------------------------#
    if config.OPTIMIZER_TYPE == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config.INIT_LEARNING_RATE, betas=(config.MOMENTUM, 0.999), weight_decay=config.WEIGHT_DECAY
        )
    if config.OPTIMIZER_TYPE == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=config.INIT_LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY, nesterov=True
        )
    if local_rank == 0:
        print("==> 优化器加载完毕... ")
    
    #---------------------------------------#
    #   判断是否需要加载之前保存的模型和相应的参数
    #   注意加载模型的时候，使用的为原始的没有被分布式的模型 
    #---------------------------------------#
    if config.LOAD_MODEL:
        checkpoint_file = config.LOAD_WEIGHT_NAME
        load_checkpoint(checkpoint_file, model, optimizer, device, config.INIT_LEARNING_RATE)
        if local_rank == 0:
            print("==> 权重加载完毕...")
    else:
        if local_rank == 0:
            print("==> 没有加载权重...")


    #---------------------------------------#
    #   加载损失函数
    #---------------------------------------#
    loss_fn = YoloLoss()
    if local_rank == 0:
        print("==> 损失函数加载完毕... ")
    
    #---------------------------------------#
    #   加载 损失 和 map 记录器
    #---------------------------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(config.SAVE_DIR, "loss/loss_" + str(time_str))
        loss_history    = Loss_Map_History(log_dir=log_dir, model=model, input_shape=config.IMAGE_SIZE)
        print("==> 损失函数日志记载函数加载完毕... ")
    else:
        loss_history    = None

    #---------------------------------------#
    #   加载先验框 3 * 3 * 2
    #---------------------------------------#
    scaled_anchors = (torch.tensor(config.ANCHORS)).cuda(local_rank)
    if local_rank == 0:
        print("==> 先验框加载完毕... ")

    #---------------------------------------#
    #   是否使用混合精度进行训练
    #---------------------------------------#
    if config.FP16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    #---------------------------------------#
    #   1、加载数据和验证集的迭代对象
    # 
    #   2、分发数据
    #---------------------------------------#
    train_annotaion_lines, val_annotation_lines = get_anno_lines(train_annotation_path=config.TRAIN_LABEL_DIR, val_annotation_path=config.VAL_LABEL_DIR)
    
    train_dataset       = Dataset_loader(annotation_lines=train_annotaion_lines, input_shape=config.IMAGE_SIZE, anchors=config.ANCHORS, 
                                    transform=train_transforms, train = True, box_mode="midpoint")
    val_dataset         = Dataset_loader(annotation_lines=val_annotation_lines, input_shape=config.IMAGE_SIZE,  anchors=config.ANCHORS,
                                    transform=val_transforms, train = True, box_mode="midpoint")

    if config.DISTRIBUTED:
        # 使用分布式的数据的加载方式
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
        batch_size      = config.BATCH_SIZE // ngpus_per_node
        config.SHUFFLR  = False

        # 训练分布，则学习率也以当适应改变
        lr_limit_max    = 1e-3 if config.OPTIMIZER_TYPE == "adam" else 5e-2
        lr_limit_min    = 3e-4 if config.OPTIMIZER_TYPE == "adam" else 5e-4
        Init_lr_train   = min(max(batch_size / config.BATCH_SIZE * config.INIT_LEARNING_RATE, lr_limit_min), lr_limit_max)
        Min_lr_train    = min(max(batch_size / config.BATCH_SIZE * config.MIN_LEARNING_RATE, lr_limit_min * 1e-2 ), lr_limit_min * 1e-2)
        # 通过最大最小学习率获得学习率下降公式
        lr_scheduler_func = get_lr_scheduler(config.LEARNING_RATE_DECAY_TYPE, Init_lr_train, Min_lr_train, config.NUM_EPOCHS)

    else:
        batch_size      = config.BATCH_SIZE
        train_sampler   = None
        val_sampler     = None
        config.SHUFFLR  = True

    train_loader        = DataLoader(train_dataset, batch_size, config.SHUFFLR, num_workers=config.NUM_WORKERS, 
                                pin_memory=config.PIN_MEMORY, drop_last=False, sampler=train_sampler)
    val_loader          = DataLoader(val_dataset, batch_size, config.SHUFFLR, num_workers=config.NUM_WORKERS, 
                                pin_memory=config.PIN_MEMORY, drop_last=False,sampler=val_sampler)
    if local_rank == 0:

        print("==> 数据集迭代器加载完毕...")

    #---------------------------------------#
    #   是否使用多卡同步 BN
    #---------------------------------------#
    model_train     = model.train()

    if config.SYNC_BN and ngpus_per_node > 1 and config.DISTRIBUTED:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    else:
        print("==> Sync_bn is not support in one gpu or not set distributed. ")

    #---------------------------------------#
    #   分布网络模型
    #---------------------------------------#
    if config.GPU:
        if config.DISTRIBUTED:
            # DDP 分布模式
            model_train     = model_train.cuda(local_rank)
            # find_unused_parameters 是用来解决定义在 forward 函数中，作用没有用的网络层，引发错误的问题
            model_train     = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
            if local_rank == 0:
                print("==> 分布式网络部署成功，使用 DDP 模式进行训练. ")
        else:
            # DP 分布模式
            model_train     = torch.nn.DataParallel(model)
            cudnn.benckmark = True
            model_train     = model_train.cuda()
            if local_rank == 0: 
                print("==> 分布式网络部署成功，使用 DP 模式进行训练. ")

    if local_rank == 0:
        print("==> 进入 epochs, 开始训练... ")
                        
    for epoch in range( config.NUM_EPOCHS ):

        #---------------------------------------#
        #   将每个 samper 设置相同的 epoch
        #---------------------------------------#
        if config.DISTRIBUTED:
            train_sampler.set_epoch(epoch)
            
        # #---------------------------------------#
        # #   验证集测试函数
        # #---------------------------------------#
        # val_mean_loss, mapval = epoch_eval_loss_map(val_loader, model_train, epoch, config.NUM_EPOCHS, scaled_anchors, loss_fn, config.CONF_THRESHOLD, config.NMS_IOU_THRESH, local_rank=local_rank )


        #---------------------------------------#
        #   主训练函数
        #---------------------------------------#
        train_mean_loss = epoch_train(train_loader, model_train, optimizer, loss_fn, scaler, scaled_anchors, epoch, config.NUM_EPOCHS, local_rank=local_rank)


        if local_rank == 0:

            #---------------------------------------#
            #   验证集测试函数
            #---------------------------------------#
            val_mean_loss, mapval = epoch_eval_loss_map(val_loader, model_train, epoch, config.NUM_EPOCHS, scaled_anchors, loss_fn, config.CONF_THRESHOLD, config.NMS_IOU_THRESH, local_rank=local_rank )

            #---------------------------------------#
            #   记录网络的 损失 和 map
            #---------------------------------------#
            loss_history.append_loss(epoch, train_mean_loss, val_mean_loss, mapval)

            #---------------------------------------#
            #   保存网络的权重
            #---------------------------------------#
            if config.SAVE_MODEL:
                if (epoch + 1) % config.WEIGHT_SAVE_PERIOD == 0 or epoch + 1 == config.NUM_EPOCHS:
                    filename = os.path.join(config.SAVE_DIR, "checkpoint/ep%03d-train_loss%.3f-val_loss%.3f.pth"% (epoch, train_mean_loss, val_mean_loss))
                    save_checkpoint(model=model, optimizer=optimizer, filename=filename)

                elif len(loss_history.val_loss) <= 1 or (val_mean_loss) <= min(loss_history.val_loss):
                    print('Save current best model to best_epoch_weights.pth')
                    filename = os.path.join(config.SAVE_DIR, "checkpoint/best_epoch_weights.pth")
                    save_checkpoint(model=model, optimizer=optimizer, filename=filename)
                
                else: # 不然就是最后一个epoch了，保存最后一个epoch
                    filename = os.path.join(config.SAVE_DIR, "checkpoint/last_epoch_weights.pth")
                    save_checkpoint(model=model, optimizer=optimizer, filename=filename)

        #---------------------------------------#
        #   设置新的学习率
        #---------------------------------------#
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)


        #---------------------------------------#
        #   等待所有分布式训练走到相同的位置
        #---------------------------------------#
        if config.DISTRIBUTED:
            dist.barrier()
    
    if local_rank == 0:
        loss_history.writer.close()
    