from loss import intervention
from utils.logger import get_temp_logger
import torch
from tqdm import *


def train(epoch, input_codes, delta_sz, delta_sn, masks, latent_type="S_ST", logger=None):
    if logger is None:
        logger = get_temp_logger("Train intervention coefficient")

    logger.info("初始化相关系数，全为0")
    intervt = intervention.StyleIntervention(latent_type)

    logger.info("初始化优化器SGD")
    optimizer = torch.optim.SGD(intervt.parameters(), lr=0.01)

    logger.info("加载损失函数")
    criterion = intervention.InterventionLoss()

    logger.info("开始训练")
    loss_list = []
    tbar = tqdm(total=epoch)
    for i in range(epoch):
        # for name, param in intervt.named_parameters():
        #     # 根据名字冻结参数
        #     if "0" in name or "1" in name:
        #         param.requires_grad = True

        delta_s = intervt(delta_sz, delta_sn)
        loss = criterion(intervt.get_coefficient(),
                         input_codes, delta_s, delta_sz, delta_sn, masks)
        loss_list.append(loss.item())
        print(f"Train Loss:{loss.item()} in Epoch {i}/{epoch}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 解冻参数
        # for param in intervt.parameters():
        #     param.requires_grad = True

        tbar.update(1)