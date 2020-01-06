import os
import math
import torch
import torch.nn as nn
import traceback

from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
from time import time

class WindowAverager:
    def __init__(self, N):
        self.val_list=[]
        for i in range(N):
            self.val_list.append(0.0)

    def get_avg(self, new_val):
        self.val_list=self.val_list[1:]
        self.val_list.append(new_val)
        nz_cnt=0
        nz_sum=0
        for i, item in enumerate(self.val_list):
            if item>1e-8:
                nz_cnt+=1
                nz_sum+=item
        avg_val=nz_sum/nz_cnt if nz_cnt!=0 else 0
        return avg_val

def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    # load embedder
    embedder_pt = torch.load(args.embedder_path)
    embedder = SpeechEmbedder(hp).cuda()
    #embedder = SpeechEmbedder(hp)
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    model = VoiceFilter(hp).cuda()
    #model = VoiceFilter(hp)
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    try:
        criterion = nn.MSELoss()
        time_averager=WindowAverager(100)
        loss_averager=WindowAverager(100)

        while True:
            model.train()
            for dvec_mels, target_mag, mixed_mag in trainloader:
                start=time()
                target_mag = target_mag.cuda()
                mixed_mag = mixed_mag.cuda()

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.cuda()
                    dvec = embedder(mel)
                    #dvec = torch.zeros([256], dtype=torch.float32).cuda()
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()
                #print("embedder time: %f" % (time()-start))

                mask = model(mixed_mag, dvec)
                output = mixed_mag * mask
                #output.tolist()
                #print("mask time: %f" % (time()-start))

                # output = torch.pow(torch.clamp(output, min=0.0), hp.audio.power)
                # target_mag = torch.pow(torch.clamp(target_mag, min=0.0), hp.audio.power)
                loss = criterion(output, target_mag)
                #print("loss time: %f" % (time()-start))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                #print("opt time: %f" % (time()-start))

                loss_val = loss.item()
                elapsed=time()-start
                avg_elapsed=time_averager.get_avg(elapsed)
                avg_loss=loss_averager.get_avg(loss_val)
                print("Step: %d, time elapsed: %f, avg elapsed: %f, loss: %f, avg loss: %f" % (step, elapsed, avg_elapsed, loss_val, avg_loss))
                if loss_val > 1e8 or math.isnan(loss_val):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss_val, step))
                    raise Exception("Loss exploded")

                # write loss to tensorboard
                if step % hp.train.summary_interval == 0:
                    writer.log_training(loss_val, step)
                    logger.info("Wrote summary at step %d" % step)

                # 1. save checkpoint file to resume training
                # 2. evaluate and save sample to tensorboard
                if step % hp.train.checkpoint_interval == 0:
                #if True:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info("Saved checkpoint to: %s" % save_path)
                    validate(audio, model, embedder, testloader, writer, step)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
