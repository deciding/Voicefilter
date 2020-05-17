import os
import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
import librosa
import numpy as np
from time import time

EPS = 1e-8

def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask

def cal_db(estimate_source, source_lengths=None):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    # Add batch size dim
    if len(estimate_source.size())==1:
        estimate_source=estimate_source.unsqueeze(0)
    # Add speaker num dim
    if len(estimate_source.size())==2:
        estimate_source=estimate_source.unsqueeze(1)
    B, C, T = estimate_source.size()
    ## Step 0. mask padding position along T
    ## [B, C, T] -> [B, 1, T] since mask is same on all speakers
    #mask = get_mask(estimate_source, source_lengths)
    #estimate_source *= mask

    # Step 1. Zero-mean norm on T dim
    #num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True)# / num_samples
    zero_mean_estimate = estimate_source - mean_estimate

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_estimate = zero_mean_estimate # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    s_estimate_power = torch.sum(s_estimate ** 2, dim=2)
    s_estimate_db = 10 * torch.log10(s_estimate_power + EPS)  # [B, C]
    db=torch.mean(s_estimate_db)

    return db

def cal_si_snr(source, estimate_source, source_lengths=None):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    import pdb;pdb.set_trace()
    assert source.size() == estimate_source.size()
    # Add batch size dim
    if len(source.size())==1:
        source=source.unsqueeze(0)
        estimate_source=estimate_source.unsqueeze(0)
    # Add speaker num dim
    if len(source.size())==2:
        source=source.unsqueeze(1)
        estimate_source=estimate_source.unsqueeze(1)
    B, C, T = source.size()
    # Step 0. mask padding position along T
    # [B, C, T] -> [B, 1, T] since mask is same on all speakers
    if source_lengths is not None:
        mask = get_mask(source, source_lengths)
        estimate_source *= mask

    # Step 1. Zero-mean norm on T dim
    #num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    num_samples = torch.tensor([T]).float().repeat(B).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    if source_lengths is not None:
        zero_mean_target *= mask
        zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target # [B, C, T]
    s_estimate = zero_mean_estimate # [B, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]
    si_snr=torch.mean(pair_wise_si_snr, dim=1)

    return si_snr

def validate(audio, model, embedder, testloader, writer, step):
    model.eval()

    criterion = nn.MSELoss()
    wav1, wav2, wav3 = None, None, None
    if not os.path.exists('logs/eval'):
        os.makedirs('logs/eval')
    sdri_list=[]
    sisnri_list=[]
    total_sisnri=0
    total_test_loss=0
    cnt=0
    neg_cnt=0
    with torch.no_grad():
        for batch in testloader:
            start=time()
            dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[0]

            dvec_mel = dvec_mel.cuda()
            target_mag = target_mag.unsqueeze(0).cuda()
            mixed_mag = mixed_mag.unsqueeze(0).cuda()

            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)
            est_mask = model(mixed_mag, dvec)
            est_mag = est_mask * mixed_mag
            test_loss = criterion(target_mag, est_mag).item()

            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            target_mag = target_mag[0].cpu().detach().numpy()
            est_mag = est_mag[0].cpu().detach().numpy()

            #mixed_wav = audio.spec2wav(mixed_mag, mixed_phase)

            est_wav = audio.spec2wav(est_mag, mixed_phase)
            est_mask = est_mask[0].cpu().detach().numpy()
            cnt+=1
            ##SDRi
            ##sdri = bss_eval_sources(target_wav, est_wav, False)[0][0] - \
            ##        bss_eval_sources(target_wav, mixed_wav, False)[0][0]
            ##SPIT
            ##sisnri=cal_db(torch.from_numpy(est_wav).float())
            #sisnri = cal_si_snr(
            #            torch.from_numpy(target_wav).float(),
            #            torch.from_numpy(est_wav).float()
            #            ) - \
            #        cal_si_snr(
            #            torch.from_numpy(target_wav).float(),
            #            torch.from_numpy(mixed_wav).float()
            #        )
            ##SDRi
            ##sdri_list.append(sdri)
            #sisnri_list.append(sisnri)
            #total_sisnri+=sisnri
            #total_test_loss+=test_loss
            #if sisnri<0.0:
            #    neg_cnt+=1
            ##SDRi
            ##print("Eval: step: %d, time: %.2f, wrong pred: %.3f, test loss: %.5f, avg: %.5f, sisnri: %.2f, avg: %.2f, median: %.2f, sdri: %.2f, avg: %.2f, median: %.2f" % \
            ##        (cnt, time()-start, float(neg_cnt)/float(cnt), test_loss, total_test_loss/cnt, sisnri, total_sisnri/cnt, np.median(sisnri_list),
            ##            sdri, np.mean(sdri_list), np.median(sdri_list)))
            #print("Eval: step: %d, time: %.2f, wrong pred: %.3f, test loss: %.5f, avg: %.5f, sisnri: %.2f, avg: %.2f, median: %.2f, sisisnri: %.2f" % \
            #        (cnt, time()-start, float(neg_cnt)/float(cnt), test_loss, total_test_loss/cnt, sisnri, total_sisnri/cnt, np.median(sisnri_list), np.mean([val for val in sisnri_list if val >0.0])))
            wav1, wav2, wav3 = mixed_wav, target_wav, est_wav
            #break
        #writer.log_evaluation(test_loss, sdr,
        #                      mixed_wav, target_wav, est_wav,
        #                      mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
        #                      step)
            librosa.output.write_wav('logs/eval1/mixed_wav_%d_%d.wav' % (step, cnt), wav1, 16000)
            librosa.output.write_wav('logs/eval1/target_wav_%d_%d.wav' % (step, cnt), wav2, 16000)
            librosa.output.write_wav('logs/eval1/est_wav_%d_%d.wav' % (step, cnt), wav3, 16000)

    np.save('db_list.npy', sisnri_list)

    model.train()
