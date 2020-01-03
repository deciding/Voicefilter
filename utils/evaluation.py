import os
import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
import librosa


def validate(audio, model, embedder, testloader, writer, step):
    model.eval()
    
    criterion = nn.MSELoss()
    wav1, wav2, wav3 = None, None, None
    if not os.path.exists('logs/eval'):
        os.makedirs('logs/eval')
    with torch.no_grad():
        for batch in testloader:
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
            sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
            writer.log_evaluation(test_loss, sdr,
                                  mixed_wav, target_wav, est_wav,
                                  mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
                                  step)
            wav1, wav2, wav3 = mixed_wav, target_wav, est_wav
            break
        librosa.output.write_wav('logs/eval/mixed_wav_%d.wav' % step, wav1, 16000)
        librosa.output.write_wav('logs/eval/target_wav_%d.wav' % step, wav2, 16000)
        librosa.output.write_wav('logs/eval/est_wav_%d.wav' % step, wav3, 16000)

    model.train()
