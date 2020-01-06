import argparse
from pathlib import Path
import numpy as np
from model.embedder import SpeechEmbedder
import torch
from utils.hparams import HParam
import librosa
from utils.audio import Audio

##python encoder_inference.py --in_dir ../vox1_test/wav/ --out_dir spkid --gpu_str 0
#python encoder_inference.py --in_dir training_libri/test_phase --gpu_str 0
if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="embedder.pt",
                        help="Path to a saved encoder")
    parser.add_argument('-c', '--config', type=str,
                        default="config/config.yaml",
                        help="yaml file for configuration")
    parser.add_argument("--in_dir", type=str, required=True, help="input data(pickle) dir")
    parser.add_argument("--out_dir", type=str, required=True, help="input data(pickle) dir")
    parser.add_argument('--gpu_str', default='0')
    args = parser.parse_args()
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_str)

    print("Preparing the encoder...")
    hp = HParam(args.config)
    embedder_pt = torch.load(args.enc_model_fpath)
    embedder = SpeechEmbedder(hp).cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()
    audio = Audio(hp)

    from glob import glob
    from tqdm import tqdm
    #../datasets/raw_libri/librispeech/train-clean-360/8194/89390/8194-89390-0041-norm.wav
    #txt_list=glob('%s/*.txt' % args.in_dir, recursive=True)
    #for txt_file in tqdm(txt_list):
    #    with open(txt_file) as f:
    #        wav_file=f.readline().strip()

    #wav_list=glob('%s/**/*.wav' % args.in_dir, recursive=True)
    wav_list=glob('%s/*dvec.wav' % args.in_dir, recursive=True)#convtasnet
    #wav_list=[wavfile for wavfile in wav_list if int(os.path.basename(wavfile).split('-')[0]) >=70000]
    for wav_file in tqdm(wav_list):
        #preprocessed_wav = encoder.preprocess_wav(wav_file)
        #norm_mean_dvector= encoder.embed_utterance(preprocessed_wav)
        dvec_wav, _ = librosa.load(wav_file, sr=hp.audio.sample_rate)
        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().cuda()
        norm_mean_dvector = embedder(dvec_mel)
        ##filename='%s.npy' % os.path.basename(txt_file.replace('.txt',''))
        #filename='%s.npy' % os.path.basename(wav_file.replace('.wav',''))
        ##spk_dir=args.in_dir
        #file_parts=wav_file.split('/')
        #spkid=file_parts[-3]
        #clipid=file_parts[-2]
        #spk_dir="%s/%s/%s" % (args.out_dir, spkid, clipid)
        #if not os.path.exists(spk_dir):
        #    os.makedirs(spk_dir)
        #npy_save_path='%s/%s' % (spk_dir, filename)

        #convtasnet
        filename='%s.npy' % os.path.basename(wav_file).replace('.wav', '')
        npy_save_path='%s/%s' % (args.in_dir, filename)
        np.save(npy_save_path, norm_mean_dvector.detach().cpu().numpy())

