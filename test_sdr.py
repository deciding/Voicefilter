import librosa
from mir_eval.separation import bss_eval_sources
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_wav1', type=str, required=True,
                    help="input wav 1")
parser.add_argument('--in_wav2', type=str, required=True,
                    help="input wav 2")
args = parser.parse_args()

srate=16000

s1, _ = librosa.load(args.in_wav1, sr=srate)
s2, _ = librosa.load(args.in_wav2, sr=srate)

min_len=min(len(s1), len(s2))
s1=s1[:min_len]
s2=s2[:min_len]

sdr = bss_eval_sources(s1, s2, False)[0][0]
print(sdr)
