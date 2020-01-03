import os
import tensorflow as tf
import torch
from tqdm import tqdm
from glob import glob
import numpy as np
from collections.abc import Iterable
from utils.hparams import HParam
#from utils.audio import Audio
#import librosa

#python encoder_inference.py --in_dir training_libri_mel/train/ --gpu_str 5
#python tfrecord_producer.py --in_dir training_libri_mel/train/ --out xyz --gpu 6
#python tfrecord_producer.py --in_dir test_libri_mel/test/ --out xyz_mel_test --gpu 5 --need_mel --need_phase

def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def convert_to_example(target, mixed, speaker,
        target_phase=None, mixed_phase=None, target_mel=None, mixed_mel=None):
    raw_target = target.tostring()
    raw_mixed = mixed.tostring()
    raw_speaker = speaker.tostring()
    raw_target_phase = target_phase.tostring() if target_phase is not None else None
    raw_mixed_phase = mixed_phase.tostring() if mixed_phase is not None else None
    raw_target_mel = target_mel.tostring() if target_mel is not None else None
    raw_mixed_mel = mixed_mel.tostring() if mixed_mel is not None else None
    #TODO: handle the case where only have phase info but no mel info
    if target_mel is not None and mixed_mel is not None:
        if target_phase is not None and mixed_phase is not None:
            example = tf.train.Example(features=tf.train.Features(feature={
                'target': bytes_feature([raw_target]),
                'mixed': bytes_feature([raw_mixed]),
                'speaker': bytes_feature([raw_speaker]),
                'target_phase': bytes_feature([raw_target_phase]),
                'mixed_phase': bytes_feature([raw_mixed_phase]),
                'target_mel': bytes_feature([raw_target_mel]),
                'mixed_mel': bytes_feature([raw_mixed_mel]),
            }))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'target': bytes_feature([raw_target]),
                'mixed': bytes_feature([raw_mixed]),
                'speaker': bytes_feature([raw_speaker]),
                'target_mel': bytes_feature([raw_target_mel]),
                'mixed_mel': bytes_feature([raw_mixed_mel]),
            }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'target': bytes_feature([raw_target]),
            'mixed': bytes_feature([raw_mixed]),
            'speaker': bytes_feature([raw_speaker]),
        }))
    return example

class TFRecordProducer:
    def remove_list(self, list1, list2):
        i,j=0,0
        tmp_list1=[]
        tmp_list2=[]
        while i<len(list1) and j<len(list2):
            item1=int(list1[i].split('/')[-1].split('-')[0])
            item2=int(list2[j].split('/')[-1].split('-')[0])
            if item1==item2:
                tmp_list1.append(list1[i])
                tmp_list2.append(list2[j])
                i+=1
                j+=1
            elif item1<item2:
                i+=1
            else:
                j+=1
        return tmp_list1, tmp_list2

    def __init__(self, in_dir, hp, args, is_train):
        def find_all(file_format):
            return sorted(glob(os.path.join(self.data_dir, file_format)))

        self.in_dir=in_dir
        self.hp = hp
        self.args = args
        self.is_train = is_train
        #self.data_dir = hp.data.train_dir if is_train else hp.data.test_dir
        self.data_dir = in_dir

        self.dvec_list = find_all(hp.form.dvec_npy)
        self.target_wav_list = find_all(hp.form.target.wav)
        self.mixed_wav_list = find_all(hp.form.mixed.wav)
        self.target_mag_list = find_all(hp.form.target.mag)
        self.mixed_mag_list = find_all(hp.form.mixed.mag)
        self.target_phase_list = find_all(hp.form.target.phase)
        self.mixed_phase_list = find_all(hp.form.mixed.phase)
        self.target_mel_list = find_all(hp.form.target.mel)
        self.mixed_mel_list = find_all(hp.form.mixed.mel)

        _, self.target_wav_list=self.remove_list(self.dvec_list, self.target_wav_list)
        _, self.mixed_wav_list=self.remove_list(self.dvec_list, self.mixed_wav_list)
        _, self.target_mag_list=self.remove_list(self.dvec_list, self.target_mag_list)
        _, self.mixed_mag_list=self.remove_list(self.dvec_list, self.mixed_mag_list)
        _, self.target_phase_list=self.remove_list(self.dvec_list, self.target_phase_list)
        _, self.mixed_phase_list=self.remove_list(self.dvec_list, self.mixed_phase_list)
        _, self.target_mel_list=self.remove_list(self.dvec_list, self.target_mel_list)
        _, self.mixed_mel_list=self.remove_list(self.dvec_list, self.mixed_mel_list)

        print(len(self.dvec_list), len(self.target_wav_list), len(self.mixed_wav_list), \
            len(self.target_mag_list), len(self.mixed_mag_list),
            len(self.target_phase_list), len(self.mixed_phase_list),
            len(self.target_mel_list), len(self.mixed_mel_list))
        assert len(self.dvec_list) == len(self.target_wav_list) == len(self.mixed_wav_list) == \
            len(self.target_mag_list) == len(self.mixed_mag_list), "number of training files must match"
        assert len(self.dvec_list) != 0, \
            "no training file found"

    def write_training_examples_to_tfrecords(self, filename, need_phase=False, need_mel=False, hp=None):
        #if hp is not None:
        #    audio=Audio(hp)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for i in tqdm(range(len(self.dvec_list))):
                target_mag = torch.load(self.target_mag_list[i]).numpy()
                mixed_mag = torch.load(self.mixed_mag_list[i]).numpy()
                dvec = np.load(self.dvec_list[i])
                if need_phase:
                    target_phase = torch.load(self.target_phase_list[i]).numpy()
                    mixed_phase = torch.load(self.mixed_phase_list[i]).numpy()
                    target_mel = torch.load(self.target_mel_list[i]).numpy()
                    mixed_mel = torch.load(self.mixed_mel_list[i]).numpy()
                    #mixed_wav = audio.spec2wav(mixed_mag, mixed_phase)
                    #librosa.output.write_wav('mixed.wav', mixed_wav, 16000)
                    writer.write(convert_to_example(target_mag, mixed_mag, dvec, target_phase, mixed_phase, target_mel, mixed_mel).SerializeToString())
                elif need_mel:
                    target_mel = torch.load(self.target_mel_list[i]).numpy()
                    mixed_mel = torch.load(self.mixed_mel_list[i]).numpy()
                    writer.write(convert_to_example(target_mag, mixed_mag, dvec, None, None, target_mel, mixed_mel).SerializeToString())
                else:
                    writer.write(convert_to_example(target_mag, mixed_mag, dvec).SerializeToString())


import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--in_dir', default='.', help='input glob str')
parser.add_argument('-c', '--config', type=str, default='config/config.yaml',
                    help="yaml file for configuration")
parser.add_argument('--out', default='.', help='output prefix of tf record')
parser.add_argument('--gpu', default=0,
                    help='Path to model checkpoint')
parser.add_argument('--need_phase', action='store_true')
parser.add_argument('--need_mel', action='store_true')
args=parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

in_dir=args.in_dir
filename=args.out
hp = HParam(args.config)
tp=TFRecordProducer(in_dir, hp, args, True)
tp.write_training_examples_to_tfrecords(filename, args.need_phase, args.need_mel, hp)

