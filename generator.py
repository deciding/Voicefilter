import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.audio import Audio
from utils.hparams import HParam

#python generator.py -c config/config.yaml -d ../datasets/raw_libri/LibriSpeech/ -o training_prepared --no_need_spec (for convtasnet)
#python generator.py -c config/config.yaml -d ../datasets/raw_libri/librispeech -o training_libri_mel (for voicefilter)
#python generator.py -c config/config.yaml -d ../datasets/raw_libri/librispeech -o test_libri_mel --need_phase (for eval voicefilter)

generate_new_field_for_prepared=False
generate_own_data=False

def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

def mix_wrapper(hp, args, audio, num, s1_dvec, s1_target, s2_target, s2_dvec, us_dvec, train):
    try:
        mix(hp, args, audio, num, s1_dvec, s1_target, s2_target, s2_dvec, us_dvec, train)
    except Exception as e:
        print(s1_dvec, s1_target, s2_target, s2_dvec, us_dvec)
        print(e)

def get_text_from_wav(wav_path):
    parent=os.path.dirname(wav_path)
    wav_id=os.path.splitext(os.path.basename(wav_path))[0][:-5]
    clip_id='-'.join(wav_id.split('-')[:2])
    trans_txt="%s/%s.trans.txt" % (parent, clip_id)
    with open(trans_txt) as f:
        for line in f:
            line=line.strip()
            fields=line.split(' ', 1)
            if fields[0]==wav_id:
                return fields[1].lower()
    assert False, "there's no transcript for the file %s" % wav_path

#def mix(hp, args, audio, num, s1_dvec, s1_target, s2, train):
#def mix(hp, args, audio, num, s1_dvec, s1_target, s2_target, s2_dvec, train):
def mix(hp, args, audio, num, s1_dvec, s1_target, s2_target, s2_dvec, us_dvec, train):
    srate = hp.audio.sample_rate
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')

    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2_target, sr=srate)
    d2, _ = librosa.load(s2_dvec, sr=srate)
    d3, _ = librosa.load(us_dvec, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == len(d2.shape) == len(d3.shape) == 1, \
        'wav files must be mono, not stereo'

    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)
    d2, _ = librosa.effects.trim(d2, top_db=20)
    d3, _ = librosa.effects.trim(d3, top_db=20)

    # run once for test data
    if True and not train:
        candidates=glob.glob(os.path.join(os.path.dirname(us_dvec), '**', hp.form.input), recursive=True)
        ref_utt=''
        while d3.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
            print("Too short dvec3 wav")
            ref_utt=random.choice(candidates)
            d3, _ = librosa.load(ref_utt, sr=srate)
            d3, _ = librosa.effects.trim(d, top_db=20)
        if ref_utt != '':
            print('changed dvec', num, ref_utt)
    if train:
        # if reference for d-vector is too short, discard it
        if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
            print("Too short dvec wav")
            return
        # if reference for d-vector is too short, discard it
        if d2.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
            print("Too short dvec2 wav")
            return
        #TODO: for future we assert d3 length.
        candidates=glob.glob(os.path.join(os.path.dirname(us_dvec), '**', hp.form.input), recursive=True)
        ref_utt=''
        while d3.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
            print("Too short dvec3 wav")
            ref_utt=random.choice(candidates)
            d3, _ = librosa.load(ref_utt, sr=srate)
            d3, _ = librosa.effects.trim(d3, top_db=20)
        if ref_utt != '':
            print('changed dvec3', num, ref_utt)

    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
    if args.vad == 1:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    if train:
        # I think random segment length will be better, but let's follow the paper first
        # fit audio to `hp.data.audio_len` seconds.
        # if merged audio is shorter than `L`, discard it
        L = int(srate * hp.data.audio_len)
        if w1.shape[0] < L or w2.shape[0] < L:
            return
        w1, w2 = w1[:L], w2[:L]
    else:
        L = max(len(w1), len(w2))
        w1 = np.pad(w1, (0, L-len(w1)), 'constant', constant_values=(0.0, 0.0))
        w2 = np.pad(w2, (0, L-len(w2)), 'constant', constant_values=(0.0, 0.0))

    mixed = w1 + w2

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm
    assert len(mixed) == len(w1) == len(w2)

    # save vad & normalized wav files
    target_wav_path = formatter(dir_, hp.form.target.wav, num)
    mixed_wav_path = formatter(dir_, hp.form.mixed.wav, num)
    librosa.output.write_wav(target_wav_path, w1, srate)
    librosa.output.write_wav(mixed_wav_path, mixed, srate)
    if True:
        dvec_wav_path = formatter(dir_, hp.form.dvec_wav, num)
        librosa.output.write_wav(dvec_wav_path, d, srate)
    flag=True #train
    if True:
        dvec3_wav_path = formatter(dir_, hp.form.dvec3_wav, num)
        librosa.output.write_wav(dvec3_wav_path, d3, srate)
    if flag:
        target2_wav_path = formatter(dir_, hp.form.target2.wav, num)
        librosa.output.write_wav(target2_wav_path, w2, srate)
    if flag:
        dvec2_wav_path = formatter(dir_, hp.form.dvec2_wav, num)
        librosa.output.write_wav(dvec2_wav_path, d2, srate)
    if flag:
        dvec3_wav_path = formatter(dir_, hp.form.dvec3_wav, num)
        librosa.output.write_wav(dvec3_wav_path, d3, srate)

    if not args.no_need_spec:
        # save magnitude spectrograms
        target_mag, target_phase = audio.wav2spec(w1)
        mixed_mag, mixed_phase = audio.wav2spec(mixed)
        target_mag_path = formatter(dir_, hp.form.target.mag, num)
        mixed_mag_path = formatter(dir_, hp.form.mixed.mag, num)
        torch.save(torch.from_numpy(target_mag), target_mag_path)
        torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    if args.need_mel:
        # save mel spectrograms
        target_mel = audio.wav2mel(w1)
        mixed_mel = audio.wav2mel(mixed)
        target_mel_path = formatter(dir_, hp.form.target.mel, num)
        mixed_mel_path = formatter(dir_, hp.form.mixed.mel, num)
        torch.save(torch.from_numpy(target_mel), target_mel_path)
        torch.save(torch.from_numpy(mixed_mel), mixed_mel_path)

    if args.need_phase:
        target_phase_path = formatter(dir_, hp.form.target.phase, num)
        mixed_phase_path = formatter(dir_, hp.form.mixed.phase, num)
        torch.save(torch.from_numpy(target_phase), target_phase_path)
        torch.save(torch.from_numpy(mixed_phase), mixed_phase_path)

    s1_text_path = formatter(dir_, '*-trans.txt', num)
    with open(s1_text_path, 'w') as f:
        f.write(get_text_from_wav(s1_target))

    # save selected sample as text file. d-vec will be calculated soon
    dvec_text_path = formatter(dir_, hp.form.dvec, num)
    with open(dvec_text_path, 'w') as f:
        f.write(s1_dvec)
    #assert os.path.exists(dvec_wav_path) and os.path.exists(dvec2_wav_path) \
    #        and os.path.exists(mixed_wav_path) \
    #        and os.path.exists(target_wav_path) and os.path.exists(mixed_mag_path) \
    #        and os.path.exists(target_mag_path) and os.path.exists(dvec_text_path) \
    #        and os.path.exists(target2_wav_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-d', '--libri_dir', type=str, default=None,
                        help="Directory of LibriSpeech dataset, containing folders of train-clean-100, train-clean-360, dev-clean.")
    parser.add_argument('-v', '--voxceleb_dir', type=str, default=None,
                        help="Directory of VoxCeleb2 dataset, ends with 'aac'")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-p', '--process_num', type=int, default=None,
                        help='number of processes to run. default: cpu_count')
    parser.add_argument('--vad', type=int, default=0,
                        help='apply vad to wav file. yes(1) or no(0, default)')
    parser.add_argument('--need_phase', action='store_true',
                        help='apply vad to wav file. yes(1) or no(0, default)')
    parser.add_argument('--need_mel', action='store_true',
                        help='apply vad to wav file. yes(1) or no(0, default)')
    parser.add_argument('--no_need_spec', action='store_true',
                        help='apply vad to wav file. yes(1) or no(0, default)')
    parser.add_argument('--train_csv', type=str, default='datasets/train_tuples_clean_all.csv',
                        help="train csv")
    parser.add_argument('--test_csv', type=str, default='datasets/dev_tuples_clean.csv',
                        help="test csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)

    hp = HParam(args.config)

    audio = Audio(hp)

    cpu_num = cpu_count() if args.process_num is None else args.process_num

    if generate_own_data:
        start_index=70000
        total_num=50000
        if args.libri_dir is None and args.voxceleb_dir is None:
            raise Exception("Please provide directory of data")

        if args.libri_dir is not None:
            train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-100', '*'))
                                if os.path.isdir(x)] + \
                            [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-360', '*'))
                                if os.path.isdir(x)]
                            # we recommned to exclude train-other-500
                            # See https://github.com/mindslab-ai/voicefilter/issues/5#issuecomment-497746793
                            # + \
                            #[x for x in glob.glob(os.path.join(args.libri_dir, 'train-other-500', '*'))
                            #    if os.path.isdir(x)]
            test_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*'))]

        elif args.voxceleb_dir is not None:
            all_folders = [x for x in glob.glob(os.path.join(args.voxceleb_dir, '*'))
                                if os.path.isdir(x)]
            train_folders = all_folders[:-20]
            test_folders = all_folders[-20:]

        train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                        for spk in train_folders]
        train_spk = [x for x in train_spk if len(x) >= 2]

        test_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                        for spk in test_folders]
        test_spk = [x for x in test_spk if len(x) >= 2]

        def train_wrapper(num):
            num+=start_index
            spk1, spk2 = random.sample(train_spk, 2)
            s1_dvec, s1_target = random.sample(spk1, 2)
            s2 = random.choice(spk2)
            mix_wrapper(hp, args, audio, num, s1_dvec, s1_target, s2, train=True)

        def test_wrapper(num):
            num+=start_index
            spk1, spk2 = random.sample(test_spk, 2)
            s1_dvec, s1_target = random.sample(spk1, 2)
            s2 = random.choice(spk2)
            mix_wrapper(hp, args, audio, num, s1_dvec, s1_target, s2, train=False)

        def generate_train_wrapper(num):
            spk1, spk2 = random.sample(train_spk, 2)
            s1_dvec, s1_target = random.sample(spk1, 2)
            #s2 = random.choice(spk2)
            s2_dvec, s2_target = random.sample(spk2, 2)
            s1_dvec=os.path.splitext(os.path.basename(s1_dvec))[0][:-5]
            s1_target=os.path.splitext(os.path.basename(s1_target))[0][:-5]
            #s2=os.path.splitext(os.path.basename(s2))[0]
            s2_dvec=os.path.splitext(os.path.basename(s2_dvec))[0][:-5]
            s2_target=os.path.splitext(os.path.basename(s2_target))[0][:-5]
            #return ','.join([s1_dvec, s1_target, s2])+'\n'
            return ','.join([s1_dvec, s1_target, s2_target, s2_dvec])+'\n'

        with open("datasets/random%d.csv" % total_num, 'w') as f:
            for i in tqdm.tqdm(range(total_num)):
                f.write(generate_train_wrapper(i))

        ##arr = list(range(10**5))
        #arr = list(range(50000))
        #with Pool(cpu_num) as p:
        #    r = list(tqdm.tqdm(p.imap(train_wrapper, arr), total=len(arr)))

        #arr = list(range(10**2))
        #with Pool(cpu_num) as p:
        #    r = list(tqdm.tqdm(p.imap(test_wrapper, arr), total=len(arr)))
    else:
        def filter_clean(files_ids3):
            file_dir1=os.path.join(args.libri_dir, 'train-clean-100')
            file_dir2=os.path.join(args.libri_dir, 'train-clean-360')
            file_dir3=os.path.join(args.libri_dir, 'dev-clean')
            file_ids=files_ids3.strip().split(',')
            for file_id in file_ids:
                file_paths=file_id.split('-')
                file_dirname=os.path.join(file_dir1, file_paths[0], file_paths[1])
                if not os.path.exists(file_dirname):
                    file_dirname=os.path.join(file_dir2, file_paths[0], file_paths[1])
                if not os.path.exists(file_dirname):
                    file_dirname=os.path.join(file_dir3, file_paths[0], file_paths[1])
                if not os.path.exists(file_dirname):
                    return False
            return True

        if generate_new_field_for_prepared:
            def get_speaker2_ref_utt(files_ids3):
                file_dir1=os.path.join(args.libri_dir, 'train-clean-100')
                file_dir2=os.path.join(args.libri_dir, 'train-clean-360')
                file_dir3=os.path.join(args.libri_dir, 'dev-clean')
                file_ids=files_ids3.strip().split(',')
                file_id=file_ids[2]
                file_paths=file_id.split('-')
                file_dirname=os.path.join(file_dir1, file_paths[0])
                if not os.path.exists(file_dirname):
                    file_dirname=os.path.join(file_dir2, file_paths[0])
                if not os.path.exists(file_dirname):
                    file_dirname=os.path.join(file_dir3, file_paths[0])
                if not os.path.exists(file_dirname):
                    raise Exception('File not found %s' % file_paths[0])
                ref_utt=file_id+'-norm.wav'
                candidates=glob.glob(os.path.join(file_dirname, '**', hp.form.input), recursive=True)
                while ref_utt == file_id+'-norm.wav':
                    ref_utt=random.choice(candidates)
                    ref_utt=os.path.basename(ref_utt)
                return ref_utt[:-9]

            file_dir1=os.path.join(args.libri_dir, 'train-clean-100')
            file_dir2=os.path.join(args.libri_dir, 'train-clean-360')
            file_dir3=os.path.join(args.libri_dir, 'dev-clean')
            all_speakers=[]
            all_speakers+=[speaker for speaker in os.listdir(file_dir1) \
                    if os.path.isdir(os.path.join(file_dir1, speaker))]
            all_speakers+=[speaker for speaker in os.listdir(file_dir2) \
                    if os.path.isdir(os.path.join(file_dir2, speaker))]
            all_speakers+=[speaker for speaker in os.listdir(file_dir3) \
                    if os.path.isdir(os.path.join(file_dir3, speaker))]

            def get_unpresented_ref_utt(files_ids3):
                file_dir1=os.path.join(args.libri_dir, 'train-clean-100')
                file_dir2=os.path.join(args.libri_dir, 'train-clean-360')
                file_dir3=os.path.join(args.libri_dir, 'dev-clean')

                file_ids=files_ids3.strip().split(',')
                speaker1=file_ids[1].split('-')[0]
                speaker2=file_ids[2].split('-')[0]
                unpresented=speaker1
                while unpresented==speaker1 or unpresented==speaker2:
                    unpresented=random.choice(all_speakers)

                file_dirname=os.path.join(file_dir1, unpresented)
                if not os.path.exists(file_dirname):
                    file_dirname=os.path.join(file_dir2, unpresented)
                if not os.path.exists(file_dirname):
                    file_dirname=os.path.join(file_dir3, unpresented)
                if not os.path.exists(file_dirname):
                    raise Exception('File not found %s' % unpresented)
                candidates=glob.glob(os.path.join(file_dirname, '**', hp.form.input), recursive=True)
                ref_utt=random.choice(candidates)
                ref_utt=os.path.basename(ref_utt)
                return ref_utt[:-9]

            with open(args.train_csv) as f:
                train_list=f.readlines()
            print("Prev train list len : %d" % len(train_list))
            train_list=[train_line for train_line in train_list if filter_clean(train_line)]
            print("Post train list len : %d" % len(train_list))
            #clean_train_csv='%s_clean.csv' % os.path.splitext(args.train_csv)[0]
            clean_train_csv='%s_unpresented.csv' % os.path.splitext(args.train_csv)[0]
            with open(clean_train_csv, 'w') as f:
                for line in tqdm.tqdm(train_list):
                    line=line.strip()
                    #ref_utt=get_speaker2_ref_utt(line)
                    ref_utt=get_unpresented_ref_utt(line)
                    line="%s,%s" % (line, ref_utt)
                    f.write(line+'\n')

            with open(args.test_csv) as f:
                test_list=f.readlines()
            print("Prev test list len : %d" % len(test_list))
            test_list=[test_line for test_line in test_list if filter_clean(test_line)]
            print("Post test list len : %d" % len(test_list))
            #clean_test_csv='%s_clean.csv' % os.path.splitext(args.test_csv)[0]
            clean_test_csv='%s_unpresented.csv' % os.path.splitext(args.test_csv)[0]
            with open(clean_test_csv, 'w') as f:
                for line in tqdm.tqdm(test_list):
                    line=line.strip()
                    #ref_utt=get_speaker2_ref_utt(line)
                    ref_utt=get_unpresented_ref_utt(line)
                    line="%s,%s" % (line, ref_utt)
                    f.write(line+'\n')
            exit()


        with open(args.train_csv) as f:
            train_list=f.readlines()
        print("Prev train list len : %d" % len(train_list))
        train_list=[train_line for train_line in train_list if filter_clean(train_line)]
        print("Post train list len : %d" % len(train_list))
        with open(args.test_csv) as f:
            test_list=f.readlines()
        print("Prev test list len : %d" % len(test_list))
        test_list=[test_line for test_line in test_list if filter_clean(test_line)]
        print("Post test list len : %d" % len(test_list))

        def get_libri_full_path_from_id(file_id):
            file_dir1=os.path.join(args.libri_dir, 'train-clean-100')
            file_dir2=os.path.join(args.libri_dir, 'train-clean-360')
            file_dir3=os.path.join(args.libri_dir, 'dev-clean')
            file_paths=file_id.split('-')
            file_suffix='-norm.wav'
            file_dirname=os.path.join(file_dir1, file_paths[0], file_paths[1])
            if not os.path.exists(file_dirname):
                file_dirname=os.path.join(file_dir2, file_paths[0], file_paths[1])
            if not os.path.exists(file_dirname):
                file_dirname=os.path.join(file_dir3, file_paths[0], file_paths[1])
            if not os.path.exists(file_dirname):
                print("Cannot find dir path %s" % file_dirname)
            file_full_path=os.path.join(file_dirname, "%s%s" % (file_id, file_suffix))
            if not os.path.exists(file_full_path):
                print("Cannot find file %s" % file_full_path)
            return file_full_path

        def prepared_train_wrapper(num):
            #s1_dvec, s1_target, s2 = train_list[num].strip().split(',')
            #s1_dvec, s1_target, s2_target, s2_dvec = train_list[num].strip().split(',')
            s1_dvec, s1_target, s2_target, s2_dvec, us_dvec = train_list[num].strip().split(',')
            s1_dvec = get_libri_full_path_from_id(s1_dvec)
            s1_target = get_libri_full_path_from_id(s1_target)
            s2_target = get_libri_full_path_from_id(s2_target)
            s2_dvec = get_libri_full_path_from_id(s2_dvec)
            us_dvec = get_libri_full_path_from_id(us_dvec)
            mix_wrapper(hp, args, audio, num, s1_dvec, s1_target, s2_target, s2_dvec, us_dvec, train=True)

        def prepared_test_wrapper(num):
            #s1_dvec, s1_target, s2 = test_list[num].strip().split(',')
            s1_dvec, s1_target, s2_target, s2_dvec, us_dvec = test_list[num].strip().split(',')
            s1_dvec = get_libri_full_path_from_id(s1_dvec)
            s1_target = get_libri_full_path_from_id(s1_target)
            s2_target = get_libri_full_path_from_id(s2_target)
            s2_dvec = get_libri_full_path_from_id(s2_dvec)
            us_dvec = get_libri_full_path_from_id(us_dvec)
            mix_wrapper(hp, args, audio, num, s1_dvec, s1_target, s2_target, s2_dvec, us_dvec, train=False)

        ##arr = list(range(10**5))
        #arr = list(range(112591))
        #with Pool(cpu_num) as p:
        #    r = list(tqdm.tqdm(p.imap(prepared_train_wrapper, arr), total=len(arr)))

        #arr = list(range(10**3))
        arr = list(range(1352))
        with Pool(20) as p:
            r = list(tqdm.tqdm(p.imap(prepared_test_wrapper, arr), total=len(arr)))
