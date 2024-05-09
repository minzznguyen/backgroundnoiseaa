import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import os
import sys
sys.path.append("DeepSpeech")
import random
from scipy.signal import butter, lfilter
from scipy.optimize import minimize
from copy import deepcopy
from scipy.optimize import dual_annealing


###########################################################################

# This section of code is credited to:
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
generation_tmp = os.path.exists
os.path.exists = lambda x: True
sample_number = 0
output_dir = sys.argv[3]

class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]

class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v

tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = generation_tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None

from util.text import ctc_label_dense_to_sparse
from tf_logits import compute_mfcc, get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

###########################################################################


def db(audio):
    # if the audio has more than 1 dimension
    if len(audio.shape) > 1:
        maxx = np.max(np.abs(audio), axis=1)
        return 20 * np.log10(maxx) if np.any(maxx != 0) else np.array([0])
    maxx = np.max(np.abs(audio))
    return 20 * np.log10(maxx) if maxx != 0 else np.array([0])

def load_wav(input_wav_file):
    # Load the inputs that we're given
    fs, audio = wav.read(input_wav_file)
    # print(fs)
    assert fs == 16000
    # print('source dB', db(audio))
    return audio

def save_wav(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))
    print('output dB', db(audio))

def save_wav_rec(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))
    print('output dB', db(audio))

def levenshteinDistance(s1, s2): 
    if len(s1) > len(s2):
        s1, s2 = s2, s1
        
    distances = range(len(s1) + 1) 
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]



###########################################################################


###########################################################################
class Genetic():
    global output_dir
    def __init__(self, bg_noise_files,input_wave_file, output_wave_file, target_phrase):
        self.pop_size = 100
        self.elite_size = 10
        self.mutation_p = 0.005
        self.noise_stdev = 40
        self.noise_threshold = 1
        self.mu = 0.9
        self.alpha = 0.001
        self.max_iters = 100
        self.num_points_estimate = 100
        self.delta_for_gradient = 100
        self.delta_for_perturbation = 1e3
        self.bg_noise_audio_sets = []
        self.counter = 0
        self.input_wave_file = input_wave_file
        self.output_dir = output_dir
        ## PROCESSING THE AUDIO FILES
        self.bg_scale = 0.1

        self.input_audio = load_wav(input_wave_file).astype(np.float32)
        for bg_noise_wave_file in os.listdir(bg_noise_files):
            bg_noise_wave_file = bg_noise_files +'/' + bg_noise_wave_file
            # print(bg_noise_wave_file)
            # load background noise
            bg_noise_audio = load_wav(bg_noise_wave_file).astype(np.float32)
            # print("MAX LENGTH OF BG", len(bg_noise_audio))
            # clip background file
            bg_noise_audio = bg_noise_audio[:len(self.input_audio)]
            # print(bg_noise_audio.shape)
            self.bg_noise_audio_sets.append(bg_noise_audio)
        # add background noise to input audio
        self.input_audio_w_bg = self.input_audio 


        # input audio from 1D array to 2D array
        self.pop = np.expand_dims(self.input_audio_w_bg, axis=0)
        # multiply into 100 candidates: (1, 32) x (100, 1) -> (100, 32)
        self.pop = np.tile(self.pop, (self.pop_size, 1))
        self.output_wave_file = output_wave_file
        self.target_phrase = target_phrase
        self.funcs = self.setup_graph(self.pop, np.array([toks.index(x) for x in target_phrase]))

    def setup_graph(self, input_audio_batch, target_phrase): 
        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320 
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)
        
        pass_in = np.clip(input_audio_batch, -2**15, 2**15-1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)
        
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            
            inputs = tf.placeholder(tf.float32, shape=pass_in.shape, name='a')
            len_batch = tf.placeholder(tf.float32, name='b')
            arg2_logits = tf.placeholder(tf.int32, shape=logits_arg2.shape, name='c')
            arg1_dense = tf.placeholder(tf.float32, shape=dense_arg1.shape, name='d')
            arg2_dense = tf.placeholder(tf.int32, shape=dense_arg2.shape, name='e')
            len_seq = tf.placeholder(tf.int32, shape=seq_len.shape, name='f')
            
            logits = get_logits(inputs, arg2_logits)
            target = ctc_label_dense_to_sparse(arg1_dense, arg2_dense, len_batch)
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=len_seq)
            decoded, _ = tf.nn.ctc_greedy_decoder(logits, arg2_logits, merge_repeated=True)
            
            sess = tf.Session()
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, "models/session_dump")
            
        func1 = lambda a, b, c, d, e, f: sess.run(ctcloss, 
            feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d, arg2_dense: e, len_seq: f})
        func2 = lambda a, b, c, d, e, f: sess.run([ctcloss, decoded], 
            feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d, arg2_dense: e, len_seq: f})
        return (func1, func2)

    def getctcloss(self, input_audio_batch, target_phrase, decode=False):
        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320 
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)
        
        pass_in = np.clip(input_audio_batch, -2**15, 2**15-1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)
        if decode:
            return self.funcs[1](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
        else:
            return self.funcs[0](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
        
    def get_fitness_score(self, input_audio_batch, target_phrase, input_audio, classify=False):
        target_enc = np.array([toks.index(x) for x in target_phrase]) 
        if classify:
            ctcloss, decoded = self.getctcloss(input_audio_batch, target_enc, decode=True)
            all_text = "".join([toks[x] for x in decoded[0].values]) 
            index = len(all_text) // input_audio_batch.shape[0] 
            final_text = all_text[:index]
        else:
            ctcloss = self.getctcloss(input_audio_batch, target_enc)
        score = -ctcloss
        if classify:
            return (score, final_text) 
        return score, -ctcloss

    # Only get the loss so that we can run optimizatiion function
    # Input: vector of weight
    # Ouput: Loss score of the original file adding the weighted background noise
    def get_only_loss(self, x):
        # in run
        # change in to scalable way
        input_audio_w_bg =  deepcopy(self.input_audio)
        for i, element in enumerate(x):
            input_audio_w_bg += self.bg_noise_audio_sets[i]*x[i]
            
        pop = np.expand_dims(input_audio_w_bg, axis=0)
        pop = np.tile(pop, (self.pop_size, 1))
        
        # in getctcloss
        pop_scores, predicted = self.get_fitness_score(pop, self.target_phrase, input_audio_w_bg, classify=True)
        ctc = pop_scores
        elite_ind = np.argsort(pop_scores)[-self.elite_size:]
        elite_pop, elite_pop_scores, elite_ctc = pop[elite_ind], pop_scores[elite_ind], ctc[elite_ind]
        
        # print out the result
        print(f"**************************** RESULT  {self.counter}****************************")
        print('Sum: ' + str(sum(x)))
        print('Loss: '+str(-elite_ctc[-1]))
        print('Predicted: '+ predicted)
        global sample_number

        # Save file each 10 iteration
        if self.counter % 10 == 0:
            with open(f'{self.output_dir}_{sample_number}/rec.txt', 'a') as file:
                file.write(f"Loss: {-elite_ctc[-1]}\n")
                temp = ''
                for el in list(x):
                    temp += ' '+str(el) 
                file.write(temp)
                file.write("\n\n")
            
            best_pop = np.tile(np.expand_dims(elite_pop[-1], axis=0), (100, 1))
            _, best_text = self.get_fitness_score(best_pop, self.target_phrase, input_audio_w_bg, classify=True)
            with open(f'{self.output_dir}_{sample_number}/rec.csv', 'a') as file:
                file.write(f"{self.counter},{best_text},{levenshteinDistance(best_text, self.target_phrase)},{round(levenshteinDistance(best_text, self.target_phrase)/len(self.target_phrase),2)}\n")
               
            save_wav_rec(elite_pop[-1], f"{self.output_dir}_{sample_number}/{self.counter}_{-elite_ctc[-1]}_{best_text}.wav")

        self.counter += 1
        
        # go to next audio file after 1000 iterations
        if self.counter == 1000:
            sample_number += 1
            main(sample_number)
        return -elite_ctc[-1]
    
   
    # used to be run    
    def run(self, log=None):
        global sample_number
        print("SAMPLE NUMBER: ", sample_number)
        def constraint(x):
            return sum(x) - 0.5
        
        ## CHANGE LENGTH
        self.bg_noise_audio_sets  = self.bg_noise_audio_sets[:200]
        len_bg = len(self.bg_noise_audio_sets)
        print("NUMBER OF BACKGROUND NOISES:",  len_bg)


        ####### Execution
        if log is not None:
            log.write('target phrase: ' + self.target_phrase + '\n')
            log.write('itr, corr, lev dist \n')
        # bound of each weight
        bounds =[(0, 0.02)]*len_bg
        options = {'eps': 1e-5}

        
        ### Run the algorithm
        ### Use recursion and keep track the number of iteration to run next file, to control the number of iteration run
        ans = dual_annealing(self.get_only_loss, bounds)
        ###
        return
    
def main(z):
    global output_dir
    directory = output_dir+'_' + str(z)
    file_name = "rec.txt"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)

    with open(file_path, "w") as file:
        file.write("")


    bg_noise_files = sys.argv[1]
    inp_wav_dir = sys.argv[2]
    print(os.path.join(os.getcwd(), f'/{inp_wav_dir}'))
    reti = 0
    # remember all characters have to be in toks
    # commands store all filenames
    commands = os.listdir(os.path.join(os.getcwd(), inp_wav_dir))
    commands.sort()

    # NEED MODIFICATION FOR EACH INPUT AUDIO FILES
    # orginal file_name : target phrase
    d = {
        "check_the_mail_please": "check the sale please",
        "heat_the_oven_to_three_hundred_fifty": "beat the oven to three hundred fifty",
        "mute_the_audio_now": "mute the audio sound",
        "open_the_garage_door":"open the garage floor",
    }

    # d = {
    #     "close_the_front_door": "lose the front door",
    #     "heat_the_oven_to_three_hundred_fifty": "beat the oven to three hundred fifty",
    #     "mute_the_audio_now": "mute the audio sound",
    #     "open_the_garage_door":"open the garage floor",
    # }

    inp_wav_file = commands[z]
    target = d[inp_wav_file[:-4]]
    reti += 1
    print('ITER:', reti)
    temp = inp_wav_file
    inp_wav_file = f'{inp_wav_dir}/{inp_wav_file}'
    out_wav_file = temp[:-4] + '_adv.wav'
    log_file = temp[:-4] + '_log.txt'

    log_file = "out/" + log_file
    out_wav_file = "out/" + out_wav_file
    print('target phrase:', target)
    print('source file:', inp_wav_file)

    g = Genetic(bg_noise_files, inp_wav_file, out_wav_file, target)
    g.run()



# bg_noise_files (args[1]): input audio directory
# inp_wav_dir (args[2]): Original audio file dir
# output_dir (args[3]): Output directory

main(sample_number)
        


