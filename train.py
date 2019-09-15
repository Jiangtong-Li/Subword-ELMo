# /home/lee/miniconda3/envs/elmo_pytorch/bin/python 
# -*- encoding = utf-8 -*-
# Author: Jiangtong Li

from typing import Callable, Optional, Tuple, List
from overrides import overrides
import os
import gc
import logging
import sys

from config import Config
from model import Bilm3
from data import SubwordBPEBatcher

import numpy as np

import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import sentencepiece as spm

torch.backends.cudnn.enabled = True

opt = Config()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logfile = opt.logfile
fh = logging.FileHandler(logfile)
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
logger.info('logfile = {}'.format(logfile))

def train(opt):
    # print parampeter
    logger.info("LSTM Parameter")
    logger.info("vocab_size: {}".format(opt.vocab_size))
    logger.info("cell_clips: {}".format(opt.cell_clips))
    logger.info("use_skip_connections: {}".format(opt.use_skip_connections))
    logger.info("num_layers: {}".format(opt.num_layers))
    logger.info("proj_clip: {}".format(opt.proj_clip))
    logger.info("projection_dim: {}".format(opt.projection_dim))
    logger.info("cell_dim: {}".format(opt.dim))
    logger.info("drop_out_p: {}".format(opt.drop_out_p))

    logger.info("CNN Parameter")
    logger.info("subword_embedding_size: {}".format(opt.subword_embedding_size))
    logger.info("subword_fliter: {}".format(opt.subword_fliter))
    logger.info("n_subword_fliter: {}".format(opt.n_subword_fliter))
    logger.info("n_subwords: {}".format(opt.n_subwords))
    logger.info("max_subwords_per_word: {}".format(opt.max_subwords_per_word))
    logger.info("cnn_activation: {}".format(opt.cnn_activation))

    logger.info("Highway Parameter")
    logger.info("n_highway: {}".format(opt.n_highway))

    logger.info("Training Parameter")
    logger.info("mode: {}".format(opt.mode))
    logger.info("reg_lambda: {}".format(opt.reg_lambda))
    logger.info("lr: {}".format(opt.lr))
    logger.info("use_gpu: {}".format(opt.use_gpu))
    logger.info("epoch: {}".format(opt.epoch))
    logger.info("batch_size: {}".format(opt.batch_size))
    logger.info("batch_size_test: {}".format(opt.batch_size_test))
    logger.info("random_shuffle: {}".format(opt.random_shuffle))
    logger.info("max_length: {}".format(opt.max_length))

    logger.info("Word Path Parameter")
    logger.info("train_file_raw: {}".format(opt.train_file_raw))
    logger.info("subwords_file: {}".format(opt.subwords_file))
    logger.info("words_file: {}".format(opt.words_file))
    logger.info("granularity: {}".format(opt.granularity))
    # load data
    train_file_raw = opt.train_file_raw
    subwords_file = opt.subwords_file
    words_file = opt.words_file
    data = SubwordBPEBatcher(subword_pkl = subwords_file, \
                             word_pkl = words_file, \
                             raw_file = train_file_raw, \
                             batch_size = opt.batch_size, \
                             word_length = opt.max_subwords_per_word, \
                             shuffle = opt.random_shuffle, 
                             bpe_model = opt.bpe_model)
    data_eval = SubwordBPEBatcher(subword_pkl = subwords_file, \
                                  word_pkl = words_file, \
                                  raw_file = train_file_raw, \
                                  batch_size = opt.batch_size_test, \
                                  word_length = opt.max_subwords_per_word, \
                                  bpe_model = opt.bpe_model)
    logger.info("Successfully load the data")
    # get model
    logger.info("The actual subword size is %d" %opt.n_subwords)
    # load pretrained model
    if opt.model_path:
        Model = torch.load(opt.model_path)
        logger.info('Load pretrained model from {}'.format(opt.model_path))
    else:
        Model = Bilm3(opt)
    logger.info(Model)
    #optimizer = torch.optim.SGD(Model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    logger.info("The model has been build")
    # load model to gpu
    if opt.use_gpu:
        Model.cuda()
        criterion.cuda()
        logger.info("Cuda is available")
        if opt.optim_path:
            optimizer = torch.load(opt.optim_path)
            logger.info('Load optimizer from {}'.format(opt.optim_path))
        else:
            optimizer = torch.optim.Adagrad(Model.parameters(), lr=opt.lr, initial_accumulator_value=1.0)
    else:
        if opt.optim_path:
            optimizer = torch.load(opt.optim_path)
            logger.info('Load optimizer from {}'.format(opt.optim_path))
        else:
            optimizer = torch.optim.Adagrad(Model.parameters(), lr=opt.lr, initial_accumulator_value=1.0)
    # training process
    logger.info('Begin Train')
    for epoch in range(opt.start_epoch,opt.epoch):
        torch.cuda.empty_cache()
        loss_f_all = []
        loss_b_all = []
        ppl_f_all = []
        ppl_b_all = []
        iteration = 0
        Model.train()
        for data_batch in data:
            iteration += 1
            # applied_idx: [batch_size, seq_length, subword_length] 
            # raw_idx: [batch_size, seq_length]
            applied_idx, length, raw_idx, mask = data_batch

            input = Variable(torch.Tensor(applied_idx).long().transpose(1,0)).contiguous()
            mask = Variable(torch.Tensor(mask).float()).contiguous()
            target = Variable(torch.Tensor(raw_idx).long()).contiguous()

            if opt.use_gpu:
                input = input.cuda()
                mask = mask.cuda()
                target = target.cuda()

            res = Model(input, length, mask)
            size = list(target.size())
            forward_batch_dis, backward_batch_dis = res
            loss_f_batch = criterion(forward_batch_dis[:, :-1].contiguous().view(size[0]*(size[1]-1), -1), 
                                     target[:, 1:].contiguous().view(size[0]*(size[1]-1)))
            loss_b_batch = criterion(backward_batch_dis[:, 1:].contiguous().view(size[0]*(size[1]-1), -1), 
                                     target[:, :-1].contiguous().view(size[0]*(size[1]-1)))
            loss_f_batch.backward(retain_graph=True)
            loss_b_batch.backward()
            if iteration%opt.n_batch == 0 and iteration:
                optimizer.step()
                optimizer.zero_grad()
            loss_f_batch_word = loss_f_batch.item()
            loss_b_batch_word = loss_b_batch.item()
            ppl_f_batch_word = np.exp(loss_f_batch_word)
            ppl_b_batch_word = np.exp(loss_b_batch_word)
            loss_f_all.append(loss_f_batch_word)
            loss_b_all.append(loss_b_batch_word)
            ppl_f_all.append(ppl_f_batch_word)
            ppl_b_all.append(ppl_b_batch_word)
            if(iteration%5000==0):
                torch.cuda.empty_cache()
                Ave_f_loss = sum(loss_f_all)/len(loss_f_all)
                Ave_f_ppl = sum(ppl_f_all)/len(ppl_f_all)
                Ave_b_loss = sum(loss_b_all)/len(loss_b_all)
                Ave_b_ppl = sum(ppl_b_all)/len(ppl_b_all)
                Ave_loss = (Ave_f_loss + Ave_b_loss)/2
                Ave_ppl = (Ave_f_ppl + Ave_b_ppl)/2
                logger.info('Epoch %d, Iteration: %d(train):\n\tAve_f_loss: %.3f, Ave_f_ppl: %.3f, \
                               \n\tAve_b_loss: %.3f, Ave_b_ppl: %.3f,\n\tAve_loss: %.3f, Ave_ppl: %.3f' 
                              %(epoch, iteration, Ave_f_loss, Ave_f_ppl, Ave_b_loss, Ave_b_ppl, Ave_loss, Ave_ppl))
                loss_f_all = []
                loss_b_all = []
                ppl_f_all = []
                ppl_b_all = []
            if(iteration%100000==0):
                torch.cuda.empty_cache()
                torch.save(Model, '%sepoch_%s_main_bpe_%s.pth_tmp' %(opt.model_prefix, epoch, opt.granularity))
                torch.save(optimizer, '%sepoch_%s_main_bpe_opt_%s.pth_tmp' %(opt.model_prefix, epoch, opt.granularity))
        Ave_f_loss = sum(loss_f_all)/len(loss_f_all)
        Ave_f_ppl = sum(ppl_f_all)/len(ppl_f_all)
        Ave_b_loss = sum(loss_b_all)/len(loss_b_all)
        Ave_b_ppl = sum(ppl_b_all)/len(ppl_b_all)
        Ave_loss = (Ave_f_loss + Ave_b_loss)/2
        Ave_ppl = (Ave_f_ppl + Ave_b_ppl)/2
        logger.info('Epoch %d, Iteration: %d(train):\n\tAve_f_loss: %.3f, Ave_f_ppl: %.3f, \
                       \n\tAve_b_loss: %.3f, Ave_b_ppl: %.3f,\n\tAve_loss: %.3f, Ave_ppl: %.3f' 
                      %(epoch, iteration, Ave_f_loss, Ave_f_ppl, Ave_b_loss, Ave_b_ppl, Ave_loss, Ave_ppl))
        torch.save(Model, '%sepoch_%s_main_bpe_%s.pth' %(opt.model_prefix, epoch, opt.granularity))
        torch.save(optimizer, '%sepoch_%s_main_bpe_opt_%s.pth' %(opt.model_prefix, epoch, opt.granularity))
        loss_f_all = []
        loss_b_all = []
        ppl_f_all = []
        ppl_b_all = []
        # Test step
        iteration = 0
        Model.eval()
        for data_batch in data_eval:
            iteration += 1
            # applied_idx: [batch_size, seq_length, subword_length] 
            # raw_idx: [batch_size, seq_length]
            applied_idx, length, raw_idx, mask = data_batch
            input = Variable(torch.Tensor(applied_idx).long().transpose(1,0)).contiguous()
            mask = Variable(torch.Tensor(mask).float()).contiguous()
            target = Variable(torch.Tensor(raw_idx).long()).contiguous()
            if opt.use_gpu:
                input = input.cuda()
                mask = mask.cuda()
                target = target.cuda()
            res = Model(input, length, mask)
            size = list(target.size())
            forward_batch_dis, backward_batch_dis = res
            # loss = loss_f + loss_b
            loss_f_batch = criterion(forward_batch_dis[:, :-1].contiguous().view(size[0]*(size[1]-1), -1), 
                                     target[:, 1:].contiguous().view(size[0]*(size[1]-1)))
            loss_b_batch = criterion(backward_batch_dis[:, 1:].contiguous().view(size[0]*(size[1]-1), -1), 
                                     target[:, :-1].contiguous().view(size[0]*(size[1]-1)))
            loss_f_batch_word = loss_f_batch.item()
            loss_b_batch_word = loss_b_batch.item()
            ppl_f_batch_word = np.exp(loss_f_batch_word)
            ppl_b_batch_word = np.exp(loss_b_batch_word)
            loss_f_all.append(loss_f_batch_word)
            loss_b_all.append(loss_b_batch_word)
            ppl_f_all.append(ppl_f_batch_word)
            ppl_b_all.append(ppl_b_batch_word)
            if(iteration%200000==0):
                break
        Ave_f_loss = sum(loss_f_all)/len(loss_f_all)
        Ave_f_ppl = sum(ppl_f_all)/len(ppl_f_all)
        Ave_b_loss = sum(loss_b_all)/len(loss_b_all)
        Ave_b_ppl = sum(ppl_b_all)/len(ppl_b_all)
        Ave_loss = (Ave_f_loss + Ave_b_loss)/2
        Ave_ppl = (Ave_f_ppl + Ave_b_ppl)/2
        logger.info('Epoch %d(test):\n\tAve_f_loss: %.3f, Ave_f_ppl: %.3f, \
                       \n\tAve_b_loss: %.3f, Ave_b_ppl: %.3f,\n\tAve_loss: %.3f, Ave_ppl: %.3f' 
                      %(epoch, Ave_f_loss, Ave_f_ppl, Ave_b_loss, Ave_b_ppl, Ave_loss, Ave_ppl))


if __name__ == '__main__':
    train(opt)
