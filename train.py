from __future__ import division

import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
from onmt.BleuCal import fetch_data, BLEU
import opts
import sys

parser = argparse.ArgumentParser(description='train.py')

## Data options
opts.model_opts(parser)
opts.translate_opts(parser)
opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.gpus:
    opt.cuda = opt.gpus[0] > -1
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit

def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches) # seqLen x bz x hz
    targets_split = torch.split(targets, opt.max_generator_batches) # seqLen x bz x vz
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2)) # (seqLen x bz) x hz
        scores_t = generator(out_t) # (seqLen x bz) x vz
        loss_t = crit(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1] # which is the greedy predict under golden trajectory 
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(onmt.Constants.PAD).data).sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward() # gradient will accumulate

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct

# Validation stage after every epoch
# Here, we consider two types of validation methods, 1). Golden based 2). Search based
## 1). Golden based: means use golden target as input, but at every step do greedy predict
def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        batch = data[i][:-1] # exclude original indices
        outputs = model(batch)
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _, num_correct = memoryEfficientLoss(
                outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words, total_num_correct / total_words

## 2). Search based: do beam search and calculate real bleu on dev data
def bleuEval(model, opt, devSrcPath, devTgtPath, dataset):
    assert model is not None
    
    translator = onmt.Translator(opt, model, dataset['dicts']['src'], dataset['dicts']['tgt'])
    srcData, references = fetch_data(devSrcPath, devTgtPath)
    srcBatch, tgtBatch, candidate = [], [], []

    # translate srcData
    start = time.time()
    lenSrcData = len(srcData)
    for i, line in enumerate(srcData):

        # Progress bar
        sys.stdout.write('\r')
        sys.stdout.write("Bleu evaluation: %s" % str(i * 100 / lenSrcData) + '%')
        sys.stdout.flush()

        srcTokens = line.split()
        srcBatch += [srcTokens]

        if (i + 1) % opt.trans_batch_size == 0:
            predBatch, _, _ = translator.translate(srcBatch, tgtBatch)
            for b in range(len(predBatch)):
                candidate += [" ".join(predBatch[b][0]) + '\n']
            srcBatch = []
        elif (i + 1) == lenSrcData:
            predBatch, _, _ = translator.translate(srcBatch, tgtBatch)
            for b in range(len(predBatch)):
                candidate += [" ".join(predBatch[b][0]) + '\n']
            srcBatch = []
        else:
            continue
        
    bleu, precisions, bp = BLEU(candidate, references)
    # Log information
    print str('BLEU: %.2f' % (bleu*100)) + ',', 'precisions:',
    for pr in precisions:
        print str('%.2f' % (pr*100)) + ',',
    print 'BP='+ str(bp)
    print 'Candidate sentences:', len(candidate), 'ref sentences:', len(references[0])

    return bleu


def trainModel(model, trainData, validData, dataset, optim):
    print(model)
    model.train()

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()
    batch_idx = 0
    def trainEpoch(epoch, batch_idx):
        # Logging
        print('In trainEpoch...')
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):
            # Logging
            # print 'i=', i
            batch_idx += 1
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1] # exclude original indices

            model.zero_grad()
            outputs = model(batch)
            targets = batch[1][1:]  # exclude <s> from targets
            loss, gradOutput, num_correct = memoryEfficientLoss(
                    outputs, targets, model.generator, criterion)

            outputs.backward(gradOutput)

            # update the parameters
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][1].data.sum()
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            # Log information
            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed  batch_idx %d" %
                      (epoch, i+1, len(trainData),
                      report_num_correct / report_tgt_words * 100,
                      math.exp(min(report_loss / report_tgt_words, 25)),
                      report_src_words/(time.time() - start),
                      report_tgt_words/(time.time() - start),
                      time.time() - start_time, batch_idx))

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                # start = time.time()

            # Validation
            if (i + 1) % opt.valid_interval == 0 and epoch >= opt.start_decay_at:
            # if i % opt.valid_interval == 0:
                print 'In validation mode...'
                model.eval()
                bleu = bleuEval(model, opt, opt.devSrcPath, opt.devTgtPath, dataset)
                # If not, will bring bug
                model.decoder.attn.clearMask()
                save_checkpoint = optim.updateLearningRate(bleu, epoch)
                # if save_checkpoint:
                #     print 'Saving checkpoint... Bad count:', optim.bad_count, 'Learning rate:', optim.lr
                #     model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
                #     model_state_dict = {k: v.cpu() for k, v in model_state_dict.items() if 'generator' not in k}
                #     # model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
                #     generator_state_dict = model.generator.module.state_dict() if len(opt.gpus) > 1 else model.generator.state_dict()
                #     generator_state_dict = {k: v.cpu() for k, v in generator_state_dict.items()}
                #     # Bug report:

                #     #  (4) drop a checkpoint
                #     checkpoint = {
                #         'model': model_state_dict,
                #         'generator': generator_state_dict,
                #         'dicts': dataset['dicts'],
                #         'opt': opt,
                #         'epoch': epoch,
                #         'optim': optim
                #     }
                #     # torch.save(checkpoint,
                #     #            '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, 100*valid_acc, valid_ppl, epoch))
                #     torch.save(checkpoint, '%s_bleu_%.2f_e%d.pt' % (opt.save_model, 100*bleu, epoch))
                #     print 'Done'
                # else:
                #     print 'Not saving checkpoint... Bad count:', optim.bad_count, 'Learning rate:', optim.lr
                model.train()

        return total_loss / total_words, total_num_correct / total_words

    print('Start training...')
    
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch, batch_idx)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % (train_acc*100))
        print('Learning rate: %f' % optim.lr)


def main():

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    dict_checkpoint = opt.train_from if opt.train_from else opt.train_from_state_dict
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    encoder = onmt.Models.Encoder(opt, dicts['src'])
    decoder = onmt.Models.Decoder(opt, dicts['tgt'])

    # Not in `Model.py` but here, create a READOUT module: generator which is a 
    # LogVocOut = log Softmax (W x h_t + b)
    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, dicts['tgt'].size()),
        nn.LogSoftmax())

    model = onmt.Models.NMTModel(encoder, decoder)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items() if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s' % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        # initialize parameters
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        # if not set through opt.pre_word_vecs_enc, opt.pre_word_vec_dec
        # the following will not work
        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim,
            opt.learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            upper_bad_count = opt.upper_bad_count
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
