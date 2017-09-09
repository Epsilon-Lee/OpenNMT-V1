import math
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None, upper_bad_count=0):
        self.last_ppl = None
        self.last_bleu = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.upper_bad_count = upper_bad_count

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, bleu, epoch):
        save_checkpoint = False
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            # self.start_decay = True
        # if self.last_ppl is not None and ppl > self.last_ppl:
        #     self.start_decay = True
            if self.last_bleu is not None and bleu <= self.last_bleu:
                self.bad_count += 1
                if self.bad_count == self.upper_bad_count:
                    self.start_decay = True
                    self.bad_count = 0
            else:
                save_checkpoint = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl
        if bleu > self.last_bleu:
            self.last_bleu = bleu
        self.optimizer.param_groups[0]['lr'] = self.lr
        return save_checkpoint
