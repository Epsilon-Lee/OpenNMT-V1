"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""

import torch
import torch.nn as nn
import math

class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def clearMask(self):
        self.mask = None

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention: inner product, essentially bilinear attention
        # attn is un-normalized
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL x 1 => batch x sourceL

        # Logging
        # if self.mask is not None:
        #     print 'mask.size:', type(self.mask), self.mask.size(), self.mask
        #     print 'attn.data.size', attn.data.size()

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = self.sm(attn) # normalization: batch x sourceL
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1) #  batch x 2dim (1000)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn
