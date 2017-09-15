import torch
import torch.nn as nn
import onmt
from onmt.BleuCal import fetch_data
import sys

if torch.cuda.is_available():
	torch.cuda.set_device(3)

checkpoint = torch.load('../Models/V1_IWSLT_Models/de2en_30k_bz64_bc5_bleu_26.06_e24.pt')
opt = checkpoint['opt']
# del(checkpoint)
opt.cuda = True

srcData, references = fetch_data('IWSLT/test.de.small.tok', 'IWSLT/test.en.small.tok')

encoder = onmt.Models.Encoder(opt, checkpoint['dicts']['src'])
decoder = onmt.Models.Decoder(opt, checkpoint['dicts']['tgt'])

model = onmt.Models.NMTModel(encoder, decoder)
model.load_state_dict(checkpoint['model'])

generator = nn.Sequential(
	nn.Linear(opt.rnn_size, checkpoint['dicts']['tgt'].size()),
	nn.LogSoftmax())

model.generator = generator
model.cuda()
opt.model = '../Models/V1_IWSLT_Models/de2en_30k_bz64_bc5_bleu_26.06_e24.pt'

translator = onmt.Translator(opt, model, checkpoint['dicts']['src'], checkpoint['dicts']['tgt'])
srcBatch, tgtBatch, candidate = [], [], []
lenSrcData = len(srcData)
for i, line in enumerate(srcData):
	
	sys.stdout.write('\r')
	sys.stdout.write("%s" % (str(i) + ' of ' + str(lenSrcData)))
	sys.stdout.flush()

	srcTokens = line.split()
	srcBatch += [srcTokens]
	if (i + 1) % opt.trans_batch_size == 0:
		predBatch, _, _ = translator.translate(srcBatch, tgtBatch)
		print 'predBatch:', len(predBatch)
		for b in range(len(predBatch)):
			candidate += [" ".join(predBatch[b][0]) + '\n']
		srcBatch = []
	elif (i + 1) == lenSrcData:
		predBatch, _, _ = translator.translate(srcBatch, tgtBatch)
		print 'predBatch:', len(predBatch)
		for b in range(len(predBatch)):
			candidate += [" ".join(predBatch[b][0]) + '\n']
		srcBatch = []
	else:
		continue

print 'candidate length:', len(candidate)
print 'referece length', len(references[0])