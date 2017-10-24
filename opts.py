import argparse

def model_opts(parser):

	parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
	parser.add_argument('-save_model', default='model',
	                    help="""Model filename (the model will be saved as
	                    <save_model>_epochN_PPL.pt where PPL is the
	                    validation perplexity""")
	parser.add_argument('-train_from_state_dict', default='', type=str,
	                    help="""If training from a checkpoint then this is the
	                    path to the pretrained model's state_dict.""")
	parser.add_argument('-train_from', default='', type=str,
	                    help="""If training from a checkpoint then this is the
	                    path to the pretrained model.""")

	parser.add_argument('-devSrcPath', default='', type=str,
	                    help="""Dev. source data path""")
	parser.add_argument('-devTgtPath', default='', type=str,
	                    help="""Dev. target data path, might be a directory which
	                    contains multiple reference files.""")

	## Model options

	parser.add_argument('-layers', type=int, default=2,
	                    help='Number of layers in the LSTM encoder/decoder')
	parser.add_argument('-rnn_size', type=int, default=500,
	                    help='Size of LSTM hidden states')
	parser.add_argument('-word_vec_size', type=int, default=500,
	                    help='Word embedding sizes')
	parser.add_argument('-input_feed', type=int, default=1,
	                    help="""Feed the context vector at each time step as
	                    additional input (via concatenation with the word
	                    embeddings) to the decoder.""")
	# parser.add_argument('-residual',   action="store_true",
	#                     help="Add residual connections between RNN layers.")
	parser.add_argument('-brnn', action='store_true',
	                    help='Use a bidirectional encoder')
	parser.add_argument('-brnn_merge', default='concat',
	                    help="""Merge action for the bidirectional hidden states:
	                    [concat|sum]""")

	## Optimization options

	parser.add_argument('-batch_size', type=int, default=64,
	                    help='Maximum batch size')
	parser.add_argument('-max_generator_batches', type=int, default=32,
	                    help="""Maximum batches of words in a sequence to run
	                    the generator on in parallel. Higher is faster, but uses
	                    more memory.""")
	parser.add_argument('-epochs', type=int, default=30,
	                    help='Number of training epochs')
	parser.add_argument('-start_epoch', type=int, default=1,
	                    help='The epoch from which to start')
	parser.add_argument('-param_init', type=float, default=0.1,
	                    help="""Parameters are initialized over uniform distribution
	                    with support (-param_init, param_init)""")
	parser.add_argument('-optim', default='sgd',
	                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
	parser.add_argument('-max_grad_norm', type=float, default=5,
	                    help="""If the norm of the gradient vector exceeds this,
	                    renormalize it to have the norm equal to max_grad_norm""")
	parser.add_argument('-dropout', type=float, default=0.3,
	                    help='Dropout probability; applied between LSTM stacks.')
	parser.add_argument('-curriculum', action="store_true",
	                    help="""For this many epochs, order the minibatches based
	                    on source sequence length. Sometimes setting this to 1 will
	                    increase convergence speed.""")
	parser.add_argument('-extra_shuffle', action="store_true",
	                    help="""By default only shuffle mini-batch order; when true,
	                    shuffle and re-assign mini-batches""")

	#learning rate
	parser.add_argument('-learning_rate', type=float, default=1.0,
	                    help="""Starting learning rate. If adagrad/adadelta/adam is
	                    used, then this is the global learning rate. Recommended
	                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
	parser.add_argument('-learning_rate_decay', type=float, default=0.5,
	                    help="""If update_learning_rate, decay learning rate by
	                    this much if (i) perplexity does not decrease on the
	                    validation set or (ii) epoch has gone past
	                    start_decay_at""")
	parser.add_argument('-start_decay_at', type=int, default=1,
	                    help="""Start decaying every epoch after and including this
	                    epoch""")
	parser.add_argument('-upper_bad_count', type=int, default=2,
	                    help="""upper bad_count which will trigger learning rate decay""")
	#pretrained word vectors

	parser.add_argument('-pre_word_vecs_enc',
	                    help="""If a valid path is specified, then this will load
	                    pretrained word embeddings on the encoder side.
	                    See README for specific formatting instructions.""")
	parser.add_argument('-pre_word_vecs_dec',
	                    help="""If a valid path is specified, then this will load
	                    pretrained word embeddings on the decoder side.
	                    See README for specific formatting instructions.""")

	# GPU
	parser.add_argument('-gpus', default=[], nargs='+', type=int,
	                    help="Use CUDA on the listed devices.")

	parser.add_argument('-log_interval', type=int, default=50,
	                    help="Print stats at this interval.")

	parser.add_argument('-valid_interval', type=int, default=1000,
	                    help="Do bleu score validation at this interval")


def translate_opts(parser):

	# parser.add_argument('-model', required=True,
	#                     help='Path to model .pt file')
	# parser.add_argument('-src',   required=True,
	#                     help='Source sequence to decode (one line per sequence)')
	# parser.add_argument('-tgt',
	#                     help='True target sequence (optional)')
	parser.add_argument('-model', type=str,
	                    help='Path to model .pt file')
	parser.add_argument('-src', type=str,
	                    help='Source sequence to decode (one line per sequence)')
	parser.add_argument('-tgt',
	                    help='True target sequence (optional)')
	parser.add_argument('-output', type=str,
	                    help="""Path to output the predictions (each line will
	                    be the decoded sequence""")
	parser.add_argument('-beam_size',  type=int, default=5,
	                    help='Beam size')
	parser.add_argument('-trans_batch_size', type=int, default=30,
	                    help='Batch size')
	parser.add_argument('-max_sent_length', type=int, default=100,
	                    help='Maximum sentence length.')
	parser.add_argument('-replace_unk', action="store_true",
	                    help="""Replace the generated UNK tokens with the source
	                    token that had the highest attention weight. If phrase_table
	                    is provided, it will lookup the identified source token and
	                    give the corresponding target token. If it is not provided
	                    (or the identified source token does not exist in the
	                    table) then it will copy the source token""")
	# parser.add_argument('-phrase_table',
	#                     help="""Path to source-target dictionary to replace UNK
	#                     tokens. See README.md for the format of this file.""")
	parser.add_argument('-verbose', action="store_true",
	                    help='Print scores and predictions for each sentence')
	parser.add_argument('-n_best', type=int, default=1,
	                    help="""If verbose is set, will output the n_best
	                    decoded sentences""")

	parser.add_argument('-gpu', type=int, default=-1,
	                    help="Device to run on")
