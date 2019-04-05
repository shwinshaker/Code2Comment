# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pickle
from collections import Counter
import random
import string
import re
import time
import math
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

from SBT_encode import Encoder as CodeEncoder
from comment_encode import Encoder as CommentEncoder


torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
	computing_device = torch.device("cuda")
	extras = {"num_workers": 1, "pin_memory": True}
	print("CUDA is supported")
else: # Otherwise, train on the CPU
	computing_device = torch.device("cpu")
	extras = False
	print("CUDA NOT supported")


# class CodeCommentDataset(Dataset):
# 	def __init__(self, inputs, target):
# 		self.inputs = inputs
# 		self.target = target

# 	def __len__(self):
# 		# Return the total number of data samples
# 		return len(self.inputs)

# 	def __getitem__(self, ind):
# 		"""Returns one-hot encoded version of the target and labels
# 		"""
# 		data = self.inputs[ind]
# 		label = self.target[ind]

# 		return torch.LongTensor(data),torch.LongTensor(label)


class CodeCommentDataset(Dataset):
	def __init__(self, data):
		self.comments, self.codes = zip(*data)
		self.code_encoder = CodeEncoder()
		self.comment_encoder = CommentEncoder()

	def __len__(self):
		# Return the total number of data samples
		return len(self.comments)

	def __getitem__(self, ind):
		"""Returns one-hot encoded version of the target and labels
		"""
		return (torch.LongTensor(self.code_encoder.encode(self.codes[ind])),
				torch.LongTensor(self.comment_encoder.encode(self.comments[ind])))


def createLoaders(batch_size=1, extras={}, debug=False):
	# load training, validation and test text
	print('-- create data loaders..')
	num_workers = 0
	pin_memory = False
	# If CUDA is available
	if extras:
		num_workers = extras["num_workers"]
		pin_memory = extras["pin_memory"]

	dataLoaders = {}
	for phase in ['train', 'valid', 'test']:
		with open('data/%s.pkl' % phase, 'rb') as f:
			if debug:
				dataset = CodeCommentDataset(pickle.load(f)[:10])
			else:
				dataset = CodeCommentDataset(pickle.load(f))
		dataLoaders[phase] = DataLoader(dataset, batch_size=batch_size,
												 num_workers=num_workers,
												 pin_memory=pin_memory)
	return dataLoaders


class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden, encoder_output):
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden, encoder_output

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_len = 3000

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_len)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(
			self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		enc_out = encoder_outputs.unsqueeze(0)
		weights = attn_weights.unsqueeze(0)[:,:,0:enc_out.size(1)]
		attn_applied = torch.bmm(weights,
								 encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# could also be use to test
def validate_model(encoder, decoder, criterion, loader, SOS_token=None, device=None, verbose=False):
	val_loss = 0
	with torch.no_grad():
		for i, (inputs, targets) in enumerate(loader, 0):
			encoder_hidden = encoder.initHidden()
			# Put the minibatch data in CUDA Tensors and run on the GPU if supported
			input_tensor = torch.LongTensor(inputs[0]).to(device)
			target_tensor = torch.LongTensor(targets[0]).to(device)
			input_length = input_tensor.size(0)
			target_length = target_tensor.size(0)

			encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

			loss = 0

			for ei in range(input_length):
				encoder_output, encoder_hidden = encoder(
					input_tensor[ei], encoder_hidden)
				encoder_outputs[ei] = encoder_output[0, 0]

			# decoder_input = torch.tensor([[SOS_token]], device=device)
			decoder_hidden = encoder_hidden

			# Teacher forcing: Feed the target as the next input
			for di in range(target_length):
				decoder_input = target_tensor[di]  # Teacher forcing
				decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, encoder_outputs)
				loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
				# decoder_input = target_tensor[di]  # Teacher forcing
			val_loss += loss.item() / target_length
		print('Validation Loss: ', val_loss / len(loader))
	return val_loss /len(loader)


def train(input_tensor, target_tensor, encoder, decoder, 
		  encoder_optimizer, decoder_optimizer, criterion,
		  SOS_token=None):

	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(
			input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0, 0]

	# decoder_input = torch.tensor([[SOS_token]], device=device)

	decoder_hidden = encoder_hidden


	# Teacher forcing: Feed the target as the next input
	for di in range(target_length):
		decoder_input = target_tensor[di]  # Teacher forcing
		decoder_output, decoder_hidden, decoder_attention = decoder(
			decoder_input, decoder_hidden, encoder_outputs)
		loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
		# decoder_input = target_tensor[di]  # Teacher forcing

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length


def get_vocab_size(encoder):
	return len(encoder.vocab_dict)


def trainIters(learning_rate=0.001):
	epochs = 1
	plot_train_losses = []
	plot_val_losses = []
	plot_loss_total = 0  # Reset every plot_every
	hidden_size = 256
	print('------- Hypers --------\n'
		  '- epochs: %i\n'
		  '- learning rate: %g\n'
		  '- hidden size: %i\n'
		  '----------------'
		  '' % (epochs, learning_rate, hidden_size))

	# set model
	vocab_size_encoder = get_vocab_size(CodeEncoder())
	vocab_size_decoder = get_vocab_size(CommentEncoder())
	print(vocab_size_encoder)
	print(vocab_size_decoder)
	print('----------------')
	# COMMENT OUT WHEN FIRST TRAINING
	# encoder, decoder = load_model()
	encoder = EncoderRNN(vocab_size_encoder, hidden_size).to(device)
	decoder = AttnDecoderRNN(hidden_size, vocab_size_decoder, dropout_p=0.1).to(device)

	# set training hypers
	criterion = nn.NLLLoss()
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

	# set data
	dataLoaders = createLoaders(extras=extras, debug=True)

	# used for initial input of decoder
	# with open('dicts/comment_dict.pkl', 'rb') as pfile:
	# 	SOS_token = pickle.load(pfile)['<SOS>']
	# since we already prepend <SOS> to the comment, don't think need this in decoder model anymore
	SOS_token = None

	# iteration
	counts = []
	best_val_loss = 100
	for eps in range(1, epochs + 1):
		print('Epoch Number', eps)
		for count, (inputs, targets) in enumerate(dataLoaders['train'], 0):
			inputs = torch.LongTensor(inputs[0])
			targets = torch.LongTensor(targets[0])
			inputs, targets = inputs.to(device), targets.to(device)

			loss = train(inputs, targets,
						 encoder, decoder,
						 encoder_optimizer, decoder_optimizer,
						 criterion, SOS_token=SOS_token)
			plot_loss_total += loss
			# if count != 0 and count % 10 == 0:
			print(count, loss)

		counts.append(eps)
		plot_loss_avg = plot_loss_total / len(dataLoaders['train'])
		plot_train_losses.append(plot_loss_avg)
		val_loss = validate_model(encoder, decoder, criterion, dataLoaders['valid'], SOS_token=SOS_token, device=device)
		if val_loss < best_val_loss:
			save_model(encoder, decoder)
			best_val_loss = val_loss
		plot_val_losses.append(val_loss)
		plot_loss_total = 0
		save_loss(plot_train_losses, plot_val_losses)
	showPlot(counts, plot_train_losses, plot_val_losses)


def save_model(encoder, decoder, type='attn'):
	with open(type+'_encoder1.ckpt', 'wb') as pfile:
		torch.save(encoder, pfile)
	with open(type + '_decoder1.ckpt', 'wb') as pfile:
		torch.save(decoder, pfile)

def load_model(type='attn'):
	with open(type+'_encoder.ckpt', 'rb') as pfile:
		encoder = pickle.load(pfile)
	with open(type + '_decoder.ckpt', 'rb') as pfile:
		decoder = pickle.load(pfile)
	return encoder, decoder

def save_loss(train_loss, val_loss):
	with open('train_loss1.pkl', 'wb') as pfile:
		pickle.dump(train_loss, pfile)
	with open('val_loss1.pkl', 'wb') as pfile:
		pickle.dump(val_loss, pfile)

######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

def showPlot(iter, train_loss, val_loss):
	plt.figure()
	plt.plot(iter, train_loss, '.-', label='train')
	plt.plot(iter, val_loss, '.-', label='val')
	fontsize = 12
	plt.legend(fontsize=fontsize)
	plt.xlabel('epoch', fontsize=fontsize)
	plt.ylabel('loss', fontsize=fontsize)
	plt.savefig('loss.png', fontsize=fontsize)

if __name__ == '__main__':
	trainIters()

######################################################################
#

# evaluateRandomly(encoder1, attn_decoder1)
#
#
# ######################################################################
# # Visualizing Attention
# # ---------------------
# #
# # A useful property of the attention mechanism is its highly interpretable
# # outputs. Because it is used to weight specific encoder outputs of the
# # input sequence, we can imagine looking where the network is focused most
# # at each time step.
# #
# # You could simply run ``plt.matshow(attentions)`` to see attention output
# # displayed as a matrix, with the columns being input steps and rows being
# # output steps:
# #
#
# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "je suis trop froid .")
# plt.matshow(attentions.numpy())
#
#
# ######################################################################
# # For a better viewing experience we will do the extra work of adding axes
# # and labels:
# #
#
# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#
#
# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)
