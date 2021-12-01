from visdial.utils import DynamicRNN

import torch
from torch import nn
import numpy as np


class DiscriminativeDecoder(nn.Module):
  def __init__(self, hparams, vocabulary):
    super().__init__()
    self.hparams = hparams
    self.voc = vocabulary
    self.word_embed = nn.Embedding(
      len(vocabulary),
      hparams.word_embedding_size,
      padding_idx=vocabulary.PAD_INDEX,
    )
    self.option_rnn = nn.LSTM(
      hparams.word_embedding_size,
      hparams.lstm_hidden_size,
      hparams.lstm_num_layers,
      batch_first=True,
      dropout=hparams.dropout,
    )

    # Options are variable length padded sequences, use DynamicRNN.
    self.option_rnn = DynamicRNN(self.option_rnn)
    self.log_softmax = nn.LogSoftmax(dim=-1)
  def tri_loss(self, batch, scores, similar_output, k=2):
    scores = scores.view(-1, 100)
    batch_size, num = scores.size()
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)
    pos_label = ranked_idx[:, :k]

    # neg_label = ranked_idx[:, -k:]
    gt = batch["ans_ind"].view(-1)
    # pos_label = torch.cat((pos_label, gt), dim=-1)
    # label = torch.zeros(batch_size, num).long().cuda()
    # for i in range(batch_size):
    #   for j in range(k+1):
    #     label[i][pos_label[j]] = 1
    # tri_loss = batch_hard_triplet_loss(label, similar_output, 10)
    anchor = torch.zeros(batch_size, k, 512).cuda()
    pos_embed = torch.zeros(batch_size, k, 512).cuda()
    neg_embed = torch.zeros(batch_size, k, 512).cuda()
    similar_output = similar_output.view(-1, 100, 512)
    for i in range(batch_size):
      for j in range(k):
        neg_id = np.random.randint(k, 100, size=1)
        pos_embed[i][j][:] = similar_output[i][pos_label[i][j]][:]
        # neg_embed[i][j][:] = similar_output[i][neg_label[i][j]][:]
        neg_embed[i][j][:] = similar_output[i][ranked_idx[i][neg_id]][:]
        anchor[i][j][:] = similar_output[i][gt[i]][:]
    tri_loss = torch.nn.functional.triplet_margin_loss(anchor.view(-1, 512), pos_embed.view(-1, 512), neg_embed.view(-1, 512))
    return tri_loss

  def cosine_distance(self, a, b):
    if a.shape != b.shape:
      raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
      a_norm = np.linalg.norm(a)
      b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
      a_norm = np.linalg.norm(a, axis=1, keepdims=True)
      b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
      raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T) / (a_norm * b_norm)
    return similiarity

  def findSimilarOption(self, batch, opt_embed,  K = 2):
    count = 0
    answer = batch['ans_ind']
    batch_size, num_round = answer.size()
    opt_embed = opt_embed.view(batch_size, num_round, 100, 20, 300)

    opt_embed = torch.sum(opt_embed, dim=-2)
    similar_matrix = torch.zeros(batch_size, num_round, 100).cuda()
    fake_label = torch.zeros(batch_size, num_round, 100).cuda()
    for i in range(batch_size):
      for j in range(num_round):
        ans_id = answer[i][j]
        ans_embed = opt_embed[i][j][ans_id][:]

        for k in range(100):
          similar_matrix[i][j][k] = torch.cosine_similarity(ans_embed, opt_embed[i][j][k][:], dim=0)

        rank_id = torch.argsort(similar_matrix[i][j])

        # if rank_id[-1] != ans_id:
        #
        #   option  = batch['opt']
        #   ans_word = option[i][j][ans_id]
        #   option_word = option[i][j][rank_id[-1]]
        #   opt_word = ''
        #   for item in option_word:
        #     if item != 0:
        #       word = self.voc.index2word[item.item()]
        #       opt_word = opt_word + ' ' + word
        #   answer_word = ''
        #   for item in ans_word:
        #     if item != 0:
        #       word = self.voc.index2word[item.item()]
        #       answer_word = answer_word + ' ' + word
        #
        #
        #   # print(ans_id)
        #   print(answer_word)
        # #   # # print(rank_id[-1])
        # #   print(rank_id[-1])
        #   print(opt_word)
        # #   # print('====wrong=====')
        # #   # input()
        #   count += 1
        for s in range(1, K+1):
          fake_id = rank_id[-s]
          if similar_matrix[i][j][fake_id] > 0.85:
            fake_label[i][j][fake_id] = similar_matrix[i][j][fake_id]
    return fake_label

  def forward(self, encoder_output, batch):
    """Given `encoder_output` + candidate option sequences, predict a score
    for each option sequence.

    Parameters
    ----------
    encoder_output: torch.Tensor
        Output from the encoder through its forward pass.
        (batch_size, num_rounds, lstm_hidden_size)
    """

    options = batch["opt"]
    batch_size, num_rounds, num_options, max_sequence_length = options.size()
    options = options.view(batch_size * num_rounds * num_options, max_sequence_length)

    options_length = batch["opt_len"]
    options_length = options_length.view(batch_size * num_rounds * num_options)

    # Pick options with non-zero length (relevant for test split).
    nonzero_options_length_indices = options_length.nonzero().squeeze()
    nonzero_options_length = options_length[nonzero_options_length_indices]
    nonzero_options = options[nonzero_options_length_indices]

    # shape: (batch_size * num_rounds * num_options, max_sequence_length,
    #         word_embedding_size)
    # FOR TEST SPLIT, shape: (batch_size * 1, num_options,
    #                         max_sequence_length, word_embedding_size)
    nonzero_options_embed = self.word_embed(nonzero_options)

    ##### make fake label ######
    # fake_label = self.findSimilarOption(batch, nonzero_options_embed)

    # shape: (batch_size * num_rounds * num_options, lstm_hidden_size)
    # FOR TEST SPLIT, shape: (batch_size * 1, num_options,
    #                         lstm_hidden_size)
    _, (nonzero_options_embed, _) = self.option_rnn(
      nonzero_options_embed, nonzero_options_length
    )

    options_embed = torch.zeros(
      batch_size * num_rounds * num_options,
      nonzero_options_embed.size(-1),
      device=nonzero_options_embed.device,
    )
    options_embed[nonzero_options_length_indices] = nonzero_options_embed

    # Repeat encoder output for every option.
    # shape: (batch_size, num_rounds, num_options, lstm_hidden_size)
    encoder_output = encoder_output.unsqueeze(2).repeat(
      1, 1, num_options, 1
    )

    # Shape now same as `options`, can calculate dot product similarity.
    # shape: (batch_size * num_rounds * num_options, lstm_hidden_size)
    encoder_output = encoder_output.view(batch_size * num_rounds * num_options, self.hparams.lstm_hidden_size)

    # shape: (batch_size * num_rounds * num_options)
    similar_output = options_embed * encoder_output
    scores = torch.sum(similar_output, 1)
    # shape: (batch_size, num_rounds, num_options)
    scores = scores.view(batch_size, num_rounds, num_options)

    return scores

