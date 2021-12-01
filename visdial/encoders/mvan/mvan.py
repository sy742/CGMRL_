import torch
from torch import nn
from torch.nn.functional import normalize
from visdial.utils import DynamicRNN
from .modules import ContextMatching, TopicAggregation, ModalityFusionTopic, ModalityFusionContext, TextAttImage
from .multi_head_attention import *


class REFER_Feature(nn.Module):
    """ This code is modified from Yu-Hsiang Huang's repository
        https://github.com/jadore801120/attention-is-all-you-need-pytorch
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.2):
        super(REFER_Feature, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q, m):
        enc_output, enc_slf_attn = self.slf_attn(q, m, m)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W) # shape [N, out_features]
        batch, N, d = h.size()

        a_input = torch.cat([h.repeat(1, 1, N).view(batch, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2 * self.out_features) # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [N,N,1] -> [N,N]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
class MVANEncoder(nn.Module):
	def __init__(self, hparams, vocabulary):
		super().__init__()
		self.hparams = hparams
		self.word_embed = nn.Embedding(
			len(vocabulary),
			hparams.word_embedding_size,
			padding_idx=vocabulary.PAD_INDEX,
		)
		self.ques_rnn = nn.LSTM(
			hparams.word_embedding_size,
			hparams.lstm_hidden_size,
			hparams.lstm_num_layers,
			batch_first=True,
			dropout=hparams.dropout,
			bidirectional=True
		)
		self.hist_rnn = nn.LSTM(
			hparams.word_embedding_size,
			hparams.lstm_hidden_size,
			hparams.lstm_num_layers,
			batch_first=True,
			dropout=hparams.dropout,
			bidirectional=True
		)
		self.hist_rnn = DynamicRNN(self.hist_rnn)
		self.ques_rnn = DynamicRNN(self.ques_rnn)
		self.context_matching = ContextMatching(self.hparams) # 1) Context Matching
		# self.topic_aggregation = TopicAggregation(self.hparams) # 2) Topic Aggregation
		self.ques_att_image = TextAttImage(self.hparams)
		self.ques_hist_att_image = TextAttImage(self.hparams)
		# Modality Fusion
		# self.modality_fusion_topic = ModalityFusionTopic(self.hparams) # Modality Fusion Topic
		# self.modality_fusion_context = ModalityFusionContext(self.hparams) # Modality Fusion Context
		# self.modality_fusion_ques = ModalityFusionContext(self.hparams)

		# 2048 + 1024 * 2 -> 512
		fusion_size = (hparams.img_feature_size + hparams.lstm_hidden_size * 2)
		self.fusion = nn.Sequential(
			nn.Dropout(p=hparams.dropout_fc),
			nn.Linear(fusion_size, hparams.lstm_hidden_size),
			nn.ReLU()
		)
		# print(self.fusion)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
		# self.refer_feature = REFER_Feature(d_model=2048, d_inner=1024, n_head=4, d_k=1024, d_v=1024, dropout=0.2)
		# self.gat = GraphAttentionLayer(2048, 2048, 0.2, 0.2)
		self.gate_coarse = nn.Sequential(
			nn.Linear(hparams.lstm_hidden_size*2 + hparams.img_feature_size, 1),
					 nn.Sigmoid()
		)
		self.gate_coarse_hard = nn.Sequential(
			nn.Linear(hparams.lstm_hidden_size * 2 + hparams.img_feature_size, 2)
		)
		self.ques_gate = nn.Sequential(
			nn.Linear(hparams.lstm_hidden_size * 2 + hparams.img_feature_size, hparams.img_feature_size + hparams.lstm_hidden_size * 2),
			nn.Sigmoid()
		)
		self.ques_hist_gate = nn.Sequential(
			nn.Linear(hparams.lstm_hidden_size * 2 + hparams.img_feature_size,
					  hparams.img_feature_size + hparams.lstm_hidden_size * 2),
			nn.Sigmoid()
		)
	def forward(self, batch):
		"""Visual Features"""
		img, img_mask = self.init_img(batch) # bs, np, 2048
		_, num_p, img_feat_size = img.size()

		"""Language Features"""
		ques_word_embed, ques_word_encoded, ques_encoded, ques_not_pad, ques_pad = self.init_q_embed(batch)
		hist_word_embed, hist_word_encoded, hist_encoded, hist_not_pad, hist_pad = self.init_h_embed(batch)
		bs, num_r, bilstm = ques_encoded.size()

		"""question features reshape"""
		# ques_word_embed = ques_word_embed.view(bs, num_r, -1, self.hparams.word_embedding_size)
		# ques_word_encoded = ques_word_encoded.view(bs, num_r, -1, bilstm)
		# ques_not_pad = ques_not_pad.view(bs, num_r, -1)
		#
		# """dialog history features reshape"""
		# hist_word_embed = hist_word_embed.view(bs, num_r, -1, self.hparams.word_embedding_size)
		# hist_word_encoded = hist_word_encoded.view(bs, num_r, -1, bilstm)
		# hist_not_pad = hist_not_pad.view(bs, num_r, -1)

		context_matching_feat = []
		context_matching_ques = []
		# topic_aggregation_feat = []
		# coref = torch.zeros(bs, num_r).cuda()
		for c_r in range(num_r):
			"""Context Matching"""
			accu_h_sent_encoded = hist_encoded[:, 0:c_r + 1, :]      # bs, num_r, bilstm
			curr_q_sent_encoded = ques_encoded[:, c_r:(c_r + 1), :]  # bs, 1, bilstm
			context_aware_feat, context_matching_score = self.context_matching(curr_q_sent_encoded, accu_h_sent_encoded)

			# ques_coref = torch.sigmoid(self.coref(curr_q_sent_encoded))
			# coref[:, c_r] = ques_coref.view(-1)
			context_aware_feat = curr_q_sent_encoded + context_aware_feat

			context_matching_feat.append(context_aware_feat)
			context_matching_ques.append(curr_q_sent_encoded)

			"""Topic Aggregation"""
			# curr_q_word_embed = ques_word_embed[:, c_r, :, :]              # bs, sl_q, word_embed_size
			# curr_q_word_encoded = ques_word_encoded[:, c_r, :, :]          # bs, sl_q, bilstm
			# accu_h_word_embed = hist_word_embed[:, 0:(c_r + 1), :, :]      # bs, nr, sl_h, bilstm
			# accu_h_word_encoded = hist_word_encoded[:, 0:(c_r + 1), :, :]  # bs, nr, sl_h, bilstm
			# accu_h_not_pad = hist_not_pad[:, 0:(c_r + 1), :]               # bs, nr, sl_h

			# topic_aware_feat = self.topic_aggregation(curr_q_word_embed, curr_q_word_encoded,
			# 																					accu_h_word_embed, accu_h_word_encoded, accu_h_not_pad,
			# 																					context_matching_score)
			# topic_aggregation_feat.append(topic_aware_feat)
		# (batch, round, dim=2048)
		context_matching = torch.cat(context_matching_feat, dim=1)
		ques_matching = torch.cat(context_matching_ques, dim=1)

		#(batch, round, max_ques_len, dim=600)
		# topic_aggregation = torch.stack(topic_aggregation_feat, dim=1)  # bs, nr, sl_q, lstm

		"""Modality Fusion"""
		# mf_topic_feat = self.modality_fusion_topic(img, topic_aggregation, ques_not_pad)          # topic-view
		ques_att_image = self.ques_att_image(img, ques_matching, img_mask)
		ques_hist_att_image = self.ques_hist_att_image(img, context_matching, img_mask) # context-view

		###### for gate #####
		ques_image = torch.cat((ques_matching, ques_att_image), dim=-1)
		ques_hist_image = torch.cat((context_matching, ques_hist_att_image), dim=-1)


		if self.hparams.hard == True:
			gate_coarse_logits = self.gate_coarse_hard(ques_image)
			# gate_coarse_logits = gate_coarse_logits.view(-1, gate_coarse_logits.size(-1))
			if self.training:
				gate_coarse = torch.nn.functional.gumbel_softmax(gate_coarse_logits, hard=True)
			else:
				index = torch.argmax(gate_coarse_logits, dim=-1)
				gate_coarse = nn.functional.one_hot(index, num_classes=2)
		else:
			gate_coarse = self.gate_coarse(ques_image)

		ques_gate = self.ques_gate(ques_image)
		ques_hist_gate = self.ques_hist_gate(ques_hist_image)

		ques_image_feat = ques_gate*ques_image
		ques_hist_image_feat = ques_hist_gate*ques_hist_image

		# gate_coarse = self.gate_coarse(ques_image_feat)
		# a =  ques_image_feat
		# b = ques_hist_image_feat
		# a_sum = torch.sum(a, dim=-1)
		# b_sum = torch.sum(b, dim=-1)
		# c_sum = a_sum - b_sum
		if self.hparams.hard == True:
			feat_all = gate_coarse[:,:,0].view(gate_coarse.size(0), gate_coarse.size(1),-1)*ques_image_feat + \
					   gate_coarse[:,:, 1].view(gate_coarse.size(0), gate_coarse.size(1),-1)*ques_hist_image_feat
		else:
			feat_all = gate_coarse * ques_image_feat + (1 - gate_coarse) * ques_hist_image_feat
		# feat_all = ques_image_feat + ques_hist_image_feat
		# feat_all = ques_image + ques_hist_image
		#feat_all = ques_hist_image

		#ques_gate = self.ques_gate(mf_context_ques)
		# feat_all = mf_context_feat

		multi_view_fusion = self.fusion(feat_all)

		return multi_view_fusion, gate_coarse

	def area(self, boxes):
		area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
		return area

	def boxlist_iou(self, boxlist1, boxlist2):

		"""Compute the intersection over union of two set of boxes.
		The box order must be (xmin, ymin, xmax, ymax).
		Arguments:
		box1: (BoxList) bounding boxes, sized [N,4].
		box2: (BoxList) bounding boxes, sized [M,4].
		Returns:
		(tensor) iou, sized [N,M].
		"""
		# N = boxlist1.shape[0]
		# M = boxlist2.shape[1]
		area1 = self.area(boxlist1)
		area2 = self.area(boxlist2)
		lt = torch.max(boxlist1[:, None, :2], boxlist2[:, :2])  # [N,M,2]
		rb = torch.min(boxlist1[:, None, 2:], boxlist2[:, 2:])  # [N,M,2]
		wh = (rb - lt).clamp(min=0)  # [N,M,2]
		inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
		iou = inter / (area1[:, None] + area2 - inter)
		return iou


	def init_img(self, batch):
		img = batch['img_feat']
		# bb = batch['bb']
		"""image feature normarlization"""
		if self.hparams.img_norm:
			img = normalize(img, dim=1, p=2)
		mask = (0 != img.abs().sum(-1)).unsqueeze(1)

		# for GAT
		# adj = batch['adj']
		# image_gat = self.gat(img, adj)
		# add self_attention for image
		# img, att = self.refer_feature(img, img)

		return img, mask

	def init_q_embed(self, batch):
		ques = batch['ques']
		bs, nr, sl_q = ques.size()
		lstm = self.hparams.lstm_hidden_size
		"""bs_q, nr_q, sl_q -> bs*nr, sl_q, 1"""
		ques_not_pad = (ques != 0).bool()
		ques_not_pad = ques_not_pad.view(-1, sl_q).unsqueeze(-1)
		ques_pad = (ques == 0).bool()
		ques_pad = ques_pad.view(-1, sl_q).unsqueeze(1)

		ques = ques.view(-1, sl_q)
		ques_word_embed = self.word_embed(ques)
		ques_word_encoded, _ = self.ques_rnn(ques_word_embed, batch['ques_len'])

		loc = batch['ques_len'].view(-1).cpu().numpy() - 1

		# sentence-level encoded
		ques_encoded_forawrd = ques_word_encoded[range(bs *nr), loc,:lstm]
		ques_encoded_backward = ques_word_encoded[:, 0,lstm:]
		ques_encoded = torch.cat((ques_encoded_forawrd, ques_encoded_backward), dim=-1)
		ques_encoded = ques_encoded.view(bs, nr, -1)

		return ques_word_embed, ques_word_encoded, ques_encoded, ques_not_pad, ques_pad

	def init_h_embed(self, batch):
		hist = batch['hist']
		bs, nr, sl_h = hist.size()
		lstm = self.hparams.lstm_hidden_size
		"""bs_q, nr_q, sl_q -> bs*nr, sl_q, 1"""
		hist_not_pad = (hist != 0).bool()
		hist_not_pad = hist_not_pad.view(-1, sl_h).unsqueeze(-1)
		hist_pad = (hist == 0).bool()
		hist_pad = hist_pad.view(-1, sl_h).unsqueeze(1)

		hist = hist.view(-1, sl_h)  # bs*nr,sl_q
		hist_word_embed = self.word_embed(hist)  # bs*nr,sl_q, emb_s
		hist_word_encoded, _ = self.hist_rnn(hist_word_embed, batch['hist_len'])

		loc = batch['hist_len'].view(-1).cpu().numpy()
		# sentence-level encoded
		hist_encoded_forawrd = hist_word_encoded[range(bs * nr), loc, :lstm]
		hist_encoded_backward = hist_word_encoded[:, 0, lstm:]
		hist_encoded = torch.cat((hist_encoded_forawrd, hist_encoded_backward), dim=-1)
		hist_encoded = hist_encoded.view(bs, nr, -1)

		return hist_word_embed, hist_word_encoded, hist_encoded, hist_not_pad, hist_pad
