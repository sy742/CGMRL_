import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import json
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import VisDialDataset
from visdial.encoders import Encoder
from visdial.decoders import Decoder
from visdial.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdial.model import EncoderDecoderModel
from visdial.utils.checkpointing import load_checkpoint
from data.preprocess.init_glove import Vocabulary
import  json

class Evaluation(object):
	def __init__(self, hparams, model = None, split = "test"):
		self.hparams = hparams
		self.model = model
		self._logger = logging.getLogger(__name__)
		# self.device = (
		# 	torch.device("cuda", self.hparams.gpu_ids[0])
		# 	if self.hparams.gpu_ids[0] >= 0
		# 	else torch.device("cpu")
		# )
		self.device = torch.device('cuda')

		self.split = split

		do_valid, do_test = False, False
		if split == "val":
			do_valid = True
		else:
			do_test = True
		self._build_dataloader(do_valid=do_valid, do_test=do_test)
		self._dataloader = self.valid_dataloader if split == 'val' else self.test_dataloader

		if model is None:
			self._build_model()

		self.sparse_metrics = SparseGTMetrics()
		self.ndcg = NDCG()
		self.vocabulary = Vocabulary(hparams.word_counts_json, min_count=hparams.vocab_min_count)

	def _build_dataloader(self, do_valid=False, do_test=False):
		if do_valid:
			split = "train" if self.hparams.dataset_version == "0.9" else "val"
			old_split= "val" if self.hparams.dataset_version == "0.9" else None

			self.valid_dataset = VisDialDataset(
				self.hparams,
				overfit=self.hparams.overfit,
				split=split,
				old_split=old_split
			)

			collate_fn = None
			if "dan" in self.hparams.img_feature_type:
				collate_fn = self.valid_dataset.collate_fn

			self.valid_dataloader = DataLoader(
				self.valid_dataset,
				batch_size=self.hparams.eval_batch_size,
				num_workers=self.hparams.cpu_workers,
				drop_last=False,
				collate_fn=collate_fn
			)

		if do_test:
			self.test_dataset = VisDialDataset(
				self.hparams,
				overfit=self.hparams.overfit,
				split="test",
			)

			collate_fn = None
			if "dan" in self.hparams.img_feature_type:
				collate_fn = self.test_dataset.collate_fn

			self.test_dataloader = DataLoader(
				self.test_dataset,
				batch_size=self.hparams.eval_batch_size,
				num_workers=self.hparams.cpu_workers,
				drop_last=False,
				collate_fn=collate_fn
			)

	def _build_model(self):
		vocabulary = self.valid_dataset.vocabulary if self.split == "val" else self.test_dataset.vocabulary
		encoder = Encoder(self.hparams, vocabulary)
		decoder = Decoder(self.hparams, vocabulary)

		# Wrap encoder and decoder in a model.
		self.model = EncoderDecoderModel(encoder, decoder).to(self.device)

		# Use Multi-GPUs
		# if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
		# 	self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)
	def base_case(self, output, batch):
		image_id = batch['img_ids']
		ans = batch['ans_ind'].view(-1)
		ques = batch['ques']
		for i in range(10):
			cur_ans = ans[0, i, :]
			cur_ques = ques[i]
			cur_out = output[0, i, :]
			_, rank_id = torch.sort(cur_out)
			predict = rank_id[-1]
			options = batch['opt']

			answer_token = options[:, i, cur_ans, :].view(-1)
			answer = ''
			for token in answer_token:
				if token == 0:
					break
				word = self.vocabulary.index2word[token.item()]
				answer = answer + ' ' + word
			if 'no' in answer.split() or 'yes' in answer.split():
				if predict == cur_ans:
					question = ''
					for token in cur_ques:
						if token == 0:
							break
						word = self.vocabulary.index2word[token.item()]
						question = question + ' ' + word
					print(image_id)
					print(question)
					print(answer)
					for m in range(5):
						cur_predict = rank_id[-1-m]
						cur_opt = ''
						opt_token = options[:, i, cur_predict, :].view(-1)
						for token in opt_token:
							if token == 0:
								break
							word = self.vocabulary.index2word[token.item()]
							cur_opt = cur_opt + ' ' + word
						print(cur_opt)
					input()

	def run_evaluate(self, evaluation_path, global_iteration_step=0,
									 tb_summary_writer:SummaryWriter=None, eval_json_path=None, eval_seed=None):

		model_state_dict, optimizer_state_dict = load_checkpoint(evaluation_path)
		print("evaluation model loading completes! ->", evaluation_path)

		self.eval_seed = self.hparams.random_seed[0] if eval_seed is None else eval_seed

		if isinstance(self.model, nn.DataParallel):
			self.model.module.load_state_dict(model_state_dict)
		else:
			self.model.load_state_dict(model_state_dict)

		self.model.eval()
		ranks_json = []
		save_dict = {}
		for i, batch in enumerate((self._dataloader)):
			for key in batch:
				batch[key] = batch[key].to(self.device)
			with torch.no_grad():
				output, ques_gate = self.model(batch)
			batch_size, num_dial, _ = batch['ques'].size()

			# self.base_case(output, batch)

			# image_id = batch['img_ids']
			# ans = batch['ans_ind'].view(-1)
			# value = []
			# for i in range(10):
			# 	# cur_ans = ans[i]
			# 	# cur_out = output[0, i, :]
			# 	# _, rank_id = torch.sort(cur_out)
			# 	# predict = rank_id[-1]
			# 	value.append(ques_gate.view(-1)[i].item())
			# 	# if cur_ans == predict:
			# 	# 	value.append(ques_gate.view(-1)[i].item())
			# 	# else:
			# 	# 	value.append(0)
			# 	# print(image_id)
			# 	# print(image_id.item())
			# save_dict[image_id.item()] = value
			# with open(save_path, 'w') as file:
			# 	json.dump(save_dict, file)
			#
			#
			# # if image_id == 388326:
			# # 	print(ques_gate)
			# # 	out = output[0, 1, :]
			# # 	rank_id = torch.sort(out)
			# # 	print(rank_id)
			#
			#
			# # for i in range(10):
			# # 	if ques_gate.view(-1)[i]<0.5:
			# # 		print('====================')
			# # 		print('image_id:', image_id)
			# # 		print(ques_gate)
			# # 		break



			ranks = scores_to_ranks(output) # bs, num_dialog, num_options

			for i in range(len(batch["img_ids"])):
				# Cast into types explicitly to ensure no errors in schema.
				# Round ids are 1-10, not 0-9
				if self.split == "test":
					ranks_json.append(
						{
							"image_id": batch["img_ids"][i].item(),
							"round_id": int(batch["num_rounds"][i].item()),
							"ranks": [rank.item() for rank in ranks[i][batch["num_rounds"][i] - 1]],
						}
					)
				else:
					for j in range(batch["num_rounds"][i]):
						ranks_json.append(
							{
								"image_id": batch["img_ids"][i].item(),
								"round_id": int(j + 1),
								"ranks": [rank.item() for rank in ranks[i][j]],
							}
						)

			if self.split == "val":
				self.sparse_metrics.observe(output, batch["ans_ind"])
				if "gt_relevance" in batch:  # version 1.0
					output = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
					self.ndcg.observe(output, batch["gt_relevance"])

		if self.split == "val":
			all_metrics = {}
			all_metrics.update(self.sparse_metrics.retrieve(reset=True))
			if self.hparams.dataset_version == '1.0':
				all_metrics.update(self.ndcg.retrieve(reset=True))

			for metric_name, metric_value in all_metrics.items():
				self._logger.info(f"{metric_name}: {metric_value}")

			if tb_summary_writer:
				tb_summary_writer.add_scalars(
					"metrics", all_metrics, global_iteration_step
				)

		# if not tb_summary_writer:
		print("Writing ranks to {}".format(self.hparams.root_dir))
		if eval_json_path is not None:
			json.dump(ranks_json, open(eval_json_path, "w"))
		else:
			json.dump(ranks_json, open(os.path.join(self.hparams.root_dir, self.hparams.model_name +
																							"_ranks_%s.json" % self.split), "w"))

		if not tb_summary_writer and self.split == "val":
			for metric_name, metric_value in all_metrics.items():
				print(f"{metric_name}: {metric_value}")