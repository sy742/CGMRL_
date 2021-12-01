import random
from collections import defaultdict

OLD_DATASET_PARAMS = defaultdict(
  dataset_version='0.9',
  visdial_json='data/v0.9/visdial_0.9_%s.json',
)

BASE_PARAMS = defaultdict(
  # Dataset reader arguments
  dataset_version='1.0',

  img_feature_type="dan_faster_rcnn_x101", # faster_rcnn_x101, dan_faster_rcnn_x101
  model_train_type="single", # single, multi
  img_features_h5='data/dan_faster_rcnn_x101/features_%s_%s.h5', # img_feature_type | train, val, test

  imgid2idx_path='data/dan_faster_rcnn_x101/%s_imgid2idx.pkl', # dan_img - train, val, test
  text_features_h5='data/visdial_1.0_text/visdial_1.0_%s_text_%s.hdf5',


  fake_label_path='data/fake_label/bert_fake_label_logits_train.pkl',
  fake_label_img_ids_path='data/fake_label/glove_fake_label_train_img_id.pkl',

  word_counts_json='data/visdial_1.0_word_counts_train.json',
  glove_npy='data/glove.npy',
  pretrained_glove='data/glove.6B.300d.txt',

  visdial_json='data/visdial_1.0_%s.json',
  valid_dense_json='data/visdial_1.0_val_dense_annotations.json',

  # Model save arguments
  root_dir='./saved_model/', # for saving logs, checkpoints and result files
  save_dirpath ='checkpoints/',

  load_pthpath = '',
  ##### q_gate #####
  img_norm=True,
  max_sequence_length=20,
  vocab_min_count=5,

  # Train related arguments
  gpu_ids=[7],
  cpu_workers=4,
  tensorboard_step=100,
  do_vaild=True,
  overfit=False,
  # random_seed=random.sample(range(1000, 10000), 1),
  random_seed = [2995],
  concat_history=True,
  hard=False,

  # Opitimization related arguments
  num_epochs=10,
  train_batch_size=16, # 32 x num_gpus is a good rule of thumb
  eval_batch_size=1,
  virtual_batch_size=32,
  training_splits="train",
  evaluation_type="disc",
  lr_scheduler="LambdaLR",
  warmup_epochs=2,
  warmup_factor=0.01,
  initial_lr=0.001,
  lr_gamma=0.1,
  lr_milestones=[5],  # epochs when lr —> lr * lr_gamma
  lr_gamma2=0.5,
  lr_milestones2=[7],  # epochs when lr —> lr * lr_gamma2

  # Model related arguments
  encoder='mvan',
  decoder='disc',  # [disc,gen]

  img_feature_size=2048,
  word_embedding_size=300,
  lstm_hidden_size=512,
  lstm_num_layers=2,
  dropout=0.4,
  dropout_fc=0.25,
  T = 1,
  alpha  = 2,
  alpha_coref  = 1,
  margin_1 = 0.1,
  margin_2 = 0.1,
  K = 1,
  M = 9,
  N = 40,
)

MVAN_MULTI_PARAMS= BASE_PARAMS.copy()
MVAN_MULTI_PARAMS.update(
  gpu_ids=[7],
  num_epochs=8,
  train_batch_size=8, # 32 x num_gpus is a good rule of thumb
  eval_batch_size=2,
  virtual_batch_size=32,
  cpu_workers=4,

  # Model related arguments
  encoder='mvan',
  decoder='disc_gen',  # [disc,gen]
  evaluation_type="disc_gen",
  aggregation_type="average",
)
