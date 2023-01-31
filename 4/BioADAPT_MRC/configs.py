seed = 42
in_domain_names = ['squad',
#                    'unsupervised_v1',
#                    'unsupervised_v2_fold1',
#                    'unsupervised_v2_fold2',
#                    'unsupervised_v2_fold3',
#                    'unsupervised_v2_fold4',
#                    'unsupervised_v2_fold5',
#                    'semisupervised_fold1',
#                    'semisupervised_fold2',
#                    'semisupervised_fold3',
#                    'semisupervised_fold4',
#                    'semisupervised_fold5',
#                    'supervised_fold1',
#                    'supervised_fold2',
#                    'supervised_fold3',
                   'supervised_fold4',
#                    'supervised_fold5',
                  ]
out_domain_names = [
#                     'otquad',
#                     'otquad_fold_1',
#                     'otquad_fold_2',
#                     'otquad_fold_3',
                    'otquad_fold_4',
#                     'otquad_fold_5',
                   ]
do_train = True
do_eval  = False
do_test  = True
#############################################################################################################################
ALTERNATE_SOURCE_TARGET = False
SOURCE_INDEX            = 0
#############################################################################################################################
_factoid_qa_generate_both_ = True
qa_loss_domain_1_alpha     = 1     # original task loss in the base model for target domain example-1
aux_layer_domain_1_gamma   = 1     # original task loss in the disc model for target domain example-1
qa_loss_domain_0_alpha     = 1     # original task loss in the base model for source domain example-2
aux_layer_domain_0_gamma   = 1     # do not change
#############################################################################################################################
domain_adaptation          = True
USE_AUX_QA_LOSS            = True
USE_TRAINED_MODEL          = True
#############################################################################################################################
SHOW_SCORES_PER_EPOCH = True
AFTER_DISC_TRAIN      = False
TRAIN_DISC            = False
#############################################################################################################################
freeze_encoder                 = False
freeze_qa_output_generator     = False
freeze_discriminator_encoder   = False
freeze_aux_qa_output_generator = False
#############################################################################################################################
qa_loss_alpha            = 1     # original task loss in the base model
reverse_layer_lambda     = 0     # gradient reversal layer 
adv_loss_beta            = 1  # discriminator triplet loss
aux_layer_gamma          = 1  # original task loss in the disc model
num_samples_per_epoch = 1_000
num_train_epochs      = 2_00 #100
patience_threshold    = 2_00 #100
learning_rate         = 5e-5
lr_multiplier         = 20
lambda_delta          = 0.01
warmup_steps          = 0
#############################################################################################################################
output_dir                    = '/net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/'
original_model_name_or_path   = 'biolinkbert'
pretrained_model_name_or_path = 'michiyasunaga/BioLinkBERT-base'
model_name_or_path            = 'michiyasunaga/BioLinkBERT-base'
output_model_dir              = output_dir+in_domain_names[1]+'/final-aug13/'
trained_model_name            = ''
#############################################################################################################################
no_cuda = False
device = "cuda:1"
# 'cuda'
n_gpu = 1
num_workers = 8
local_rank  = -1
train_batch_size = 1 # overwritten during training
per_gpu_train_batch_size = 1 #7
per_gpu_eval_batch_size  = 8
gradient_accumulation_steps = 35 #5
validation_batch_size = 64
weight_decay  = 0.0
adam_epsilon  = 1e-8
initializer_range = 0.02
max_grad_norm = 1.0
max_steps = -1
tokenizer_name = ''
config_name = '' 
model_type = 'bert' 
cache_dir = ''
overwrite_output_dir = True
overwrite_output_model_dir = True
overwrite_cache = True
server_ip = ''
server_port = ''
threads = 512
n_best_size       = 5
max_query_length  = 64
max_answer_length = 200
verbose_logging   = False
max_seq_length    = 384
doc_stride        = 128
emb_dim           = 768
do_lower_case     = True #False
lang_id = 0
dataset_name_domain_0 = in_domain_names[0]
dataset_name_domain_1 = in_domain_names[1]
null_score_diff_threshold = 0.0
version_2_with_negative   = False
fp16 = False
fp16_opt_level = 'O1'
eval_all_checkpoints     = False
evaluate_during_training = False
logging_steps = 50_000_000
save_steps    = 50_000_000
