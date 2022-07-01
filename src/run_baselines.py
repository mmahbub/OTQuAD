import os

os.system('python3 finetune_model.py --model_type biolinkbert --orig_model_name_or_path biolinkbert --model_name_or_path michiyasunaga/BioLinkBERT-base --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/biolinkbert-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/ --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel --do_lower_case --do_train --max_seq_length 384 --doc_stride 128 --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5 --overwrite_output_dir --overwrite_output_model_dir')

os.system('python3 finetune_model.py  --model_type pubmedbert   --orig_model_name_or_path pubmedbert  --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/pubmedbert-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/  --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel  --do_lower_case --do_train --max_seq_length 384   --doc_stride 128  --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5  --overwrite_output_dir  --overwrite_output_model_dir') 

os.system('python3 finetune_model.py  --model_type bioelectra   --orig_model_name_or_path bioelectra  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed-pmc-lt  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/bioelectra-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/  --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel  --do_train --max_seq_length 384   --doc_stride 128  --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5  --overwrite_output_dir  --overwrite_output_model_dir') 

os.system('python3 finetune_model.py  --model_type biobert   --orig_model_name_or_path biobert  --model_name_or_path dmis-lab/biobert-base-cased-v1.1  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/biobert-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/  --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel  --do_train --max_seq_length 384   --doc_stride 128  --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5  --overwrite_output_dir  --overwrite_output_model_dir') 

os.system('python3 finetune_model.py  --model_type scibert   --orig_model_name_or_path scibert  --model_name_or_path allenai/scibert_scivocab_uncased  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/scibert-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/  --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel  --do_lower_case --do_train --max_seq_length 384   --doc_stride 128  --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5  --overwrite_output_dir  --overwrite_output_model_dir') 

os.system('python3 finetune_model.py  --model_type clinicalbert   --orig_model_name_or_path clinicalbert  --model_name_or_path emilyalsentzer/Bio_ClinicalBERT --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/clinicalbert-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/  --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel  --do_train --max_seq_length 384   --doc_stride 128  --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5  --overwrite_output_dir  --overwrite_output_model_dir') 

os.system('python3 finetune_model.py  --model_type roberta   --orig_model_name_or_path roberta  --model_name_or_path roberta-base --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/roberta-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/  --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel  --do_train --max_seq_length 384   --doc_stride 128  --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5  --overwrite_output_dir  --overwrite_output_model_dir') 

os.system('python3 finetune_model.py  --model_type bluebert   --orig_model_name_or_path bluebert  --model_name_or_path bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/bluebert-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/  --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel  --do_train --max_seq_length 384   --doc_stride 128  --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5  --overwrite_output_dir  --overwrite_output_model_dir') 

os.system('python3 finetune_model.py  --model_type bert   --orig_model_name_or_path bert  --model_name_or_path bert-base-uncased --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/ --output_model_dir /net/kdinxidk03/opt/NFS/75y/data/qa/output/bert-emrqa-rel/ --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/emrqa/datasets/  --train_file relation-train-sampled-0.03.json  --dataset_name emrqa_rel  --do_train --do_lower_case --max_seq_length 384   --doc_stride 128  --per_gpu_train_batch_size 24 --num_train_epochs 3.0 --learning_rate  3e-5  --overwrite_output_dir  --overwrite_output_model_dir')