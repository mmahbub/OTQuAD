import os

os.system('python3 load_dataset.py  --orig_model_name_or_path biolinkbert  --model_name_or_path michiyasunaga/BioLinkBERT-base  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_lower_case --do_evaluate --max_seq_length 384   --doc_stride 128  --overwrite_cache')

os.system('python3 load_dataset.py  --orig_model_name_or_path pubmedbert  --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_lower_case --do_evaluate --max_seq_length 384   --doc_stride 128  --overwrite_cache')

os.system('python3 load_dataset.py  --orig_model_name_or_path bioelectra  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed-pmc-lt  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_evaluate --max_seq_length 384   --doc_stride 128  --overwrite_cache')

os.system('python3 load_dataset.py  --orig_model_name_or_path biobert  --model_name_or_path dmis-lab/biobert-base-cased-v1.1  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_evaluate --max_seq_length 384   --doc_stride 128  --overwrite_cache')

os.system('python3 load_dataset.py  --orig_model_name_or_path scibert  --model_name_or_path allenai/scibert_scivocab_uncased  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_lower_case --do_evaluate --max_seq_length 384   --doc_stride 128       --overwrite_cache')

os.system('python3 load_dataset.py  --orig_model_name_or_path clinicalbert  --model_name_or_path emilyalsentzer/Bio_ClinicalBERT --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_evaluate --max_seq_length 384   --doc_stride 128  --overwrite_cache')

os.system('python3 load_dataset.py  --orig_model_name_or_path roberta  --model_name_or_path roberta-base --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_evaluate --max_seq_length 384   --doc_stride 128  --overwrite_cache')

os.system('python3 load_dataset.py  --orig_model_name_or_path bluebert  --model_name_or_path bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_evaluate --max_seq_length 384   --doc_stride 128  --overwrite_cache')

os.system('python3 load_dataset.py  --orig_model_name_or_path bert  --model_name_or_path bert-base-uncased --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/quac/  --predict_file test_quac.json --dataset_name quac  --do_lower_case --do_evaluate --max_seq_length 384   --doc_stride 128  --overwrite_cache')