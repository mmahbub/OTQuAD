import os

os.system('python3 load_dataset.py  --orig_model_name_or_path biolinkbert  --model_name_or_path michiyasunaga/BioLinkBERT-base  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/otquad/  --train_file train_squad_otquad_fold1.json --dataset_name squad_otquad_fold1  --do_lower_case --do_train --max_seq_length 384   --doc_stride 128')

os.system('python3 load_dataset.py  --orig_model_name_or_path biolinkbert  --model_name_or_path michiyasunaga/BioLinkBERT-base  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/otquad/  --train_file train_squad_otquad_fold2.json --dataset_name squad_otquad_fold2  --do_lower_case --do_train --max_seq_length 384   --doc_stride 128')

os.system('python3 load_dataset.py  --orig_model_name_or_path biolinkbert  --model_name_or_path michiyasunaga/BioLinkBERT-base  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/otquad/  --train_file train_squad_otquad_fold3.json --dataset_name squad_otquad_fold3  --do_lower_case --do_train --max_seq_length 384   --doc_stride 128')

os.system('python3 load_dataset.py  --orig_model_name_or_path biolinkbert  --model_name_or_path michiyasunaga/BioLinkBERT-base  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/otquad/  --train_file train_squad_otquad_fold4.json --dataset_name squad_otquad_fold4  --do_lower_case --do_train --max_seq_length 384   --doc_stride 128')

os.system('python3 load_dataset.py  --orig_model_name_or_path biolinkbert  --model_name_or_path michiyasunaga/BioLinkBERT-base  --output_dir /net/kdinxidk03/opt/NFS/75y/data/OTMRC_PAPER/  --data_dir /net/kdinxidk03/opt/NFS/75y/data/qa/dataset_pos/otquad/  --train_file train_squad_otquad_fold5.json --dataset_name squad_otquad_fold5  --do_lower_case --do_train --max_seq_length 384   --doc_stride 128')
