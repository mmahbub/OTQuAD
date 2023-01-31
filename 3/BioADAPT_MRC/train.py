# coding=utf-8

""" Finetuning the library models for question-answering """

import sys
sys.path.append('../')

import glob
import logging
import os
import random
import timeit

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used

import numpy as np
import pandas as pd
import csv

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    get_raw_scores,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import BioADAPT_MRC.configs as configs
from BioADAPT_MRC.siamese_adv_model import siamese_adv_net
from BioADAPT_MRC.data_generator import qa_dataset, qa_collate

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(configs):
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    if configs.n_gpu > 0:
        torch.cuda.manual_seed_all(configs.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(model, tokenizer):
    """Train the model"""

    if not os.path.exists(configs.output_model_dir):
        os.makedirs(configs.output_model_dir)

    set_seed(configs)
    
    if configs.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    configs.train_batch_size = configs.per_gpu_train_batch_size * max(1, configs.n_gpu)

    train_dataset = qa_dataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=configs.train_batch_size,
                                                   shuffle=False,
                                                   num_workers=configs.num_workers,
                                                   drop_last=False,
                                                   collate_fn=qa_collate)

    if configs.max_steps > 0:
        t_total = configs.max_steps
        configs.num_train_epochs = configs.max_steps // (
                len(train_dataloader) // configs.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // configs.gradient_accumulation_steps * configs.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
#     no_decay = []
    
    # check the parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": configs.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=configs.learning_rate,
                      eps=configs.adam_epsilon
                     )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=configs.warmup_steps,
                                                num_training_steps=t_total*configs.lr_multiplier
                                               ) ################################################################################################################################################################################################################
    
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(configs.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(configs.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(configs.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(configs.model_name_or_path, "scheduler.pt")))

    if configs.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=configs.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if configs.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if configs.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[configs.local_rank], output_device=configs.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", configs.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", configs.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        configs.train_batch_size
        * configs.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if configs.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", configs.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    # Check if continuing training from a checkpoint
    if os.path.exists(configs.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = configs.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // configs.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // configs.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(configs.num_train_epochs), desc="Epoch", disable=configs.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(configs)

    adv_loss_list, aux_qa_loss_list, original_qa_loss_list, total_loss_list, lr_list, lambda_list, val_loss_list, cosDist_source_list, cosDist_target_list, squad_em_list, squad_f1_list, val_loss_all_list, squad_em_all_list, squad_f1_all_list, test_em_list, test_f1_list = [np.inf], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    

    ite = 0
    patience = configs.patience_threshold

    fieldnames      = [f'sample_{i}' for i in range(5000)]

    with open(f"{configs.output_model_dir}df_val_loss.csv", "w") as csvFileLoss, open(f"{configs.output_model_dir}df_val_em.csv", "w") as csvFileEM, open(f"{configs.output_model_dir}df_val_f1.csv", "w") as csvFileF1:
#         writer1 = csv.DictWriter(csvFileLoss, fieldnames=fieldnames_loss)
#         writer1.writeheader()
        writer2 = csv.DictWriter(csvFileEM, fieldnames=fieldnames)
        writer2.writeheader()
        writer3 = csv.DictWriter(csvFileF1, fieldnames=fieldnames)
        writer3.writeheader()

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=configs.local_rank not in [-1, 0])

            adv_loss_tr, aux_qa_loss_tr, original_qa_loss_tr, total_loss_tr, cosDist_source_tr, cosDist_target_tr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            local_step = 0

            for step, batch in enumerate(epoch_iterator):
                local_step += 1

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()

                outputs = model(batch)

                # model outputs are always tuple in transformers (see doc)
                encodings, factoid_qa_outputs, aux_qa_outputs, \
                adv_loss, aux_qa_loss, original_qa_loss, loss, cosDist_source, cosDist_target = outputs  # want to make a list of the losses per epoch

                total_loss = loss

                if configs.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                    adv_loss = adv_loss.mean()
                    original_qa_loss = original_qa_loss.mean()
                    aux_qa_loss = aux_qa_loss.mean()
                    cosDist_source = cosDist_source.mean()
                    cosDist_target = cosDist_target.mean()


    #             if local_step == 33:
    #                 print(original_qa_loss)
    #             if local_step == 33:
    #                 exit(1)

                if configs.gradient_accumulation_steps > 1:
                    loss = loss / configs.gradient_accumulation_steps
    #                 adv_loss = adv_loss / configs.gradient_accumulation_steps
    #                 original_qa_loss = original_qa_loss / configs.gradient_accumulation_steps
    #                 aux_qa_loss = aux_qa_loss / configs.gradient_accumulation_steps

                if configs.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                total_loss_tr+= total_loss.item()
                adv_loss_tr += adv_loss.item()
                original_qa_loss_tr += original_qa_loss.item()
                aux_qa_loss_tr += aux_qa_loss.item()
                cosDist_source_tr += cosDist_source
                cosDist_target_tr += cosDist_target

                if (step + 1) % configs.gradient_accumulation_steps == 0:
                    if configs.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), configs.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), configs.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if configs.local_rank in [-1, 0] and configs.logging_steps > 0 and global_step % configs.logging_steps == 0:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if configs.local_rank == -1 and configs.evaluate_during_training:
                            results = evaluate(model, tokenizer, None, in_domain=None, out_domain=None, evaluate_all=False, evaluate_domain_0=False)
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / configs.logging_steps, global_step)
                        logging_loss = tr_loss

                    # Save model checkpoint
                    if configs.local_rank in [-1, 0] and configs.save_steps > 0 and global_step % configs.save_steps == 0:

                        output_dir = os.path.join(configs.output_model_dir, "checkpoint-{}".format(global_step))

                        if not os.path.exists(output_dir) and configs.local_rank in [-1, 0]:
                            os.makedirs(output_dir)

                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), f'{output_dir}/model.pt')
                        tokenizer.save_pretrained(output_dir)

                        torch.save(configs, os.path.join(output_dir, "training_configs.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if configs.max_steps > 0 and global_step > configs.max_steps:
                    epoch_iterator.close()
                    break

            print('Current LR    :', scheduler.get_last_lr())
            print('Current Lambda:', configs.reverse_layer_lambda)
                    
# ######################################################################################################

            print(ite)
            if ite > 83:
                print('Exit training ....')
                exit(1)

            if configs.SHOW_SCORES_PER_EPOCH:

                val_loss_all, squad_em_all, squad_f1_all, loss_dict, em_dict, f1_dict = validation_loss_all_squad(model, tokenizer)
        
                squad_em_all_list.append(squad_em_all)
                squad_f1_all_list.append(squad_f1_all)
                
                if ite>=0:
                    test_em, test_f1, test_em_dict, test_f1_dict = get_test_score(model, tokenizer)
#                     if squad_f1_all_list.index(max(squad_f1_all_list))==ite:
#                         test_em, test_f1, test_em_dict, test_f1_dict = get_test_score(model, tokenizer)
#                     else:
#                         test_em, test_f1, test_em_dict, test_f1_dict = 0, 0, None, None
                else:
                    test_em, test_f1, test_em_dict, test_f1_dict = 0, 0, None, None


                if not os.path.exists(configs.output_model_dir):
                    os.makedirs(configs.output_model_dir)
                    
                if ite == 83:

#                     if squad_f1_all_list.index(max(squad_f1_all_list))==ite:

                    print(f'Saving model to {configs.output_model_dir}model_epoch_{ite}.pt')

                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), f'{configs.output_model_dir}model_epoch_{ite}.pt')
                    tokenizer.save_pretrained(configs.output_model_dir)

                    torch.save(optimizer.state_dict(), os.path.join(configs.output_model_dir, f"optimizer_epoch_{ite}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(configs.output_model_dir, f"scheduler_epoch_{ite}.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", configs.output_model_dir)                      

                if ite>0:
                    if not (squad_f1_all_list[-1] > squad_f1_all_list[-2]):
                        patience -= 1
                        if patience <= 0:
                            print('Exit training ....')
                            exit(1)


            if configs.TRAIN_DISC:
                adv_loss_tracker = adv_loss_tr/local_step
                if adv_loss_tracker < adv_loss_list[-1]:
                    if not os.path.exists(configs.output_model_dir):
                        os.makedirs(configs.output_model_dir)

                    print(f'Saving model to {configs.output_model_dir}model_epoch_{ite}.pt')

                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), f'{configs.output_model_dir}model_epoch_{ite}.pt')
                    tokenizer.save_pretrained(configs.output_model_dir)

                    torch.save(optimizer.state_dict(), os.path.join(configs.output_model_dir, f"optimizer_epoch_{ite}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(configs.output_model_dir, f"scheduler_epoch_{ite}.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", configs.output_model_dir)

            adv_loss_list.append(adv_loss_tr/local_step)
            original_qa_loss_list.append(original_qa_loss_tr/local_step)
            total_loss_list.append(total_loss_tr/local_step)
            aux_qa_loss_list.append(aux_qa_loss_tr/local_step)
            cosDist_source_list.append(cosDist_source_tr/local_step)
            cosDist_target_list.append(cosDist_target_tr/local_step)
            lr_list.append(scheduler.get_last_lr()[0])
            lambda_list.append(configs.reverse_layer_lambda)
            val_loss_all_list.append(val_loss_all)
            test_em_list.append(test_em)
            test_f1_list.append(test_f1)

#             writer1.writerow(loss_dict)
            writer2.writerow(em_dict)
            writer3.writerow(f1_dict)

            file = open(configs.output_model_dir+f'loss_epochs.txt', 'w')
            file.write(f'{lr_list}\n{adv_loss_list}\n{aux_qa_loss_list}\n{original_qa_loss_list}\n{total_loss_list}\n{lambda_list}\n{val_loss_list}\n{cosDist_source_list}\n{cosDist_target_list}\n{squad_em_list}\n{squad_f1_list}\n{val_loss_all_list}\n{squad_em_all_list}\n{squad_f1_all_list}\n{test_em_list}\n{test_f1_list}\n')
            file.close()

            print(configs.reverse_layer_lambda)

            ite+=1

            if (ite%10==0):
                if (configs.reverse_layer_lambda < 0.04):
                    configs.reverse_layer_lambda = configs.reverse_layer_lambda+configs.lambda_delta

            if configs.max_steps > 0 and global_step > configs.max_steps:
                train_iterator.close()
                break

        if configs.local_rank in [-1, 0]:
            tb_writer.close()
            
#     print(global_step, adv_loss_list, aux_qa_loss_list, 
#           original_qa_loss_list, total_loss_list)
          
    return global_step, tr_loss / global_step


def get_test_score(model, tokenizer):
    
    model.eval()
    
    input_dir = configs.output_dir if configs.output_dir else "."
        
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            'dev',
            list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
            str(configs.max_seq_length),
            'test',
            f'{configs.out_domain_names[0]}',
            ),
    )

    # logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
        )
    
    em, f1, em_dict, f1_dict = val_evaluate(model, tokenizer, dataset, features, examples,configs.out_domain_names[0])

    return em, f1, em_dict, f1_dict


def validation_loss_all_squad(model, tokenizer):
    
    model.eval()
    
    input_dir = configs.output_dir if configs.output_dir else "."
#     cached_features_file = os.path.join(
#         input_dir,
#         "cached_{}_{}_{}_{}_{}".format(
#             'train',
#             list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
#             str(configs.max_seq_length),
#             'train',
#             f'squad_test_sample',
#             ),
#     )
    
#     # logger.info("Loading features from cached file %s", cached_features_file)
#     features_and_dataset = torch.load(cached_features_file)
#     features, dataset, examples = (
#         features_and_dataset["features"],
#         features_and_dataset["dataset"],
#         features_and_dataset["examples"],
#         )

#     eval_batch_size = configs.validation_batch_size
#     # Note that DistributedSampler samples randomly
#     eval_sampler = SequentialSampler(dataset)
#     eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
#                                  batch_size=eval_batch_size,
#                                 num_workers = 32)
        
#     loss_ls = []
#     for batch in tqdm(eval_dataloader, desc="Evaluating"):
#         with torch.no_grad():
#             batch = tuple(torch.autograd.Variable(t.to(configs.device).long()) for t in batch)
#             inputs = {
#                 "input_ids": batch[0],
#                 "attention_mask": batch[1],
#                 "token_type_ids": batch[2],
#             }
#             encoded_output = model.encoder(inputs)
#             outputs = model.factoid_qa_output_generator(encoded_output, batch[3], batch[4])
            
#         loss_ls.append(outputs.loss.cpu())
        
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            'dev',
            list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
            str(configs.max_seq_length),
            'test',
            f'squad_sample',
            ),
    )

    # logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
        )
    
    em, f1, em_dict, f1_dict = val_evaluate(model, tokenizer, dataset, features, examples,configs.in_domain_names[0]+'_all')

#     sample_num = [f'sample_{i}' for i in range(len(loss_ls))]
#     loss_dict = dict(zip(sample_num, loss_ls))
    
    return None, em, f1, None, em_dict, f1_dict


def validation_loss(model, tokenizer):
     
    model.eval()
    
    n_fold = random.randint(1,10)
    input_dir = configs.output_dir if configs.output_dir else "."

    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            'train',
            list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
            str(configs.max_seq_length),
            'train',
            f'squad_fold_{n_fold}_seed{configs.seed}',
            ),
    )

    # logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
        )

    eval_batch_size = configs.validation_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                                 batch_size=eval_batch_size)
        
    loss_ls = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            batch = tuple(torch.autograd.Variable(t.to(configs.device).long()) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            encoded_output = model.encoder(inputs)
            outputs = model.factoid_qa_output_generator(encoded_output, batch[3], batch[4])
            
        loss_ls.append(outputs.loss.cpu())
        
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            'dev',
            list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
            str(configs.max_seq_length),
            'test',
            f'squad_fold_{n_fold}_seed{configs.seed}',
            ),
    )

    # logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
        )
    
    em, f1, _, _ = val_evaluate(model, tokenizer, dataset, features, examples, configs.in_domain_names[0])
    
    return np.mean(loss_ls), em, f1


def val_evaluate(model, tokenizer, dataset, features, examples, prefix):

    eval_batch_size = configs.validation_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                                 batch_size=eval_batch_size)
    
    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(configs.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            feature_indices = batch[3]

#             encoded_output = model.module.encoder(inputs)
#             outputs = model.module.factoid_qa_output_generator(encoded_output)
            encoded_output = model.encoder(inputs)
            outputs = model.factoid_qa_output_generator(encoded_output)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs[:2]]

            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)
    
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(configs.output_model_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(configs.output_model_dir, "nbest_predictions_{}_ours.json".format(prefix))

    if configs.version_2_with_negative:
        output_null_log_odds_file = os.path.join(configs.output_model_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None
        
    if os.path.isfile(output_nbest_file):
        os.remove(output_nbest_file)
        
    print(features[1].__dict__['unique_id'])
    
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        configs.n_best_size,
        configs.max_answer_length,
        configs.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        configs.verbose_logging,
        configs.version_2_with_negative,
        configs.null_score_diff_threshold,
        tokenizer,
    )
            
    em, f1 = get_raw_scores(examples, predictions)
    
    print(len(em))
    sample_num = [f'sample_{i}' for i in range(len(list(em.values())))]
    em_dict = dict(zip(sample_num, list(em.values())))
    f1_dict = dict(zip(sample_num, list(f1.values())))

    return np.mean(list(em.values())), np.mean(list(f1.values())), em_dict, f1_dict


def load_all_examples(n_domain, in_domain, out_domain, output_examples, validation=False):
    # Load data features from cache or dataset file
    input_dir = configs.output_dir if configs.output_dir else "."
    
    print(out_domain, n_domain)
    if in_domain:
        dataset_name = configs.in_domain_names[n_domain]
    elif out_domain:
        dataset_name = configs.out_domain_names[n_domain]

    if configs.do_eval:
        prefix = f'valid_{dataset_name}'
    elif configs.do_test:
        prefix = f'test_{dataset_name}'
    else:
        prefix = f'validation_{dataset_name}'

    if validation:
        cached_features_file = os.path.join(
            input_dir,
            "cached_{}_{}_{}_{}_{}".format(
                'train',
                list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
                str(configs.max_seq_length),
                'train',
                str(dataset_name),
                ),
            )
    else:
        cached_features_file = os.path.join(
            input_dir,
            "cached_{}_{}_{}_{}_{}".format(
                "dev",
                list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
                str(configs.max_seq_length),
                "valid" if configs.do_eval else "test" if configs.do_test else 'NEED AT LEAST ONE',
                str(dataset_name),
                ),
            )

    # Init features and dataset from cache if it exists
    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
        )

    if output_examples:
        return dataset, examples, features, prefix, dataset_name
    return dataset, prefix, dataset_name


def load_and_cache_examples(evaluate_domain_0=True, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = configs.output_dir if configs.output_dir else "."
    if evaluate_domain_0:
        dataset_name = configs.dataset_name_domain_0
    else:
        dataset_name = configs.dataset_name_domain_1

    if configs.do_eval:
        prefix = f'valid_{dataset_name}'
    elif configs.do_test:
        prefix = f'test_{dataset_name}'

    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "dev",
            list(filter(None, configs.original_model_name_or_path.split("/"))).pop(),
            str(configs.max_seq_length),
            "valid" if configs.do_eval else "test" if configs.do_test else 'NEED AT LEAST ONE',
            str(dataset_name),
            ),
        )

    print(cached_features_file)
    
    # Init features and dataset from cache if it exists
    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
        )

    if output_examples:
        return dataset, examples, features, prefix, dataset_name
    return dataset, prefix, dataset_name


def evaluate(model, tokenizer, n_domain, in_domain, out_domain, evaluate_all=False, evaluate_domain_0=True):
    if evaluate_all:
        dataset, examples, features, prefix, dataset_name = load_all_examples(n_domain, in_domain, out_domain, output_examples=True)
    else:
        dataset, examples, features, prefix, dataset_name = load_and_cache_examples(evaluate_domain_0, output_examples=True)

    if not os.path.exists(configs.output_dir) and configs.local_rank in [-1, 0]:
        os.makedirs(configs.output_dir)

    eval_batch_size = configs.per_gpu_eval_batch_size * max(1, configs.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # multi-gpu evaluate
    if configs.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(configs.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if configs.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if configs.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * configs.lang_id).to(configs.device)}
                    )
#             encoded_output = model.module.encoder(inputs)
#             outputs = model.module.factoid_qa_output_generator(encoded_output)
            encoded_output = model.encoder(inputs)
            outputs = model.factoid_qa_output_generator(encoded_output)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs[:2]]

            #             print(len(outputs), outputs[0].shape, outputs[1].shape)

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(configs.output_model_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(configs.output_model_dir, "nbest_predictions_{}_ours.json".format(prefix))

    if configs.version_2_with_negative:
        output_null_log_odds_file = os.path.join(configs.output_model_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None
        
    if os.path.isfile(output_nbest_file):
        os.remove(output_nbest_file)
        

    # XLNet and XLM use a more complex post-processing procedure
    if configs.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            configs.n_best_size,
            configs.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            configs.version_2_with_negative,
            tokenizer,
            configs.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            configs.n_best_size,
            configs.max_answer_length,
            configs.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            configs.verbose_logging,
            configs.version_2_with_negative,
            configs.null_score_diff_threshold,
            tokenizer,
        )
        
        
    em, f1 = get_raw_scores(examples, predictions)
    
    print(len(em))
    
    return np.mean(list(em.values())), np.mean(list(f1.values()))       


def main():

    set_seed(configs)

    if configs.doc_stride >= configs.max_seq_length - configs.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
            os.path.exists(configs.output_dir)
            and os.listdir(configs.output_dir)
            and configs.do_train
            and not configs.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                configs.output_dir
            )
        )

    # Setup distant debugging if needed
    if configs.server_ip and configs.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(configs.server_ip, configs.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if configs.local_rank == -1 or configs.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not configs.no_cuda else "cpu")
        configs.n_gpu = 0 if configs.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(configs.local_rank)
        device = torch.device("cuda", configs.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        configs.n_gpu = 1
    configs.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if configs.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        configs.local_rank,
        device,
        configs.n_gpu,
        bool(configs.local_rank != -1),
        configs.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(configs.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    # Set seed
    set_seed(configs)

    # Load pretrained model and tokenizer
    if configs.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    configs.model_type = configs.model_type.lower()

    tokenizer = AutoTokenizer.from_pretrained(
        configs.tokenizer_name if configs.tokenizer_name else configs.model_name_or_path,
        do_lower_case=configs.do_lower_case,
        cache_dir=configs.cache_dir if configs.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )

    if configs.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model = siamese_adv_net()
    
    if configs.AFTER_DISC_TRAIN:
        model.load_state_dict(torch.load(f'{configs.model_name_or_path}model.pt'))
        
    model.to(configs.device)

    logger.info("Training/evaluation parameters %s", configs)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if configs.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if configs.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


######################################################################################################################
    # Training
    if configs.do_train:      
        global_step, tr_loss = train(model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if configs.do_train and (configs.local_rank == -1 or torch.distributed.get_rank() == 0):

        if not os.path.exists(configs.output_model_dir):
            os.makedirs(configs.output_model_dir)

        logger.info("Saving model checkpoint to %s", configs.output_model_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        
        torch.save(model_to_save.state_dict(), f'{configs.output_model_dir}model.pt')

        tokenizer.save_pretrained(configs.output_model_dir)

#         # Good practice: save your training arguments together with the trained model
#         torch.save(configs, os.path.join(configs.output_model_dir, "training_configs.bin"))

#         # Load a trained model and vocabulary that you have fine-tuned
#         model = siamese_adv_net()
#         model.load_state_dict(torch.load(f'{configs.output_model_dir}model.pt'))  # , force_download=True)

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out
        tokenizer = AutoTokenizer.from_pretrained(configs.output_model_dir,
                                                  do_lower_case=configs.do_lower_case,
                                                  use_fast=False)
#         model.to(configs.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
#     results_domain_0, results_domain_1 = {}, {}
    if (configs.do_eval or configs.do_test) and configs.local_rank in [-1, 0]:
        if configs.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [configs.output_model_dir]
            if configs.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(configs.output_model_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", configs.model_name_or_path)
            checkpoints = [configs.output_model_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = siamese_adv_net()
            
            if configs.USE_TRAINED_MODEL:
                model.load_state_dict(torch.load(f'{checkpoint}{configs.trained_model_name}'))  # , force_download=True)

            model.to(configs.device)
            
            em, f1, _, _ = get_test_score(model, tokenizer)
            
            print('-----RESULTS-----\n')
            print(f'{em}\t{f1}')
                

if __name__ == "__main__":
    main()
