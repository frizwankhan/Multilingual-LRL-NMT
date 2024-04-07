'''
torchrun --nproc_per_node=2 --nnodes=1 --master_port 29600 train.py
python3 -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port 29600 train.py
'''

from config import get_default_config
import os
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import adapters
from adapters import list_adapters, BnConfig, LoRAConfig

from dataset import MultilingualDataset, custom_collate

from utils import get_sacrebleu, label_smoothed_nll_loss
from model import get_model, get_tokenizer

from torch.distributed import init_process_group, destroy_process_group, barrier , all_reduce


class Trainer:
    def __init__(self, config, train_files, dev_files):
        self.tok = get_tokenizer(config)
        self.config = config
        self.train_files = train_files
        self.dev_files = dev_files
        
        torch.cuda.set_device(config.local_rank)
        
        self.total_params = None
        self.total_trainable_params = None
        self.get_model(config)
        
        if config.local_rank == 0:
            wandb.init(
                project=config.wb_project,
                name=config.wb_run,
                config={**config.__dict__, "Total Model Params": self.total_params, "Total Trainable Params": self.total_trainable_params}
            )
        
        self.get_optimizer_and_scheduler(config)
        
        torch.cuda.empty_cache()
        barrier() 
        
        self.train_dataset = MultilingualDataset(train_files, config)
        self.device = torch.device(f"cuda:{config.local_rank}")
        
        
    def get_model(self, config):
        self.model = get_model(self.config, self.tok)
        
        # resizing model so that 2 extra tokens are accomodated
        self.tok.add_tokens(["<2si>", "<2ne>"])
        self.model.resize_token_embeddings(len(self.tok)+2)
        
        if self.config.local_rank == 0:
            # print("Optimizing", [n for n, p in self.model.named_parameters() if p.requires_grad])
            num_params_to_optimize = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            num_model_params = sum(p.numel() for p in self.model.parameters())
            print("Number of model parameters:", num_model_params)
            print("Total number of params to be optimized are: ", num_params_to_optimize)
            print("Percentage of parameters to be optimized: ", 100*num_params_to_optimize/num_model_params)

        if config.adapter_type == "bn":
            adap_config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=4, non_linearity="gelu")#, leave_out=[0])
            adapters.init(self.model)
            self.model.add_adapter("bn_adapter", config=adap_config)
            self.model.train_adapter("bn_adapter")
        elif config.adapter_type == "lora":
            adap_config = LoRAConfig(selfattn_lora=True, intermediate_lora=True)
            adapters.init(self.model)
            self.model.add_adapter("lora_adapter", config=adap_config)  
            self.model.train_adapter("lora_adapter")
            
        for name, param in self.model.named_parameters():
            temp = name.split(".") 
            if "embed" in name:
                param.requires_grad = True
            if "shared" in name:
                param.requires_grad = True
            if temp[1]=="encoder" and "adapter" not in name:
                if "layers" in temp:
                    if int(temp[3]) in config.unfreeze_params["encoder"]:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            if temp[1]=="decoder" and "adapter" not in name:
                if "layers" in temp :
                    if int(temp[3]) in config.unfreeze_params["decoder"]:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        
        
        if self.config.local_rank == 0:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    print(n)
                    
            num_params_to_optimize = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            num_model_params = sum(p.numel() for p in self.model.parameters())
            self.total_params = num_model_params
            self.total_trainable_params = num_params_to_optimize
            print("Number of model parameters after adapter:", num_model_params)
            print("Total number of params to be optimized are after adapter: ", num_params_to_optimize)
            print("Percentage of parameters to be optimized after adapter: ", 100*num_params_to_optimize/num_model_params)
        
        self.model.cuda(self.config.local_rank)
        print("Memory consumed after moving model to GPU", round(torch.cuda.memory_allocated(self.config.local_rank)/(1024**3), 2), "GB on rank", self.config.local_rank)
        
        # freeze_params(model, args.freeze_exception_list, rank) # May be freeze params of needed?    
        ### NOTE: Please freeze params before wrapping the model in DDP.
        
        # Convert the model to DistributedDataParallel
        self.model = DistributedDataParallel(self.model, device_ids=[self.config.local_rank])
        if config.local_rank == 0:
            print("Memory consumed after wrapping model in DDP", round(torch.cuda.memory_allocated(config.local_rank)/(1024**3), 2), "GB on rank", config.local_rank)
        
    def get_optimizer_and_scheduler(self, config):
        no_decay = ["bias", "LayerNorm.weight"]
        # To read about weight decay: https://arxiv.org/pdf/1711.05101.pdf, baisically it decouples L2 reg in Adam Optimizer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.00001,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ] ## We suppose that weight decay will be used except for biases and layer norm weights.
    
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=1e-9)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=config.warmup_steps, 
            num_training_steps=config.num_batches
            )
        
        # Setting a minimum lr
        while self.scheduler.get_last_lr()[0] < 1e-7:
            self.scheduler.step()
            
        if config.local_rank == 0:
            print("Initial LR is:", self.scheduler.get_lr()[0], ", max LR is:", config.lr, ", warmup steps are:", config.warmup_steps, ", total number of batches/steps are:", config.num_batches)
   
    def run_evaluation(self, config):
        self.model.eval()
        individual_sbleu_history = [[dev_pair, 0] for dev_pair, _ in self.dev_files]
        sbleus = []
        print("Evaluating on dev set")
        for dev_idx, [dev_pair, dev_pair_info] in enumerate(self.dev_files):
            print("Evaluating on", dev_pair)
            slangtlang =dev_pair.strip().split("-")
            slang=slangtlang[0]
            tlang=slangtlang[1]
            
            dataset = MultilingualDataset(dev_pair_info, config, dataset_type="dev", lang_pair=dev_pair)
            val_dataloader = DataLoader(
                dataset, 
                batch_size=config.dev_batch_size, 
                shuffle=False,
                collate_fn=lambda b: custom_collate(b, config, self.tok)
            )
            predictions = [[] for dev_pair, dev_pair_info in self.dev_files]
            target_sentences = [[] for dev_pair, dev_pair_info in self.dev_files]
            for dev_input_ids, dev_input_masks, _, __, tgt_sentences in val_dataloader:
                dev_input_ids = dev_input_ids.to(self.device) 
                dev_input_masks = dev_input_masks.to(self.device) 
                
                with torch.no_grad():
                    translations = self.model.module.generate(
                        dev_input_ids, 
                        use_cache=True, 
                        num_beams=1, 
                        max_length=int(len(dev_input_ids[0])*2), 
                        min_length=int(len(dev_input_ids[0])*0.1), 
                        early_stopping=True, 
                        attention_mask=dev_input_masks, 
                        pad_token_id=self.tok.pad_token_id, 
                        eos_token_id=self.tok(["</s>"], add_special_tokens=False).input_ids[0][0], 
                        decoder_start_token_id=self.tok(["<2"+tlang+">"], add_special_tokens=False).input_ids[0][0], 
                        bos_token_id=self.tok(["<s>"], add_special_tokens=False).input_ids[0][0]
                    )
                del dev_input_ids 
                del dev_input_masks 
                translations = translations.to('cpu')
                
                for translation in translations:
                    translation  = self.tok.decode(translation, skip_special_tokens=True, clean_up_tokenization_spaces=False) ### Get the raw sentences.
                    predictions[dev_idx].append(translation)
                target_sentences[dev_idx].extend(tgt_sentences)
            
            # for i in range(20):
            #     print("translation", predictions[dev_idx][i], "---- target", target_sentences[dev_idx][i])
            # print(len(target_sentences[dev_idx])==len(predictions[dev_idx]))
            del translations
            
            sbleu = get_sacrebleu(target_sentences[dev_idx], predictions[dev_idx])
            individual_sbleu_history[dev_idx][1] = sbleu
            sbleus.append(sbleu)
            print("Evaluation on", dev_pair, "done.", sbleu)
            
        sbleus = sum(sbleus)/len(sbleus)
        
        return sbleus, individual_sbleu_history

    def train(self, config):
        return
        # wandb_run_id = None
        step = 0
        epoch = 0
        if config.finetuned_model_path is not None and os.path.exists(config.finetuned_model_path+".checkpoint_dict"):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % config.local_rank}
            CHECKPOINT_PATH = config.model_path
            checkpoint_dict = torch.load(CHECKPOINT_PATH+".checkpoint_dict", map_location=map_location)
            self.model.load_state_dict(checkpoint_dict['model'])
            print("Reloading optimizer")
            self.optimizer.load_state_dict(checkpoint_dict['optimizer']) 
            print("Reloading scheduler")
            self.scheduler.load_state_dict(checkpoint_dict['scheduler']) 
            print("Reloading step. This means we resume training.")
            step = checkpoint_dict['step'] + 1
            # wandb_run_id = checkpoint_dict['wandb_run_id']
            epoch = checkpoint_dict['epoch'] 
        
        # if config.local_rank == 0:
        #     if wandb_run_id is not None:
        #         wandb.init(
        #             project=config.wb_project,
        #             id=wandb_run_id,
        #             config=config.__dict__
        #         )
        #     else:
        #         wandb.init(
        #             project=config.wb_project,
        #             name = config.wb_run,
        #             config=config.__dict__
        #         )
        
        barrier()
        
        device = torch.device(f"cuda:{config.local_rank}")
    
        max_individual_sbleu = [[dev_pair, 0] for dev_pair, _ in self.dev_files]
        max_individual_sbleu_step = [[dev_pair, 0] for dev_pair, _ in self.dev_files]
        max_global_sbleu = 0
        # max_global_sbleu_step = 0
        start = time.time()
        batch_stats = torch.zeros(7, dtype=torch.long, device=device)
        avg_memory_stats = torch.zeros(2, dtype=torch.float, device=device)
        
        self.model.train()
        for epoch in range(epoch, config.num_epochs):
            if config.local_rank==0:
                print("Starting EPOCH", epoch)
            self.train_dataset.reload_data()
            train_dataloader = DataLoader(
                self.train_dataset, 
                batch_size=config.batch_size, 
                shuffle=False, 
                sampler=DistributedSampler(self.train_dataset, shuffle=False), 
                collate_fn=lambda b: custom_collate(b, config, self.tok)
                )
            for input_ids, input_masks, decoder_input_ids, labels, _ in train_dataloader:
                
                if step%config.eval_every==0 and config.local_rank==0:
                    CHECKPOINT_PATH = config.model_path
                    checkpoint_dict = {
                        'model': self.model.state_dict(), 
                        'optimizer': self.optimizer.state_dict(), 
                        'scheduler': self.scheduler.state_dict(), 
                        'step': step,
                        'epoch': epoch,
                        # "wandb_run_id": wandb.run.id
                        }
                    
                    sbleus, individual_sbleu_history = self.run_evaluation(config)
                    save_model = False
                    
                    for lang_idx, (lang_pair, sbleu) in enumerate(individual_sbleu_history):
                        print("BLEU score using ScareBLEU after", step, "iterations is", round(sbleu, 2), "for language pair", lang_pair)

                        if sbleu > max_individual_sbleu[lang_idx][1]: ## Update the best score and step number. If the score has improved then save a model copy for this pair. Although we will stop on the global score (average across scores over all pairs) we save these models if we want a model that performs the best on a single pair.
                            max_individual_sbleu[lang_idx][1] = sbleu
                            max_individual_sbleu_step[lang_idx][1] = step
                            print("New peak reached for", lang_pair,". Saving.")
                            save_model=True
                    wandb.log({f"{lang_pair}_sbleu": sbleu for lang_pair, sbleu in individual_sbleu_history} )
                        # wandb.log({f"{lang_pair}_sbleu": sbleu}, step=step)
                        
                    if save_model:
                        # torch.save(checkpoint_dict, CHECKPOINT_PATH+".best_dev_bleu."+lang_pair+"."+str(step))
                        torch.save(self.model.module.state_dict(), CHECKPOINT_PATH) 
                        torch.save(checkpoint_dict, CHECKPOINT_PATH+".checkpoint_dict")
                    print(f"Overall BLEU score: {sbleus}")
                    wandb.log({"sbleu": sbleus}, step=step)
                    
                    if sbleu > max_global_sbleu: 
                        max_global_sbleu = sbleu
                        # max_global_sbleu_step = curr_eval_step
                    
                    # TODO: we can do early stopping here
                    
                self.model.train()
                    
                input_ids=input_ids.to(device) 
                input_masks=input_masks.to(device) 
                decoder_input_ids=decoder_input_ids.to(device) 
                labels=labels.to(device) 
                
                self.optimizer.zero_grad(set_to_none=True)
                    
                output = self.model(
                    input_ids=input_ids, 
                    attention_mask=input_masks,
                    decoder_input_ids=decoder_input_ids
                    ) ## Run the model and get logits. 
                
                # print(output.logits.shape, labels.shape) # (16, 45, 64014)
                logits = output.logits
                lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                
                loss = label_smoothed_nll_loss(
                    lprobs, labels, config.label_smoothing, ignore_index=self.tok.pad_token_id
                )
                fwd_memory = torch.cuda.memory_allocated(f"cuda:{config.local_rank}")/(1024**3)
                loss.backward()
                bwd_memory=torch.cuda.memory_allocated(f"cuda:{config.local_rank}")/(1024**3)
                
                avg_memory_stats += torch.tensor([fwd_memory, bwd_memory], dtype=torch.float, device=device)
                padding_tokens_dec = (decoder_input_ids == self.tok.pad_token_id).sum().item()
                padding_tokens_enc = (input_ids == self.tok.pad_token_id).sum().item()
                total_tokens_dec = decoder_input_ids.numel()
                total_tokens_enc = input_ids.numel()
                non_padding_tokens_dec = total_tokens_dec - padding_tokens_dec
                non_padding_tokens_enc = total_tokens_enc - padding_tokens_enc
                num_sequences = input_ids.size()[0]
                batch_stats += torch.tensor([non_padding_tokens_enc, padding_tokens_enc, total_tokens_enc, non_padding_tokens_dec, padding_tokens_dec, total_tokens_dec, num_sequences], dtype=torch.long, device=device)
                
                self.optimizer.step()
                self.scheduler.step()
                

                if config.local_rank==0:
                    wandb.log({"loss": loss.detach().cpu().numpy()}, step=step)
                    wandb.log({"learning rate": self.scheduler.get_lr()[0]}, step=step)
                    
                if step%100==0:
                    end = time.time()
                    all_reduce(batch_stats)
                    all_reduce(avg_memory_stats)
                    avg_memory_stats = avg_memory_stats/config.world_size/100
                    fwd_memory, bwd_memory = [round(mem,2) for mem in avg_memory_stats.tolist()]
                    non_padding_tokens_enc, padding_tokens_enc, total_tokens_enc, non_padding_tokens_dec, padding_tokens_dec, total_tokens_dec, num_sequences = batch_stats.tolist()
                    non_padding_percent_enc = round(non_padding_tokens_enc/total_tokens_enc*100, 2)
                    non_padding_percent_dec = round(non_padding_tokens_dec/total_tokens_dec*100, 2)
                    
                    if config.local_rank % config.world_size == 0:
                        print("Step:", step, "| Loss:", round(loss.item(),2), " | Time:", round(end-start, 2), "s/100 batches. | Fwd-Bwd (avg):", fwd_memory, ",", bwd_memory, "GB. | Enc Non-pad, Pad, Total tokens, Non pad percentage:", non_padding_tokens_enc, ",", padding_tokens_enc, ",", total_tokens_enc, ",", non_padding_percent_enc, "% | Dec Non-pad, Pad, Total tokens, Non pad percentage:", non_padding_tokens_dec, ",", padding_tokens_dec, ",", total_tokens_dec, ",", non_padding_percent_dec, "% | Num sequences:", num_sequences)
                        sys.stdout.flush()
                    batch_stats = torch.zeros(7, dtype=torch.long, device=device)
                    avg_memory_stats = torch.zeros(2, dtype=torch.float, device=device)
                    start = time.time()
                    
                del input_ids 
                del input_masks 
                del decoder_input_ids 
                del labels 
                del output, loss
                    
                step+=1
                
        CHECKPOINT_PATH = config.model_path
        checkpoint_dict = {
            'model': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(), 
            'scheduler': self.scheduler.state_dict(), 
            'step': step,
            'epoch': epoch,
            # "wandb_run_id": wandb.run.id
            }
        torch.save(self.model.module.state_dict(), CHECKPOINT_PATH+".final") 
        torch.save(checkpoint_dict, CHECKPOINT_PATH+".checkpoint_dict.final")
        
    

            
if __name__=="__main__":
    config = get_default_config()
    
    config.local_rank = int(os.environ['LOCAL_RANK'])
    config.global_rank = int(os.environ['RANK'])
    config.world_size = int(os.environ['WORLD_SIZE'])
    
    init_process_group(backend='nccl')
    
    train_files = [(slang+"-"+tlang, (train_src, train_tgt)) 
                   for slang, tlang, train_src, train_tgt in zip(config.train_src_languages, config.train_tgt_languages, config.train_src_files, config.train_tgt_files)]
    
    dev_files = [(slang+"-"+tlang, (dev_src, dev_tgt)) 
                   for slang, tlang, dev_src, dev_tgt in zip(config.dev_src_languages, config.dev_tgt_languages, config.dev_src_files, config.dev_tgt_files)]
    
    if config.local_rank == 0:
        print("Training Files are: ", train_files)
        print("Dev Files are: ", dev_files)
        
    if config.local_rank == 0:
        wandb.login()    
    
    trainer = Trainer(config, train_files, dev_files)
                
    trainer.train(config)
    # print(config.__dict__)
    
    destroy_process_group()