# torchrun --nproc_per_node=2 --nnodes=1 --master_port 29600 train.py

from dataclasses import dataclass, field
import os

@dataclass
class Config:
    train_src_languages : list
    train_tgt_languages : list
    dev_src_languages : list
    dev_tgt_languages : list
    train_src_files : list
    train_tgt_files : list
    dev_src_files : list
    dev_tgt_files : list
    encoder_layers : int
    decoder_layers : int
    encoder_attention_heads : int
    decoder_attention_heads : int
    encoder_ffn_dim : int
    decoder_ffn_dim : int
    d_model : int
    pretrained_model: str
    tokenizer_name_or_path: str
    wb_project: str
    wb_run: str
    
    unseen_dev_src_files : list
    unseen_dev_tgt_files : list
    no_languages : int
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    positional_encodings: bool = False
    activation_function: str = "gelu"
    lr: float = 0.001
    num_batches: int = 500000
    warmup_steps: int = 16000
    max_src_length: int = 200
    max_tgt_length: int = 200
    hard_truncate_length: int = 800
    batch_size: int = 16
    label_smoothing: float = 0.1
    eval_every: int = 1000
    model_path: str = "model.ft"
    dev_batch_size: int = 32
    num_epochs: int = 100
    finetuned_model_path: str = None
    adapter_type = "bn"
    unfreeze_params = {"encoder":[], "decoder":[0,1,2,3,4,5]}
    unfreeze_embeddings =False
    
    
    
def get_default_config():
    setting = "M2O"
    
    languages = ["bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
    # languages = ['si', 'ne']
    dataset_path = "dataset"
    unseen_dataset_path = "Dataset_Unseen"
    no_languages = 10
    
    if setting == "M2O":
        train_src_languages = languages
        train_tgt_languages = ["en"]*len(languages)
        dev_src_languages = languages
        dev_tgt_languages = ["en"]*len(languages)
        train_src_files = [os.path.join(dataset_path, f"train_data/en-{lang}/train.{lang}") for lang in languages]
        train_tgt_files = [os.path.join(dataset_path, f"train_data/en-{lang}/train.en") for lang in languages]
        dev_src_files = [os.path.join(dataset_path, f"dev_data/dev.{lang}") for lang in languages]
        dev_tgt_files = [os.path.join(dataset_path, f"dev_data/dev.en") for lang in languages]
        if len(languages) == 10:
            unseen_dev_src_files = [os.path.join(unseen_dataset_path, f"dev_data/dev.{lang}") for lang in languages]
            unseen_dev_tgt_files = [os.path.join(unseen_dataset_path, f"dev_data/dev.en") for lang in languages]
    elif setting == "O2M":
        train_src_languages = ["en"]*len(languages)
        train_tgt_languages = languages
        dev_src_languages = ["en"]*len(languages)
        dev_tgt_languages = languages
        train_tgt_files = [os.path.join(dataset_path, f"train_data/en-{lang}/train.{lang}") for lang in languages]
        train_src_files = [os.path.join(dataset_path, f"train_data/en-{lang}/train.en") for lang in languages]
        dev_tgt_files = [os.path.join(dataset_path, f"dev_data/dev.{lang}") for lang in languages]
        dev_src_files = [os.path.join(dataset_path, f"dev_data/dev.en") for lang in languages]
        if len(languages) == 10:
            unseen_dev_src_files = [os.path.join(unseen_dataset_path, f"dev_data/dev.{lang}") for lang in languages]
            unseen_dev_tgt_files = [os.path.join(unseen_dataset_path, f"dev_data/dev.en") for lang in languages]
    
    config = Config(
        train_src_languages = train_src_languages,
        train_tgt_languages = train_tgt_languages,
        dev_src_languages = dev_src_languages,
        dev_tgt_languages = dev_tgt_languages,
        train_src_files = train_src_files,
        train_tgt_files = train_tgt_files,
        dev_src_files = dev_src_files,
        dev_tgt_files = dev_tgt_files,
        unseen_dev_src_files = unseen_dev_src_files,
        unseen_dev_tgt_files = unseen_dev_tgt_files,
        no_languages = no_languages,

        
        encoder_layers = 6,
        decoder_layers = 6,
        encoder_attention_heads = 16,
        decoder_attention_heads = 16,
        encoder_ffn_dim = 4096,
        decoder_ffn_dim = 4096,
        d_model = 1024,
        pretrained_model="indicbart_model.ckpt",
        tokenizer_name_or_path="albert-indicunified64k",
        wb_project="indic-bart-adapter",
        wb_run="bn-adapter-10-lang-case-1",
        warmup_steps=16000,
        dropout=0.1,
        batch_size=16,
        model_path="bn-adapter-case1-model-all-lang.ft",
        eval_every=1000,
        finetuned_model_path=None
    )
    return config