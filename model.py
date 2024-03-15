from transformers import AlbertModel, AlbertTokenizer, MBartConfig, MBartForConditionalGeneration
import torch

def get_tokenizer(config):
    tokenizer = None
    if "indicbart" in config.pretrained_model:
        # model = AlbertModel.from_pretrained(config.pretrained_model)
        tokenizer = AlbertTokenizer.from_pretrained(config.tokenizer_name_or_path, do_lower_case=False, use_fast=False, keep_accents=True)
    return tokenizer

def get_model(config, tok):
    model = None
    if "indicbart" in config.pretrained_model:
        mbart_config = MBartConfig(vocab_size=len(tok), target_vocab_size=0, 
                             init_std=0.02, 
                             initialization_scheme="static", 
                             encoder_layers=config.encoder_layers, 
                             decoder_layers=config.decoder_layers, 
                             dropout=config.dropout, 
                             attention_dropout=config.attention_dropout, 
                             activation_dropout=config.activation_dropout, 
                             encoder_attention_heads=config.encoder_attention_heads, 
                             decoder_attention_head=config.decoder_attention_heads, 
                             encoder_ffn_dim=config.encoder_ffn_dim, 
                             decoder_ffn_dim=config.decoder_ffn_dim, 
                             d_model=config.d_model, 
                             pad_token_id=tok.pad_token_id, 
                             eos_token_id=tok(["</s>"], add_special_tokens=False).input_ids[0][0], 
                             bos_token_id=tok(["<s>"], add_special_tokens=False).input_ids[0][0], 
                             softmax_temperature=1, 
                             positional_encodings=config.positional_encodings, 
                             activation_function=config.activation_function, 
                             tokenizer_class="AlbertTokenizer"
                             )
        model = MBartForConditionalGeneration(mbart_config)
        state_dict = torch.load("indicbart_model.ckpt")
        model.load_state_dict(state_dict) 
        # Here include the code to load model when already saved
        
    return model