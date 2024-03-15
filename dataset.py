from torch.utils.data import Dataset
import numpy as np

# This
class MultilingualDataset(Dataset):
    def __init__(self, files, config, dataset_type="train", lang_pair=None):
        self.dataset_type = dataset_type
        self.files = files
        self.lang_pair = lang_pair
        self.config = config
        
        if dataset_type == "train":
            self.src_target_lang_generators = []
            self.language_list = [lang for lang, _ in files]
            self.language_indices = [i for i in range(len(self.language_list))]
            
            self.total_sent = 0
            probs = []
            for lang, file_details in files:
                src_file_content = open(file_details[0], "r").readlines()
                tgt_file_content = open(file_details[1], "r").readlines()
                probs.append(len(src_file_content))
                self.total_sent += len(src_file_content)
                
                src_tgt_content = list(zip(src_file_content, tgt_file_content))
                self.src_target_lang_generators.append(self.yield_data_lang(src_tgt_content, lang))
                
            probs = [p/sum(probs) for p in probs]
            # Temperature > 1: Flattens the distribution, making probabilities more similar.
            # Temperature < 1: Sharpens the distribution, emphasizing larger probabilities and reducing smaller ones
            data_sampling_temp = 5
            probs = [p**(1/data_sampling_temp) for p in probs]
            self.probs = [p/sum(probs) for p in probs]
            
        elif dataset_type=="dev":
            file_details = files
            slangtlang =lang_pair.strip().split("-")
            slang=slangtlang[0]
            tlang=slangtlang[1]
            self.data_src = open(file_details[0], "r").readlines()
            self.data_tgt = open(file_details[1], "r").readlines()
            self.data_lang = lang_pair
            self.total_sent = len(self.data_src)
        
    def reload_data(self):
        self.sample_data(self.files, self.dataset_type, self.lang_pair)
            
    def sample_data(self, files, dataset_type="train", lang_pair=None):
        if dataset_type == "train":
            
            self.files = files
            self.data_src = []
            self.data_tgt = []
            self.data_lang = []
            
            for i in range(self.total_sent):
                langugage_index = np.random.choice(self.language_indices, p=self.probs)
                language = self.language_list[langugage_index]
                src_sent, tgt_sent = next(self.src_target_lang_generators[langugage_index])
                self.data_src.append(src_sent)
                self.data_tgt.append(tgt_sent)
                self.data_lang.append(language)
                
        
        
    def yield_data_lang(self, corpus, lang):
        epoch_counter = 0
        num_lines = len(corpus)
        
        while True:
            for src_line, tgt_line in corpus:
                yield src_line, tgt_line
            epoch_counter += 1
            print(f"Epoch {epoch_counter} completed for {lang} corpus")
    
    def __len__(self):
        return self.total_sent
    
    def __getitem__(self, index):
        if self.dataset_type == "train":
            language = self.data_lang[index]
        else:
            language = self.data_lang
            
        src_sent, tgt_sent = self.data_src[index], self.data_tgt[index]
        
        return src_sent, tgt_sent, language
        
        
        
def custom_collate(batch, config, tok):
    
    encoder_input_batch = []
    decoder_input_batch = []
    decoder_label_batch = []
    tgt_sentences = []
    
    for src_sent, tgt_sent, language in batch:
        tgt_sentences.append(tgt_sent)
        src_sent_split = src_sent.split(" ")
        tgt_sent_split = tgt_sent.split(" ")
        tgt_sent_len = len(tgt_sent_split)
        src_sent_len = len(src_sent_split)
        
        if src_sent_len < 1 or tgt_sent_len < 1:
            continue
        else:   # Initial truncation
            if src_sent_len >= config.max_src_length:
                src_sent_split = src_sent_split[:config.max_src_length]
                src_sent = " ".join(src_sent_split)
                src_sent_len = config.max_src_length
            if tgt_sent_len >= config.max_tgt_length:
                tgt_sent_split = tgt_sent_split[:config.max_tgt_length]
                tgt_sent = " ".join(tgt_sent_split)
                tgt_sent_len = config.max_tgt_length
                
        iids = tok(src_sent, add_special_tokens=False).input_ids
        curr_src_sent_len = len(iids)
        if curr_src_sent_len > config.hard_truncate_length:
            src_sent = tok.decode(iids[:config.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            curr_src_sent_len = config.hard_truncate_length
        
        iids = tok(tgt_sent, add_special_tokens=False).input_ids
        curr_tgt_sent_len = len(iids)
        if curr_tgt_sent_len > config.hard_truncate_length:
            tgt_sent = tok.decode(iids[:config.hard_truncate_length], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            curr_tgt_sent_len = config.hard_truncate_length
        
        slangtlang = language.strip().split("-")
        slang = "<2"+slangtlang[0]+">"
        tlang = "<2"+slangtlang[1]+">"
        encoder_input_batch.append(src_sent + " </s> " + slang)
        decoder_input_batch.append(tlang + " " + tgt_sent)
        decoder_label_batch.append(tgt_sent + " </s>")
        
    input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    input_masks = (input_ids != tok.pad_token_id).int()
    decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    
    return input_ids, input_masks, decoder_input_ids, labels, tgt_sentences
