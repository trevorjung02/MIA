import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class DetectGPTPerturbation():
    def __init__(self):
        super().__init__()
        self.name = "detectgpt"
        self.mask_filling_model_name = "t5-large"
        
        print(f"Loading in model {self.mask_filling_model_name}")
        CACHE_DIR = '/mmfs1/gscratch/ark/tjung2/continual-knowledge-learning/huggingface'
        self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(self.mask_filling_model_name,device_map='auto', cache_dir=CACHE_DIR)
        try:
            self.n_positions = self.mask_model.config.n_positions
        except AttributeError:
            self.n_positions = 512
        
        self.mask_tokenizer = AutoTokenizer.from_pretrained(self.mask_filling_model_name, model_max_length=self.n_positions)
        # self.mask_model.to()
        
        # default arguments from detectgpt
        self.n_spans = None
        self.mask_top_p = 1.0
        self.buffer_size = 1
        self.chunk_size = 20
        self.min_seq_length = 20
        self.max_spans = 25
        self.max_seq_length = 256
        
    def perturb(self, text):
        tokens = text.split(' ')
        if len(tokens) > 25:
            span_length = 2
        else:
            span_length = 1
        perturbed = self.perturb_texts(text, span_length, 0.3, ceil_pct=True)
        return perturbed
    
    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False):
        orig_text = text
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        if self.n_spans is None:
            n_spans = pct * len(tokens) / (span_length + self.buffer_size * 2)
            if ceil_pct:
                n_spans = np.ceil(n_spans)
            n_spans = int(n_spans)
            # print(f"Filling {n_spans} spans")
        else:
            n_spans = self.n_spans
            
        
        if n_spans > self.max_spans:
            print(f"Warning: reducing from {n_spans} to {self.max_spans}")
            n_spans = self.max_spans
        
        n_masks = 0
    
        while n_masks < n_spans:
            if len(tokens) - span_length <= 0:
                return " ".join(tokens)
            
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text


    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
        
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=self.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        pattern = re.compile(r"<extra_id_\d+>")
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills


    def apply_extracted_fills(self, masked_texts, extracted_fills, accept_faulty_perturb=False):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n and not accept_faulty_perturb:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts


    def perturb_texts_(self, texts, span_length, pct, ceil_pct=False):
        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
            
            if attempts > 3:
                span_length = 1
                pct = 0.2
                ceil_pct = False
                # import pdb; pdb.set_trace()
                
            if attempts > 6:
                span_length = 5
                pct = 0.3
                ceil_pct = False
                # import pdb; pdb.set_trace()
                    
            elif attempts > 10:
                print('Unable to perturb, keeping this as is vanilla :()')
                import pdb; pdb.set_trace()
                break
        return perturbed_texts


    def perturb_texts(self, text, span_length, pct, ceil_pct=False):

        tokens = text.split(" ")
        if len(tokens) > self.max_seq_length:
            texts = []
            for i in range(0, len(tokens), self.max_seq_length):
                texts.append(" ".join(tokens[i:i+self.max_seq_length]))

                
        else:
            texts = [text]
                
        chunk_size = self.chunk_size
        if '11b' in self.mask_filling_model_name:
            chunk_size //= 2

        outputs = []
        # for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        for i in range(0, len(texts), chunk_size):
            outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
        output = " ".join(outputs)
        
        return output

if __name__ == "__main__":
    detect_gpt = DetectGPTPerturbation()
    perturbed = detect_gpt.perturb("To further investigate the biases of our model on the gender category, we look at the WinoGender benchmark (Rudinger et al., 2018), a co-reference resolution dataset. WinoGender is made of Winograd schema, and biases are evaluated by determining if a model co-reference resolution performance is impacted by the gender of the pronoun")
    print("perturbed: ", perturbed)
