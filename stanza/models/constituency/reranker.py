import numpy as np
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoConfig

class GPTReranker:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = GPT2Tokenizer.from_pretrained("/home/john/gpt2/tokenizer_gpt_neo_vi")

        configuration_GPT2_neo = GPTNeoConfig.from_pretrained("NlpHUST/gpt-neo-vi-small", output_hidden_states=False)
        configuration_GPT2_neo.bos_token_id = tokenizer.bos_token_id
        configuration_GPT2_neo.eos_token_id = tokenizer.eos_token_id
        configuration_GPT2_neo.vocab_size = tokenizer.vocab_size

        model = GPTNeoForCausalLM.from_pretrained("NlpHUST/gpt-neo-vi-small", config=configuration_GPT2_neo)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load("/home/john/gpt2/official-neo-16_gpt-neo-vi-small.pt"))
        model = model.to(device)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        
    def sent_scoring(self, sentence):
        #if tokenizer_type == "bert":
        #    encodings = torch.tensor(tokenizer.encode(sentence, truncation = True, max_length = 1000)).unsqueeze(0)
        #else:
        #    words = sentence.split()
        #    processed_sent = ' '.join([word.replace("_", " ") if ("(_" not in word and ")_" not in word) else word for word in words]).strip()
        #    encodings = torch.tensor(tokenizer.encode("<s> " + processed_sent + " </s>", truncation = True, max_length = 1000)).unsqueeze(0)

        words = sentence.split()
        processed_sent = ' '.join([word.replace("_", " ") if ("(_" not in word and ")_" not in word) else word for word in words]).strip()
        encodings = torch.tensor(self.tokenizer.encode("<s> " + processed_sent + " </s>", truncation = True, max_length = 1000)).unsqueeze(0)

        if any(v is None for v in encodings):
            raise RuntimeError("Could not process sentence!")

        # input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        input_ids = encodings.to('cuda')
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            sentence_prob = loss.item()
            return sentence_prob
            #return np.exp(sentence_prob)

    def rerank(self, sentences):
        sentences = ["{:L}".format(sent) for sent in sentences]
        #for sent in sentences:
        #    print(sent)
        scores = [self.sent_scoring(sent) for sent in sentences]
        return np.argmin(scores), scores

def main():
    reranker = GPTReranker()
    print(reranker.sent_scoring("(_ROOT (_S (_NP (_NOUN Nhật_ký )_NOUN (_PART ơi )_PART )_NP (_PUNCT ! )_PUNCT )_S )_ROOT"))

if __name__ == '__main__':
    main()
