from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

class EthicalGuardTokenizerBuilder:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # Instantiate a clean, high-performance Byte-Pair Encoding model from scratch
        self.bpe_model = models.BPE(unk_token="<unk>")
        self.tokenizer = Tokenizer(self.bpe_model)
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    def compile_tokenizer(self, training_corpus_path: str, save_directory: str):
        """ Trains our Causal Language Modeling tokenizer on our safety taxonomy data structures """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>", "System:", "User Context Block:"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        # Execute raw BPE sequence extraction pass
        self.tokenizer.train(files=[training_corpus_path], trainer=trainer)
        
        # Wrap into standard Hugging Face asset serialization wrapper
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>"
        )
        os.makedirs(save_directory, exist_ok=True)
        hf_tokenizer.save_pretrained(save_directory)
        print(f"[SUCCESS] Custom Tokenizer saved securely to: {save_directory}")

if __name__ == "__main__":
    print("EthicalGuard Tokenizer Ingestion Pipeline Framework Initialized.")
