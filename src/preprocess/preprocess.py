from multiprocessing import Pool
from pathlib import Path

import consts
import utils
from tqdm import tqdm


class Preprocessor:
    def __init__(self, path_corpus, num_cores=8, use_cache=True):
        self.use_cache = use_cache
        self.num_cores = num_cores

        # establish preprocess folder
        self.path_corpus = Path(path_corpus)
        self.dir_corpus = self.path_corpus.parent
        self.dir_preprocess = self.dir_corpus / f"preprocess-{consts.LM_NAME_SUFFIX}"
        self.dir_preprocess.mkdir(exist_ok=True)

        # path_tokenized_corpus: wordpieces tokenized with huggingface LM tokenizer
        # path_tokenized_id_corpus: tokenized wordpiece ids with word boundaries
        self.path_tokenized_corpus = (
            self.dir_preprocess / f"tokenized.{self.path_corpus.name}"
        )
        self.path_tokenized_id_corpus = (
            self.dir_preprocess / f"tokenized.id.{self.path_corpus.name}"
        )

    @staticmethod
    def _par_tokenize_doc(doc):
        docid = doc["_id_"]
        sents = doc["sents"]

        # tokenize
        # For GPT-2: add space before each raw sentence to tokenize the first token with PREFIX_TOKEN for phrase matching
        tokenized_sents = [
            consts.LM_TOKENIZER.tokenize(" " + s, add_special_tokens=False)
            for s in sents
        ]
        # # For BERT: add the PREFIX_TOKEN token manually based on SPLIT_TOKEN
        # tokenized_sents = [consts.LM_TOKENIZER.tokenize(s) for s in sents]
        # for i, tokens in enumerate(tokenized_sents):
        #     trans_tokens = list()  # transition tokens
        #     start_split = False
        #     for token in tokens:
        #         if (consts.SPLIT_TOKEN in token) & (start_split == False):
        #             trans_tokens.append(consts.PREFIX_TOKEN + token)
        #             start_split = True
        #         elif (consts.SPLIT_TOKEN in token) & (start_split == True):
        #             trans_tokens.append(token)
        #         elif (consts.SPLIT_TOKEN not in token) & (start_split == True):
        #             trans_tokens.append(token)
        #             start_split = False
        #         elif (consts.SPLIT_TOKEN not in token) & (start_split == False):
        #             trans_tokens.append(consts.PREFIX_TOKEN + token)
        #     tokenized_sents[i] = trans_tokens

        cleaned_tokenized_sents = []
        for tokens in tokenized_sents:
            tokens_batch = utils.get_batches(tokens, batch_size=consts.MAX_SENT_LEN)
            cleaned_tokenized_sents += tokens_batch
        tokenized_doc = {
            "_id_": docid,
            "sents": [" ".join(tokens) for tokens in cleaned_tokenized_sents],
        }

        tokenized_id_doc = {"_id_": doc["_id_"], "sents": []}
        for tokens in cleaned_tokenized_sents:
            widxs = [
                i
                for i, token in enumerate(tokens)
                if token.startswith(consts.PREFIX_TOKEN)
            ]  # the indices of start of words
            trans_tokens = tokens.copy()
            trans_tokens = [
                token.replace(consts.PREFIX_TOKEN, "") for token in trans_tokens
            ]
            # For GPT-2
            ids = consts.LM_TOKENIZER.convert_tokens_to_ids(tokens)
            # # For BERT
            # ids = consts.LM_TOKENIZER.convert_tokens_to_ids(trans_tokens)
            tokenized_id_doc["sents"].append({"ids": ids, "widxs": widxs})

        return tokenized_doc, tokenized_id_doc

    def tokenize_corpus(self):
        if (
            self.use_cache
            and utils.IO.is_valid_file(self.path_tokenized_corpus)
            and utils.IO.is_valid_file(self.path_tokenized_id_corpus)
        ):
            print(f"[Preprocessor] Use cache: {self.path_tokenized_corpus}")
            return
        docs = utils.JsonLine.load(self.path_corpus)
        pool = Pool(processes=self.num_cores)
        pool_func = pool.imap(func=Preprocessor._par_tokenize_doc, iterable=docs)
        doc_tuples = list(
            tqdm(
                pool_func,
                total=len(docs),
                ncols=100,
                desc=f"[Tokenize] {self.path_corpus}",
            )
        )
        tokenized_docs = [doc for doc, iddoc in doc_tuples]
        tokenized_id_docs = [iddoc for doc, iddoc in doc_tuples]
        pool.close()
        pool.join()
        utils.JsonLine.dump(tokenized_docs, self.path_tokenized_corpus)
        utils.JsonLine.dump(tokenized_id_docs, self.path_tokenized_id_corpus)
