"""Tweaked version of corresponding AllenNLP file"""
import logging
from collections import defaultdict
from typing import Dict, List, Callable

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides
from transformers import AutoTokenizer

from utils.helpers import START_TOKEN

logger = logging.getLogger(__name__)

# TODO(joelgrus): Figure out how to generate token_type_ids out of this token indexer.

# This is the default list of tokens that should not be lowercased.
_NEVER_LOWERCASE = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']


class WordpieceIndexer(TokenIndexer[int]):  #  进行wordpiece 编码.
    """
    A token indexer that does the wordpiece-tokenization (e.g. for BERT embeddings).
    If you are using one of the pretrained BERT models, you'll want to use the ``PretrainedBertIndexer``
    subclass rather than this base class.

    Parameters
    ----------
    vocab : ``Dict[str, int]``
        The mapping {wordpiece -> id}.  Note this is not an AllenNLP ``Vocabulary``.
    wordpiece_tokenizer : ``Callable[[str], List[str]]``
        A function that does the actual tokenization.
    namespace : str, optional (default: "wordpiece")
        The namespace in the AllenNLP ``Vocabulary`` into which the wordpieces
        will be loaded.
    use_starting_offsets : bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    max_pieces : int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    do_lowercase : ``bool``, optional (default=``False``)
        Should we lowercase the provided tokens before getting the indices?
        You would need to do this if you are using an -uncased BERT model
        but your DatasetReader is not lowercasing tokens (which might be the
        case if you're also using other embeddings based on cased tokens).
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    separator_token : ``str``, optional (default=``[SEP]``)
        This token indicates the segments in the sequence.
    truncate_long_sequences : ``bool``, optional (default=``True``)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 vocab: Dict[str, int],
                 bpe_ranks: Dict,
                 byte_encoder: Dict,
                 wordpiece_tokenizer: Callable[[str], List[str]],
                 namespace: str = "wordpiece",
                 use_starting_offsets: bool = False,
                 max_pieces: int = 512,
                 max_pieces_per_token: int = 3,
                 is_test=False,
                 do_lowercase: bool = False,
                 never_lowercase: List[str] = None,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 truncate_long_sequences: bool = True,
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self.vocab = vocab

        # The BERT code itself does a two-step tokenization:
        #    sentence -> [words], and then word -> [wordpieces]
        # In AllenNLP, the first step is implemented as the ``BertBasicWordSplitter``,
        # and this token indexer handles the second.
        self.wordpiece_tokenizer = wordpiece_tokenizer
        self.max_pieces_per_token = max_pieces_per_token
        self._namespace = namespace
        self._added_to_vocabulary = False
        self.max_pieces = max_pieces
        self.use_starting_offsets = use_starting_offsets ########## 使用开始index进行计算offset!
        self._do_lowercase = do_lowercase
        self._truncate_long_sequences = truncate_long_sequences
        self.max_pieces_per_sentence = 180 #----------------------这个地方写大一点.
        self.is_test = is_test
        self.cache = {}
        self.bpe_ranks = bpe_ranks
        self.byte_encoder = byte_encoder


        with open('nameofweight') as f: # 这种读文件的都是管最上乘调用的目录结构.作为基地路径.跟当前py无关.
            t=f.readline().strip()
        print('用的模型名字是',t)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=t, do_lower_case=True, do_basic_tokenize=False)








        if self.is_test:
            self.max_pieces_per_token = None

        if never_lowercase is None:
            # Use the defaults
            self._never_lowercase = set(_NEVER_LOWERCASE) # 设置哪些不用小写的关键位置token
        else:
            self._never_lowercase = set(never_lowercase)

        # Convert the start_tokens and end_tokens to wordpiece_ids
        self._start_piece_ids = [vocab[wordpiece]
                                 for token in (start_tokens or [])# 表示左边的有就取左边,否则就让他等于[]
                                 for wordpiece in wordpiece_tokenizer(token)]  # 把 cls进行编码成4个数字了!!!!!!!!!!!!!!!!!!
        self._end_piece_ids = [vocab[wordpiece]
                               for token in (end_tokens or []) # wordpiece_tokenizer 这个是拆分算法.
                               for wordpiece in wordpiece_tokenizer(token)]  # 把 sep进行了编码成4个数字了!!!!!!!!!!!!!!!!!!!!

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.???????????????奇怪怎么跑的. 写个pass, 但是还是又作用.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    def get_pairs(self, word):
        """Return set of symbol pairs in a word.

        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair,
                                                                    float(
                                                                        'inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def bpe_tokenize(self, text):
        """ Tokenize a string."""
        bpe_tokens = []
        for token in text.split():
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        text = (token.text.lower()
                if self._do_lowercase and token.text not in self._never_lowercase
                else token.text
                for token in tokens)

        token_wordpiece_ids = []
        for token in text:
            if self.bpe_ranks != {}:
                wps = self.bpe_tokenize(token)
            else:
                wps = self.wordpiece_tokenizer(token)
            limited_wps = [self.vocab[wordpiece] for wordpiece in wps][:self.max_pieces_per_token]
            token_wordpiece_ids.append(limited_wps)
# token_wordpiece_ids 获得所有编码, 因为这里面使用的是每一个单独的汉子,一个一个编码,而不是句子直接编码,所以答案会大量存在19.
        flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]
# 进行max_len切割.
        while not self.is_test and len(flat_wordpiece_ids) > \
                self.max_pieces_per_sentence - len(self._start_piece_ids) - len(self._end_piece_ids):
            max_pieces = max([len(row) for row in token_wordpiece_ids])
            token_wordpiece_ids = [row[:max_pieces - 1] for row in token_wordpiece_ids]
            flat_wordpiece_ids = [wordpiece for token in token_wordpiece_ids for wordpiece in token]
        window_length = self.max_pieces - len(self._start_piece_ids) - len(self._end_piece_ids)
        stride = window_length // 2

        offsets = []

        offset = len(self._start_piece_ids) if self.use_starting_offsets else len(self._start_piece_ids) - 1

        for token in token_wordpiece_ids:

            next_offset = 1 if self.use_starting_offsets else 0
            if self._truncate_long_sequences and offset >= window_length + next_offset:
                break

            if self.use_starting_offsets:
                offsets.append(offset)
                offset += len(token)
            else:
                offset += len(token)
                offsets.append(offset)

        if len(flat_wordpiece_ids) <= window_length:
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids)] #  前面加上cls 和sep后面
        elif self._truncate_long_sequences:
            logger.warning("Too many wordpieces, truncating sequence. If you would like a sliding window, set"
                           "`truncate_long_sequences` to False %s", str([token.text for token in tokens]))
            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids[:window_length])]
        else:

            wordpiece_windows = [self._add_start_and_end(flat_wordpiece_ids[i:i + window_length])
                                 for i in range(0, len(flat_wordpiece_ids), stride)]

            last_window = wordpiece_windows[-1][1:]
            penultimate_window = wordpiece_windows[-2]
            if last_window == penultimate_window[-len(last_window):]:
                wordpiece_windows = wordpiece_windows[:-1]

#
        wordpiece_ids = [wordpiece for sequence in wordpiece_windows for wordpiece in sequence]

        mask = [1 for _ in offsets]

        '''
        下面开始给出自己对于中文的改进算法, 节省计算19这个数值.!!!!!!!!!!!!!!!!
        '''



        # print(111111111111)


        text2=[token.text.lower()
                if self._do_lowercase and token.text not in self._never_lowercase
                else token.text
                for token in tokens]
        xlnetEncode=self.bert_tokenizer.encode(''.join(text2))  # 正规xlnet编码
        xlnetEncode2=self.bert_tokenizer.tokenize(''.join(text2))  # 正规xlnet编码 # 这个地方是大小写问题,无所谓.直接过.
        wordpiece_ids  # 原始编码
        # print(1)


        # 其实就是wordpiece_ids 去掉中间的19即可.
        out_qianzhui=wordpiece_ids[:4]
        token_wordpiece_ids # 改对这个进行处理.=  这里面居然有空数组!!!!!!!!!!!
        # token_wordpiece_ids=[out_qianzhui]+ token_wordpiece_ids# 改对这个进行处理.=
        offset2=[]
        out2=[]
        out2.append(out_qianzhui)
        for i in token_wordpiece_ids:
            if [j for j in i if j !=19]:
               out2.append([j for j in i if j !=19])
        savingfordebug_shuju=out2


        offsets4=[]
        tmp=0
        for i in out2[:-1]:
            tmp+=len(i)
            offsets4.append(tmp)

        [token.text.lower()
         if self._do_lowercase and token.text not in self._never_lowercase
         else token.text
         for token in tokens]

        # print(11111111)
        # pass


        mask = [1 for _ in offsets4]
        # 最后进行flatten
        out3=[]
        for i in out2:
            out3+=i
        out22=out3

# 替换为我自己优化的版本.
        return {index_name: out22,
                f"{index_name}-offsets": offsets4,
                "mask": mask}




        return {index_name: wordpiece_ids,
                f"{index_name}-offsets": offsets,
                "mask": mask}

    def _add_start_and_end(self, wordpiece_ids: List[int]) -> List[int]:
        return self._start_piece_ids + wordpiece_ids + self._end_piece_ids

    def _extend(self, token_type_ids: List[int]) -> List[int]:
        """
        Extend the token type ids by len(start_piece_ids) on the left
        and len(end_piece_ids) on the right.
        """
        first = token_type_ids[0]
        last = token_type_ids[-1]
        return ([first for _ in self._start_piece_ids] +
                token_type_ids +
                [last for _ in self._end_piece_ids])

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name, f"{index_name}-offsets", f"{index_name}-type-ids", "mask"]


class PretrainedBertIndexer(WordpieceIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    truncate_long_sequences : ``bool``, optional (default=``True``)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    """

    def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512,
                 max_pieces_per_token=5,
                 is_test=False,
                 truncate_long_sequences: bool = True,
                 special_tokens_fix: int = 0) -> None:
        if pretrained_model.endswith("-cased") and do_lowercase:
            logger.warning("Your BERT model appears to be cased, "
                           "but your indexer is lowercasing tokens.")
        elif pretrained_model.endswith("-uncased") and not do_lowercase:
            logger.warning("Your BERT model appears to be uncased, "
                           "but your indexer is not lowercasing tokens.")
# 根据xlnet这个名字,获取token数据.里面有url,可以自动下载, transformer库包可以直接下载这些模型提供使用.
        bert_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, do_lower_case=do_lowercase, do_basic_tokenize=False)
#  bert_tokenizer        bert_tokenizer.get_vocab() 调取整个字典!!!!!!!!
        # to adjust all tokenizers
        if hasattr(bert_tokenizer, 'encoder'):
            bert_tokenizer.vocab = bert_tokenizer.encoder
        if hasattr(bert_tokenizer, 'sp_model'): # sentence_piece model
            bert_tokenizer.vocab = defaultdict(lambda: 1) #当遇到没见过的就当他是句子开始.
            for i in range(bert_tokenizer.sp_model.get_piece_size()): # 对sp里面的数据进行index编码
                bert_tokenizer.vocab[bert_tokenizer.sp_model.id_to_piece(i)] = i

        if special_tokens_fix: # 是否进行token修改. # 就是添加一个位置.
            bert_tokenizer.add_tokens([START_TOKEN])
            bert_tokenizer.vocab[START_TOKEN] = len(bert_tokenizer) - 1

        if "roberta" in pretrained_model:
            bpe_ranks = bert_tokenizer.bpe_ranks
            byte_encoder = bert_tokenizer.byte_encoder
        else:
            bpe_ranks = {}
            byte_encoder = None
#调用父类的构造方法,来进行初始化参数, 其中使用的参数就都是上面得到的那些数据.
        super().__init__(vocab=bert_tokenizer.vocab,
                         bpe_ranks=bpe_ranks,
                         byte_encoder=byte_encoder,
                         wordpiece_tokenizer=bert_tokenizer.tokenize,
                         namespace="bert",
                         use_starting_offsets=use_starting_offsets,
                         max_pieces=max_pieces,
                         max_pieces_per_token=max_pieces_per_token,
                         is_test=is_test,
                         do_lowercase=do_lowercase,
                         never_lowercase=never_lowercase,
                         start_tokens=["[CLS]"] if not special_tokens_fix else [],  # bert里面句子第一个词记做cls. classification
                         end_tokens=["[SEP]"] if not special_tokens_fix else [], # 最后一个词记做sep,这个地方注意一下后续是不是大写,因为vocab里面的东西是小写的.
                         truncate_long_sequences=truncate_long_sequences)
